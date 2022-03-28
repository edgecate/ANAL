#xhost local:root
# Show Red for x
# Don't show duplicate plates
import threading
from multiprocessing import Pool
from edgetpu.detection.engine import DetectionEngine

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget,QTableWidgetItem, QDialog
from PyQt5.QtCore import QTimer, QSize
import time
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

import PIL
from PIL import Image
from openalpr import Alpr

import argparse
import os
import re
import time

import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup

class FrameGrabber(QtCore.QThread):
    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        # Main
        self.terminate = False
        self.file_list = []
        self.plate_list = []
        
        self.default_model_dir = '../all_models'
        self.default_model = 'plates.tflite'  #'plates.tflite' #'plates_v2.tflite' #'ssd_v2_lp_etpu_500.tflite'
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', help='.tflite model path',
                            default=os.path.join(self.default_model_dir,self.default_model))
        self.parser.add_argument('--top_k', type=int, default=3,
                            help='number of classes with highest score to display')
        self.parser.add_argument('--threshold', type=float, default=0.49,
                            help='class score threshold')
        self.args = self.parser.parse_args()
        self.engine = DetectionEngine(self.args.model)
        threading.Timer(5.0,self.take_snapshot).start()
    signal = QtCore.pyqtSignal(QtGui.QImage)
    
    def run(self):
        qformat = QImage.Format_RGB888
        global cap
        global frame
        global objs
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        while cap.isOpened():
            ret, frame = cap.read()
            frame=cv2.flip(frame,-1)
            if ret:
                self.pil_im = Image.fromarray(frame)
                objs = self.engine.detect_with_image(self.pil_im, threshold=self.args.threshold,
                                                      keep_aspect_ratio=True, relative_coord=True,
                                                      top_k=self.args.top_k)
                frame = self.append_objs_to_img(frame, objs, False)
                img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], qformat)
                img = img.rgbSwapped()
                self.signal.emit(img)
    
    def append_objs_to_img(self, frame, objs, snap):
        height, width, channels = frame.shape
        for self.i, obj in enumerate(objs):
            x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if snap == True:
                self.now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
                obj_filename = self.now + '-' + str(self.i)
                self.ROI = frame[y0:y1,x0:x1]
                cv2.imwrite('./detected/' + obj_filename+'.jpg', self.ROI)
                self.file_list.append('./detected/' + obj_filename+'.jpg')
                print('snap taken')
                self.run_alpr(self.file_list)
        return frame
    
    def take_snapshot(self):
        #print('loading camera')
        if self.terminate == False:
            self.append_objs_to_img(frame, objs, True)
            threading.Timer(5.0,self.take_snapshot).start()

    def run_alpr(self, file_list):
        df_check = pd.read_csv('rego_simple.csv')
        df_check = df_check.iloc[-15:].sort_index(ascending=False)
        for j, each_file in enumerate(self.file_list):
            print('Each file: ' + each_file)
            try:
                img = Image.open(each_file)
                alpr = Alpr('au', '/etc/openalpr/openalpr.conf', '/home/mendel/openalpr/runtime_data/')
                alpr.set_top_n(10)
                alpr.set_default_region('vic')
                results = alpr.recognize_file(each_file)
                #k=0
                for plate in results['results']:
                    #k += 1
                    #print('Plate #%d' % k)
                    print('  %12s %12s' % ('Plate', 'Confidence'))
                    for candidate in plate['candidates']:
                        #if candidate['matches_template']==1:
                        print(candidate)
                        prefix = '-'
                        if candidate['matches_template']==1:
                            prefix = '*'
                            print('  %s %12s%12f' % (prefix, candidate['plate'], candidate['confidence']))
                            if df_check['p'].isin([candidate['plate']]).any() == False:
                                print(candidate['plate'], 'not checked recently')
                                self.plate_list.append(candidate['plate'])
                            else:
                                print(candidate['plate'], 'recently checked - skipping check')
                            alpr.unload()
                            break
            except:
                print('image corrupted')
        self.file_list = []
        self.CheckRego(self.plate_list)
        self.plate_list = []
        

    def CheckRego(self, plate_list):
        if plate_list != []:
            for rego in plate_list:
                print('checking plate: ', plate_list)
                # VicRoads Rego Check URL
                url = 'https://www.vicroads.vic.gov.au/registration/buy-sell-or-transfer-a-vehicle/check-vehicle-registration/vehicle-registration-enquiry'
                get_headers = {
                    'accept': '*/*',
                    'accept-encoding': 'gzip,deflate',
                    'accept-language': 'en-US,en;q=0.8',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362'
                }
                # Create initial GET request session
                s = requests.Session()
                res = s.get(url, headers=get_headers, timeout=10)
                soup = BeautifulSoup(res.content, 'html.parser')

                # Body parameters for the POST request
                params = {
                    '__EVENTTARGET': '',
                    '__EVENTARGUMENT': '',
                    '__VIEWSTATE': soup.select('#__VIEWSTATE')[0]['value'],
                    '__VIEWSTATEGENERATOR': soup.select('#__VIEWSTATEGENERATOR')[0]['value'],
                    '__VIEWSTATEENCRYPTED': '',
                    'site-search-head': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$PersonalEmail$EmailAddress': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$PersonalPassword$SingleLine_CtrlHolderDivShown': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$OrganisationEmail$EmailAddress': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$OrganisationPassword$SingleLine_CtrlHolderDivShown': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$PartnerEmail$EmailAddress': '',
                    'ph_pagebody_0$phheader_0$_FlyoutLogin$PartnerPassword$SingleLine_CtrlHolderDivShown': '',
                    'ph_pagebody_0$phthreecolumnmaincontent_1$panel$VehicleSearch$vehicle-type': soup.select('#ph_pagebody_0_phthreecolumnmaincontent_1_panel_VehicleSearch_vehicle_type_car_truck')[0]['value'],
                    'ph_pagebody_0$phthreecolumnmaincontent_1$panel$VehicleSearch$vehicle-identifier-type': soup.select('#ph_pagebody_0_phthreecolumnmaincontent_1_panel_VehicleSearch_vehicle_identifier_type_registration_number')[0]['value'],
                    'ph_pagebody_0$phthreecolumnmaincontent_1$panel$VehicleSearch$RegistrationNumberCar$RegistrationNumber_CtrlHolderDivShown': rego,
                    'honeypot': '',
                    'ph_pagebody_0$phthreecolumnmaincontent_1$panel$btnSearch': 'Search'
                }

                # POST request with session, headers, and body
                res2 = s.post(url, data=params, headers={'Referer': res.url}, timeout=10)

                # Parse Response text into a BS object 
                input_soup = BeautifulSoup(res2.text, 'html.parser')

                # Open CSV which is the input and structure for our DF
                df = pd.read_csv(r'rego.csv')
                df_simple = pd.read_csv(r'rego_simple.csv')
                simp_data = []
                # Date and Time have to be obtained from a description in a H2 tag
                try:
                    date = input_soup.find('h2',{'class':'legend'}).text[(input_soup.find('h2',{'class':'legend'}).text).find(' as at ',0)+7:(input_soup.find('h2',{'class':'legend'}).text).find(' as at ',0)+7+10]
                    time = input_soup.find('h2',{'class':'legend'}).text[-10:-5]
                    row_data = [date, time]

                    details = input_soup.find_all('div',{'class':'display'})
                    for count, each_display in enumerate(details):
                        if count == 1:
                            status = each_display.text[0:each_display.text.find('-',0)].strip()
                            row_data.append(status)
                            if status == 'Current':
                                simp_data.append('o')
                            else:
                                simp_data.append('x')
                            expiry = each_display.text[-10:]
                            row_data.append(expiry)
                        elif count == 2:
                            row_data.append(each_display.text.split(None,1)[0])
                            row_data.append(each_display.text.split(None,2)[1])
                            row_data.append(each_display.text.split(None,3)[2])
                            row_data.append(each_display.text.split(None,4)[3])
                        else:
                            if count == 0:
                                simp_data.append(each_display.text.strip())
                            elif count == 7:
                                sanctions = each_display.text.strip()
                                if sanctions == 'None':
                                    simp_data.append('o')
                                else:
                                    simp_data.append('x')
                            row_data.append(each_display.text.strip())
                        if count == 9:
                            break

                    # Add to the last row of the DF
                    df.loc[len(df)] = row_data
                    df_simple.loc[len(df_simple)] = simp_data
                    # Export to CSV
                    df.to_csv(r'rego.csv', index=False)
                    df_simple.to_csv(r'rego_simple.csv', index=False)
                except:
                    print(plate_list[0], 'record not found')
                    df.loc[len(df)] = [datetime.now().strftime('%d/%m/%Y'),datetime.now().strftime('%H:%M'),plate_list[0],'RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF','RNF']
                    df_simple.loc[len(df_simple)] = [plate_list[0], 'x','x']
                    df.to_csv(r'rego.csv', index=False)
                    df_simple.to_csv(r'rego_simple.csv', index=False)
                print(df.iloc[-1:])

            
    def appExec(self):
        app.exec_()
        print('exiting')
        self.terminate = True
        cap.release()
        cv2.destroyAllWindows()

        
class Ui_MainWindow(QtWidgets.QMainWindow, QtCore.QThread):
    def __init__(self, MainWindow):
        super().__init__()
        self.terminate = False
        self.cwd = os.getcwd()
        self.video_size=QSize(640,480) #600,480
        
        self.MainWindow = MainWindow
        self.setupUi(self.MainWindow)
        self.grabber = FrameGrabber()
        self.grabber.signal.connect(self.updateFrame)
        self.grabber.start()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        MainWindow.setGeometry(0, 30, 840, 480) # x,y,w,h
        #icon = QtGui.QIcon()
        #icon.addPixmap(QtGui.QPixmap("e_logo_no_bg_icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Camera
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480)) #600,480
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")  
        
        # Table
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640, 0, 200, 480))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setHorizontalHeaderLabels(['Plate','Rego','Sanc'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)#Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.RefreshTable()
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Edgecate LPR"))
        
    @QtCore.pyqtSlot(QtGui.QImage)
    def updateFrame(self, image):
        self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def RefreshTable(self):
        df = pd.read_csv('rego_simple.csv')
        df = df.iloc[-15:].sort_index(ascending=False)
        for each_row in range(len(df)):
            self.tableWidget.setItem(each_row,0,QTableWidgetItem(df.iloc[each_row][0]))
            self.tableWidget.setItem(each_row,1,QTableWidgetItem(df.iloc[each_row][1]))
            if df.iloc[each_row][1] == 'x':
                self.tableWidget.item(each_row,1).setBackground(QtGui.QColor(255,0,0))
            self.tableWidget.setItem(each_row,2,QTableWidgetItem(df.iloc[each_row][2]))
            if df.iloc[each_row][2] == 'x':
                self.tableWidget.item(each_row,2).setBackground(QtGui.QColor(255,0,0))
                self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)#Stretch)
                self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
                self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        
    def appExec(self):
        self.Terminate = True
        self.grabber.appExec()
        
if __name__ == "__main__":
    import sys
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()

    # LP Table
    timer = QTimer()
    timer.timeout.connect(ui.RefreshTable)
    timer.start(5000)
    
    sys.exit(ui.appExec())
