from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtWidgets import QTableWidgetItem

import argparse
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

import threading
from datetime import datetime

from openalpr import Alpr

import pandas as pd
import numpy as np

class FrameGrabber(QtCore.QThread):
    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.default_model_dir = 'examples-camera/all_models'
        self.default_model = 'plates.tflite'
        self.default_labels = 'plates.txt'
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(self.default_model_dir,self.default_model))
        self.parser.add_argument('--labels', help='label file path',
                        default=os.path.join(self.default_model_dir, self.default_labels))
        self.parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
        self.parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 2)
        self.parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
        self.args = self.parser.parse_args()

        print('Loading {} with {} labels.'.format(self.args.model, self.args.labels))
        self.interpreter = make_interpreter(self.args.model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(self.args.labels)
        self.inference_size = input_size(self.interpreter)
        threading.Timer(10.0,self.take_snapshot).start()
    signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        global cv2_im
        global objs
        cap = cv2.VideoCapture(self.args.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                cv2_im = frame
                cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
                run_inference(self.interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(self.interpreter, self.args.threshold)[:self.args.top_k]
                cv2_im = self.append_objs_to_img(cv2_im, self.inference_size, objs, self.labels, False)
                image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                self.signal.emit(image)

    def append_objs_to_img(self, cv2_im, inference_size, objs, labels, take_photo):
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        for self.i, obj in enumerate(objs):
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            #percent = int(100 * obj.score)
            #label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            if take_photo:
                self.now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
                obj_filename = f'''{self.now}-{self.i}'''
                print(obj_filename)
                self.roi = cv2_im[y0:y1,x0:x1]
                obj_path = f'''./detected/{obj_filename}.jpg'''
                cv2.imwrite(f'''{obj_path}''',self.roi)
                self.run_alpr(obj_path)
                #self.file_list.append(f'''./detected/{obj_filename}.jpg''')
        return cv2_im
    
    def take_snapshot(self):
        print('take snapshot init')
        self.append_objs_to_img(cv2_im, self.inference_size, objs, self.labels, True)
        thread = threading.Timer(5.0,self.take_snapshot)
        thread.daemon = True
        thread.start()

    def run_alpr(self, plate_image):
        alpr=None
        try:
            alpr = Alpr('au', '/etc/openalpr/openalpr.conf', '/usr/share/openalpr/runtime_data/')

            if not alpr.is_loaded():
                print("Error loading OpenALPR")
            else:
                print("Using OpenALPR " + alpr.get_version())

            alpr.set_top_n(7)
            alpr.set_default_region("vic")
            alpr.set_detect_region(False)
            jpeg_bytes = open(plate_image, "rb").read()
            results = alpr.recognize_array(jpeg_bytes)

            # Uncomment to see the full results structure
            # import pprint
            # pprint.pprint(results)

            print("Image size: %dx%d" %(results['img_width'], results['img_height']))
            print("Processing Time: %f" % results['processing_time_ms'])

            i = 0
            j = 0
            for plate in results['results']:
                i += 1
                print("Plate #%d" % i)
                print("   %12s %12s" % ("Plate", "Confidence"))
                
                for candidate in plate['candidates']:
                    prefix = "-"
                    if candidate['matches_template']:
                        prefix = "*"
                        if j == 0:
                            self.check_rego(candidate['plate'])
                        print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
                        j += 1
        
        finally:
            if alpr:
                alpr.unload()

    def check_rego(self, plate_text):
        #write API or post request to check Government website
        print(f'''{plate_text} being checked''')
        print(f'''{plate_text} being written to CSV''')
        f=open('plates.csv','a')
        f.write(f'''{plate_text},o,o\n''')
        f.close()
    
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setupUi(self.MainWindow)
        self.grabber = FrameGrabber()
        self.grabber.signal.connect(self.updateFrame)
        self.grabber.start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(840, 480)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640, 0, 200, 480))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setHorizontalHeaderLabels(['Plate','Rego','Sanc'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Camera
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480)) #600,480
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")

    def refresh_table(self):
        df = pd.read_csv('plates.csv')
        df = df.iloc[-10:].sort_index(ascending=False)
        for each_row in range(len(df)):
            self.tableWidget.setItem(each_row,0,QTableWidgetItem(df.iloc[each_row][0]))
            self.tableWidget.setItem(each_row,1,QTableWidgetItem(df.iloc[each_row][1]))
            self.tableWidget.setItem(each_row,2,QTableWidgetItem(df.iloc[each_row][2]))
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ANAL"))

    @QtCore.pyqtSlot(QtGui.QImage)
    def updateFrame(self, image):
        self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def appExec(self):
        self.grabber.appExec()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    
    timer = QTimer()
    timer.timeout.connect(ui.refresh_table)
    timer.start(5000)
    
    sys.exit(app.exec_())
