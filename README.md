# ANAL
AI Number-Plate Analytics Link
Code for my AI Number-Plate Analytics Link (ANAL) software for the Asus Tinker Edge T

This is going to be a WIP for awhile.
I can't begin to list down all the dependencies and packages required to make this work because...I made this over 2 years ago, revisited the code, and for the life of me, cannot get it work properly.
But I'll eventually get there.

For now, here are the rough steps to get this thing work (and it's not even half of it).

Install OpenCV4
Install Tesseract OCR
Install Tesseract OCR Dev

On Windows:
Download Python 3.9 (3.10 onwards isn't compatible yet)
pip install pyqt5
pip install pyqt5-tools
open C:\Python39\Lib\site-packages\qt5_applications\Qt\bin\designer.exe

In Designer:
File > New > Main Window
Geometry = 840x480 (change x,y to 0,30 later)
Add QLabel for video (640x480)
Add Table Widget (640,0,200,480)
Remove Menu
Remove Window

In CMD:
cd C:\Python39\Lib\site-packages\qt5_applications\Qt\bin\
pyuic5 -x elpr.ui -o elpr.py
(x to generate Python code to display the class, o for output file)

Transfer elpr.py to ATET:
cd C:\Python39\Lib\site-packages\qt5_applications\Qt\bin
mdt push elpr.py
sudo rm -r '\home\mendel'
sudo cp '\home\mendel\elpr.py' elpr.py
sudo rm '\home\mendel\elpr.py'

In ATET:
cd home\mendel
sudo nano rego_simple.csv (contents=p,r,s)

sudo apt install libtesseract-dev
sudo apt-get install python3-pyqt5
sudo pip3 install openalpr
sudo apt-get install python3-pandas
sudo apt-get install python3-bs4
sudo apt-get install xorg
xhost local:root
sudo python3 elpr.py

Stuck:
CURL missing
