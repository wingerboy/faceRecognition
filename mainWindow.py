#!/usr/bin/python3
#-*-coding:utf-8-*-
# @Time    : 18-10-29 下午2:00
# @Author  : wingerliu
# @FileName: mainWindow.py

import sys
import cv2
import time
from main import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow



class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.timer_camera = QtCore.QTimer(self)  # 本地摄像头定时器
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.time = time

        self.cameraP.clicked.connect(self.local_camera_start)
        # self.connect(self.timer_camera, SIGNAL())
        # self.timer_camera.timeout.connect(self.show_local_camera)

    def local_camera_start(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                print("please open camera!")
                self.cameraP.setText('start')
            else:
                self.timer_camera.start(30)
                self.cameraP.setText('stop')


    def show_local_camera(self):
        flag, image = self.cap.read()
        pic_show = cv2.resize(image, (640, 480))
        pic_show = cv2.cvtColor(pic_show, cv2.COLOR_BGR2RGB)
        showimage = QtGui.QImage(pic_show.data, pic_show.shape[1], pic_show.shape[0], QtGui.QImage.Format_RGB888)
        self.graphicsView.close()
        self.cameraL.setPixmap(self.pix.fromImage(showimage))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
