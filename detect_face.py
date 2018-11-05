import sys
import cv2
import os
import datetime
import dlib
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from faceClassifier import FaceClassifier


facePath = './person_faces/'


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 需要定时器刷新摄像头界面
        self.cap = cv2.VideoCapture()
        self.cap_num = 0
        self.set_ui()  # 初始化UI界面
        self.slot_init()  # 初始化信号槽
        self.detect_flag = 0  # 人脸检测开关变量
        self.cap_face = None
        self.detector = FaceClassifier('dlib')

    def set_ui(self):
        # 布局设置
        self.layout_main = QHBoxLayout()  # 整体框架是水平布局
        self.layout_button = QVBoxLayout()

        # 按钮设置
        self.btn_open_cam = QPushButton('打开相机')        # self.btn_open_cam.move(10, 10)
        self.btn_detection_face = QPushButton('人脸检测')
        self.btn_capture_face = QPushButton('人脸捕获')
        self.btn_save_face = QPushButton('人脸保存')
        self.name_input = QTextEdit() #名字输入框
        self.quit = QPushButton('退出')
        self.name_label = QLabel('输入名字')
        self.name_label.setFixedSize(100, 20)
        # self.btn_close_cam.move(10, 30)

        # 显示视频
        self.label_show_camera = QLabel()
        self.label_move = QLabel()
        self.label_save_face = QLabel()
        self.label_move.setFixedSize(100, 50)
        self.name_input.setFixedSize(100,40)
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.label_save_face.setFixedSize(100, 100)
        self.label_save_face.setAutoFillBackground(False)

        # 布局
        self.layout_button.addWidget(self.btn_open_cam)
        self.layout_button.addWidget(self.btn_detection_face)
        self.layout_button.addWidget(self.btn_capture_face)
        self.layout_button.addWidget(self.btn_save_face)
        self.layout_button.addWidget(self.quit)
        self.layout_button.addWidget(self.name_label)
        self.layout_button.addWidget(self.name_input)
        self.layout_button.addWidget(self.label_save_face)
        self.layout_button.addWidget(self.label_move)

        self.layout_main.addLayout(self.layout_button)
        self.layout_main.addWidget(self.label_show_camera)


        self.setLayout(self.layout_main)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("人脸识别软件")


    # 信号槽设置
    def slot_init(self):
        self.btn_open_cam.clicked.connect(self.btn_open_cam_click)
        self.btn_detection_face.clicked.connect(self.detect_face)
        self.btn_capture_face.clicked.connect(self.capture_face)
        self.btn_save_face.clicked.connect(self.save_face)
        self.timer_camera.timeout.connect(self.show_camera)
        self.quit.clicked.connect(self.close)

    def btn_open_cam_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.cap_num)
            if flag == False:
                msg = QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.btn_open_cam.setText(u'关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.btn_open_cam.setText(u'打开相机')

    def show_camera(self):
        if self.detect_flag == 0:
            ret, self.image = self.cap.read()
            show = cv2.resize(self.image, (640, 480))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            # int height, Format format, QImageCleanupFunction cleanupFunction = 0, void *cleanupInfo = 0)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))
        else:
            ret_1, self.image_1 = self.cap.read()
            if ret_1:
                detect_image = self.detector.face_detect(self.image_1)
                self.label_show_camera.setPixmap(QPixmap.fromImage(detect_image))
            else:
                print("please open camera!!!")


    def detect_face(self):
        if self.detect_flag == 0:
            self.detect_flag = 1
            self.btn_detection_face.setText(u'关闭人脸检测')
        else:
            self.detect_flag = 0
            self.btn_detection_face.setText(u'人脸检测')


    def capture_face(self):
        self.cap_face = self.detector.detected_face  # BGR cv格式

        show_face = cv2.cvtColor(self.cap_face, cv2.COLOR_BGR2RGB)
        show_face = cv2.resize(show_face, (100, 100))
        show_face = QImage(show_face.data, show_face.shape[1], show_face.shape[0], QImage.Format_RGB888)
        self.label_save_face.setPixmap(QPixmap.fromImage(show_face))


    def save_face(self):
        label = self.name_input.toPlainText()

        # 判断输入框中是否有异常字符
        if label:
            if not os.path.exists(facePath + label):
                os.makedirs(facePath + label)
            time_now = datetime.datetime.now().strftime('%H%M%S')
            faceDir = facePath + label + '/' + label + '_' + time_now + '.jpg'
            cv2.imwrite(faceDir, self.cap_face)

            save_flag = self.detector.face_save(self.cap_face, label)
            if save_flag:
                self.name_input.setPlainText('')
                self.label_save_face.clear()
        else:
            print('请输入正确字符！！！')


    def closeEvent(self, QCloseEvent):

        reply = QMessageBox.question(self, u"Warning", "Are you sure quit ?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.cap.release()
            self.timer_camera.stop()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())