import cv2
import dlib
from PyQt5.QtGui import *


class FaceClassifier():
    def __init__(self, classifier='cnn'):
        self.detect_class = classifier
        self.detected_face = None
        self.save_flag = 0
        self.sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        if classifier == "cv":
            harr_filepath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # 系统安装的是opencv-contrib-python
            self.detector = cv2.CascadeClassifier(harr_filepath)  # 加载人脸特征分类器
        elif classifier == "dlib":
            self.detector = dlib.get_frontal_face_detector()
        elif classifier == "cnn":
            self.detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


    def reco_cv2(self, img):
        faces = self.detector.detectMultiScale(img, 1.3, 5)  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        return faces


    def reco_dlib(self, img):
        dets = self.detector(img, 1)
        return dets


    def reco_dlib_cnn(self, img):
        dets = self.detector(img, 1)
        return dets


    #输入图像为摄像头直接读取的图像
    def face_detect(self, img):
        show_0 = cv2.resize(img, (640, 480))
        show_1 = cv2.cvtColor(show_0, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(show_0, cv2.COLOR_BGR2GRAY)
        if self.detect_class == "cv":
            dets = self.reco_cv2(gray_image)
            if len(dets)==1:
                (x, y, w, h) = dets[0]
                self.detected_face = show_0[y:y+h,x:x+w]
                self.save_flag = 1
            for (x, y, w, h) in dets:
                cv2.rectangle(show_1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出人脸

        elif self.detect_class == "dlib":
            dets = self.reco_dlib(show_1)
            if len(dets)==1:
                self.detected_face = show_0[dets[0].left():dets[0].right(), dets[0].top():dets[0].bottom()]
                self.save_flag = 1
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                cv2.rectangle(show_1, (x2, x1), (y2, y1), (0, 255, 0), 2)  # 画出人脸

        elif self.detect_class == "cnn":
            dets = self.reco_dlib_cnn(show_1)
            if len(dets)==1:
                self.detected_face = show_0[dets.rect.left():dets.rect.right(), dets.rect.top():dets.rect.bottom()]
                self.save_flag = 1
            for i, d in enumerate(dets):
                x1 = d.rect.top() if d.rect.top() > 0 else 0
                y1 = d.rect.bottom() if d.rect.bottom() > 0 else 0
                x2 = d.rect.left() if d.rect.left() > 0 else 0
                y2 = d.rect.right() if d.rect.right() > 0 else 0
                cv2.rectangle(show_1, (x2, x1), (y2, y1), (0, 255, 0), 2)  # 画出人脸

        detect_image = QImage(show_1.data, show_1.shape[1], show_1.shape[0],
                              QImage.Format_RGB888)
        return detect_image


    #保存人脸及人脸标签 输入图像为cv格式
    def face_save(self, face, label):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 如果图片太大则压缩图像
        if face.shape[0] * face.shape[1] > 500000:
            face = cv2.resize(face, (0, 0), fx = 0.5, fy = 0.5)



        return True

    #返回一张图像多张人脸的 128D 特征, 输入img为分割后的人脸图像
    def get_128d_features(self, img):
        shape = self.sp(img)
        face_des.append(self.facerec.compute_face_descriptor(img, shape))
        else:
            face_des = []
        return face_des

