# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Faces.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys,os
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog, QWidget
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QDir
from src import FaceCut, face_train, face_recognitation

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(601, 472)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolBox = QtWidgets.QToolBox(self.centralwidget)
        self.toolBox.setObjectName("toolBox")
        self.pic_get_page = QtWidgets.QWidget()
        self.pic_get_page.setGeometry(QtCore.QRect(0, 0, 581, 291))
        self.pic_get_page.setObjectName("pic_get_page")
        self.gridLayout = QtWidgets.QGridLayout(self.pic_get_page)
        self.gridLayout.setObjectName("gridLayout")
        self.input_name_label = QtWidgets.QLabel(self.pic_get_page)
        self.input_name_label.setObjectName("input_name_label")
        self.gridLayout.addWidget(self.input_name_label, 2, 0, 1, 1)
        self.input_video_path = QtWidgets.QLineEdit(self.pic_get_page)
        self.input_video_path.setObjectName("input_video_path")
        self.gridLayout.addWidget(self.input_video_path, 0, 2, 1, 1)
        self.input_name_lineEdit = QtWidgets.QLineEdit(self.pic_get_page)
        self.input_name_lineEdit.setObjectName("input_name_lineEdit")
        self.gridLayout.addWidget(self.input_name_lineEdit, 2, 2, 1, 1)
        self.slt_input_video = QtWidgets.QPushButton(self.pic_get_page)
        self.slt_input_video.setObjectName("slt_input_video")
        self.gridLayout.addWidget(self.slt_input_video, 0, 0, 1, 1)
        self.slt_classifer = QtWidgets.QPushButton(self.pic_get_page)
        self.slt_classifer.setObjectName("slt_classifer")
        self.gridLayout.addWidget(self.slt_classifer, 1, 0, 1, 1)
        self.classifer_path = QtWidgets.QLineEdit(self.pic_get_page)
        self.classifer_path.setObjectName("classifer_path")
        self.gridLayout.addWidget(self.classifer_path, 1, 2, 1, 1)
        self.face_cut = QtWidgets.QPushButton(self.pic_get_page)
        self.face_cut.setObjectName("face_cut")
        self.gridLayout.addWidget(self.face_cut, 2, 3, 1, 1)
        self.toolBox.addItem(self.pic_get_page, "")
        self.train_page = QtWidgets.QWidget()
        self.train_page.setGeometry(QtCore.QRect(0, 0, 581, 291))
        self.train_page.setObjectName("train_page")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.train_page)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.slt_cnn = QtWidgets.QPushButton(self.train_page)
        self.slt_cnn.setObjectName("slt_cnn")
        self.gridLayout_2.addWidget(self.slt_cnn, 1, 0, 1, 1)
        self.slt_input_picfiles = QtWidgets.QPushButton(self.train_page)
        self.slt_input_picfiles.setObjectName("slt_input_picfiles")
        self.gridLayout_2.addWidget(self.slt_input_picfiles, 0, 0, 1, 1)
        self.train = QtWidgets.QPushButton(self.train_page)
        self.train.setObjectName("train")
        self.gridLayout_2.addWidget(self.train, 2, 3, 1, 1)
        self.train_end_file = QtWidgets.QLineEdit(self.train_page)
        self.train_end_file.setObjectName("train_end_file")
        self.gridLayout_2.addWidget(self.train_end_file, 0, 2, 1, 1)
        self.cnn_path = QtWidgets.QLineEdit(self.train_page)
        self.cnn_path.setObjectName("cnn_path")
        self.gridLayout_2.addWidget(self.cnn_path, 1, 2, 1, 1)
        self.toolBox.addItem(self.train_page, "")
        self.face_re_page = QtWidgets.QWidget()
        self.face_re_page.setGeometry(QtCore.QRect(0, 0, 581, 291))
        self.face_re_page.setObjectName("face_re_page")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.face_re_page)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.re_search = QtWidgets.QPushButton(self.face_re_page)
        self.re_search.setObjectName("re_search")
        self.gridLayout_3.addWidget(self.re_search, 3, 2, 1, 1)
        self.model_path = QtWidgets.QLineEdit(self.face_re_page)
        self.model_path.setObjectName("model_path")
        self.gridLayout_3.addWidget(self.model_path, 0, 1, 1, 1)
        self.slt_model = QtWidgets.QPushButton(self.face_re_page)
        self.slt_model.setObjectName("slt_model")
        self.gridLayout_3.addWidget(self.slt_model, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.speed_button_2 = QtWidgets.QRadioButton(self.face_re_page)
        self.speed_button_2.setObjectName("speed_button_2")
        self.horizontalLayout.addWidget(self.speed_button_2)
        self.speed_button_4 = QtWidgets.QRadioButton(self.face_re_page)
        self.speed_button_4.setObjectName("speed_button_4")
        self.horizontalLayout.addWidget(self.speed_button_4)
        self.speed_button_8 = QtWidgets.QRadioButton(self.face_re_page)
        self.speed_button_8.setObjectName("speed_button_8")
        self.horizontalLayout.addWidget(self.speed_button_8)
        self.speed_button_16 = QtWidgets.QRadioButton(self.face_re_page)
        self.speed_button_16.setObjectName("speed_button_16")
        self.horizontalLayout.addWidget(self.speed_button_16)
        self.gridLayout_3.addLayout(self.horizontalLayout, 2, 0, 1, 3)
        self.slt_re_video = QtWidgets.QPushButton(self.face_re_page)
        self.slt_re_video.setObjectName("slt_re_video")
        self.gridLayout_3.addWidget(self.slt_re_video, 1, 0, 1, 1)
        self.re_video_lineEdit = QtWidgets.QLineEdit(self.face_re_page)
        self.re_video_lineEdit.setObjectName("re_video_lineEdit")
        self.gridLayout_3.addWidget(self.re_video_lineEdit, 1, 1, 1, 1)
        self.toolBox.addItem(self.face_re_page, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 601, 28))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.input_video = QtWidgets.QAction(MainWindow)
        self.input_video.setObjectName("input_video")
        self.input_model = QtWidgets.QAction(MainWindow)
        self.input_model.setObjectName("input_model")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.train_model = QtWidgets.QAction(MainWindow)
        self.train_model.setObjectName("train_model")
        self.search = QtWidgets.QAction(MainWindow)
        self.search.setObjectName("search")
        self.examine_pic = QtWidgets.QAction(MainWindow)
        self.examine_pic.setObjectName("examine_pic")
        self.set_pic_pos = QtWidgets.QAction(MainWindow)
        self.set_pic_pos.setObjectName("set_pic_pos")
        self.set_model_pos = QtWidgets.QAction(MainWindow)
        self.set_model_pos.setObjectName("set_model_pos")
        self.set_speed = QtWidgets.QAction(MainWindow)
        self.set_speed.setObjectName("set_speed")
        self.man = QtWidgets.QAction(MainWindow)
        self.man.setObjectName("man")
        self.menu.addAction(self.input_video)
        self.menu.addAction(self.input_model)
        self.menu.addSeparator()
        self.menu.addAction(self.exit)
        self.menu_2.addAction(self.examine_pic)
        self.menu_2.addAction(self.train_model)
        self.menu_2.addAction(self.search)
        self.menu_3.addAction(self.set_pic_pos)
        self.menu_3.addAction(self.set_model_pos)
        self.menu_3.addAction(self.set_speed)
        self.menu_4.addAction(self.man)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())

        self.retranslateUi(MainWindow)
        self.exit.triggered.connect(MainWindow.close)
        self.input_video.triggered.connect(self.open_video_file)
        self.slt_input_video.clicked.connect(self.open_video_file)
        self.slt_classifer.clicked.connect(self.select_claasifer)
        self.face_cut.clicked.connect(self.faceCut)
        self.slt_input_picfiles.clicked.connect(self.open_pics_file)
        self.train.clicked.connect(self.start_train)
        self.slt_model.clicked.connect(self.select_model)
        self.slt_re_video.clicked.connect(self.select_re_video)
        self.re_search.clicked.connect(self.re_video)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_name_label.setText(_translate("MainWindow", "输入样本名称"))
        self.slt_input_video.setText(_translate("MainWindow", "选择样本视频"))
        self.slt_classifer.setText(_translate("MainWindow", "选择分类器"))
        self.face_cut.setText(_translate("MainWindow", "开始获取"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.pic_get_page), _translate("MainWindow", "样本获取"))
        self.slt_cnn.setText(_translate("MainWindow", "CNN"))
        self.slt_input_picfiles.setText(_translate("MainWindow", "样本文件夹"))
        self.train.setText(_translate("MainWindow", "开始训练"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.train_page), _translate("MainWindow", "训练模型"))
        self.re_search.setText(_translate("MainWindow", "开始检索"))
        self.slt_model.setText(_translate("MainWindow", "选择模型"))
        self.speed_button_2.setText(_translate("MainWindow", "2X"))
        self.speed_button_4.setText(_translate("MainWindow", "4X"))
        self.speed_button_8.setText(_translate("MainWindow", "8X"))
        self.speed_button_16.setText(_translate("MainWindow", "16X"))
        self.slt_re_video.setText(_translate("MainWindow", "选择视频"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.face_re_page), _translate("MainWindow", "人物检索"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.menu_3.setTitle(_translate("MainWindow", "设置"))
        self.menu_4.setTitle(_translate("MainWindow", "帮助"))
        self.input_video.setText(_translate("MainWindow", "导入视频"))
        self.input_model.setText(_translate("MainWindow", "导入模型"))
        self.exit.setText(_translate("MainWindow", "退出"))
        self.train_model.setText(_translate("MainWindow", "训练模型"))
        self.search.setText(_translate("MainWindow", "视频检索"))
        self.examine_pic.setText(_translate("MainWindow", "查看样本"))
        self.set_pic_pos.setText(_translate("MainWindow", "样本存储位置"))
        self.set_model_pos.setText(_translate("MainWindow", "模型存储位置"))
        self.set_speed.setText(_translate("MainWindow", "检索速度设置"))
        self.man.setText(_translate("MainWindow", "查看使用帮助"))

    # 选择输入样本按钮
    def open_video_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择样本视频',
                                                  QDir.homePath(),
                                                  'Videos (*.mp4 *.ts *.avi *.mpeg *.mpg *.mkv *.VOB *.m4v)')
        if fileName != '':
            print(fileName)
            self.input_video_path.setText(fileName)

    # 选择分类器按钮
    def select_claasifer(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择分类器',
                                                  QDir.homePath(),
                                                  '*.xml')
        if fileName != '':
            print(fileName)
            self.classifer_path.setText(fileName)

    # 样本获取按钮
    def faceCut(self):
        people_name = self.input_name_lineEdit.text()
        save_path = r'/home/venidi/FaceRecognition/test/LinuxProject/faces/' + people_name
        # os.mkdir(save_path)
        vedio_path = self.input_video_path.text()
        classifer_path = self.classifer_path.text()
        print('vedio path->'+vedio_path)
        print('save_path->'+save_path)
        print('classifer_path->'+classifer_path)
        # FaceCut.face_cut(vedio_path,save_path,classifer_path)

        # 打开获取的样本，手动删除
        fileNames, _ = QFileDialog.getOpenFileNames(self, '选择要删除的图片', save_path, '*.jpg')
        print(fileNames)
        if fileNames != '':
            for name in fileNames:
                os.remove(name)

    def open_pics_file(self):
        fileDir = QFileDialog.getExistingDirectory(self, "选择文件夹", QDir.homePath())
        if fileDir != '':
            self.train_end_file.setText(fileDir)
            self.cnn_path.setText('18 层网络模型')

    # 模型训练
    def start_train(self):
        full_path = self.train_end_file.text()
        print('full_path->' + full_path)
        cut_full_path = full_path.split('/')
        # print('cut_full_path->' + cut_full_path)
        end_name =cut_full_path[-1]
        print('end_name->' + end_name)
        save_name = r'../models/' + end_name + '.model.h5'
        print('save_name->' + save_name)
        face_train.train(end_name,model_save_path=save_name)

    # 选择模型
    def select_model(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择模型',
                                                  QDir.homePath(),
                                                  '*.h5')
        if fileName != '':
            print(fileName)
            self.model_path.setText(fileName)

    # 选择视频
    def select_re_video(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择视频',
                                                  QDir.homePath(),
                                                  'Videos (*.mp4 *.ts *.avi *.mpeg *.mpg *.mkv *.VOB *.m4v)')
        if fileName != '':
            print(fileName)
            self.re_video_lineEdit.setText(fileName)

    # 开始检索
    def re_video(self):
        model_path = self.model_path.text()
        video_path = self.re_video_lineEdit.text()
        clifer_path = r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml'
        face_recognitation.face_re(model_path, video_path, clifer_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())