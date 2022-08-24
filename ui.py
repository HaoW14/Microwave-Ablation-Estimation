# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import numpy as np
from torchvision import transforms as T
from PIL import Image
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from model import lstm, lstm2, Unet



class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(783, 510)
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(260, 90, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(110, 90, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 170, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(260, 170, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_3.setGeometry(QtCore.QRect(440, 170, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_4.setGeometry(QtCore.QRect(600, 170, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(440, 310, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(430, 240, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(170, 380, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(470, 380, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(250, 10, 311, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(260, 380, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(260, 450, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.time = QtWidgets.QLineEdit(Dialog)
        self.time.setGeometry(QtCore.QRect(580, 450, 141, 31))
        self.time.setObjectName("time")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(460, 450, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(220, 170, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(410, 170, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(560, 170, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.lineEdit_5 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_5.setGeometry(QtCore.QRect(580, 240, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.fib4 = QtWidgets.QLineEdit(Dialog)
        self.fib4.setGeometry(QtCore.QRect(580, 380, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.fib4.setFont(font)
        self.fib4.setObjectName("fib4")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 240, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(50, 290, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.comboBox_3 = QtWidgets.QComboBox(Dialog)
        self.comboBox_3.setGeometry(QtCore.QRect(260, 310, 141, 31))
        font = QtGui.QFont()
        font.setKerning(True)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.lineEdit_7 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_7.setGeometry(QtCore.QRect(580, 320, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setObjectName("lineEdit_7")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.pushButton_2.clicked.connect(self.get_needle_distance)   ##botton event
        self.pushButton.clicked.connect(self.get_time)

    def retranslateUi(self, Dialog): #from designer
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "image location"))
        self.label_2.setText(_translate("Dialog", "tumor size（cm）"))
        self.label_3.setText(_translate("Dialog", "power（W）："))
        self.label_4.setText(_translate("Dialog", "needle_distance（cm）："))
        self.label_5.setText(_translate("Dialog", "needle_type："))
        self.label_6.setText(_translate("Dialog", "FIB4  ："))
        self.label_7.setText(_translate("Dialog", "Planning for Microwave Ablation Thermal Field Parameters"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "1"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "2"))
        self.pushButton.setText(_translate("Dialog", "get time"))
        self.label_8.setText(_translate("Dialog", "pred_time(s)："))
        self.label_9.setText(_translate("Dialog", "a:"))
        self.label_10.setText(_translate("Dialog", "b:"))
        self.label_11.setText(_translate("Dialog", "c:"))
        self.pushButton_2.setText(_translate("Dialog", "get_needle_distance"))
        self.label_13.setText(_translate("Dialog", "Important adjoining structures lead to："))
        self.comboBox_3.setItemText(0, _translate("Dialog", "heat loss"))
        self.comboBox_3.setItemText(1, _translate("Dialog", "normal"))
        self.comboBox_3.setItemText(2, _translate("Dialog", "heat buildup"))

    def get_needle_distance(self):
        flag = 0
        try:            #check the input
            image_path = self.lineEdit.text()
            image = Image.open(image_path)
        except:
            flag = 1
            msg = "Sorry, the file " + self.lineEdit.text() + " does not exist."
            msg_box = QMessageBox(QMessageBox.Warning, 'warning', msg)
            msg_box.exec_()
        try:
            a = float(self.lineEdit_2.text())
            b = float(self.lineEdit_3.text())
            c = float(self.lineEdit_4.text())
        except:
            flag = 1
            msg = "data input error！"
            msg_box = QMessageBox(QMessageBox.Warning, 'warning', msg)
            msg_box.exec_()

        if flag == 0:
            input = np.array([a, b, c])
            norm = [4., 4, 4]
            input = input / norm
            input = torch.from_numpy(input.reshape(1, 1, 3).astype('float32'))
            model = lstm2()
            model.load_state_dict(torch.load("needle_distance.pkl", map_location='cpu'))
            with torch.no_grad():
                needle_distance = format(model(input).item() * 1.8, '.2f')
            self.lineEdit_5.setText(str(needle_distance))

    def get_time(self):
        flag = 0
        try:
            image_path = self.lineEdit.text()
            image = Image.open(image_path)
            transform = T.Compose([T.Resize([128,128]),T.Grayscale(1),T.ToTensor()])
            image = transform(image).float().reshape([1,1,128,128])
        except:
            flag = 1
            msg = "Sorry, the file " + self.lineEdit.text() + " does not exist."
            msg_box = QMessageBox(QMessageBox.Warning, 'warning', msg)
            msg_box.exec_()

        try:
            a = float(self.lineEdit_2.text())
            b = float(self.lineEdit_3.text())
            c = float(self.lineEdit_4.text())
            Adjacent_structure = float(self.comboBox_3.currentIndex()) - 1
            power = float(self.lineEdit_7.text())
            needle_distance = float(self.lineEdit_5.text())
            needle_type = float(self.comboBox_2.currentIndex()) + 1
            FIB4 = float(self.fib4.text())
        except:
            flag = 1
            msg = "data input error！"
            msg_box = QMessageBox(QMessageBox.Warning, 'warning', msg)
            msg_box.exec_()


        if flag == 0:  ##data input correctly
            input = np.array([a, b, c, power, needle_type, needle_distance, Adjacent_structure, FIB4])
            norm = [3.7, 3.4, 3.1, 60., 2., 1.8, 1., 22.57] #the defined norm during training
            input = input / norm
            input = torch.from_numpy(input.reshape(1, 1, input.shape[0]).astype('float32'))

            model = Unet(2)
            model.load_state_dict(torch.load("unet.pkl", map_location='cpu'))

            model.eval()
            with torch.no_grad():
                time,_ = model(image, input)

            self.time.setText(str(int(time.item() * 600)))
            if float(self.time.text()) > 480 and needle_distance == 0:
                msg = "time > 8min，double needle ablation is recommended！"
                msg_box = QMessageBox(QMessageBox.Warning, 'Note', msg)
                msg_box.exec_()
            if float(self.time.text()) < 240 and needle_distance != 0:
                msg = "time < 4min，Single needle ablation is recommended！"
                msg_box = QMessageBox(QMessageBox.Warning, 'Note', msg)
                msg_box.exec_()
            if float(self.time.text()) > 540:
                msg = "time > 9min，Multi-needle ablation is recommended！"
                msg_box = QMessageBox(QMessageBox.Warning, 'Note', msg)
                msg_box.exec_()








app =QApplication([])
window = QWidget()
ui = Ui_Dialog()

ui.setupUi(window)

window.show()

app.exec()
