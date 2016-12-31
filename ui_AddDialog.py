# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AddDialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AddDialog(object):
    def setupUi(self, AddDialog):
        AddDialog.setObjectName("AddDialog")
        AddDialog.resize(274, 310)
        self.gridLayout = QtWidgets.QGridLayout(AddDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(AddDialog)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(AddDialog)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.pictureHolder = QtWidgets.QLabel(AddDialog)
        self.pictureHolder.setObjectName("pictureHolder")
        self.verticalLayout.addWidget(self.pictureHolder)
        self.buttonBox = QtWidgets.QDialogButtonBox(AddDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(AddDialog)
        self.buttonBox.accepted.connect(AddDialog.accept)
        self.buttonBox.rejected.connect(AddDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(AddDialog)

    def retranslateUi(self, AddDialog):
        _translate = QtCore.QCoreApplication.translate
        AddDialog.setWindowTitle(_translate("AddDialog", "Dialog"))
        self.label.setText(_translate("AddDialog", "Who are you :"))
        self.pictureHolder.setText(_translate("AddDialog", "text"))

