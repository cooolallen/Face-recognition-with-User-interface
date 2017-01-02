# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Table.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Table(object):
    def setupUi(self, Table):
        Table.setObjectName("Table")
        Table.resize(334, 264)
        Table.setStyleSheet("Table{\n"
"background:url(./figures/background.jpg);\n"
"}")
        self.gridLayout_2 = QtWidgets.QGridLayout(Table)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.TableHolder = QtWidgets.QTableWidget(Table)
        self.TableHolder.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.TableHolder.setObjectName("TableHolder")
        self.TableHolder.setColumnCount(2)
        self.TableHolder.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(16)
        item.setFont(font)
        self.TableHolder.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(16)
        item.setFont(font)
        self.TableHolder.setHorizontalHeaderItem(1, item)
        self.horizontalLayout.addWidget(self.TableHolder)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.RemoveButton = QtWidgets.QPushButton(Table)
        self.RemoveButton.setObjectName("RemoveButton")
        self.verticalLayout.addWidget(self.RemoveButton)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Table)
        QtCore.QMetaObject.connectSlotsByName(Table)

    def retranslateUi(self, Table):
        _translate = QtCore.QCoreApplication.translate
        Table.setWindowTitle(_translate("Table", "Dialog"))
        item = self.TableHolder.horizontalHeaderItem(0)
        item.setText(_translate("Table", "Index"))
        item = self.TableHolder.horizontalHeaderItem(1)
        item.setText(_translate("Table", "Name"))
        self.RemoveButton.setText(_translate("Table", "Remove"))

