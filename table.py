# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\users\cjs\ccc\table.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(908, 862)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(20, 60, 861, 531))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 3))
        self.tableWidget.setRowCount(12)
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.File = QtWidgets.QMenuBar(MainWindow)
        self.File.setGeometry(QtCore.QRect(0, 0, 908, 31))
        self.File.setDefaultUp(False)
        self.File.setNativeMenuBar(False)
        self.File.setObjectName("File")
        self.menuFile = QtWidgets.QMenu(self.File)
        self.menuFile.setObjectName("menuFile")
        self.menuEidt = QtWidgets.QMenu(self.File)
        self.menuEidt.setObjectName("menuEidt")
        self.menuView = QtWidgets.QMenu(self.File)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.File)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionView = QtWidgets.QAction(MainWindow)
        self.actionView.setObjectName("actionView")
        self.actionRedo = QtWidgets.QAction(MainWindow)
        self.actionRedo.setObjectName("actionRedo")
        self.actionZoon_In = QtWidgets.QAction(MainWindow)
        self.actionZoon_In.setObjectName("actionZoon_In")
        self.actionZoom_Out = QtWidgets.QAction(MainWindow)
        self.actionZoom_Out.setObjectName("actionZoom_Out")
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuEidt.addAction(self.actionView)
        self.menuEidt.addAction(self.actionRedo)
        self.menuView.addAction(self.actionZoon_In)
        self.menuView.addAction(self.actionZoom_Out)
        self.File.addAction(self.menuFile.menuAction())
        self.File.addAction(self.menuEidt.menuAction())
        self.File.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEidt.setTitle(_translate("MainWindow", "Eidt"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionView.setText(_translate("MainWindow", "Undo"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionZoon_In.setText(_translate("MainWindow", "Zoon In"))
        self.actionZoom_Out.setText(_translate("MainWindow", "Zoom Out"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

