import cv2
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtCore
from PyQt5.QtGui import *
import PyQt5.QtWidgets as widgets
from PyQt5.QtGui import QKeyEvent
import time
import numpy as np


from ui_AddDialog import Ui_AddDialog
from ui_Table import Ui_Table
from dictionary import Database as db

global data, nameList, features, tableDialog

class Table(widgets.QDialog):
    def __init__(self,parent=None):
        super(Table, self).__init__(parent)
        self.ui = Ui_Table()
        self.ui.setupUi(self)
        self.buildTable()
        self.setWindowTitle('Identity Table')
        self.show()

        # Connection
        self.ui.RemoveButton.clicked.connect(self.removeName)

    def buildTable(self):
        nameList = data.name_list()
        name_sz = data.num_in_dict()

        self.ui.TableHolder.clear()
        self.ui.TableHolder.setColumnCount(2)
        self.ui.TableHolder.setColumnWidth(0, 50)
        self.ui.TableHolder.setColumnWidth(1, 120)
        self.ui.TableHolder.setRowCount(name_sz)
        # test = widgets.QTableWidget()
        # test.cellChanged().connect(self.nameEdit(x,y))
        # test.cellChanged(p_int,p_int_1)
        # test.item


        for key, name in enumerate(nameList):
            newitem = widgets.QTableWidgetItem(str(key+1))
            self.ui.TableHolder.setItem(key,0,newitem)
            self.ui.TableHolder.item(key,0).setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            newitem = widgets.QTableWidgetItem(name)
            self.ui.TableHolder.setItem(key,1,newitem)

        self.ui.TableHolder.cellChanged.connect(self.nameEdit)

    def nameEdit(self,x,y):
        # Edit Cell
        name = self.ui.TableHolder.item(x,y).text()
        data.edit(x,name)

    def removeName(self):
        test = widgets.QTableWidget()
        test.currentRow()
        cR = self.ui.TableHolder.currentRow()
        print('CR =',cR)
        if cR >=0:
            data.del_one(cR)
            self.buildTable()
        else:
            widgets.QMessageBox.warning(self, 'None Select Any Row', "You did't select any row\nData Update Failed!!")
class AddDialog(widgets.QDialog):
    def __init__(self, frame, parent=None):
        super(AddDialog, self).__init__(parent)
        self.ui = Ui_AddDialog()
        self.ui.setupUi(self)
        self.setPicture(frame)
        self.accepted.connect(self.ADialogAccept)
        self.setWindowTitle('Add New Identity')
        self.show()

    def setPicture(self, frame):
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

        height, width, channel = frame.shape
        print('height', height, 'width', width, 'channel', channel)
        bytesPerLine = 3 * width
        QI = QImage(frame, width, height, bytesPerLine, QImage.Format_RGB888)

        self.ui.pictureHolder.setScaledContents(True)
        self.ui.pictureHolder.setPixmap(QPixmap.fromImage(QI))

    def ADialogAccept(self):
        print('accept and do something')
        # self.ui.lineEdit.setText('gggggg')
        message = self.ui.lineEdit.text()
        print(message)
        #Prevent Stupid Error
        if message=='':
            widgets.QMessageBox.warning(self,'Empty Warning','You did not enter anything!!!\nData Update Failed!!')
        else:
            print('saving new data')
            # Save message and feature in database.
            # Reload data
            data.add_one(message,np.array([23234,34,32344]))

            # Trigger TableHolder to update
            tableDialog.buildTable()


# Click event definition.
click_state = False
ix,iy = -1,-1
def clickEvent(event,x,y,flags,param):
    global ix,iy,click_state

    if event == cv2.EVENT_LBUTTONDOWN:
        click_state = True
        ix,iy = x,y
        # print('x=',ix,'y=',iy,'flags=',flags)


def frameCropper(frame,faces,chosen):
    x,y,w,h = faces[chosen]
    frame_crop = frame[y:y+h,x:x+w]

    print("frame = ",type(frame))
    print("frame_crop = ",type(frame_crop))
    frame_crop.astype('uint8')
    # frame_crop = frame

    frame_crop = cv2.resize(frame_crop,(w,h))
    print(type(frame_crop))
    return frame_crop


if __name__ == "__main__":
    import sys
    app = widgets.QApplication(sys.argv)

    data = db()
    nameList = data.name_list()
    features = data.feat_list()
    tableDialog = Table()

    # Face Detection
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', clickEvent)

    while True:
        # Capture frame-by-frame, (height,width) = 720,1280
        ret, frame = video_capture.read()

        frame = cv2.resize(frame,(640,360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print(faces.shape)
        # Draw a rectangle around the faces
        cond = False
        # nameList = {0:"Allen1",1:"Allen2"}


        for box_ind,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv.PutText(img, text, org, font, color) -> None
            cv2.putText(frame,nameList[box_ind],(x+w,y+h),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0, 0, 255),3,8)

            if (ix >= x and ix <= x + w):
                if (iy >= y and iy <= y + h):
                    cond = True
                    chosen_ind = box_ind


        if (click_state):
            click_state = False
            ix = -1
            iy = -1
            if cond:
                addDialog = AddDialog(frameCropper(frame,faces,chosen_ind))
                # addDialog = AddDialog(frame)
            else:
                print("out")

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    sys.exit(app.exec_())