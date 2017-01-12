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
#from dictionary import Database as db
from FaceRecognizer import *


global FR,tableDialog,db_save_path

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
        nameList = FR.list_database()
        print('nameList', nameList)
        name_sz = len(nameList)
        try:
            self.ui.TableHolder.cellChanged.disconnect(self.nameEdit)
        except:
            pass
        self.ui.TableHolder.clear()
        self.ui.TableHolder.setColumnCount(2)
        self.ui.TableHolder.setHorizontalHeaderItem(0, widgets.QTableWidgetItem("Index"))
        self.ui.TableHolder.setHorizontalHeaderItem(1, widgets.QTableWidgetItem("Name"))
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
        nameList = FR._database.identity_list

        new_name = self.ui.TableHolder.item(x,y).text()
        # data.edit(x,name)
        if new_name=="":
            self.ui.TableHolder.item(x,y).setText(nameList[x])
        else:
            FR.rename_identity(nameList[x],new_name)
            FR.save_database(db_load_path)
            self.buildTable()

    def removeName(self):
        test = widgets.QTableWidget()
        test.currentRow()
        cR = self.ui.TableHolder.currentRow()
        del_name = self.ui.TableHolder.item(cR,1).text()
        print('CR =',cR)
        if cR >=0:
            FR.remove_identity(del_name)
            FR.save_database(db_load_path)
            self.buildTable()
        else:
            widgets.QMessageBox.warning(self, 'None Select Any Row', "You did't select any row\nData Update Failed!!")

class AddDialog(widgets.QDialog):
    def __init__(self, frame,embs, parent=None):
        super(AddDialog, self).__init__(parent)
        self.ui = Ui_AddDialog()
        self.ui.setupUi(self)
        self.setPicture(frame)
        self.accepted.connect(self.ADialogAccept)
        self.setWindowTitle('Add New Identity')
        self.current_embs = embs
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
        newName = self.ui.lineEdit.text()
        print(newName)
        #Prevent Stupid Error
        if newName=='':
            widgets.QMessageBox.warning(self,'Empty Warning','You did not enter anything!!!\nData Update Failed!!')
        else:
            print('saving new data')
            # Save message and feature in database.

            FR.add_identity(newName,self.current_embs)

            #Dump database
            FR.save_database(db_save_path)
            # data.add_one(message,np.array([23234,34,32344]))

            # Trigger TableHolder to update
            # Reload data
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


def frameCropper(frame,bb):
    x,y,w,h = bb[0:4]
    x = int(x); y = int(y); w = int(w); h = int(h)
    frame_crop = frame[y:h,x:w]

    print("frame = ",type(frame))
    print("frame_crop = ",type(frame_crop))
    frame_crop.astype('uint8')
    # frame_crop = frame

    frame_crop = cv2.resize(frame_crop,(160,160))
    print(type(frame_crop))
    return frame_crop


if __name__ == "__main__":
    import sys
    app = widgets.QApplication(sys.argv)



    # Face Detection
    # cascPath = 'haarcascade_frontalface_default.xml'
    # faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    video_capture.set(3,1280)
    video_capture.set(4,720)

    facenet_model_dir = './pretrained_models/FaceNet/'
    mtcnn_model_dir = './pretrained_models/MTCNN/'
    database_verbose = False
    db_load_path = './data/database/database.pkl'
    db_save_path = './data/database/database.pkl'
    FR = FaceRecognizer(facenet_model_dir,
                        mtcnn_model_dir,
                        db_load_path=db_load_path,
                        database_verbose=database_verbose,
                        resize_factor = 0.2)
    # Build the model
    tic = time.clock()
    FR.build()
    toc = time.clock()
    build_time = toc - tic
    print('build time: {}'.format(build_time))
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Camera', clickEvent)
    cv2.moveWindow('Camera', 0, 0)
    tableDialog = Table()




    while True:
        # Capture frame-by-frame, (height,width) = 720,1280
        ret, frame = video_capture.read()
        # inference
        tic = time.clock()
        bb, bb_names, image = FR.inference(frame)
        cur_embeds = FR.get_current_embeddings()
        
        toc = time.clock()

        print('{}'.format(toc - tic))

#        print(FR.list_database())

##################################
        # box clicked detection
        cond = False
        if bb is not None:
            for box_ind, shit in enumerate(bb):
                x, y, w, h  = shit[0:4]
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv.PutText(img, text, org, font, color) -> None
            # cv2.putText(frame,bb_names[box_ind],(x+w,y+h),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0, 0, 255),3,8)

                if (ix >= x and ix <= w):
                    if (iy >= y and iy <= h):
                        cond = True
                        chosen_ind = box_ind


            if (click_state):
                click_state = False
                ix = -1
                iy = -1
                if cond:
#                    print('in')
                    addDialog = AddDialog(frameCropper(image,bb[chosen_ind]), cur_embeds[chosen_ind,:])
                else:
                    pass
#                    print("out")
##########################
        # Display the resulting frame
        cv2.imshow('Camera', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    sys.exit(app.exec_())
