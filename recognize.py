import datetime
import os
import time
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
from config import USE_MTCNN
from utils import Annotator

def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"TrainingImageLabel\Trainner.yml")
    harcascadePath = r"haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv(r"PersonDetail\PersonDetail.csv", encoding='cp1252')
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    prev_frame_time = 0
    new_frame_time = 0

    if USE_MTCNN:
        mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
        while(True):
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = f'FPS: {str(int(fps))}'
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes, conf, points_list = mtcnn.detect(frame, landmarks=True)
                annotator = Annotator(frame, 5, 25, pil=True)
                annotator.text((0, 30), fps, (0, 255, 0))
                if boxes is None:
                    confidence = 0
                    boxes = []
                
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    w, h = x2 - x1, y2 - y1
                    # frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0), 3)
                    # cv2.putText(frame, fps, (0, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if boxes is not None:
                
                        Id, conf = recognizer.predict(gray[y1: y2, x1: x2])
                        
                        if conf < 100:
                            
                            aa = df.loc[df['Id'] == Id]['Name'].values
                            confstr = "  {0}%".format(round(100 - conf))
                            tt = str(Id)+" - "+aa
                        else:
                            Id = '  Unknown  '
                            tt = str(Id)
                            confstr = "  {0}%".format(round(100 - conf))

                        if (100-conf) > 65:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            aa = str(aa)[2:-2]
                            attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                        

                        tt = str(tt)[2:-2]
                        pa = "[Pass]"
                        if(100-conf) > 65:
                            annotator.text(xy=(x1 + w - 5, y1 - 5), text=pa, txt_color=(255, 255, 255))
                        else:
                            annotator.text(xy=(x1 + w - 10, y1 - 5), text='', txt_color=(255, 255, 255))

                        if (100-conf) > 65:
                            annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 255, 0))
                        elif (100-conf) > 50:
                            annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 255, 255))
                        else:
                            annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 0, 255))

                        annotator.box_label(bbox, tt)
                    
                if len(boxes) == 0:
                    frame = frame
                else:    
                    frame = annotator.result()

                attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
                cv2.imshow('Tham gia', frame)
                if (cv2.waitKey(1) == ord('q')):
                    break
    else:
        while True:
            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = f'FPS: {str(int(fps))}'
            annotator = Annotator(frame, 5, 25, pil=True)
            annotator.text((0, 30), fps, (0, 255, 0))
            for bbox in boxes:
                x, y, w ,h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                bbox = [x1, y1, x2, y2]
                if boxes is not None:
                    Id, conf = recognizer.predict(gray[y1: y2, x1: x2])
                    
                    if conf < 100:

                        aa = df.loc[df['Id'] == Id]['Name'].values
                        confstr = "  {0}%".format(round(100 - conf))
                        tt = str(Id)+" - "+aa
                    else:
                        Id = '  Unknown  '
                        tt = str(Id)
                        confstr = "  {0}%".format(round(100 - conf))

                    if (100-conf) > 60:
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        aa = str(aa)[2:-2]
                        attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                    tt = str(tt)[2:-2]
                    pa = "[Pass]"
                    if(100-conf) > 60:
                        annotator.text(xy=(x1 + w - 5, y1 - 5), text=pa, txt_color=(255, 255, 255))
                    else:
                        annotator.text(xy=(x1 + w - 10, y1 - 5), text='', txt_color=(255, 255, 255))

                    if (100-conf) > 60:
                        annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 255, 0))
                    elif (100-conf) > 50:
                        annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 255, 255))
                    else:
                        annotator.text(xy=(x1 + 5, y1 + h - 5), text=str(confstr), txt_color=(0, 0, 255))

                    annotator.box_label(bbox, tt)

            if len(boxes) == 0:
                frame = frame
            else:    
                frame = annotator.result()

            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            cv2.imshow('Tham gia', frame)
            if (cv2.waitKey(1) == ord('q')):
                break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = r"Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_attendence()


