import csv
from telnetlib import X3PAD
import cv2
import os
import os.path
from config import USE_MTCNN
from facenet_pytorch import MTCNN

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def takeImages(Id, name):

    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        os.makedirs(f"TrainingImage\{Id}", exist_ok=True)
        if USE_MTCNN:
            mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 460)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)
            while(True):
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes, conf, points_list = mtcnn.detect(frame, landmarks=True)
                if boxes is None:
                    confidence = 0
                    boxes = []
                else:
                    confidence = '%.3f'%(conf[0])
                
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0), 3)
                    
                    sampleNum = sampleNum + 1
                    cv2.imwrite(f"TrainingImage/{Id}" + os.sep + name + "."+Id + '.' +
                                str(sampleNum) + ".jpg", gray[y1: y2, x1: x2])
                cv2.imshow('Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif sampleNum > 100:
                    break
        else:
            
            while(True):
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
                for(x,y,w,h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    sampleNum = sampleNum + 1
                    cv2.imwrite(f"TrainingImage/{Id}" + os.sep + name + "."+Id + '.' +
                                str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                    cv2.imshow('frame', frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum > 100:
                    break

        cam.release()
        cv2.destroyAllWindows()
        
        header=["Id", "Name"]
        row = [Id, name]
        if(os.path.isfile("PersonDetail"+os.sep+"PersonDetail.csv")):
            with open("PersonDetail"+os.sep+"PersonDetail.csv", 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(j for j in row)
            csvFile.close()
        else:
            with open("PersonDetail"+os.sep+"PersonDetail.csv", 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(i for i in header)
                writer.writerow(j for j in row)
            csvFile.close()
    else:
        if(is_number(Id)):
            print("Enter Alphabetical Name")
        if(name.isalpha()):
            print("Enter Numeric ID")
    return Id, name


