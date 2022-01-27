import time
import cv2
from facenet_pytorch import MTCNN
from scipy.misc import face
from config import USE_MTCNN
import re
from utils import Annotator

def mtcnn():
    prev_frame_time = 0
    new_frame_time = 0

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 460)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)
    while True:
        _, frame = cap.read()
        boxes, confs, points_list = mtcnn.detect(frame, landmarks=True)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        fps = f'FPS: {str(int(fps))}'
        annotator = Annotator(frame, 5, 25, pil=True)
        annotator.text((0, 30), fps, (0, 255, 0))
        if boxes is None:
            confidence = 0
            boxes = []
                
        for conf, box in zip(confs, boxes):
            if conf < 0.8:
                continue
            confidence = '%.3f'%(conf)
            bbox = list(map(int,box.tolist()))
            
            if boxes is not None:
                
                annotator.box_label(bbox, str(confidence))
        if len(boxes) == 0:
            frame = frame
        else:    
            frame = annotator.result()
        
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def haar():
    prev_frame_time = 0
    new_frame_time = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
        if faces is None or levelWeights == ():
            levelWeights = ()
            confidence = 0
            faces = []
           
        confs = levelWeights
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = f'FPS: {str(int(fps))}'
        
        annotator = Annotator(frame, 5, 25, pil=True)
        annotator.text((0, 30), fps, (0, 255, 0))
        for conf, (x, y, w, h) in zip(confs, faces):
            box = [x, y, x + w, y + h]
            
            conf = conf / 10
            if conf > 1:
                conf = 1
            confidence = '%.3f'%(conf)
            
            if faces is not None:
                annotator.box_label(box, str(confidence))
        if len(faces) == 0:
            frame = frame
        else:    
            frame = annotator.result()
        cv2.imshow('Webcam Check', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if USE_MTCNN:
        mtcnn()
    else:
        haar()
