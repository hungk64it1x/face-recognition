import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def UpdateDB(id):
    print(id)
    path2images = r"TrainingImage"
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    
    harcascadePath = r"haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    try:
        recognizer.read(r"TrainingImageLabel\Trainner.yml")
    except:
        print('No exist database')
    faces, Id = getImagesAndLabels(f"TrainingImage\{id}")
    Thread(target = recognizer.update(faces, np.array(Id))).start()
    Thread(target = counter_img(f"TrainingImage\{id}")).start()
    recognizer.save(r"TrainingImageLabel\Trainner.yml")

def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        # print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

