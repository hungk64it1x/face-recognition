import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

from sklearn.feature_extraction import image


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        for image_id in os.listdir(imagePath):
            image_id_path = os.path.join(imagePath, image_id)
            pilImage = Image.open(image_id_path).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(image_id.split('.')[1])
            faces.append(imageNp)
            Ids.append(Id)
    return faces, Ids

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    
    harcascadePath = r"haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels(fr"TrainingImage")
    Thread(target=recognizer.train(faces, np.array(Id))).start()
    recognizer.save(r"TrainingImageLabel"+os.sep+"Trainner.yml")
    
if __name__ == '__main__':
    TrainImages()

