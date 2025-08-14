import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml') #classifier 

""" features = np.load('features.npy')
labels = np.load('labels.npy') """
  
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

ppl = []
DIR = r'/Users/junaleeiscool/Desktop/Project4/faces_train'

for i in os.listdir(DIR):
    ppl.append(i) 
del ppl[0]
 
img = cv.imread('/Users/junaleeiscool/Desktop/Project4/faces_train/EmmaStone/cinque.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Who is it?', img)

#Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y: y+h, x: x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f"{label}\n")
    print(f'Label = {ppl[label]}')

    cv.putText(img, str(ppl[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 1)

cv.imshow('Detected Face', img)

cv.waitKey(0)