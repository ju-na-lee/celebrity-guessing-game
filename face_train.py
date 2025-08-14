#training openCV's face recognizer using haar cascade
import os
import cv2 as cv
import numpy as np

ppl = []
DIR = r'/Users/junaleeiscool/Desktop/Project4/faces_train'

for i in os.listdir(DIR):
    ppl.append(i) 
del ppl[0]

features = [] #image array of faces
labels = [] #who's face? 

haar_cascade = cv.CascadeClassifier('haar_face.xml') #classifier 

#This function takes image paths and turns them grayscale, draws rectangle face 
def create_train():
    for person in ppl:
        path = os.path.join(DIR, person)
        label = ppl.index(person)
    
        img_path = []
        for img in os.listdir(path):
            img_path.append(os.path.join(path, img))

            for i in range(len(img_path)):
                if img_path[i].endswith('.DS_Store'):
                    del img_path[i]

        for i in img_path:
            img_array = cv.imread(i)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #turns grayscale

            #haar_cascade classifier
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
            
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y: y+h, x: x+w] #crop image
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training done --------------")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#training the recognizer using features list and labels list
features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml') #saves as a yaml source file 

np.save('features.npy', features)
np.save('labels.npy', labels)