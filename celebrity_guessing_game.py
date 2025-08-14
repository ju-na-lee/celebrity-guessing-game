import os
import numpy as np
import cv2 as cv

#Global Variables
DIR = r'/Users/junaleeiscool/Desktop/Project4/dataset'

celebrities = [] 
for i in os.listdir(DIR):
    celebrities.append(i)

features = [] #image array of faces
labels = [] #who's face in index format

haar_cascade = cv.CascadeClassifier('haar_face.xml') #face classifier

#Takes in an image path and turns them grayscale, finds region of interest
def create_train():
    for cel in celebrities:
        path = os.path.join(DIR, cel)
        label = celebrities.index(cel)
    
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
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 6)
            
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y: y+h, x: x+w] #crop image
                features.append(faces_roi)
                labels.append(label)

create_train()
print("---------- Training complete ----------")

#Training the recognizer with features & labels list
face_recognizer = cv.face.LBPHFaceRecognizer_create() #uses local binary patterns (LBPs) 
features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer.train(features, labels)
face_recognizer.save('celebrity_game_trained.yml') #saves as a yaml source file 

""" 
#Recognizing the face 
face_recognizer.read('celebrity_game_trained.yml')

#testing with Angelina Jolie 
img = cv.imread('/Users/junaleeiscool/Desktop/Project4/dataset/Angelina Jolie/001_fe3347c0.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Who is it?', img)

#detect face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y: y + h, x: x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label OG = {label}, Label = {celebrities[label]}, Confidence = {confidence}")

    cv.putText(img, str(celebrities[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), thickness = 1)

cv.imshow('Detected Face', img)

cv.waitKey(0)

#UPDATE IT TO BE A PORTION OF THEIR FACE??
#MAKE A GAME WINDOW USING TKINTER
#CONSIDER MINIMIZING SIZE OF IMAGE, FIX CONFIDENCE LEVEL""" 