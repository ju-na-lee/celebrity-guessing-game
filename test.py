import random as r
import cv2 as cv
import os

DIR = r'/Users/junaleeiscool/Desktop/Project4/dataset'
img = "/Users/junaleeiscool/Desktop/Project4/dataset/Angelina Jolie/001_fe3347c0.jpg"
haar_cascade = cv.CascadeClassifier('haar_face.xml') 

celebrities = [] 
for i in os.listdir(DIR):
    celebrities.append(i)

# Takes in an image path and turns them grayscale, finds region of interest
def create_train(celebrities, features, labels):
    for cel in celebrities:
        path = os.path.join(DIR, cel)
        # Finds celebrity as an index in DIR
        label = celebrities.index(cel)
    
        img_path = []
        
        for img in os.listdir(path):
            img_path.append(os.path.join(path, img))

            for i in range(len(img_path)):
                if img_path[i].endswith('.DS_Store'):
                    del img_path[i]

        # Changes image to grayscale 
        for i in img_path:
            img_array = cv.imread(i)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #turns grayscale

            # Haar_cascade classifier
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
            
            # Gets region of interest (ROI) and crops image 
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y: y+h, x: x+w] 
                features.append(faces_roi)
                labels.append(label)
    
    print("---------- Training complete ----------")