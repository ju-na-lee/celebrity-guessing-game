import os 
import cv2 as cv
import numpy as np
import random as r
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import Canvas, Label

# Global variables
DIR = r'/Users/junaleeiscool/Desktop/Project4/dataset'
haar_cascade = cv.CascadeClassifier('haar_face.xml') 

# List of celebrities
celebrities = [] 
for i in os.listdir(DIR):
    celebrities.append(i)

features = [] #image array of faces
labels = [] #who's face in index format

#Takes in an image path and turns them grayscale, finds region of interest
def create_train():
    for cel in celebrities:
        path = os.path.join(DIR, cel)
        # Finds celebrity as an index in DIR
        label = celebrities.index(cel)
    
        img_path = []
        for img in os.listdir(path):
            img_path.append(os.path.join(path, img))

            """ for i in range(len(img_path)):
                if img_path[i].endswith('.DS_Store'):
                    del img_path[i] """
        
        # Changes image to grayscale 
        for i in img_path:
            img_array = cv.imread(i)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #turns grayscale

            # Haar_cascade classifier
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
            
            # Gets region of interest (ROI) and crops image 
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y: y+h, x: x+w] #crop image
                features.append(faces_roi)
                labels.append(label)

create_train()
print("---------- Training complete ----------")

# Training the recognizer with features & labels list
face_recognizer = cv.face.LBPHFaceRecognizer_create() #uses local binary patterns (LBPs) 
features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer.train(features, labels)
face_recognizer.save('celebrity_game_trained.yml') #saves as a yaml source file 

# Reading the trained Haar cascade 
face_recognizer.read('celebrity_game_trained.yml')

# Root frame 
root = ctk.CTk() 
root.geometry("700x700") 
root.title("Guess the Celebrity")
title_label = ctk.CTkLabel(root, text = "Guess Who?", font = ctk.CTkFont(size = 30, weight = "bold"))
title_label.pack(pady = 20)

# Game description
description = ctk.CTkLabel(root, text = "This is a celebrity guessing game. \n"
                           "The image dataset was taken from Kaggle by Vishesh Thakur: \n"
                           "kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset \n\n"
                           "Haar cascade was used to train 1800+ images for face recognition. \n"
                           " \n"
                           "github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml\n"
                           "Customtkinter library was used to code the GUI. \n\n"
                           "Learning objectives: OpenCV, face recognition, image processing, Tkinter GUI. \n\n"
                           "ENGR 130 Project 4 \n" 
                           "Author: Ju Na Lee (April 2023)", font = ctk.CTkFont(size = 15)) #EXPAND THE DESCRIPTION 

description.pack(pady = 20)
 
#random image picker 
def RIG(DIR, subjects):
   """ Random image picker.

   Keyword arguments:
   DIR -- folder directory with images
   subjects -- os.listdir(DIR), the different celebrities 
   """
   # First random picks celebrity
   ind = r.choice(subjects) 
   path1 = os.path.join(DIR, ind)
   files = os.listdir(path1) 

   # Deletes DS Store files (no directory)
   for i in range(len(files)):
         if files[i].endswith('.DS_Store'):
               del files[i]

    # Second random picks image 
   img = r.choice(files) 
   path2 = os.path.join(path1, img)

   return(path2, img)

#takes in user input
def user_input():
    user = ctk.CTkEntry(root) #add placeholder text
    user.pack(pady = 10)
    input = user.get()

    return(input)

""" #displays result using user input
def dispOutput(input, cel): #logic is incorrect, input the face recognition program in here
    if input == cel:
        result = "You are correct :)"
    else:
        result = "Incorrect :/"

    print_res = ctk.CTkLabel(root, text = result, ) """
    
"""     button = ctk.CTkButton(root, width = 300, height = 40, text = "Next Round", font = ctk.CTkFont(size = 15), command = )
    button.pack(padx = 100, pady = 20)  """
    
#displays image and resizes to 450 by 450
def show_img():
    img_path, cel = RIG(DIR, celebrities)
    #might need canvas
    img = Image.open(img_path)
    resize = img.resize((450, 450))
    img1 = ImageTk.PhotoImage(resize)
    label = Label(root, image = img1)
    label.image = img1 #https://github.com/ythy/blog/issues/302
    label.pack(pady = 20)

    instr = ctk.CTkLabel(root, text = 'Press Enter key when done.', font = ctk.CTkFont(size = 15))
    instr.pack(pady = 5)

    #root.bind('<Return>', dispOutput(user_input(), cel))

#transition from home to game frame 
def home2game():
    description.pack_forget()
    button.pack_forget()
    show_img()

button = ctk.CTkButton(root, width = 300, height = 40, text = "Start Game", font = ctk.CTkFont(size = 15), command = home2game)
button.pack(pady = 20)

root.mainloop()

#TURN IT ALL INTO A CLASS!!!
#change image resizing to be by scale (not to scale right now)
#fix DS_Store issue -> go through loop in beginning of function 
#change image to show a portion?
#make a main function