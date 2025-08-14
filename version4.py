import os 
import cv2 as cv
import numpy as np
import random as r
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import Canvas, Label

#------ Part 1: Implementing trained algorithm
# Global variables
DIR = r'/Users/junaleeiscool/Desktop/Project4/dataset'
haar_cascade = cv.CascadeClassifier('haar_face.xml') 

# List of celebrities
celebrities = [] 
for i in os.listdir(DIR):
    celebrities.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml') #classifier
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Reading the trained Haar cascade 
face_recognizer.read('celebrity_game_trained.yml')

# Training the recognizer with features & labels list
face_recognizer = cv.face.LBPHFaceRecognizer_create() #uses local binary patterns (LBPs) 

#------ Part 2: GUI design 
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

# Takes in user input
def user_input(label, faces_rect, confidence, img):
    user = ctk.CTkEntry(root)
    user.pack(pady=10)
    input = user.get()

    def on_return(event):
        dispOutput(input, label, faces_rect, confidence, img)

    user.bind('<Return>', on_return)

def haar_res(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
    
    return(gray, faces_rect)

def ROI(gray, faces_rect):
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y: y + h, x: x + w]

        # Compute the confidence level
        confidence = 1.0 - float(w * h) / float(img.shape[0] * img.shape[1])
        label = face_recognizer.predict(faces_roi)

    return(x, y, w, h)

#displays result using user input
def dispOutput(input, label, faces_rect, confidence, img): #logic is incorrect, input the face recognition program in here
    input = user_input(label, faces_rect, confidence, img)
    if input == label:
        result = "You are correct :)"
        print_res = ctk.CTkLabel(root, text = result, font = ctk.CTkFont(size = 10), text_color = (255, 0, 0)) 
        print_res.pack(pady = 10)
        
    else:
        result = "Incorrect :/"
        print_res = ctk.CTkLabel(root, text = result, font = ctk.CTkFont(size = 10), text_color = (0, 255, 0)) 
        print_res.pack(pady = 10)
    
    gray, faces_rect = haar_res(img)
    
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y: y + h, x: x + w]
        cv.putText(img, str(label), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness = 1)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 1)
   
    # Displaying OpenCV image in Tkinter
    b, g, r = cv.split(img)
    img = cv.merge((r, g, b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image = im)

    label = Label(root, image = imgtk)
    label.image = imgtk
    label.pack(pady = 20)

""" button = ctk.CTkButton(root, width = 300, height = 40, text = "Next Round", font = ctk.CTkFont(size = 15), command = )
    button.pack(padx = 100, pady = 20)"""
    
# Displays image and resizes to 450 by 450
def show_img():
    img_path, cel = RIG(DIR, celebrities)
    
    img = cv.imread(img_path)
    b, g, r = cv.split(img)
    img = cv.merge((r, g, b))

    im = Image.fromarray(img)
    resize = im.resize((450, 450))
    imgtk = ImageTk.PhotoImage(image = resize)
    label = Label(root, image = imgtk)
    label.image = imgtk
    label.pack(pady = 20)  
    
    faces_rect, confidence, label, img = haar_res(img_path)

    instr = ctk.CTkLabel(root, text = 'Press Enter key when done.', font = ctk.CTkFont(size = 15))
    instr.pack(pady = 10)
    user_input(label, faces_rect, confidence, img)
        
#transition from home to game frame 
def home2game():
    description.pack_forget()
    button.pack_forget()
    show_img()

button = ctk.CTkButton(root, width = 300, height = 40, text = "Start Game", font = ctk.CTkFont(size = 15), command = home2game)
button.pack(pady = 20)

root.mainloop()

#Notes: 
#TURN IT ALL INTO A CLASS!!!
#change image resizing to be by scale (not to scale right now)
#fix DS_Store issue -> go through loop in beginning of function 
#change image to show a portion?
#make a main function
#make gray and image ROI into a separate function
#ADD SOME INTRODUCTION WITH FIXES & ERRORS
#change min neighbors for algorithm sensitivity