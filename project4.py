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

# Random image picker 
def RIG(subjects):
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

   return(path2, ind)

# GUI page
def GUI(celebrities):
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
    
    #------ Part 2: Implementing trained algorithm & finding ROI
    def haar_analysis(img_path):
        haar_cascade = cv.CascadeClassifier('haar_face.xml') # Classifier
        face_recognizer = cv.face.LBPHFaceRecognizer_create()  # Uses local binary patterns (LBPs)

        # Reading the trained Haar cascade  
        face_recognizer.read('celebrity_game_trained.yml')
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

        confidence = 0 # Initializing confidence value

        for (x, y, w, h) in faces_rect: 
            faces_roi = gray[y: y + h, x: x + w]
            faces_roi = np.array(faces_roi, dtype=np.uint8)
            # Who the algorithm thinks it is
            index, conf = face_recognizer.predict(faces_roi)
            name = celebrities[index]
            # Calculating confidence level
            confidence = 1.0 - float(w * h) / float(img.shape[0] * img.shape[1])
            # Draws rectangle over detected face
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 1)
        
        # Showing an OpenCV image on Tkinter
        b, g, r = cv.split(img)
        img = cv.merge((r, g, b))

        im = Image.fromarray(img)
        resize = im.resize((450, 450))
        imgtk = ImageTk.PhotoImage(image = resize)
        label = Label(root, image = imgtk)
        label.image = imgtk
        label.pack(pady = 20)  

        return(name, confidence)

    # Upon pressing enter, displays outputs
    def on_submit(entry, img_path, celeb, img_label, instr):
        instr.pack_forget()
        img_label.pack_forget()
        input = entry.get()
        entry.pack_forget()

        def check_special_char(string):
            for c in string:
                if not (c.isalpha() or c.isdigit() or c == ' '):
                    return True
            return False

        name, confidence = haar_analysis(img_path)

        # Checks if user input has special characters 
        if check_special_char(input):
            result = ctk.CTkLabel(root, text = "ERROR: input contains special characters."
                                  "Restart the program.", font = ctk.CTkFont(size = 15), text_color = "#357A38")
            result.pack(pady = 5)
        
        # Checks user input against algorithm and real answer
        if input == celeb or input.lower == celeb:
            result2 = (f"Your answer was: {input}.\n"
                       f"You are correct, the algorithm detected {name}\n"
                                  f"with a confidence of {confidence: .2f}.\n"
                                  f"The actual answer is {celeb}.")
            result = ctk.CTkLabel(root, text = result2, font = ctk.CTkFont(size = 15), text_color = "#357A38")
            result.pack(pady = 5)
        else:
            result2 = (f"Your answer was: {input}.\n"
                       f"You are incorrect, the algorithm detected {name}\n"
                                  f"with a confidence of {confidence: .2f}.\n"
                                  f"The actual answer is {celeb}.")
            result = ctk.CTkLabel(root, text = result2, font = ctk.CTkFont(size = 15), text_color = "#D30000")
            result.pack(pady = 5)

    # Takes in user input
    def user_input(img_path, celeb, img_label, instr):
        entry = ctk.CTkEntry(root, placeholder_text = "Make a guess...", width = 250)
        entry.pack(pady = 15)
        entry.bind("<Return>", lambda event: on_submit(entry, img_path, celeb, img_label, instr)) #ENTER EVENT!!!

    # Displays random celebrity image generated from RIG function
    def disp_img(button, description):
        button.pack_forget()
        description.pack_forget()

        img_path, celeb = RIG(celebrities) # "celeb" is the real answer from file directory
        
        img = cv.imread(img_path)
        b, g, r = cv.split(img)
        img = cv.merge((r, g, b))

        # Showing an OpenCV image on Tkinter
        im = Image.fromarray(img)
        resize = im.resize((450, 450))
        imgtk = ImageTk.PhotoImage(image = resize)
        img_label = Label(root, image = imgtk)
        img_label.image = imgtk
        img_label.pack(pady = 20)  

        instr = ctk.CTkLabel(root, text = "Press Enter key when done.\n"
                             "Make sure to capitalize the first letters of their first & last name\n"
                             "(e.g. Tom Cruise, not tom cruise).", 
                             font = ctk.CTkFont(size = 15))
        instr.pack(pady = 5)

        user_input(img_path, celeb, img_label, instr)

    button = ctk.CTkButton(root, width = 300, height = 40, text = "Start Game", font = ctk.CTkFont(size = 15), command = lambda: disp_img(button, description))
    button.pack(pady = 20) 

    root.mainloop()
    
#------ Main Function
def main():
    #------ Part 1: Creating trained Haar cascade classifier
    # List of celebrities
    celebrities = [] 
    for i in os.listdir(DIR):
        celebrities.append(i)
    
    #------ Part 3: GUI design 
    GUI(celebrities)
    
if __name__ == '__main__':
    main()

#Notes: 
#TURN IT ALL INTO A CLASS!!!
#change image resizing to be by scale (not to scale right now)
#fix DS_Store issue -> go through loop in beginning of function 
#change image to show a portion?
#make a main function
#make gray and image ROI into a separate function
#ADD SOME INTRODUCTION WITH FIXES & ERRORS
#change min neighbors for algorithm sensitivity