#Drawing Shapes & writing text on images
import cv2 as cv
import numpy as np
 
#showing a blank image
blank = np.zeros((500, 500, 3), dtype = 'uint8')
""" #1. Paint the image a certain color
blank[200 :] = 0, 255, 0 #referencing all pixels
#cv.imshow('Green', blank)   """

""" #2. Draw a rectangle
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness = -1)
#cv.imshow('Rectangle', blank)

#3. Draw a circle 
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness = -1 )
#cv.imshow('Circle', blank)

#3. Draw a line
cv.line(blank, (100, 250), (300, 450), (255, 255, 255), thickness = 3)
#cv.imshow('Line', blank) """

#4. Write text 
cv.putText(blank, 'Hello Lakshme', (0, 255), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
cv.imshow('Text', blank)

""" img = cv.imread('pexels-pixabay-45201.jpeg')
cv.imshow('Cat', img) """
 
cv.waitKey(0)