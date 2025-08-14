import cv2 as cv
import numpy as np

#splitting color channels
img = cv.imread('rome.jpeg')
cv.imshow('Rome', img)

b, g, r = cv.split(img)

""" cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r) """

#methods of blurring
#1. Averaging
average = cv.blur(img, (3, 3))
#cv.imshow("Average", average)

#edge detection method
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

#canny edge detection
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny) 

cv.waitKey(0)