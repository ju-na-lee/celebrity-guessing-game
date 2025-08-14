#countour = boundaries of an object
import cv2 as cv
import numpy as np

img = cv.imread('pexels-pixabay-45201.jpeg')
cv.imshow('cat', img)

blank = np.zeros(img.shape, dtype = 'uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

#finds edges
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny) 

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) #makes it either black (125) or white(255) in binary
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv.drawContours(blank, contours, -1, (0, 0, 255), thickness = 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)