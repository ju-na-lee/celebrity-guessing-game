#BASIC OPENCV FUNCTIONS

import cv2 as cv  
img = cv.imread('pexels-pixabay-45201.jpeg')
img2 = cv.imread('rome.jpeg')
cv.imshow('Romne', img2)

#1. converting image to greyscale (see intensity of color) 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Grayscale', gray)

#2. blur - reducing image noise
blur = cv.GaussianBlur(img2, (5, 5), cv.BORDER_DEFAULT)
#cv.imshow('Rome', blur)

#3. edge cascade
canny = cv.Canny(blur, 125, 175) #adding blur reduces edges 
#cv.imshow('edge', canny)

#4. dilating the image
dilated = cv.dilate(canny, (7, 7), iterations = 3)
#cv.imshow('Dilated', dilated)

#5. Eroding
eroded = cv.erode(dilated, (7, 7), iterations = 3)
#cv.imshow('Eroded', eroded)

#6. Resize
resized = cv.resize(img2, (500, 500), interpolation = cv.INTER_LINEAR)
cv.imshow('resize', resized)

#7. Cropping
cropped = img2[0:50, 0:100]
cv.imshow('crop', cropped)

cv.waitKey(0)