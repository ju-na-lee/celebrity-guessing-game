import cv2 as cv

#multiline comment = Shift + Option + A

#Reading in images
img = cv.imread("pexels-pixabay-45201.jpeg") #reads as matrix of pixels 
cv.imshow('Cat', img)
cv.waitKey(0)

""" 
#Reading videos 
#capture = cv.VideoCapture('.mp4') #takes int args or path to video file 
#use while loop to read frame by frame
while True:
    isTrue, frame = capture.read()
    cv.limshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows() """