import cv2 as cv

groupofppl = cv.imread('groupofppl2.jpeg')
lady = cv.imread('lady.jpeg')
cv.imshow('Group of People', groupofppl)

gray = cv.cvtColor(groupofppl, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml') #classifier 
 
#detectMultiScale is an instance of cascadeclassifier class, returns rectangular coords of face as a list 
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 7) #minNeighbors change noise
print(len(faces_rect))

#haarCascades are sensitive to noise

for (x, y, w, h) in faces_rect:
    cv.rectangle(groupofppl, (x, y), (x + w, y + h), (0, 255, 0), thickness = 1)

cv.imshow('Detected Faces', groupofppl)

cv.waitKey(0) 