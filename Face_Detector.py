import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier("Python_open_cv/Harrcascade_Face_Detection.xml")
while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    for (x,y,w,h) in face_rect:
        new_frame = cv.rectangle(frame, (x,y), (x+w,y+h), color=(0,0,255), thickness=2)
        cv.imshow("Images",new_frame)

        if cv.waitKey(1) == ord("q"):
            break
    
