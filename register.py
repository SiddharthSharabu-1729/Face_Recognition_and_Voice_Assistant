import os
import time
import numpy as np
import cv2
#import FaceTrain

s=input('name')
os.mkdir("dataset/{}".format(s))

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    frame = cv2.flip(frame,-1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
    for (x,y,w,h) in face:
        roi_gray = gray[y:y+h, x:x+w]
        for i in range(7):
            cv2.imwrite("dataset/{}/{}.png".format(s,(s+str(i))), roi_gray)
            time.sleep(0.5)
        break
    #cv2.imshow('frame',frame)
    break
time.sleep(3)
#FaceTrain.TrainFace()
