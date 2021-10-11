import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer  = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name:": 1}

with open("labels.pickel", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        gray = gray[y:y+h,x:x+w]
        #color = frame[y:y+h, x:x+h]

        id_,conf = recognizer.predict(gray)
        if conf>80:# and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0,255,0)
            strokes = 2
            cv2.putText(frame , name, (x,y), font, 1, color, strokes, cv2.LINE_AA)
        else :
            pass

        #cv2.imwrite('images/me1.png',color)

        color = (255,0,0)
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame, (x,y),(width,height),color, stroke)
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
