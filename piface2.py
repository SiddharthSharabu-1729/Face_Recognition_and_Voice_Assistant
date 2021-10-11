import os
import cv2
import pickle
import face_recognition

encodings = 'encodings.pickle'

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    frame = cv2.flip(frame,-1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
    for (x,y,w,h) in face:
        roi_gray = gray[y:y+h+10, x:x+w+10]
        cv2.imwrite("temp.png", roi_gray)
        print('ok')
        break
    break
data = pickle.loads(open(encodings, "rb").read())

frame = cv2.imread('temp.png')

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
print(boxes)
encodings = face_recognition.face_encodings(rgb, boxes)
for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"],encoding)
    name = "Unknown"
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    print(name)
    break

os.remove('temp.png')