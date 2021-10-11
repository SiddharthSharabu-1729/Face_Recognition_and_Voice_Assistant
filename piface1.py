import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

encodings = 'encodings.pickle'
data = pickle.loads(open(encodings, "rb").read())
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

'''cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    frame = cv2.flip(frame,-1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
    for (x,y,w,h) in face:
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite("temp.png".format(s,(s+str(i))), roi_gray)
        break
    break'''

frame = cv2.imread('dataset/sid/sid0.png')

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
print(boxes)
encodings = face_recognition.face_encodings(rgb, boxes)
names = []
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
    names.append(name)
    print(name)
'''for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    print(name)'''
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
    cv2.destroyAllWindows()
