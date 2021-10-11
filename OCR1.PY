import cv2
import pytesseract

cap = cv2.VideoCapture(1)
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'


while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        cv2.imwrite('temp/temp.png',frame)
        break


img_ = cv2.imread('temp/temp.png')
img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
boxes = pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        #print(b)
        if len(b)==12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img, (x,y), (w+x,h+y), (0,255,0), 1)
            cv2.putText(img, b[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),1)
            print(b[11],end=' ')
print('\n')