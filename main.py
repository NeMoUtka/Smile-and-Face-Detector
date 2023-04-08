import cv2

cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('face.xml')
smile = cv2.CascadeClassifier('smile.xml')

while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = face.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=8)


    for (x, y, w, h) in res:
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 200, 50), thickness=4)
        face_gray = img_gray[y:y + h , x:x + w]
        face_color = img[y:y + h, x:x + w]
    
    
        res_smile = smile.detectMultiScale(face_gray, scaleFactor= 1.1, minNeighbors=40)
        for (x2, y2, w2, h2) in res_smile:
            cv2.rectangle(face_color, (x2, y2), (x2+w2, y2+h2), (50, 50, 200), thickness=4)
            cv2.putText(img, 'Smile', (x,y-7), 3, 1.2, (169, 7, 126), 2, cv2.LINE_AA)
        
    
    cv2.imshow('Res', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 