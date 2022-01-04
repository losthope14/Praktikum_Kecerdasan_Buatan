import cv2, os, numpy as np

cam = cv2.VideoCapture(0)
dirData = 'data wajah'
dirTrain = 'training data'

pengenal_wajah = cv2.CascadeClassifier('haarcascade_frontal_face.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(dirTrain+'/training.xml')
font = cv2.FONT_HERSHEY_PLAIN


names = {0:'Unknown', 55120060:'Basthian', 12345:'King'}

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = pengenal_wajah.detectMultiScale(frame, 1.2, 5)


    for (x, y, w, h) in face:
        frm = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = faceRecognizer.predict(grey[y:y+h, x:x+w])

        if confidence<=50:
            nameID = names[0]
            confidenceTxt = '{0}%'.format(round(confidence))
        else:
            nameID = names[id]
            confidenceTxt = '{0}%'.format(round(confidence))
            print(confidence)
            print(id)

        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5, y+h-5), font, 1, (255,255,0), 1)


    cv2.imshow('Recognition', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()