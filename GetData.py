import cv2

cam = cv2.VideoCapture(0)
dir = 'data wajah'

pengenal_wajah = cv2.CascadeClassifier('haarcascade_frontal_face.xml')
faceId = input("Id wajah : ")
getData = 1

while True:
    ret, frame = cam.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = pengenal_wajah.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in face:
        frm = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        namaFile = 'Wajah.' + str(faceId) + '.' + str(getData) + '.jpg'
        cv2.imwrite(dir + '/' + namaFile, frame)
        getData+=1

    cv2.imshow('Web Cam', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    elif getData == 31:
        print("Pengambilan data selesai")
        break

cam.release()
cv2.destroyAllWindows()
