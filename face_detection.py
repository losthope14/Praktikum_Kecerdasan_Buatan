import cv2

pixels = cv2.imread()

classiffier = cv2.CascadeClassifier('haarcascade_frontal_face.xml')

bboxes = classiffier.detectMultiScale(pixels, 1.1, 3)

for box in bboxes:
    x, y, width, height = box
    x2, y2 = x + width, y + height

    cv2.rectangle(pixels, (x,y), (x2,y2), (0,0,255), 1)

cv2.imshow('Face detection', pixels)

cv2.waitKey(0)
cv2.destroyAllWindows()