import cv2, os, numpy as np
from PIL import Image

dirSrc = 'data wajah'
dirDest = 'training data'

def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []

    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)

        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)

        return faceSamples, faceIDs

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontal_face.xml')

faces, IDs = getImageLabel(dirSrc)
faceRecognizer.train(faces, np.array(IDs))

faceRecognizer.write(dirDest + '/training.xml')