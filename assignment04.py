"""
Assignment 04
=============

The goal of this assignment is to ignore eye regions of interest (ROI) that are not placed within face ROI.

The code you will write in this file will be similar to main.py code, but will include additional rectangles filtering.

Run this code with

    > invoke run assignment04.py
"""


from array import array

import cv2
import time
from tqdm import tqdm

FACE_MODEL_FILE = 'models/haarcascade_frontalface_default.xml'
EYES_MODEL_FILE = 'models/haarcascade_eye.xml'

PLATE_FILES = [
    'models/haarcascade_licence_plate_rus_16stages.xml',
    'models/haarcascade_russian_plate_number.xml',
]

eye_name = "eye_model"
face_name = "face_model"


def main():
    # load haar cascades model
    faces = cv2.CascadeClassifier(FACE_MODEL_FILE)
    eyes = cv2.CascadeClassifier(EYES_MODEL_FILE)
    plates = [cv2.CascadeClassifier(p) for p in PLATE_FILES]

    # connect to camera
    camera = cv2.VideoCapture(0)
    while not camera.isOpened():
        time.sleep(0.5)

    # read and show frames
    progress = tqdm()
    while True:

        ret, frame = camera.read()
        frame = process(frame, [
            (faces, (255, 255, 0), dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)), face_name),
            (eyes, (0, 0, 255), dict(scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)), eye_name),
        ])
        #        frame = process(frame, [
        #            (model, (0, 255, 0), dict(scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)))
        #            for model in plates
        #            ])
        cv2.imshow('Objects', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        progress.update()

    # gracefully close
    camera.release()
    cv2.destroyWindows()
    tqdm.close()


def process(frame, models):
    """Process initial frame and tag recognized objects."""

    # 1. Convert initial frame to grayscale
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceObject = []

    eyeObject = []

    elseObject = []

    allObjects = [faceObject, eyeObject, elseObject]

    for model, color, parameters, name in models:

        # 2. Apply model, recognize objects
        objects = model.detectMultiScale(grayframe, **parameters)

        # 3. For every recognized object, insert them into their storage
        if name == face_name and len(objects) > 0:
            faceObject.append((color, toList(objects)))
        elif name == eye_name:
            eyeObject.append((color, toList(objects)))
        else:
            elseObject.append((color, toList(objects)))

    def filterEyeObjects():

        removeEyeObjects = True
        (color, eyeObjects) = eyeObject[0]
        for eyeCorrd in eyeObjects[:]:
           (x, y, w, h) = eyeCorrd

           if len(faceObject) > 0:
            (color, faceObjects) = faceObject[0]
            for faceCoord in faceObjects[:]:
                (x2, y2, w2, h2) = faceCoord
                if x2 < x < (x2 + w2) and y2 < y < (y2 + h):
                  removeEyeObjects = False
                  break
            if removeEyeObjects:
                removeEyeObjects = False
                eyeObjects.remove(eyeCorrd)
           else:
               removeEyeObjects = False
               eyeObjects.remove(eyeCorrd)

    # 4. Filter eye rectangles
    filterEyeObjects()

    for specialObjects in allObjects:
        for (color, objects) in specialObjects:
            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # BGR

    # 5. Return initial color frame with rectangles
    return frame

def toList(nddArray):
    nextArr = []
    for next in nddArray:
        nextArr.append(next)
    return nextArr

if __name__ == '__main__':
    main()

