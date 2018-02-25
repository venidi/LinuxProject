# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

cv.namedWindow('PeopleRecognition',0)
video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')

classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml')

color = (0,255,0)
while video.isOpened():
    ok,frame = video.read()
    if not ok:
        break
    gery = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faceRects = classifier.detectMultiScale(gery, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv.rectangle(frame, (x-10,y-10),(x+w+10,y+h+10),color,2)
    cv.imshow('PeopleRecognition',frame)
    c = cv.waitKey(10)
    if c & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
