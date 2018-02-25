# -*- coding: utf-8 -*-

import cv2 as cv

cv.namedWindow('PeopleRecognition', cv.WINDOW_NORMAL)
video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')
path_name = r'./faces/faceO'
classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt.xml')

color = (0, 255, 0)
num = 0
while video.isOpened():
    ok,frame = video.read()
    if not ok:
        break
    gery = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faceRects = classifier.detectMultiScale(gery, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            img_name = '%s/%d.jpg' % (path_name, num)
            image = frame[y-10: y+h+10, x-10: x+w+10]
            cv.imwrite(img_name,image)
            num += 1
            if num > 10000:
                break
            # 画出矩形框
            cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

            # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

    # 超过指定最大保存数量结束程序
    if num > 10000:
        break

    cv.imshow('PeopleRecognition', frame)
    c = cv.waitKey(1)
    if c & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
