# -*- coding: utf-8 -*-


# 样本视频处理，获取视频中人脸作为识别样本

import cv2 as cv


# def face_cut(video_path, save_path):
def face_cut(video_path, save_path, cfier_path):
    cv.namedWindow('PeopleRecognition', cv.WINDOW_NORMAL)
    # video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')
    video = cv.VideoCapture(video_path)
    # path_name = r'./faces/faceO'
    path_name = save_path
    # openCV的人脸分类器
    # classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt.xml')
    classifier = cv.CascadeClassifier(cfier_path)

    color = (0, 135, 255)
    num = 0

    # 逐帧检测
    while (1):
        ok, frame = video.read()
        if not ok:
            break
        # 转换成灰度图像便于计算
        gery = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classifier.detectMultiScale(gery, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 截取保存
                img_name = '%s/%d.jpg' % (path_name, num)
                # print(img_name)
                image = frame[y-10: y+h+10, x-10: x+w+10]
                cv.imwrite(img_name,image)
                num += 1
                if num > 1000:
                    break
                # # 框出人脸
                cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                # 显示当前照片编号
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num > 1000:
            break

        cv.imshow('PeopleRecognition', frame)
        c = cv.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'/home/venidi/FaceRecognition/data/videos/toRe.mp4'
    save_path = r'/home/venidi/FaceRecognition/test/LinuxProject/faces/face_zz'
    clfier_path = r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt.xml'
    face_cut(video_path, save_path, clfier_path)