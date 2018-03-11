# import cv2
# import numpy as np
#
#
# def show():
#     cap = cv2.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbj.mp4')
#     timeF = 3
#     c = 1
#     while(1):
#         # get a frame
#
#         ret, frame = cap.read()
#         if c % timeF == 0:
#             # show a frame
#             cv2.imshow("capture", frame)
#         c = c + 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     show()

# -*- coding: utf-8 -*-


# 样本视频处理，获取视频中人脸作为识别样本

# import cv2 as cv
#
#
# def face_cut(video_path,save_path):
#     cv.namedWindow('PeopleRecognition', cv.WINDOW_NORMAL)
#     # video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')
#     video = cv.VideoCapture(video_path)
#     # path_name = r'./faces/faceO'
#     path_name = save_path
#     # openCV的人脸分类器
#     classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt.xml')
#
#     color = (0, 135, 255)
#     num = 0
#
#     TIME = 3
#     C = 1
#     # 逐帧检测
#     while (1):
#         ok,frame = video.read()
#         if not ok:
#             break
#         # 转换成灰度图像便于计算
#         if C % TIME == 0:
#             gery = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
#             faceRects = classifier.detectMultiScale(gery, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
#             if len(faceRects) > 0:
#                 for faceRect in faceRects:
#                     x, y, w, h = faceRect
#                     # 截取保存
#                     img_name = '%s/%d.jpg' % (path_name, num)
#                     # print(img_name)
#                     image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
#                     cv.imwrite(img_name, image)
#                     num += 1
#                     if num > 1000:
#                         break
#                     # # 框出人脸
#                     cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
#
#                     # 显示当前照片编号
#                     # font = cv.FONT_HERSHEY_SIMPLEX
#                     # cv.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
#         C = C + 1
#         # 超过指定最大保存数量结束程序
#         if num > 1000:
#             break
#
#         cv.imshow('PeopleRecognition', frame)
#         c = cv.waitKey(1)
#         if c & 0xFF == ord('q'):
#             break
#     video.release()
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     video_path = r'/home/venidi/FaceRecognition/data/videos/toRe.mp4'
#     save_path = r'/home/venidi/FaceRecognition/test/LinuxProject/faces/face'
#     face_cut(video_path, save_path)

# import cv2 as cv
# from src.face_train import Model
#
#
# if __name__ == '__main__':
#     model = Model()
#     # 加载训练完的模型
#     model.load_model(file_path=r'/home/venidi/FaceRecognition/test/LinuxProject/models/face_xi.model.h5')
#
#     color = (0, 255, 0)
#     cv.namedWindow('PeopleRecognition', 0)
#     video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbr_full.mp4')
#
#     classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml')
#
#     # 抽帧
#     TIME = 3
#     C = 1
#     path_name = r'/home/venidi/FaceRecognition/test/LinuxProject/faces/face'
#     num = 0
#     while video.isOpened():
#         ok,frame = video.read()
#         if not ok:
#             print('not ok')
#             break
#         if C % TIME == 0:
#             frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             faceRects = classifier.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
#             if len(faceRects) > 0:
#                 for faceRect in faceRects:
#                     x, y, w, h = faceRect
#                     image_name = '%s/%d.jpg' % (path_name, num)
#                     num = num + 1
#
#                     image = frame[y-10: y+h+10, x-10: x+w+10]
#
#                     faceID = model.face_predict(image)
#                     # 识别并显示
#                     if faceID == 0:
#                         cv.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
#                         cv.putText(frame, 'FACE_D', (x+30, y+30), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 28, 33), 2)
#                         cv.imwrite(image_name, image)
#                     else:
#                         pass
#                         # cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
#             cv.imshow('result', frame)
#             key = cv.waitKey(1)
#
#             if key & 0xFF == ord('q'):
#                 break
#         C = C + 1
#
#     video.release()
#     cv.destroyAllWindows()

# -*- coding: utf-8 -*-


# 样本视频处理，获取视频中人脸作为识别样本

# import cv2 as cv
# from src.face_train import Model
#
# def face_cut(video_path, save_path):
#     model = Model()
#     # 加载训练完的模型
#     # model.load_model(file_path='./models/faceD_2.model.h5')
#     model.load_model(file_path=r'/home/venidi/FaceRecognition/test/LinuxProject/models/face_xi.model.h5')
#     cv.namedWindow('PeopleRecognition', cv.WINDOW_NORMAL)
#     # video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')
#     video = cv.VideoCapture(video_path)
#     # path_name = r'./faces/faceO'
#     path_name = save_path
#     # openCV的人脸分类器
#     classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt.xml')
#
#     color = (0, 135, 255)
#     num = 0
#     TIME = 5
#     C = 1
#     # 逐帧检测
#     while (1):
#         ok, frame = video.read()
#         if not ok:
#             break
#         if C % TIME == 0:
#             # 转换成灰度图像便于计算
#             gery = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#             # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
#             faceRects = classifier.detectMultiScale(gery, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
#             if len(faceRects) > 0:
#                 for faceRect in faceRects:
#                     x, y, w, h = faceRect
#                     # 截取保存
#                     img_name = '%s/%d.jpg' % (path_name, num)
#
#                     # print(img_name)
#                     image = frame[y-10: y+h+10, x-10: x+w+10]
#                     faceID = model.face_predict(image)
#                     # 识别并显示
#                     if faceID == 0:
#                         cv.imwrite(img_name,image)
#                         num += 1
#                         cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
#                     if num > 1000:
#                         break
#                     # # 框出人脸
#             # 超过指定最大保存数量结束程序
#             if num > 1000:
#                 break
#         C = C + 1
#         cv.imshow('PeopleRecognition', frame)
#         c = cv.waitKey(1)
#         if c & 0xFF == ord('q'):
#             break
#     video.release()
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     video_path = r'/home/venidi/FaceRecognition/data/videos/toRe.mp4'
#     save_path = r'/home/venidi/FaceRecognition/test/LinuxProject/faces/face_zz'
#     face_cut(video_path, save_path)
import cv2 as cv
from src.face_train import Model

# def face_re(model_path, video_path, clifer_path):
import cv2 as cv
from src.face_train import Model

# def face_re(model_path, video_path, clifer_path):
def face_re(model_path, video_path, clifer_path):
    model = Model()
    # 加载训练完的模型
    # model.load_model(file_path='./models/faceD_2.model.h5')
    model.load_model(file_path=model_path)
    color = (0, 255, 0)
    cv.namedWindow('PeopleRecognition', 0)
    # video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbr_full.mp4')
    video = cv.VideoCapture(video_path)
    # classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml')
    classifier = cv.CascadeClassifier(clifer_path)
    # 抽帧
    flag = 0
    if flag == 0:
        TIME = 30
    else:
        TIME = 5
    C = 1
    while video.isOpened():
        ok,frame = video.read()
        if not ok:
            print('not ok')
            break
        if C % TIME == 0:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    image = frame[y-10: y+h+10, x-10: x+w+10]
                    faceID = model.face_predict(image)
                    # 识别并显示
                    if faceID == 0:
                        flag = 1
                        cv.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
                        cv.putText(frame, 'FACE_D', (x+30, y+30), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 28, 33), 2)
                    else:
                        # pass
                        cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
            flag = flag
            cv.imshow('result', frame)
            key = cv.waitKey(1)

            if key & 0xFF == ord('q'):
                break
        C = C + 1

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    model_path = r'/home/venidi/FaceRecognition/test/LinuxProject/models/face_xi.model.h5'
    video_path = r'/home/venidi/FaceRecognition/data/videos/toRe.mp4'
    clifer_path = r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml'
    face_re(model_path, video_path,clifer_path)