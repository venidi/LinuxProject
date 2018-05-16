import cv2 as cv
from src.face_train import Model

# def face_re(model_path, video_path, clifer_path):
def face_re(model_path, video_path, clifer_path, time):
    model = Model()
    # 加载训练完的模型
    # model.load_model(file_path='./models/faceD_2.model.h5')
    model.load_model(file_path=model_path)
    color = (0, 255, 0)
    cv.namedWindow('PeopleRecognition', 0)
    # video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbr_full.mp4')
    video = cv.VideoCapture(video_path)
    classifier = cv.CascadeClassifier(clifer_path)
    flag = 0
    # 抽帧
    # TIME = 10
    TIME = time
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
                        cv.putText(frame, 'target', (x+30, y+30), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 28, 33), 2)
                    else:
                        # pass
                        cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
            cv.imshow('result', frame)
            key = cv.waitKey(1)

            if key & 0xFF == ord('q'):
                break
        C = C + 1

    video.release()
    cv.destroyAllWindows()
    return flag


# if __name__ == '__main__':
#     model_path = r'/home/venidi/FaceRecognition/test/LinuxProject/models/face_xi.model.h5'
#     video_path = r'/home/venidi/FaceRecognition/data/videos/zeng.mp4'
#     clifer_path = r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml'
#     face_re(model_path, video_path,clifer_path)