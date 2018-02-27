import cv2 as cv
from src.face_train import Model


if __name__ == '__main__':
    model = Model()
    model.load_model(file_path='./models/faceD_2.model.h5')

    color = (0, 255, 0)
    cv.namedWindow('PeopleRecognition', 0)
    video = cv.VideoCapture(r'/home/venidi/FaceRecognition/data/videos/kbjdkr.mp4')

    classifier = cv.CascadeClassifier(r'/opt/opencv34/data/haarcascades/haarcascade_frontalface_alt2.xml')

    while video.isOpened():
        ok,frame = video.read()
        if not ok:
            print('not ok')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faceRects = classifier.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                image = frame[y-10: y+h+10, x-10: x+w+10]
                faceID = model.face_predict(image)
                if faceID == 0:
                    cv.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
                    cv.putText(frame, 'FACE_D', (x+30, y+30), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 28, 33), 2)
                else:
                    # pass
                    cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
        cv.imshow('result', frame)
        key = cv.waitKey(10)
        if key & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
