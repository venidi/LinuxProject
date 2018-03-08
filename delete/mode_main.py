from src.face_train import Dataset
from src.face_train import Model

# 模型验证
if __name__ == '__main__':
    dataset = Dataset(r'/home/venidi/FaceRecognition/test/LinuxProject/faces')
    dataset.load()

    model = Model()
    model.load_model(file_path='./models/faceD.model.h5')
    model.evaluate(dataset)