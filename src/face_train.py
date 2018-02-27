# -*- coding:UTF-8 -*-

import random
import numpy as np
import h5py
from src.load_face_data import resize_image, load_images, IMAGE_SIZE
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

class Dataset:
    def __init__(self, path_name):
        #训练
        self.train_images = None
        self.train_lables = None
        #验证
        self.valid_images = None
        self.valid_lables = None
        #测试
        self.test_images = None
        self.test_lables = None

        self.path_name = path_name

        self.input_shape = None

    def load(self, img_row = IMAGE_SIZE, img_col = IMAGE_SIZE, img_chl = 3, nb_classes = 2):
        images, lables = load_images(self.path_name)
        print('load images ok')
        train_images, valid_images, train_lables, valid_lables = train_test_split(images, lables, test_size = 0.3, random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, lables, test_size=0.5, random_state=random.randint(0, 100))

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_chl, img_row, img_col)
            valid_images = valid_images.reshape(valid_images.shape[0], img_chl, img_row, img_col)
            test_images = test_images.reshape(test_images.shape[0], img_chl, img_row, img_col)
            self.input_shape = (img_chl, img_row, img_col)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_row, img_col, img_chl)
            valid_images = valid_images.reshape(valid_images.shape[0], img_row, img_col, img_chl)
            test_images = test_images.reshape(test_images.shape[0], img_row, img_col, img_chl)
            self.input_shape = (img_row, img_col, img_chl)

        print(test_images.shape[0], 'train')
        print(valid_images.shape[0], 'valid')
        print(test_images.shape[0], 'test')

        # 使用categorical_crossentropy作为损失函数
        train_lables = np_utils.to_categorical(train_lables, nb_classes)
        valid_lables = np_utils.to_categorical(valid_lables, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_lables = train_lables
        self.valid_lables = valid_lables
        self.test_lables = test_labels

# CNN Model
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 2):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = dataset.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()


    # train
    def train(self, dataset, betch_size = 20, epochs = 10, data_augementation = True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        if not data_augementation:
            self.model.fit(dataset.train_images,
                           dataset.train_lables,
                           betch_size = betch_size,
                           epochs = epochs,
                           validation_data = (dataset.valid_images, dataset.valid_lables),
                           suffle = True)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                samplewise_std_normalization=False,
                featurewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )
            datagen.fit(dataset.train_images)

            self.model.fit_generator(datagen.flow(dataset.train_images,
                                                  dataset.train_lables,batch_size=betch_size),
                                     # samples_per_epoch= dataset.train_images.shape[0],
                                     steps_per_epoch=26,
                                     epochs = epochs,
                                     validation_data=(dataset.valid_images,dataset.valid_lables))
    MODEL_PATH = r'./models/faceD_2.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_lables, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)

        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)

        # 返回类别预测结果
        return result[0]


if __name__ == '__main__':
    dataset = Dataset('./faces/')
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.evaluate(dataset)
    model.save_model(file_path=r'./models/faceD_2.model.h5')
