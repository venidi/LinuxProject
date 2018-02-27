# -*- coding:UTF-8 -*-

import random
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


class DataSet:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None

        self.path_name = path_name
        # 维度顺序，因后台而异
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_row = IMAGE_SIZE, img_col = IMAGE_SIZE, img_chl = 3, nb_classes = 2):
        # 加载到内存
        images, labels = load_images(self.path_name)
        # print('load images ok')

        # 随机交叉验证，根据test_size参数按比例划分数据集
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))

        # 根据keras库要求的维度顺序重组训练数据集,tensorflow或theano
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

        # 输出训练集、验证集、测试集的数量
        print(test_images.shape[0], 'train')
        print(valid_images.shape[0], 'valid')
        print(test_images.shape[0], 'test')

        # 使用categorical_crossentropy作为损失函数
        # 根据类别数量nb_classes将，类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_lables = np_utils.to_categorical(train_labels, nb_classes)
        valid_lables = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        # 像素数据浮点化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 归一化,提升网络收敛速度
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_lables = train_lables
        self.valid_lables = valid_lables
        self.test_lables = test_labels


# 卷积神经网络模型
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 添加CNN网络需要的各层，一个add就是一个网络层
        # 1 2维卷积层 利用卷积核逐个像素、顺序进行计算
        self.model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = dataset.input_shape))
        # 2 激活函数层，采用的relu（Rectified Linear Units，修正线性单元）函数->f(x)=max(0,x),收敛快
        self.model.add(Activation('relu'))

        # 3 2维卷积层
        self.model.add(Conv2D(32, (3, 3)))
        # 4 激活函数层
        self.model.add(Activation('relu'))

        # 5 池化层，缩小输入的特征图，简化网络计算复杂度；同时进行特征压缩，突出主要特征
        # 选取覆盖区域的最大值作为区域主要特征组成新的缩小后的特征图，2*2池化: 64*64->32*32
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        # 6 Dropout层,随机断开一定百分比的输入神经元链接，以防止过拟合
        self.model.add(Dropout(0.25))

        # 7  2维卷积层
        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        # 8  激活函数层
        self.model.add(Activation('relu'))

        # 9  2维卷积层
        self.model.add(Conv2D(64, (3, 3)))
        # 10 激活函数层
        self.model.add(Activation('relu'))

        # 11 池化层
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        # 12 Dropout层
        self.model.add(Dropout(0.25))

        # 13 Flatten层,把输入数据“压扁”成一维后才能进入全连接层
        self.model.add(Flatten())
        # 14 全连接层,分类，保留了512个特征输出到下一层
        self.model.add(Dense(512))
        # 15 激活函数层
        self.model.add(Activation('relu'))
        # 16 Dropout层
        self.model.add(Dropout(0.5))
        # 17 Dense层
        self.model.add(Dense(nb_classes))
        # 18 分类层，输出最终结果
        self.model.add(Activation('softmax'))

        self.model.summary()

    # train
    def train(self, dataset, betch_size = 20, epochs = 10, data_augementation = True):
        # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # 完成实际的模型配置工作
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 不使用数据提升
        if not data_augementation:
            self.model.fit(dataset.train_images,
                           dataset.train_lables,
                           betch_size = betch_size,
                           epochs = epochs,
                           validation_data = (dataset.valid_images, dataset.valid_lables),
                           suffle = True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen
            datagen = ImageDataGenerator(
                featurewise_center=False,             # 是否使输入数据去中心化
                samplewise_center=False,              # 是否使输入数据的每个样本均值为0
                samplewise_std_normalization=False,   # 是否数据标准化（输入数据除以数据集的标准差）
                featurewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,                  # 是否对输入数据施以ZCA白化
                rotation_range=20,                    # 数据提升时图片随机转动的角度
                width_shift_range=0.2,                # 数据提升时图片水平偏移的幅度
                height_shift_range=0.2,               # 数据提升时图片垂直偏移的幅度
                horizontal_flip=True,                 # 是否进行随机水平翻转
                vertical_flip=False                   # 是否进行随机垂直翻转
            )
            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images,
                                                  dataset.train_lables,batch_size=betch_size),
                                     # samples_per_epoch= dataset.train_images.shape[0],
                                     steps_per_epoch=26,
                                     epochs = epochs,
                                     validation_data=(dataset.valid_images,dataset.valid_lables))

    MODEL_PATH = r'./models/faceD_2.model.h5'

    # 模型保存
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    # 模型加载
    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    # 模型评估
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_lables, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = resize_image(image)
            # 与模型训练不同，这次只是针对1张图片进行预测
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
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
    dataset = DataSet('./faces/')
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.evaluate(dataset)
    model.save_model(file_path=r'./models/faceD_2.model.h5')
