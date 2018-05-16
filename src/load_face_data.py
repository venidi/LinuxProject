# -*- coding:utf-8 -*-

# 将分类好的数据加载到内存编号

import cv2 as cv
import os
import numpy as np
import sys

IMAGE_SIZE = 64


# 按照指定图像大小调整尺寸 64*64
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像长宽

    h, w, _ = image.shape

    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv.resize(constant, (height, width))


# load images
images = []
labels = []


def read_path(path_name):
    # 绝对路径
    for dir_one in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_one))

        if os.path.isdir(full_path):
            read_path(full_path)
        else:  # isfile
            if dir_one.endswith('.jpg'):
                image = cv.imread(full_path)
                print(full_path)

                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                # print('img ok')

                # append
                images.append(image)
                # print(images.shape)
                labels.append(path_name)
                # print(labels.shape)

    return images, labels


def load_images(path_name, end_name):
    images, labels = read_path(path_name)

    # 图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)

    # 标注数据,选中的全部指定为0，
    # 另外的文件夹，全部指定为1
    labels = np.array([0 if label.endswith(end_name) else 1 for label in labels])

    return images, labels


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_images(sys.argv[1])
        for label in labels:
            print(label)
        print(labels.shape)

