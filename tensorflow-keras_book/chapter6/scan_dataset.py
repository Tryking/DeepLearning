import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()

print('train data=', len(X_train_image))
print(' test data=', len(X_test_image))

print('x_train_image:', X_train_image.shape)
print('y_train_label:', y_train_label.shape)

# One-Hot Encoding 转换
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print(y_TrainOneHot)


def plot_image(image):
    fig = plt.gcf()
    # 设置显示图像的大小
    fig.set_size_inches(2, 2)
    # binary: 以黑白灰度显示
    plt.imshow(image, cmap='binary')
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = 'label=' + str(labels[idx])
        if len(prediction) > 0:
            title += ', predict=' + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


if __name__ == '__main__':
    print(type(X_train_image[0]))
    print(X_train_image[0])
    plot_images_labels_prediction(images=X_train_image, labels=y_train_label, prediction=[], idx=0)
