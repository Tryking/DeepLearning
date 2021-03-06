#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import backend as K

num_classes = 10
img_rows, img_cols = 28, 28

(trainX, trainY), (testX, testY) = mnist.load_data()

# 因为不同的底层（TensorFlow或者MXNet）对输入的要求不一样，所以这里需要根据对图像编码的格式要求来设置输入层的格式
if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    # 因为MNIST中的图片是黑白的，所以第一位的取值为1
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 将图像像素转化为 0 到 1 之间的实数
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

# 将标准答案转化为需要的格式（one-hot编码）
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

# 使用Keras API定义模型
model = Sequential()
# 一层深度为32，过滤器大小为5*5的卷积层
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
# 一层过滤器大小为2*2的最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# 一层深度为64，过滤器大小为5*5的卷积层
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
# 一层过滤器大小为2*2的最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# 将卷积层的输出拉直后作为下面全连接层的输入
model.add(Flatten())
# 全连接层，有500个节点
model.add(Dense(500, activation='relu'))
# 全连接层，得到最后的输出
model.add(Dense(num_classes, activation='softmax'))

# 定义损失函数、优化函数和评测方法
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

# 自动完成模型训练过程
model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))

# 在测试数据上计算准确率
score = model.evaluate(testX, testY)
print('Test loss:', score[0])
print('test accuracy:', score[1])
