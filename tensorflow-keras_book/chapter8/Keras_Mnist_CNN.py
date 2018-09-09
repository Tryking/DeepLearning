"""
卷积神经网络  CNN （ Convolutional Neural Network）
"""
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
print(x_Train.shape)

# 转化为四维矩阵
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')
print(x_Train4D.shape)

# 标准化
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

# One-Hot Encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

# 建立 Sequential 模型
model = Sequential()

# 建立卷积层1
"""
filter = 16     建立16个滤镜
kernel_size=(5, 5)     每一个滤镜为 5 X 5
padding='same' 此设置让卷积运算产生的卷积图像大小不变
"""
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))

# 建立池化层1
model.add(MaxPooling2D(pool_size=(2, 2)))

# 建立卷积层2
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))

# 建立池化层2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 加入Dropout
model.add(Dropout(0.25))

# 建立平坦层
"""
将之前已经建立的池化层2，共有 36 个 7 * 7 的图像转换为一维的向量，长度是 36 * 7 * 7 = 1764
"""
model.add(Flatten())

# 建立隐藏层
model.add(Dense(units=128, activation='relu'))

# 加入Dropout：每次训练迭代时，会随机地在神经网络中放弃 50% 的神经元，以免过度拟合
model.add(Dropout(0.5))

# 建立输出层
model.add(Dense(units=10, activation='softmax'))

# 查看模型摘要
print(model.summary())

# 进行训练
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_Train4D_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=300, verbose=2)


# 显示训练过程
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history=train_history, train='acc', validation='val_acc')

show_train_history(train_history=train_history, train='loss', validation='val_loss')
