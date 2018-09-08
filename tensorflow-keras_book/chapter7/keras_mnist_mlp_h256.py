from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
print(x_train_image.shape)
x_Train = x_train_image.reshape(60000, 784).astype('float32')
print(x_Train.shape)
x_Test = x_test_image.reshape(10000, 784).astype('float32')

# 标准化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

# One-Hot Encoding
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

# 建立模型
model = Sequential()
# Dense特色：所有的上一层与下一层的神经元都完全连接
"""
units：定义隐藏层的神经元个数为256
kernel_initializer：使用 normal distribution 正态分布的随机数来初始化 weight（权重）和bias（偏差）
"""
# 建立输入层和隐藏层
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))

# 加入 Dropout功能
model.add(Dropout(0.5))

# 第二个隐藏层
model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))

# 加入 Dropout功能
model.add(Dropout(0.5))

# 建立输出层
"""
softmax 可以将神经元的输出转换为预测每一个数字的概率
"""
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 查看模型的摘要
print(model.summary())

# 进行训练
"""
loss：设置损失函数，在深度学习中使用 cross_entropy(交叉熵)训练的效果比较好
optimizer：在深度学习中使用 adam 优化器可以让训练更快收敛，并提高准确率
metrics：设置评估模型的方式是准确率
"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 开始训练
"""
使用 48000 项训练数据进行训练，分为每一个批次200项，所以大约分成 240 个批次进行训练
verbose=2：显示训练过程

训练过程会存储在 train_history 变量中
"""
train_history = model.fit(x=x_Train_normalize, y=y_Train_OneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)


# 显示训练过程
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
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


# 评估模型准确率
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=', scores[1])
print(scores)

# 进行预测
prediction = model.predict_classes(x_Test)
# print(prediction)

plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)

# 建立混淆矩阵
crosstab = pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
print(crosstab)

# 建立真实值与预测DataFrame
df = pd.DataFrame(data={'label': y_test_label, 'predict': prediction})
# 真实值是 5 但预测值是 3 的数据：
# df[(df.label==5)&(df.predict==3)]
print(df)

if __name__ == '__main__':
    show_train_history(train_history=train_history, train='acc', validation='val_acc')
    show_train_history(train_history=train_history, train='loss', validation='val_loss')
