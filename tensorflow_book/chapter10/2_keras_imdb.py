#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# 最多使用的单词数
max_features = 20000
# 循环神经网络的截断长度
maxlen = 80
batch_size = 32

# 加载数据并将单词转化为ID，max_features 给出了最多使用的单词数。和自然语言模型类似，会将出现频率较低的单词替换为统一的ID。
# 通过Keras封装的API会生成25000条训练数据和25000条测试数据，每一条数据可以被看成一段话，并且每段话都有一个好评或者差评的标签。
(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
print(len(trainX), 'train sequences')
print(len(testX), 'test sequence')
print()
print(trainX[0])
print(trainY[0])
print()
print(testX[0])
print(testY[0])

# 处理长度
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)

print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

# 构建模型
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=15, validation_data=(testX, testY))

score = model.evaluate(testX, testY, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
