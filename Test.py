import pandas as pd

import pyprind
import time
import logging

# 进度条指标
# for progress in pyprind.prog_bar(20):
#     time.sleep(1)
#
# # 百分比指标
# for progress in pyprind.prog_percent(range(20)):
#     time.sleep(1)

# bar = pyprind.ProgBar(100)
# for i in range(100):
#     time.sleep(0.5)
#     bar.update()
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

max_feature = 100

data_file_train_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv'
data_file_test_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_test.csv'
train_data = pd.read_csv(data_file_train_path)
test_data = pd.read_csv(data_file_test_path)
train_data = train_data.sample(frac=0.05)
test_data = test_data.sample(frac=0.1)
print(train_data)
train_data_Y = train_data['sentiment']
test_data_Y = test_data['sentiment']

# print(csv['review'])
tokenizer = Tokenizer(num_words=max_feature, lower=True, split=' ')
tokenizer.fit_on_texts(train_data['review'].values)
# 查看词向量字典
# print(tokenizer.word_index)
X_train = tokenizer.texts_to_sequences(train_data['review'].values)
X_test = tokenizer.texts_to_sequences(test_data['review'].values)
# print(X)
# 填充序列到相同的长度    Pads sequences to the same length
X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
# print(X)

embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(Embedding(input_dim=max_feature, output_dim=embed_dim))
model.add(LSTM(units=embed_dim, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, train_data_Y, batch_size=batch_size, epochs=1, verbose=5, validation_data=(X_test, test_data_Y))
score, acc = model.evaluate(X_test, test_data_Y, verbose=2, batch_size=batch_size)
print('Score: %.2f' % score)
print('Validation Accuracy: %.2f' % acc)
