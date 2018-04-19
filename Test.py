import pandas as pd

import pyprind
import time

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

data_file_train_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv'
csv = pd.read_csv(data_file_train_path)[0:50000]
print(csv['review'])
tokenizer = Tokenizer(num_words=10, lower=True, split=' ')
tokenizer.fit_on_texts(csv['review'].values)
# 查看词向量字典
# print(tokenizer.word_index)
X = tokenizer.texts_to_sequences(csv['review'].values)
print(type(X))
# print(X)
# 填充序列到相同的长度    Pads sequences to the same length
X = pad_sequences(X)
print(type(X))
# print(X)

embed_dim = 218
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(Embedding(input_dim=10, output_dim=embed_dim, input_length=X.shape[1]))
model.add(LSTM(units=lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

Y = pd.get_dummies(data=csv['sentiment'].values)
print(Y)

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.20, random_state=36)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=5)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print('Score: %.2f' % score)
print('Validation Accuracy: %.2f' % acc)
