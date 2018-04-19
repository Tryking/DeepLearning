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
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

data_file_train_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv'
csv = pd.read_csv(data_file_train_path)
print(csv[0:3]['review'])
tokenizer = Tokenizer(num_words=10, lower=True, split=' ')
tokenizer.fit_on_texts(csv[0:3]['review'])
# 查看词向量字典
# print(tokenizer.word_index)
X = tokenizer.texts_to_sequences(csv[0:3]['review'].values)
print(type(X))
print(X)
# 填充序列到相同的长度    Pads sequences to the same length
X = pad_sequences(X)
print(type(X))
print(X)

embed_dim = 218
lstm_out = 200
batch_size = 32

model = Sequential()
model.add()









