from __future__ import print_function

import os
import pandas as pd
import numpy as np
import pyprind
from keras.preprocessing.text import Tokenizer
from libs import clean_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


# max_features = 20000
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(x_train[0])
# print(y_train)


class IMDB_LSTM(object):
    def __init__(self):
        self.pbar = pyprind.ProgBar(50000)

    def load_data(self):
        data_file_train_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv'
        data_file_test_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_test.csv'
        if os.path.isfile(data_file_train_path) and os.path.isfile(data_file_test_path):
            print('data exist')
        else:
            print('generate data...')
            labels = {'pos': 1, 'neg': 0}
            data_train = pd.DataFrame()
            data_test = pd.DataFrame()
            for train_type in ('test', 'train'):
                for sentiment in ('pos', 'neg'):
                    path = '/home/dengkaiting/pycharm_project/aclImdb/%s/%s' % (train_type, sentiment)
                    print(path)
                    for file in os.listdir(path):
                        with open(os.path.join(path, file), 'r', encoding='UTF-8') as f:
                            content = f.read()
                            if train_type is 'train':
                                data_train = data_train.append([[content, labels[sentiment]]], ignore_index=True)
                            else:
                                data_test = data_test.append([[content, labels[sentiment]]], ignore_index=True)
                            self.pbar.update()
            data_train.columns = ['review', 'sentiment']
            np.random.seed(0)
            data_train.reindex(np.random.permutation(data_train.index))
            data_train.to_csv("/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv", index=False)

            data_test.columns = ['review', 'sentiment']
            np.random.seed(0)
            data_test.reindex(np.random.permutation(data_test.index))
            data_test.to_csv("/home/dengkaiting/pycharm_project/aclImdb/move_data_test.csv", index=False)
        return pd.read_csv(data_file_train_path), pd.read_csv(data_file_test_path)

    @staticmethod
    def clean_data(data):
        new_data = pd.DataFrame()
        for row in data.itertuples(index=True, name='Pandas'):
            clean_str = clean_data.basic_clean_str(getattr(row, 'review'))
            clean_str = clean_data.customized_clean_str(clean_str)
            new_data = new_data.append([[clean_str, getattr(row, 'sentiment')]], ignore_index=True)
        new_data.columns = ['review', 'sentiment']
        return new_data

    @staticmethod
    def interface(data):
        # 转化为词向量
        tokenizer = Tokenizer(num_words=10, lower=True, split=' ')
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        print(sequences)


if __name__ == '__main__':
    imdb_lstm = IMDB_LSTM()
    data_train, data_test = imdb_lstm.load_data()
    data_train = imdb_lstm.clean_data(data_train)
    imdb_lstm.interface(data_train[0:500]['review'].values)
    print('over')
