from __future__ import print_function

import os
import pandas as pd
import numpy as np
import pyprind
import logging
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from libs import clean_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


class IMDB_LSTM(object):
    def __init__(self):
        self.pbar = pyprind.ProgBar(50000)

    def load_data(self):
        data_file_train_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_train.csv'
        data_file_test_path = '/home/dengkaiting/pycharm_project/aclImdb/move_data_test.csv'
        if os.path.isfile(data_file_train_path) and os.path.isfile(data_file_test_path):
            logging.info('data exist')
        else:
            logging.info('generate data...')
            labels = {'pos': 1, 'neg': 0}
            data_train = pd.DataFrame()
            data_test = pd.DataFrame()
            for train_type in ('test', 'train'):
                for sentiment in ('pos', 'neg'):
                    path = '/home/dengkaiting/pycharm_project/aclImdb/%s/%s' % (train_type, sentiment)
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
    def interface(train_data, validation_data):
        train_data_Y = train_data['sentiment']
        validation_data_Y = validation_data['sentiment']
        # 转化为词向量
        max_feature = 100

        tokenizer = Tokenizer(num_words=max_feature, lower=True, split=' ')
        tokenizer.fit_on_texts(train_data['review'].values)

        X_train = tokenizer.texts_to_sequences(train_data['review'].values)
        X_validation = tokenizer.texts_to_sequences(validation_data['review'].values)

        # 填充序列到相同的长度    Pads sequences to the same length
        X_train = pad_sequences(X_train)
        X_validation = pad_sequences(X_validation)
        logging.info('train sequences :%s' % len(X_train))
        logging.info('validation sequences :%s' % len(X_validation))

        embed_dim = 3
        batch_size = 32

        model = Sequential()
        model.add(Embedding(input_dim=max_feature, output_dim=embed_dim))
        model.add(LSTM(units=embed_dim, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=1, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logging.info(model.summary())

        model.fit(X_train, train_data_Y, batch_size=batch_size, epochs=1, verbose=5, validation_data=(X_validation, validation_data_Y))
        score, acc = model.evaluate(X_validation, validation_data_Y, verbose=2, batch_size=batch_size)
        logging.info('Score: %.2f' % score)
        logging.info('Validation Accuracy: %.2f' % acc)


if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    imdb_lstm = IMDB_LSTM()
    data_train, data_test = imdb_lstm.load_data()
    data_train = data_train.sample(frac=0.05)
    data_test = data_test.sample(frac=0.5)
    data_train = imdb_lstm.clean_data(data_train)
    data_test = imdb_lstm.clean_data(data_test)
    imdb_lstm.interface(data_train, data_test)
    logging.info('over')
