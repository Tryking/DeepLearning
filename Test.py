import collections

import pandas as pd

# import pyprind
import time
import logging
# import tensorflow as tf

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

import re


def build_words_dataset(words=None, vocabulary_size=50000, printable=True, unk_key='UNK'):
    if words is None:
        raise Exception("words : list of str or byte")

    count = [[unk_key, -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    if printable:
        logging.info('Real vocabulary size    %d' % len(collections.Counter(words).keys()))
        logging.info('Limited vocabulary size {}'.format(vocabulary_size))
    if len(collections.Counter(words).keys()) < vocabulary_size:
        raise Exception(
            "len(collections.Counter(words).keys()) >= vocabulary_size , the limited vocabulary_size must be less than or equal to the read vocabulary_size")
    return data, count, dictionary, reverse_dictionary


words = ['您好呀', '您好', '是的', '是的', 'okey']
build_words_dataset(words=words)
