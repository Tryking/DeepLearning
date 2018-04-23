import logging
import math
import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split


def load_dataset(files, test_size=0.2):
    x = []
    y = []
    for file in files:
        data = np.load(file)
        if x == [] or y == []:
            x = data['x']
            y = data['y']
        else:
            np.append(x, data['x'], axis=0)
            np.append(y, data['y'], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test


def network(x, keep=0.8):
    n_hidden = 64  # Hidden layer num of features
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DynamicRNNLayer(network,
                                        cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=n_hidden,
                                        dropout=keep, sequence_length=tl.layers.retrieve_seq_length_op(x),
                                        return_seq_2d=True, return_last=True, name='dynamic_rnn')
    network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity, name='output')
    network.outputs_op = tf.argmax(tf.nn.softmax(network.outputs), 1)
    return network


def load_checkpoint(sess, ckpt_file):
    index = ckpt_file + '.index'
    meta = ckpt_file + '.meta'
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.train.Saver().restore(sess, ckpt_file)


def save_checkpoint(sess, ckpt_file):
    path = os.path.dirname(os.path.abspath(ckpt_file))
    if not os.path.isdir(path):
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_file)


def train(sess, x, network):
    learning_rate = 0.1
    n_classes = 1
    y = tf.placeholder(tf.int64, [None, ], name='labels')
    cost = tl.cost.cross_entropy(network.outputs, y, 'xentropy')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(network.outputs_op, y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 使用TensorBoard可视化loss与准确率
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    writter_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writter_test = tf.summary.FileWriter('./logs/test')

    x_train, y_train, x_test, y_test = load_dataset(
        ["../word2vec/output/sample_seq_pass.npz",
         "../word2vec/output/sample_seq_spam.npz"])

    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, ckpt_file=ckpt_file)

    n_epoch = 2  # 所有样本重复训练2次
    batch_size = 128
    test_size = 128
    display_step = 10
    step = 0
    total_step = math.ceil(len(x_train) / batch_size) * n_epoch
    logging.info('batch_size: %d', batch_size)
    logging.info('Start training the network...')

    for epoch in range(n_epoch):



if __name__ == '__main__':
    ckpt_file = "./rnn_checkpoint/rnn.ckpt"
