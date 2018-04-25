import logging
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import pyprind
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat


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
    test_size = 1280
    display_step = 10
    step = 0
    total_step = math.ceil(len(x_train) / batch_size) * n_epoch
    logging.info('batch_size: %d', batch_size)
    logging.info('Start training the network...')
    bar = pyprind.ProgBar(total_step)
    for epoch in range(n_epoch):
        for batch_x, batch_y in tl.iterate.minibatches(
                inputs=x_train, targets=y_train, batch_size=batch_size, shuffle=True):
            start_time = time.time()
            max_seq_len = max([len(d) for d in batch_x])
            for i, d in enumerate(batch_x):
                batch_x[i] += [np.zeros(200) for i in range(max_seq_len - len(d))]
            batch_x = list(batch_x)
            feed_dict = {x: batch_x, y: batch_y}
            sess.run(optimizer, feed_dict=feed_dict)

            # TensorBoard打点
            summary = sess.run(merged, feed_dict=feed_dict)
            writter_train.add_summary(summary=summary, global_step=step)

            # 计算测试机准确率
            start = random.randint(0, len(x_test) - test_size)
            test_data = x_test[start:(start + test_size)]
            test_label = y_test[start:(start + test_size)]
            max_seq_len = max([len(d) for d in test_data])
            for i, d in enumerate(test_data):
                test_data[i] += [np.zeros(200) for i in range(max_seq_len - len(d))]
            test_data = list(test_data)
            summary = sess.run(merged, {x: test_data, y: test_label})
            writter_test.add_summary(summary=summary, global_step=step)

            # 每十步输出loss值与准确率
            if step == 0 or (step + 1) % display_step == 0:
                logging.info("Epoch %d/%d Step %d/%d took %fs",
                             epoch + 1, n_epoch, step + 1, total_step, time.time() - start_time)
                loss = sess.run(cost, feed_dict=feed_dict)
                acc = sess.run(accuracy, feed_dict=feed_dict)
                logging.info("Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                save_checkpoint(sess=sess, ckpt_file=ckpt_file)
            step += 1
            bar.update()


def export(model_version, model_dir, sess, x, y_op):
    """
    导出tensorflow_serving可用的模型
    :param model_version:
    :param model_dir:
    :param sess:
    :param x:
    :param y_op:
    :return:
    """
    if model_version <= 0:
        logging.warning('Please specify a positive value of version')
        sys.exit()

    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        logging.warning('(%s) not exists, making directories...', path)
        os.makedirs(path)

    export_path = os.path.join(compat.as_bytes(model_dir), compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        logging.warning('(%s) exists, removing dirs...', export_path)
        shutil.rmtree(export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y_op)

    prediction_signature = signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'y': tensor_info_y},
                                                                   method_name=signature_constants.PREDICT_METHOD_NAME)
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={'predict_text': prediction_signature,
                                                            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature})
    builder.save()


if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)

    ckpt_file = "./rnn_checkpoint/rnn.ckpt"
    x = tf.placeholder('float', shape=[None, None, 200], name='inputs')
    sess = tf.InteractiveSession()

    """
        flags用于支持接受命令行传递参数，相当于接受argv
    """
    flags = tf.flags
    """
        mode:参数名称；train：默认值；train or export：description
    """
    flags.DEFINE_string('mode', 'export', 'train or export')
    FLAGS = flags.FLAGS

    if FLAGS.mode == 'train':
        network = network(x)
        train(sess=sess, x=x, network=network)
        logging.info('Optimization Finished!')
    elif FLAGS.mode == 'export':
        model_version = 1
        model_dir = './output/rnn_model'
        network = network(x=x, keep=1.0)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess=sess, ckpt_file=ckpt_file)
        export(model_version, model_dir=model_dir, sess=sess, x=x, y_op=network.outputs_op)
        logging.info('Servable Export Finished!')
