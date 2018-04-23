import logging

import collections
import os
import tarfile

import pyprind
import tensorlayer as tl
import tensorflow as tf


def load_dataset():
    prj = 'http://github.com/pakrchen/text-antispam'
    if not os.path.exists('data/msglog'):
        tl.files.maybe_download_and_extract(
            filename='msglog.tar.gz', working_directory='data',
            url_source=prj + '/raw/master/word2vec/data/')
        tarfile.open('data/msglog.tar.gz', 'r').extractall('data')
    files = ['data/msglog/msgpass.log.seg', 'data/msglog/msgspam.log.seg']
    words = []
    for file in files:
        with open(file) as f:
            for line in f:
                for word in line.strip().split(' '):
                    if word != '':
                        words.append(word)
    return words


def get_vocabulary_size(words, min_freq=3):
    size = 1  # 有一个UNK
    counts = collections.Counter(words).most_common()
    for word, c in counts:
        if c >= min_freq:
            size += 1
    return size


def save_checkpoint(ckpt_file_path):
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if not os.path.isdir(path):
        logging.warning('(%s) not exists, making directories...' % path)
        os.makedirs(path)
    tf.train.Saver().save(sess=sess, save_path=ckpt_file_path + '.ckpt')


def load_checkpoint(ckpt_file_path):
    ckpt = ckpt_file_path + '.ckpt'
    index = ckpt + '.index'
    meta = ckpt + '.meta'
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.train.Saver().restore(sess=sess, save_path=ckpt)


def save_embedding(dictionary, network, embedding_file_path):
    words, ids = zip(*dictionary.items())
    params = network.normalized_embeddings
    embeddings = tf.nn.embedding_lookup(params=params, ids=tf.constant(ids, dtype=tf.int32)).eval()
    wv = dict(zip(words, embeddings))
    path = os.path.dirname(os.path.abspath(embedding_file_path))
    if not os.path.isdir(path):
        logging.warning('(%s) not exists, making directories...' % path)
        os.makedirs(path)
    tl.files.save_any_to_npy(save_dict=wv, name=embedding_file_path + '.npy')


def train(model_name):
    words = load_dataset()
    data_size = len(words)
    vocabulary_size = get_vocabulary_size(words=words, min_freq=3)
    batch_size = 500  # 一次Forward运算以及BP运算中所需要的训练样本数组
    embedding_size = 200  # 词向量维度
    skip_window = 5  # 上下文窗口，单词前后各取5个词
    num_skips = 10  # 从窗口中选取多少个预测对
    num_sapmled = 64  # 负采样个数
    learning_rate = 0.1  # 学习率
    n_epoch = 10  # 所有样本重复训练10次
    num_steps = int((data_size / batch_size) * n_epoch)  # 总迭代次数

    data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words=words, vocabulary_size=vocabulary_size)
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        emb_net = tl.layers.Word2vecEmbeddingInputlayer(inputs=train_inputs, train_labels=train_labels, vocabulary_size=vocabulary_size
                                                        , embedding_size=embedding_size, num_sampled=num_sapmled)
        loss = emb_net.nce_cost
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    tl.layers.initialize_global_variables(sess=sess)
    ckpt_file_path = 'checkpoint_temp/' + model_name
    load_checkpoint(ckpt_file_path=ckpt_file_path)

    pbar = pyprind.ProgBar(num_steps)
    step = data_index = 0
    loss_vals = []
    while step < num_steps:
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
            data=data, batch_size=batch_size, num_skips=num_skips
            , skip_window=skip_window, data_index=data_index)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        loss_vals.append(loss_val)
        if step != 0 and step % 2000 == 0:
            logging.info('(%d/%d) latest average loss: %f.', step, num_steps, sum(loss_vals) / len(loss_vals))
            save_checkpoint(ckpt_file_path)
            embedding_file_path = 'output/' + model_name
            save_embedding(dictionary=dictionary, network=emb_net, embedding_file_path=embedding_file_path)
        step += 1
        pbar.update()


if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)

    sess = tf.InteractiveSession()  # 默认TensorFlow Session
    train('model_word2vec_200')
    sess.close()
