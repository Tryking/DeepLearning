import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

# 生成一个写文件的writer，并将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter('/home/dengkaiting/pycharm_project/DeepLearning/tensorflow_book/logs', tf.get_default_graph())
writer.close()


