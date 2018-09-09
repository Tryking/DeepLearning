"""
TensorFlow 卷积神经网络识别手写数字
"""
import time

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

"""
建立共享函数
"""


# 定义weight函数，用于建立权重张量
def weight(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='W')


# 定义 bias 函数，用于建立偏差张量
def bias(shape):
    return tf.Variable(initial_value=tf.constant(value=1.0, shape=shape), name='b')


# 定义 conv2d 函数，用于进行卷积计算
def conv2d(x, W):
    """
    strides：滤镜的步长，其格式是[1, stride, stride, 1]，也就是滤镜每次移动时，从左到右，从上到下各一步
    padding：设置为"SAME"模式，此模式会在边界外自动补0，在进行计算时，让输入与输出图像的大小相同
    """
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


# 定义 max_pool_2x2 函数，用于建立池化层
def max_pool_2x2(x):
    """
    ksize：缩减采样窗口的大小，其格式为[1, height, width, 1]，也就是高度=2，宽度=2的窗口
    strides：缩减采样窗口的跨步，其格式是[1, stride, stride, 1]，
            也就是缩减采样窗口，从左到右、从上到下移动时的步长各两步
    """
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


"""
建立模型
"""
# 输入层input（28x28 的图像共1层）
with tf.name_scope(name='Input_Layer'):
    x = tf.placeholder('float32', shape=[None, 784], name='x')
    """
        第1维是-1：因为后续训练时要通过placeholder输入的项数不固定，所以设置为-1
        第2，3维是28，28：输入的数字图像大小是28 x 28
        第4维是1：因为是单色，所以设置为1，如果是彩色，就要设置为3
    """
    x_image = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])

# 卷积层1（C1_Conv，28x28的图像共16层）
with tf.name_scope(name='C1_Conv'):
    """
        第1，2维均是5：代表滤镜（filter weight）的大小是 5x5
        第3维是1：因为数字图像是单色的，所以设置为1，如果是彩色的，就要设置为3
        第4维是16：要产生16个图像
    """
    w1 = weight(shape=[5, 5, 1, 16])
    b1 = bias(shape=[16])
    Conv1 = conv2d(x=x_image, W=w1) + b1
    C1_Conv = tf.nn.relu(Conv1)

# 池化层1（C1_Pool，14x14的图像共16层）
with tf.name_scope(name='C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

# 卷积层2（C2_Conv，14x14的图像共36层）
with tf.name_scope(name='C2_Conv'):
    W2 = weight([5, 5, 16, 36])
    b2 = bias([36])
    Conv2 = conv2d(x=C1_Pool, W=W2) + b2
    C2_Conv2 = tf.nn.relu(Conv2)

# 池化层2（C2_Pool，7x7的图像共36层）
with tf.name_scope(name='C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv2)

# 平坦层（D_Flat 764个神经元）
with tf.name_scope(name='D_Flat'):
    """
        第1维是-1，因为后续会传入不限定项数的训练数据---数字图像
        第2维是1764，因为C2_Pool是36个 7x7 的图像，要转换为1维的向量，长度是 36x7x7=1764
    """
    D_Flat = tf.reshape(tensor=C2_Pool, shape=[-1, 1764])

# 隐藏层（D_Hidden_Dropout 128个神经元）
with tf.name_scope(name='D_Hidden_Layer'):
    W3 = weight(shape=[1764, 128])
    b3 = bias(shape=[128])
    D_Hidden = tf.nn.relu(features=tf.matmul(a=D_Flat, b=W3) + b3)
    D_Hidden_Droupout = tf.nn.dropout(x=D_Hidden, keep_prob=0.8)

# 输出层（y_predict 10个神经元）
with tf.name_scope('Output_Layer'):
    W4 = weight(shape=[128, 10])
    b4 = bias(shape=[10])
    y_predict = tf.nn.softmax(logits=tf.matmul(a=D_Hidden_Droupout, b=W4) + b4)

"""
定义训练方式
"""
with tf.name_scope(name='optimizer'):
    y_label = tf.placeholder(dtype='float32', shape=[None, 10], name='y_label')
    loss_function = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss_function)

# 定义评估模型准确率的方式
with tf.name_scope(name='evaluate_model'):
    correct_prediction = tf.equal(x=tf.argmax(input=y_predict, axis=1), y=tf.argmax(input=y_label, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype='float32'))

"""
进行训练
"""
# 定义训练参数
trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
epoch_list = []
accuracy_list = []
loss_list = []
start_time = time.time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 进行训练
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batchSize)
        sess.run(fetches=optimizer, feed_dict={x: batch_x, y_label: batch_y})
    loss, acc = sess.run(fetches=[loss_function, accuracy],
                         feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print('Train Epoch:', '%02d' % (epoch + 1), ' Loss=', '{:.9f}'.format(loss), ' Accuracy=', acc)
duration = round((time.time() - start_time), ndigits=2)
print('Train Finished takes: ', duration)

# 将要显示的计算图写入log文件
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN', sess.graph)


