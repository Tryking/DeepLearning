import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项的损失函数中的系数
TRAINING_STEPS = 300000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 通过变量管理定义神经网络
def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1', reuse=reuse):
        # 根据传进来的reuse来判断是创建新变量还是使用已经创建好的。
        # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 第二层
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    # 返回最后的前向传播结果
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数下神经网络前向传播的结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
    # 在使用TensorFlow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类（可以加快训练早期变量的更新速度）
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量不需要。tf.trainable_variables 返回的就是图上的集合
    # GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的，学习率在这个基础上递减
        global_step=global_step,  # 当前迭代的轮数
        decay_steps=mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        decay_rate=LEARNING_RATE_DECAY  # 学习率衰减速度
    )

    # 使用GradientDescentOptimizer优化算法来优化损失函数。这里损失函数包括了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值
    # 为了一次完成多个操作，TensorFlow提供了tf.control_dependencies和tf.group两种机制
    # train_op = tf.group(train_step,variables_average_op)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(fetches=accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy '
                      'using average model is %g ' % (i, validate_acc))

                # 产生这一轮使用的一个batch的训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)

        print('After %d training step(s), test accuracy using average '
              'model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
