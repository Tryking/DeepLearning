import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# stddev:标准差，seed：随机种子，可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape 的一个维度上使用None可以方便使用不同的batch大小。在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。
# 当数据集较小时这样比较方便测试，但数据集较大时，将大量数据放入一个batch可能会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
# 使用sigmoid函数将y转换成0~1之间的数值。转换后y代表预测是正样本的概率，1-y代表预测是负样本的概率
y = tf.sigmoid(y)
# cross_entropy定义了真实值和预测值之间的交叉熵（cross entropy）
"""
    tf.clip_by_value可以将一个张量中的数值限制在一个范围之内，这样可以避免一些运算错误。小于1e-10的换成1e-10，大于1的换成1
    tf.log：对张量中所有元素依次求对数
    * ： 将矩阵对应元素之间直接相乘，和tf.matmul函数不一样
"""
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并跟新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
