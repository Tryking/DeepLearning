"""
TensorFlow张量运算仿真神经网络
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = tf.Variable([[0.4, 0.2, 0.4]])
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])

b = tf.Variable([[0.1, 0.2]])

XWb = tf.matmul(a=X, b=W) + b

y = tf.nn.relu(tf.matmul(a=X, b=W) + b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb :', sess.run(fetches=XWb))
    print('y :', sess.run(fetches=y))

# 以正态分布的随机数生成权重与偏差的初始值
W = tf.Variable(initial_value=tf.random_normal([3, 2]))
b = tf.Variable(initial_value=tf.random_normal(shape=[1, 2]))
X = tf.Variable(initial_value=[[0.4, 0.2, 0.4]])
y = tf.nn.relu(tf.matmul(a=X, b=W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    (b, W, y) = sess.run(fetches=(b, W, y))
    print('b: ', b)
    print('W ', W)
    print('y: ', y)

# 正太分布的随机数 tf.random_normal
ts_norm = tf.random_normal([10000])
with tf.Session() as sess:
    normal_data = ts_norm.eval()
print(normal_data[:5])
plt.hist(x=normal_data)
plt.show()

# 以 placeholder 传入 X 值
W = tf.Variable(initial_value=tf.random_normal([3, 2]))
b = tf.Variable(initial_value=tf.random_normal([1, 2]))
X = tf.placeholder(dtype='float32', shape=[None, 3])
y = tf.nn.relu(features=tf.matmul(a=X, b=W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.3, -0.4, 0.5]])
    (_b, _W, _X, _y) = sess.run(fetches=(b, W, X, y), feed_dict={X: X_array})
    print('b: ', _b)
    print('W: ', _W)
    print('X: ', _X)
    print('y: ', _y)


# 创建layer函数以矩阵运算仿真神经网络
def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(initial_value=tf.random_normal(shape=[input_dim, output_dim]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_dim]))
    XWb = tf.matmul(a=inputs, b=W) + b
    if activation:
        outputs = activation(XWb)
    else:
        outputs = XWb
    return outputs


# 使用layer函数建立3层神经网络
X = tf.placeholder(dtype='float32', shape=[None, 4])
h = layer(output_dim=3, input_dim=4, inputs=X, activation=tf.nn.relu)
y = layer(output_dim=2, input_dim=3, inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (layer_X, layer_h, layer_y) = sess.run(fetches=(X, h, y), feed_dict={X: X_array})
    print('input Layer X: ', layer_X)
    print('hidden Layer h: ', layer_h)
    print('output Layer y: ', layer_y)


# 建立layer_debug 函数显示权重与偏差
def layer_debug(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal(shape=[input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation:
        outputs = activation(XWb)
    else:
        outputs = XWb
    return outputs, W, b


# 使用layer_debug函数建立3层类神经网络，并显示第一层的W1与b1，以及第二层
X = tf.placeholder(dtype='float32', shape=[None, 4])
h, W1, b1 = layer_debug(output_dim=3, input_dim=4, inputs=X, activation=tf.nn.relu)
y, W2, b2 = layer_debug(output_dim=2, input_dim=3, inputs=h)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(fetches=init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (layer_X, layer_h, W1, b1, W2, b2) = sess.run(fetches=(X, h, W1, b1, W2, b2), feed_dict={X: X_array})
    print('input Layer X: ', layer_X)
    print('W1: ', W1)
    print('b1: ', b1)
    print('hidden Layer h: ', layer_h)
    print('W2: ', W2)
    print('b2: ', b2)
    print('output Layer y: ', layer_y)
