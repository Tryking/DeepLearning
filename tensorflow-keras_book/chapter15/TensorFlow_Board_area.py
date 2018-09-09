import tensorflow as tf

width = tf.placeholder(dtype='int32', name='width')
height = tf.placeholder(dtype='int32', name='height')
area = tf.multiply(x=width, y=height, name='area')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area = ', sess.run(fetches=area, feed_dict={width: 6, height: 8}))

    # 将所有要显示的数据整合
    tf.summary.merge_all()
    # 将所有要显示在 TensorBoard 的数据写入log文件
    train_writer = tf.summary.FileWriter('log/area', sess.graph)

# 建立一维张量（向量）
ts_X = tf.Variable([0.4, 0.2, 0.4])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X = sess.run(ts_X)
    print(X)
    print(X.shape)

# 建立二维张量
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(fetches=init)
    W_array = sess.run(fetches=W)
    print(W_array)

# 矩阵乘法与加法
X = tf.Variable([[1., 1., 1.]])
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])

b = tf.Variable(initial_value=[[0.1, 0.2]])

XWb = tf.matmul(a=X, b=W) + b
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(fetches=init)
    print('X:', sess.run(X))
    print('W:', sess.run(W))
    print('XWb:', sess.run(fetches=XWb))
