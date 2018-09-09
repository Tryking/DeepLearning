import time

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# 建立layer 函数
def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(a=inputs, b=W) + b
    if activation:
        outputs = activation(XWb)
    else:
        outputs = XWb
    return outputs


# 建立输入层
X = tf.placeholder(dtype='float32', shape=[None, 784])
# 建立隐藏层
h1 = layer(output_dim=256, input_dim=784, inputs=X, activation=tf.nn.relu)
# 建立输出层
y_predict = layer(output_dim=10, input_dim=256, inputs=h1, activation=None)

# 建立训练数据label真实值的placeholder
y_label = tf.placeholder('float32', shape=[None, 10])
# 定义损失函数
"""
深度学习模型训练中使用 cross_entropy 交叉熵训练的效果好
"""
loss_function = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
# 定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss_function)

# 定义评估模型准确率的方式
"""
计算每一项数据是否预测正确
"""
correct_prediction = tf.equal(x=tf.argmax(y_label, 1), y=tf.argmax(y_predict, 1))
"""
计算预测正确结果的平均值
先使用 tf.cast 转换为 float，再使用 tf.reduce_mean 将所有数值平均
"""
accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype='float32'))

# 进行训练
"""
定义训练参数
"""
# 训练周期
trainEpochs = 15
# 每一批次项数
batchSize = 100
loss_list = []
epoch_list = []
accuracy_list = []
# 计算每个训练周期
totalBatches = int(mnist.train.num_examples / batchSize)
start_time = time.time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
进行训练
"""
for epoch in range(trainEpochs):
    for i in range(totalBatches):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batchSize)
        sess.run(fetches=optimizer, feed_dict={X: batch_x, y_label: batch_y})
    loss, acc = sess.run(fetches=[loss_function, accuracy],
                         feed_dict={X: mnist.validation.images, y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print('Train Epoch: ', '%02d' % (epoch + 1), 'Loss=', '{:.9f}'.format(loss), ' Accuracy=', acc)

duration = round((time.time() - start_time), ndigits=2)
print('Train Finished takes: ', duration, ' s')

# 画出误差执行结果
fig = plt.gcf()
fig.set_size_inches(w=(4, 2))
plt.plot(epoch_list, loss_list, label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()

# 画出准确率执行结果
plt.plot(epoch_list, accuracy_list, label='accuracy')
fig.set_size_inches(w=(4, 2))
plt.ylim(0.8, 1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
