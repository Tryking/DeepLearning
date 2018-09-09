import tensorflow as tf

# 建立"计算图"
# 常量
ts_c = tf.constant(value=2, name='ts_c')
# 变量
ts_x = tf.Variable(initial_value=ts_c + 5, name='ts_x')

# 执行"计算图"
"""
执行"计算图"前必须先建立 Session（会话），在TensorFlow中Session代表在客户端和执行设备之间建立连接。
有了这个连接，就可以在设备中执行"计算图"，后续任何与设备之间沟通都必须通过这个Session，并且可取得执行后的结果。
"""
with tf.Session() as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    print('ts_c = ', sess.run(fetches=ts_c))
    print('ts_x = ', sess.run(fetches=ts_x))
    print('ts_x = ', ts_x.eval(session=sess))

width = tf.placeholder(dtype='int32')
height = tf.placeholder(dtype='int32')
area = tf.multiply(x=width, y=height, name='test_placeholder')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area = ', sess.run(fetches=area, feed_dict={width: 6, height: 8}))
