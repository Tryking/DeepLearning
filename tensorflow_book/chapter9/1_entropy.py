import tensorflow as tf

word_labels = tf.constant([2, 0])

predict_logits = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])

# 使用sparse_softmax_cross_entropy_with_logits计算交叉熵
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)

# 运行程序，计算loss的结果，对应两个预测的perplexity损失
sess = tf.Session()
sess.run(loss)
print(loss)

# sotfmax_cross_entropy_with_logits与上面的函数类似，但是需要将预测目标以概率分布的形式给出
word_prob_distribution = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=word_prob_distribution, logits=predict_logits)
sess.run(loss)
print(loss)
