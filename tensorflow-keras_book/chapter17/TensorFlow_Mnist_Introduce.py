import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 查看 Mnist 数据
print('train: ', mnist.train.num_examples,
      '\nvalidation: ', mnist.validation.num_examples,
      '\ntest: ', mnist.test.num_examples)

# 查看训练数据
print('train images: ', mnist.train.images.shape, '\nlabels: ', mnist.train.labels.shape)


def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


plot_image(mnist.train.images[0])


# 查看多项训练数据 images 与 labels
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(w=(12, 14))
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        # TensorFlow的MNIST数据集image为784，必须转换为 28 * 28 才能显示出来
        ax.imshow(X=np.reshape(images[idx], (28, 28)), cmap='binary')
        # TensorFlow的MNIST数据集是 One-Hot Encoding 格式，必须转换为数字才能显示
        title = 'label=' + str(np.argmax(labels[idx]))
        if len(prediction) > 0:
            title += ', predict=' + str(prediction[idx])

        ax.set_title(label=title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(images=mnist.train.images, labels=mnist.train.labels, prediction=[], idx=0)

# 批次读取 MNIST 数据
batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=100)
print(len(batch_images_xs), len(batch_labels_ys))
