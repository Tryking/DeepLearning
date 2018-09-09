"""
识别 CIFAR-10 数据集
"""
from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

print('train:', len(x_img_train))
print(' test:', len(x_img_test))

print(x_img_train.shape)
print(y_label_train.shape)

print(x_img_test[0])

label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(images=x_img_train, labels=y_label_train, prediction=[], idx=10, num=20)
