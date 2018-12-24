"""
Yolo V1
"""
import numpy as np
import tensorflow as tf


def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


class Yolo(object):
    def __init__(self, weights_file, verbose=True):
        self.verbose = verbose
        # detection params
        self.S = 7  # cell size
        self.B = 2  # boxes per cell
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.C = len(self.classes)  # number of classes

        # offset for box center (top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B), [self.B, self.S, self.S]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.threshold = 0.2  # confidence scores threshold
        self.iou_threshold = 0.4
        # the maximum number of boxes to be selected by non max suppression
        self.max_output_size = 10

        self.sess = tf.Session()
        self._build_net()
        self._build_detector()
        self._load_weights(weights_file)

    def _build_net(self):
        """ build the network"""
        if self.verbose:
            print('Start to build the network ...')
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])
        net = self._conv_layer(x=self.images, id=1, num_filters=64, filter_size=7, stride=2)
        net = self._maxpool_layer(x=net, id=1, pool_size=2, stride=2)
        net = self._conv_layer(x=net, id=2, num_filters=192, filter_size=3, stride=1)
        net = self._maxpool_layer(x=net, id=2, pool_size=2, stride=2)
        net = self._conv_layer(x=net, id=3, num_filters=128, filter_size=1, stride=1)
        net = self._conv_layer(x=net, id=4, num_filters=256, filter_size=3, stride=1)
        net = self._conv_layer(x=net, id=5, num_filters=256, filter_size=1, stride=1)
        net = self._conv_layer(x=net, id=6, num_filters=512, filter_size=3, stride=1)
        net = self._maxpool_layer(x=net, id=6, pool_size=2, stride=2)
        net = self._conv_layer(net, 7, 256, 1, 1)
        net = self._conv_layer(net, 8, 512, 3, 1)
        net = self._conv_layer(net, 9, 256, 1, 1)
        net = self._conv_layer(net, 10, 512, 3, 1)
        net = self._conv_layer(net, 11, 256, 1, 1)
        net = self._conv_layer(net, 12, 512, 3, 1)
        net = self._conv_layer(net, 13, 256, 1, 1)
        net = self._conv_layer(net, 14, 512, 3, 1)
        net = self._conv_layer(net, 15, 512, 1, 1)
        net = self._conv_layer(net, 16, 1024, 3, 1)
        net = self._maxpool_layer(net, 16, 2, 2)
        net = self._conv_layer(net, 17, 512, 1, 1)
        net = self._conv_layer(net, 18, 1024, 3, 1)
        net = self._conv_layer(net, 19, 512, 1, 1)
        net = self._conv_layer(net, 20, 1024, 3, 1)
        net = self._conv_layer(net, 21, 1024, 3, 1)
        net = self._conv_layer(net, 22, 1024, 3, 2)
        net = self._conv_layer(net, 23, 1024, 3, 1)
        net = self._conv_layer(net, 24, 1024, 3, 1)
        net = self._flatten(x=net)
        net = self._fc_layer(x=net, id=25, num_out=512, activation=leak_relu)
        net = self._fc_layer(x=net, id=26, num_out=4096, activation=leak_relu)
        net = self._fc_layer(x=net, id=27, num_out=self.S * self.S * (self.C + 5 * self.B))
        self.predicts = net

    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        """Conv Layer"""
        in_channels = x.get_shape().as_list()[-1]
        # truncated_normal 返回一个截断的正态分布
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters, ]))
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        # strides: [batch, height, width, channels]，确定了滑动窗口在各个维度上移动的步数
        # padding：‘VALID’：最后一个不足时舍弃，‘SAME’：最后一个不足时补齐
        conv = tf.nn.conv2d(input=x_pad, filter=weight, strides=[1, stride, stride, 1], padding='VALID')
        output = leak_relu(tf.nn.bias_add(value=conv, bias=bias))
        if self.verbose:
            print('     Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s' % (
                id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _maxpool_layer(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(value=x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='SAME')
        if self.verbose:
            print('     Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s' % (
                id, pool_size, stride, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """ flatten the x """
        # channel first mode
        tran_x = tf.transpose(x, [0, 3, 1, 2])
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])

    def _fc_layer(self, x, id, num_out, activation=None):
        """ fully connected layer """
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out, ]))
        output = tf.nn.xw_plus_b(x=x, weights=weight, biases=bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("     Layer %d: type=Fc, num_out=%d, output_shape=%s" % (
                id, num_out, str(output.get_shape())))
        return output

    def _build_detector(self):
        """Interpret the net output and get the predicted boxes"""
        # the width and height of original image
        self.width = tf.placeholder(tf.float32, name='img_w')
        self.height = tf.placeholder(tf.float32, name='img_h')
        # get class prob, confidence, boxes from net output
        idx1 = self.S * self.S * self.C
        idx2 = idx1 + self.S * self.S * self.B
        # class prediction
        class_probs = tf.reshape(self.predicts[0, :idx1], [self.S, self.S, self.C])
        # confidence
        confs = tf.reshape(self.predicts[0, idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x,y,w,h)
        boxes = tf.reshape(self.predicts[0, idx2:], [self.S, self.S, self.B, 4])

        # convert the x,y to the coordinates relative to the top left point of the image
        # the predictions of w,h are the square root
        # multiply the width and height of image
        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S * self.width,
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S * self.height,
                          tf.square(boxes[:, :, :, 2]) * self.width,
                          tf.square(boxes[:, :, :, 3]) * self.height], axis=3)

        pass


if __name__ == '__main__':
    yolo = Yolo(weights_file=None)
