# !/usr/bin/env python3
# coding=utf-8

"""
AlexNet Using TensorFlow

Author : Chai Zheng, Ph.D.@Zhejiang University, Hangzhou
Email  : zchaizju@gmail.com
Blog   : http://blog.csdn.net/chai_zheng/
Github : https://github.com/Chai-Zheng/
Date   : 2018.4.1
"""

import tensorflow as tf
import numpy as np


class AlexNet(object):
    def __init__(self, x, keep_prob, skip_layer, weights_path='bvlc_alexnet.npy'):
        self.x = x
        self.keep_prob = keep_prob
        self.skip_layer = skip_layer
        self.weights_path = weights_path
        self.build_AlexNet()

    def build_AlexNet(self):
        conv1 = conv_layer(self.x, 96, 11, 11, 4, 4, 'conv1', groups=1, padding='VALID')
        norm1 = LRN_layer(conv1, 2, 1e-4, 0.75, 'norm1')
        pool1 = max_pool_layer(norm1, 3, 3, 2, 2, 'pool1')

        conv2 = conv_layer(pool1, 256, 5, 5, 1, 1, 'conv2', groups=2)
        norm2 = LRN_layer(conv2, 2, 1e-4, 0.75, 'norm2')
        pool2 = max_pool_layer(norm2, 3, 3, 2, 2, 'pool2', padding='VALID')

        conv3 = conv_layer(pool2, 384, 3, 3, 1, 1, 'conv3')

        conv4 = conv_layer(conv3, 384, 3, 3, 1, 1, 'conv4', groups=2)

        conv5 = conv_layer(conv4, 256, 3, 3, 1, 1, 'conv5', groups=2)
        pool5 = max_pool_layer(conv5, 3, 3, 2, 2, 'pool5', 'VALID')
        pool5_flatted = tf.reshape(pool5, [-1, 6*6*256], 'pool5_flatted')

        fc6 = fc_layer(pool5_flatted, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        self.fc8 = fc_output_layer(dropout7, 4096, 1000, name='fc8')

    def load_weights(self, sess):
        weights_dict = np.load(self.weights_path, encoding='bytes').item()
        for name in weights_dict:
            if name not in self.skip_layer:
                with tf.variable_scope(name, reuse=True):
                    for p in weights_dict[name]:
                        if len(p.shape) == 1:    # bias
                            var = tf.get_variable('b', trainable=False)
                            sess.run(var.assign(p))
                        else: # weights
                            var = tf.get_variable('w', trainable=False)
                            sess.run(var.assign(p))


def weights(shape):
    return tf.get_variable('w', shape, trainable=True)


def bias(shape):
    return tf.get_variable('b', shape, trainable=True)


def conv_layer(x, filter_num, filter_height, filter_width, stride_x, stride_y, name, groups=1, padding='SAME'):
    channel = int(x.shape[-1])
    conv2d = lambda a, b: tf.nn.conv2d(input=a, filter=b, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = weights([filter_height, filter_width, int(channel/groups), filter_num])
        b = bias([filter_num])

        x_split = tf.split(value=x, num_or_size_splits=groups, axis=3)
        w_split = tf.split(value=w, num_or_size_splits=groups, axis=3)

        conv_split = [conv2d(m, n) for m, n in zip(x_split, w_split)]
        conv_merge = tf.concat(conv_split, axis=3)

        return tf.nn.relu(conv_merge+b, name='scope.name')


def LRN_layer(x, R, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, name=name, bias=bias)


def max_pool_layer(x, filter_height, filter_width, stride_x, stride_y, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob, name)


def fc_layer(x, input_num, output_num, name):
    with tf.variable_scope(name) as scope:
        w = weights([input_num, output_num])
        b = bias([output_num])
        return tf.nn.relu(tf.matmul(x, w)+b)


def fc_output_layer(x, input_num, output_num, name):
    with tf.variable_scope(name) as scope:
        w = weights([input_num, output_num])
        b = bias([output_num])
        return tf.nn.softmax(tf.matmul(x, w)+b)
