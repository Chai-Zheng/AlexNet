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

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import AlexNet
import caffe_classes

path = 'test_images'

withPath = lambda f: '{}/{}'.format(path, f)
test_images = dict((f, cv2.imread(withPath(f))) for f in os.listdir(path) if os.path.isfile(withPath(f)))

if test_images.values():
    dropout_prob = 1.0
    skip_layer = []

    image_mean = np.array([104, 117, 124], np.float)
    x = tf.placeholder('float', [1, 227, 227, 3])

    trained_model = AlexNet.AlexNet(x, dropout_prob, skip_layer)
    y_predict = trained_model.fc8

    fig = plt.figure()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trained_model.load_weights(sess)

        j = 0
        for i, img in test_images.items():
            image_resized = cv2.resize(img.astype(np.float), (227, 227))-image_mean
            probs = sess.run(y_predict, feed_dict={x: image_resized.reshape(1, 227, 227, 3)})
            max_prob = np.max(probs)
            y_pre = caffe_classes.class_names[np.argmax(probs)]

            fig.add_subplot(1, 3, j+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('class:{} probability:{}'.format(y_pre, max_prob))
            j += 1

        plt.show()
