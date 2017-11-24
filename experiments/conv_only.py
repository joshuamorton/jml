"""A CNN model to recognize the centers of circles.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import seaborn

import sys

from utils.circles import random_boxed_circle

def make_single(vals, shape=(10, 10)):
    cumprod = np.cumprod([1] + list(shape))[:-1][::-1]
    return [sum(val * cumprod) for val in vals]

conv_filter = tf.get_variable('conv_filter', shape=[3,3,1,1], 
        initializer=tf.contrib.layers.xavier_initializer())

# 10x10 greyscale image
image = tf.placeholder(tf.float32, [None, 10, 10, 1], name='image')
conv = tf.nn.conv2d(
        image,
        filter=conv_filter,
        strides=[1,1,1,1],
        padding='SAME',
        name='conv',
    )
bias = tf.Variable(tf.zeros([1]))
one_hot = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, bias)), [-1, 100])
correct = tf.placeholder(tf.int32, [None])


loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=correct,
            logits=one_hot,
            )
train_step = tf.train.GradientDescentOptimizer(.05).minimize(loss)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in range(10000):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ycords = zip(*[random_boxed_circle(10, 10, 1) for i in range(10)])
        xs = [x.reshape([10, 10, 1]).transpose((1, 0, 2)) for x in xs]
        ys = make_single(ycords, shape=(10, 10))
        sess.run(train_step, feed_dict={image: xs, correct: ys})
        if _ % 100 == 0:
            mses.append(sess.run(loss, feed_dict={image: xs, correct: ys}))

    plt.plot(mses)
    print(mses)
    plt.show()

