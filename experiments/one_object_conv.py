"""A CNN model to recognize the centers of circles.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import seaborn

from utils.circles import random_boxed_circle, circle

global_step = tf.Variable(0, trainable=False)
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
conv_bias = tf.Variable(tf.zeros([1]))
one_hot = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, conv_bias)), [-1, 100])

W = tf.get_variable('W', shape=[100, 2],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([2]), name='b')
coords = tf.nn.softmax(tf.matmul(one_hot, W) + b)

correct = tf.placeholder(tf.float32, [None, 2])

loss = tf.reduce_mean(tf.squared_difference(coords, correct))
learning_rate = tf.train.exponential_decay(
        .75, global_step, 1000, 0.8, staircase=True)
train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

dims = (range(1,9), range(1,9))
test_x = [circle(x, y, 1, shape=(10, 10)) for x, y in itertools.product(*dims)]
test_y = [(x/10, y/10) for x, y in itertools.product(*dims)]

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in tqdm(range(10000)):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_boxed_circle(10, 10, 1) for i in range(100)])
        xs = [x.reshape([10, 10, 1]) for x in xs]
        sess.run(train_step, feed_dict={image: xs, correct: ys})
        if _ % 100 == 0:
            mses.append(
                    sess.run(loss, feed_dict={image: test_x, correct: test_y}))

    pairs = sess.run(coords, feed_dict={image: test_x, correct: test_y})
    plt.plot(mses)
    plt.show()

