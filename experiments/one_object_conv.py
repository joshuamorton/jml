"""A CNN model to recognize the centers of circles.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn

import sys

from utils.circles import random_boxed_circle

# 10x10 greyscale image
image = tf.placeholder(tf.float32, [None, 10, 10, 1])
conv = tf.nn.conv2d(image, tf.Variable(tf.zeros([3,3,1,1])), [1,1,1,1], 'SAME')
bias = tf.Variable(tf.zeros([1]))
layer = tf.nn.relu(tf.nn.bias_add(conv, bias))
W = tf.Variable(tf.zeros([100, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(tf.reshape(layer, [-1, 100]), W) + b)
y_prime = tf.placeholder(tf.float32, [None, 2])

mse = tf.reduce_mean(tf.squared_difference(y, y_prime))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(mse)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in range(10000):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_boxed_circle(10, 10, 1) for i in range(50)])
        xs = [x.reshape([10, 10, 1]) for x in xs]
        sess.run(train_step, feed_dict={image: xs, y_prime: ys})
        if _ % 100 == 0:
            mses.append(sess.run(mse, feed_dict={image: xs, y_prime: ys}))

print(xs)
plt.plot(mses)
plt.show()

