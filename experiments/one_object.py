"""A machine learning model to recognize the centers of circles.

The general outline is that we can create arbitrary circles and pass them into
a tensorflow model which will locate their centers. This should be a relatively
simple task, since at first at least there will be no partial circles (meaning
that the entire model can be described as learning the average function over
our 2d input space).

Unsurprisingly, this model performs (very) poorly.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn

import sys

from utils.circles import random_boxed_circle

# 100x100 greyscale image
x = tf.placeholder(tf.float32, [None, 10000])
W = tf.Variable(tf.zeros([10000, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_prime = tf.placeholder(tf.float32, [None, 2])
mse = tf.reduce_mean(tf.squared_difference(y, y_prime))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mse)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in range(10000):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_boxed_circle(100, 100, 5) for i in range(50)])
        xs = [x.flatten() for x in xs]
        sess.run(train_step, feed_dict={x: xs, y_prime: ys})
        if _ % 100 == 0:
            mses.append(sess.run(mse, feed_dict={x: xs, y_prime: ys}))

plt.plot(mses)
plt.show()
