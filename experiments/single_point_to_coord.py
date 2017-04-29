"""A CNN model to recognize the centers of circles.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn

from utils.point import random_point

# 10x10 greyscale image
x = tf.placeholder(tf.float32, [None, 100])
W = tf.Variable(tf.random_normal([100, 2], .5, .16))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_prime = tf.placeholder(tf.float32, [None, 2])

mse = tf.reduce_mean(tf.squared_difference(y, y_prime))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(mse)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in range(10000):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_point(10, 10) for i in range(50)])
        xs = [x.reshape([100]) for x in xs]
        ys = [(o/10, t/10) for o, t in ys]
        sess.run(train_step, feed_dict={x: xs, y_prime: ys})
        if _ % 100 == 0:
            mses.append(sess.run(mse, feed_dict={x: xs, y_prime: ys}))
            print(mses[-1] * 100)

    print(W.eval())
    plt.plot(mses)
    plt.axhline(y=0.57 ** 2)
    plt.show()

