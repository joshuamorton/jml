"""A CNN model to recognize the centers of circles.
"""

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from utils.point import point, random_point

DIMS = 10

# 10x10 greyscale image
x = tf.placeholder(tf.float32, [None, DIMS ** 2])
W = tf.Variable(tf.random_normal([DIMS ** 2, 2], .5, .16))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_prime = tf.placeholder(tf.float32, [None, 2])

mse = tf.reduce_mean(tf.squared_difference(y, y_prime))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(mse)

test_x, test_y = zip(
    *[point(DIMS, DIMS, n % DIMS, n // DIMS) for n in range(DIMS ** 2)]
)
test_x = [x.reshape([DIMS ** 2]) for x in test_x]
test_y = [(o / DIMS, t / DIMS) for o, t in test_y]

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in range(100000):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_point(DIMS, DIMS) for i in range(50)])
        xs = [x.reshape([DIMS ** 2]) for x in xs]
        ys = [(o / DIMS, t / DIMS) for o, t in ys]
        sess.run(train_step, feed_dict={x: xs, y_prime: ys})
        if _ % 100 == 0:
            mses.append(sess.run(mse, feed_dict={x: test_x, y_prime: test_y}))
            print(mses[-1] * 100)

    print(W.eval())
    plt.plot(mses)
    # This magic number is the average distance between two points in a unit
    # square, squared. The exact formula is
    # (1/15 * (sqrt(2) + 2 + 5 * ln(1 + sqrt(2)))) ** 2, because...duh
    # plt.axhline(y=.27186)
    plt.show()
