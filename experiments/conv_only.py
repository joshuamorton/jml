"""A CNN model to recognize the centers of circles.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tensorflow as tf
from tqdm import tqdm
from utils import flatten_indicies
from utils.circles import circle, random_boxed_circle

conv_filter = tf.get_variable(
    "conv_filter",
    shape=[3, 3, 1, 1],
    initializer=tf.contrib.layers.xavier_initializer(),
)

# 10x10 greyscale image
image = tf.placeholder(tf.float32, [None, 10, 10, 1], name="image")
conv = tf.nn.conv2d(
    image, filter=conv_filter, strides=[1, 1, 1, 1], padding="SAME", name="conv"
)
bias = tf.Variable(tf.zeros([1]))
one_hot = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, bias)), [-1, 100])
correct = tf.placeholder(tf.int32, [None])


loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct, logits=one_hot)
train_step = tf.train.GradientDescentOptimizer(.05).minimize(loss)

dims = (range(1, 9), range(1, 9))
test_x = [circle(x, y, 1, shape=(10, 10)) for x, y in itertools.product(*dims)]
test_y = flatten_indicies(itertools.product(*dims))

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in tqdm(range(10000)):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ycords = zip(*[random_boxed_circle(10, 10, 1) for i in range(100)])
        xs = [x.reshape([10, 10, 1]) for x in xs]
        ys = flatten_indicies(ycords, shape=(10, 10))
        sess.run(train_step, feed_dict={image: xs, correct: ys})
        if _ % 100 == 0:
            mses.append(sess.run(loss, feed_dict={image: test_x, correct: test_y}))

    plt.plot(mses)
    plt.show()
