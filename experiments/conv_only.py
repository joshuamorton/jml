"""A CNN model to recognize the centers of circles.
"""

import itertools

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
import utils
from tqdm import tqdm
from utils.circles import circle, random_boxed_circle

global_step = tf.Variable(0, trainable=False)
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
conv_bias = tf.Variable(tf.zeros([1]))
one_hot = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, conv_bias)), [-1, 100])

correct = tf.placeholder(tf.int32, [None])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct, logits=one_hot))
learning_rate = tf.train.exponential_decay(.005, global_step, 1000, 0.8, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step
)

dims = (range(1, 9), range(1, 9))
test_x = [circle(x, y, 1, shape=(10, 10)) for x, y in itertools.product(*dims)]
test_y = list(utils.flatten_indicies(itertools.product(*dims)))

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mses = []
    for _ in tqdm(range(10000)):
        # zip(*i) is a fast way to transpose the iterable "i", so this takes
        # [(img, (x, y)), ...] and converts it to ([img, ...], [(x, y), ...])
        xs, ys = zip(*[random_boxed_circle(10, 10, 1) for i in range(100)])
        xs = [x.reshape([10, 10, 1]) for x in xs]
        ys = list(utils.flatten_indicies(ys))
        sess.run(train_step, feed_dict={image: xs, correct: ys})
        error = sess.run(loss, feed_dict={image: test_x, correct: test_y})
        if _ % 100 == 0:
            mses.append(error)

    plt.plot(mses)
    plt.show()
