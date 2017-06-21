# -*- coding: utf-8 -*-
import tensorflow as tf
import time

import input_data
import common

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(x)
print(W)
print(b)
print(y)
time.sleep(2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # print(sess.run(x))
    print(sess.run(W))
    print(sess.run(b))
    # print(sess.run(y))
    print('aaaaaa')
    time.sleep(3)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print('bbbbb')
        print(batch_xs)
        print(batch_ys)
        print('ccccccc')
        # common.print_images(batch_xs, batch_ys, 10)
        # time.sleep(2)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            # print(sess.run(x))
            print(common.print_matrix(sess.run(W)))
            print(sess.run(b))
            # print(sess.run(y))
            print('ddddd')
            print('')

    time.sleep(5)
    print('Evaluate')
    print(sess.run([accuracy, y], feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# time.sleep(5)
# print('Evaluate')
#
#
# with tf.Session() as sess2:
#     print(sess2.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
