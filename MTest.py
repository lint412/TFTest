# -*- coding: utf-8 -*-
import time

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

time.sleep(5)

print("begin")
print("\n\n\n")

print(mnist.train.images)
print('\n')
print(mnist.train.labels)
print("\n\n")
time.sleep(5)
count = 500  # mnist.train.images.shape[0]
count1 = mnist.train.images.shape[1]
print(count)
print(count1)
for i in range(count):
    for l in range(10):
        temp = mnist.train.labels.flat[i * 10 + l]
        if temp == 1.:
            print(l)
            break
    for j in range(28):
        for k in range(28):
            va = mnist.train.images.flat[i * 784 + 28 * j + k]
            if va == 0.:
                res = 0
            else:
                res = 1
            print(res, end='')
        print('', end='\n')
    print('')
