# -*- coding: utf-8 -*-

def print_images(images, labels, den):
    print('print images')
    count = images.shape[0]
    count1 = images.shape[1]
    print(count)
    print(count1)
    for i in range(count):
        for l in range(den):
            temp = labels.flat[i * den + l]
            if temp == 1.:
                print(l)
                break
        for j in range(28):
            for k in range(28):
                va = images.flat[i * 784 + 28 * j + k]
                if va == 0.:
                    res = 0
                else:
                    res = 1
                print(res, end='')
            print('', end='\n')
        print('')

    print('print images end')

def print_matrix(matrix, change=True):
    print('print matrix')
    col = matrix.shape[0]
    row = matrix.shape[1]
    print(col)
    print(row)
    for i in range(row):
        for j in range(784):
            if j % 28 == 0:
                print('')
            va = matrix.flat[i + j * row]
            print(va, end='\t')
        print('')

    print('print matrix end')