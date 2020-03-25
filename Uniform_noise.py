# -*- coding: utf-8 -*-

__author__ = 'uniform symmetry noisy'

import numpy as np
import random
import os
import struct
import numpy as np


def uniform_noise(Cifar10_Y, noise_ratio):
    #
    array1 = Cifar10_Y.tolist()
    array = Cifar10_Y.tolist()
    array2 = Cifar10_Y

    ratio=5*noise_ratio
    noisy_ratio= ratio * 10
    for class_number in range(10):
    #选取所有同类标签位置
        print(class_number)
        ss=array.count(class_number)
        print(ss)
        first_pos=0
        find_out=[]
        for i in range(array.count(class_number)):
            new_list = array[first_pos:]
            next_pos = new_list.index(class_number) + 1
            # print ('find ', first_pos + new_list.index(2))
            find_out.append(first_pos + new_list.index(class_number))
            first_pos += next_pos

        # print(find_out.shape)
        #随机选择加入标签位置
        label_choose_index = random.sample(find_out,noisy_ratio)

        Noise_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #Noise_list.remove(class_number)
        noisy_label=Noise_list*ratio
        random.shuffle(noisy_label)
        print(len(noisy_label))


        number=0
        for index_label in label_choose_index:
            array1[index_label]=noisy_label[number]
            # print(array1[index_label])
            # print(Cifar10_Y[index_label])
            number+=1
        # print(array.shape)


    # num=array1-Cifar10_Y
    # print(array1==Cifar10_Y)
    # num = array1 - Y_train_Mnist
    # print(Y_train == noisy_label_max)
    # number = num.nonzero()
    # print(number)
    # print(sum(mnist_labels == original2))
    array1=np.array(array1, dtype = int)
    return array1




