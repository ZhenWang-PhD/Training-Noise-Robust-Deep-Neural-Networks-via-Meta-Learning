# -*- coding: utf-8 -*-

__author__ = 'uniform symmetry noisy'

import numpy as np
import random

def flip_to_one_noise(Cifar10_Y, noise_ratio):
    array1 = Cifar10_Y.tolist()
    array = Cifar10_Y.tolist()
    array2 = Cifar10_Y

    noisy_ratio=50*noise_ratio

    for class_number in range(10):

        ss=array.count(class_number)

        first_pos=0
        find_out=[]
        for i in range(array.count(class_number)):
            new_list = array[first_pos:]
            next_pos = new_list.index(class_number) + 1

            find_out.append(first_pos + new_list.index(class_number))
            first_pos += next_pos


        label_choose_index = random.sample(find_out,noisy_ratio)

        Noise_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Noise_list.remove(class_number)

        noisy_label=int(random.sample(Noise_list, 1)[0])
        print (len(label_choose_index))

        number=0
        for index_label in label_choose_index:
            array1[index_label]=noisy_label

            number+=1

    array1=np.array(array1, dtype = int)

    return array1



