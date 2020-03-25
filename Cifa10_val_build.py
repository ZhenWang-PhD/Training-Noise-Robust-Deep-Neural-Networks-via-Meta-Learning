from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random





def val_set_select(test_labels, num):


    # val_ratio = num *10
    array1 = test_labels[:]
    # array = array1
    array = test_labels[:]
    label_choose_index = []
    for class_number in range(10):

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

        label_choose_each = random.sample(find_out,num)
        label_choose_index.extend(label_choose_each)

    return label_choose_index