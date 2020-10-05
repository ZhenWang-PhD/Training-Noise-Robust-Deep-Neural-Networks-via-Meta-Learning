import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import torch
from PIL import Image
import torch.utils.data as Data
import torchvision
from Uniform_noise import *
from Flip_random_noise import *
from Flip_to_one_noise import *
from Cifa10_val_build import *



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def get_data(train, noise_type,  noisy_ratio):

    base_dataset = torchvision.datasets.CIFAR10('./data/', train=True, download=True)

    data = None
    labels = None
    label_1s = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle('./data/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])
            if i == 1:
                labels_original = batch[b'labels']
                
            else:
                labels_original = np.concatenate([labels_original, batch[b'labels']])     

        if (noise_type==1):
            noise_type = 'uniform'
            labels = uniform_noise(labels_original, noisy_ratio)
        elif (noise_type==2):
            noise_type = 'flip'
            labels = flip_random_noise(labels_original, noisy_ratio)
        else:
            noise_type = 'flip_to_one'
            labels = flip_to_one_noise(labels_original, noisy_ratio)

    else:
        batch = unpickle('./data/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']


    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



class Cifar10_Dataset(Data.Dataset):
    def __init__(self, train=True, Val_choose=None, datas=None, lables=None, transform=None, target_transform=None, noise_type=None, noisy_ratio=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = datas, lables
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))

            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            
        else:
            Noise_list = []
            for i in range(10000):
                Noise_list.append(i)

            #aa = Val_choose.tolist()
            cc = list(set(Noise_list).difference(set(Val_choose)))

            dd = np.array(cc)
            np.random.shuffle(dd)
            self.test_data1, self.test_labels1 = get_data(False,0,0)
            self.test_data1 = self.test_data1.reshape((10000, 3, 32, 32))
            self.test_data1 = self.test_data1.transpose((0, 2, 3, 1))

            self.test_data = self.test_data1[dd, :]
            self.test_labels = np.array(self.test_labels1)[dd]
        pass

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):

        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



def Cifar10_Val_1(Val_choose, sample_number):


    test_data1, test_labels1 = get_data(False, 0, 0)

    np.random.shuffle(Val_choose)
    test_data1 = test_data1.reshape((10000, 3, 32, 32))
    val_data_NEW = torch.Tensor(sample_number *10,3,32,32)
    val_labels_NEW = torch.Tensor(sample_number*10)
    test_data1 = test_data1.transpose((0, 2, 3, 1))
    val_data = test_data1[Val_choose, :]
    val_labels = np.array(test_labels1)[Val_choose]
    val_labels = val_labels.tolist()


    for index in range(len(val_data)):
        img, label = val_data[index], val_labels[index]
        img = Image.fromarray(img)
        img = transform(img)
        val_data_NEW[index] = img
        val_labels_NEW[index] = target_transform(val_labels[index])
    return val_data_NEW, val_labels_NEW


if __name__ == '__main__':
    sample_number = 50
    test_datas, test_lables = get_data(False, 0, 0)
    Val_choose = val_set_select(test_lables, sample_number)
    for noise_type in [3]:
        for noise_ratio in [10, 20, 30, 40]:
            train_datas, train_lables = get_data(True, noise_type,  noise_ratio)
            test_datas, test_lables = get_data(False, noise_type, noise_ratio)

            print('ok')
