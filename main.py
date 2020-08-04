# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse, sys
import numpy as np
import datetime
import shutil
from tqdm import tqdm


# load backbone
from meta_wrn import Wide_ResNet
########## Load dataset ########################

from load_cifar10_data import *

parser = argparse.ArgumentParser(description='PyTorch Uncertainty Training')
#Network
parser.add_argument('--lr_1', default=1e-1, type=float, help='learning_rate')
parser.add_argument('--lr_2', default=1e-4, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
#
#Summary
args = parser.parse_args()

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


############### Build model ######################################
def build_model():

    net =Wide_ResNet(40, 2, 10)
    print(net)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    return net


def train_warm_up(Val_choose, train_datas, train_lables,  test_datas, test_lables, type, ratio):
    model = build_model()
    optimizer = torch.optim.SGD(model.params(), lr=args.lr_1, momentum=args.momentum, weight_decay=args.weight_decay)
    #print(model)
    if (type==1):
        type = 'uniform'
    elif (type==2):
        type = 'flip'
    else:
        type = 'flip_to_one'
    ##########    load data  ##########################################
    save_root = './Cifar10_warmup_CE_'+str(type)+str(ratio)+'.pth'
    print('noise ratio is '+ str(ratio)+'%')
    train_data = Cifar10_Dataset(True, Val_choose, train_datas, train_lables, transform, target_transform, noise_type=type, noisy_ratio=ratio)
    print('size of train_data:{}'.format(train_data.__len__()))
    test_data = Cifar10_Dataset(False, Val_choose, test_datas, test_lables, transform, target_transform)
    print('size of test_data:{}'.format(test_data.__len__()))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

    ####################### define variable ##############################################
    best_accuracy = 0

    plot_step = 100
    for i in tqdm(range(2000)):
        model.train()

        image, labels = next(iter(train_loader))


        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        pre = model(image)

        loss = F.cross_entropy(pre, labels)

        model.zero_grad()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            model.eval()
            with torch.no_grad():

                acc = []
                pre_correct = 0.0
                for itr, (test_img, test_label) in enumerate(test_loader):
                    test_img = to_var(test_img, requires_grad=False)
                    test_label = to_var(test_label, requires_grad=False)

                    output = model(test_img)
                    pre = torch.max(output, 1)[1]

                    pre_correct = pre_correct + float(torch.sum(pre == test_label))

                test_acc = (pre_correct / float(10000)) * 100
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(),save_root)

            print('| Cifar10 ' + 'Baseline type: '+ str(type) + ' ratio:' + str(ratio)+ '% ' + 'Test Best_Test_Acc: %.2f%% Test_Acc@1: %.2f%%' % (best_accuracy, test_acc))


def train_MLC(Val_choose, train_datas, train_lables,  test_datas, test_lables, type, ratio):
    main_model= build_model()
    optimizer = torch.optim.SGD(main_model.params(), lr=args.lr_2, momentum=args.momentum, weight_decay=args.weight_decay)

    if (type==1):
        type = 'uniform'
    elif (type==2):
        type = 'flip'
    else:
        type = 'flip_to_one'
    ##########    load data  ##########################################

    print('noise ratio is '+ str(ratio)+'%')
    train_data = Cifar10_Dataset(True, Val_choose, train_datas, train_lables, transform, target_transform, noise_type=type, noisy_ratio=ratio)
    print('size of train_data:{}'.format(train_data.__len__()))
    test_data = Cifar10_Dataset(False, Val_choose, test_datas, test_lables, transform, target_transform)
    print('size of test_data:{}'.format(test_data.__len__()))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)


    model_root = './Cifar10_warmup_CE_'+str(type)+str(ratio)+'.pth'

    main_model.load_state_dict(torch.load(model_root))
    ## load small validation set

    X_Val, Y_Val =Cifar10_Val_1(Val_choose, 50)
    print('size of val_data:{}'.format(len(X_Val)))
    Y_Val = Y_Val.int()
    Y_Val = Y_Val.long()
    X_Val = to_var(X_Val, requires_grad=False)
    Y_Val = to_var(Y_Val, requires_grad=False)

    ####################### define variable ##############################################
    best_accuracy = 0

    plot_step = 100
    for i in tqdm(range(20000)):
        main_model.train()

        image, labels = next(iter(train_loader))

        meta_net = Wide_ResNet(40, 2, 10)

        meta_net.load_state_dict(main_model.state_dict())
        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)
        T = to_var(torch.eye(10, 10))

        y_f_hat  = meta_net(image)

        pre2 = torch.mm(y_f_hat, T)

        l_f_meta = torch.sum(F.cross_entropy(pre2,labels, reduce=False))

        meta_net.zero_grad()
        

        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(1e-3, source_params=grads)
        

        y_g_hat = meta_net(X_Val)

        l_g_meta = F.cross_entropy(y_g_hat,Y_Val)

        grad_eps = torch.autograd.grad(l_g_meta, T, only_inputs=True)[0]

        T = torch.clamp(T-0.11*grad_eps,min=0)
        norm_c = torch.sum(T, 0)


        for j in range(10):
            if norm_c[j] != 0:
                T[:, j] /= norm_c[j]

        y_f_hat = main_model(image)
        pre2 = torch.mm(y_f_hat, T)

        l_f = torch.sum(F.cross_entropy(pre2,labels, reduce=False))

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        if i % plot_step == 0:
            main_model.eval()
            with torch.no_grad():

                acc = []
                pre_correct = 0.0
                for itr, (test_img, test_label) in enumerate(test_loader):
                    test_img = to_var(test_img, requires_grad=False)
                    test_label = to_var(test_label, requires_grad=False)

                    output = main_model(test_img)
                    pre = torch.max(output, 1)[1]

                    pre_correct = pre_correct + float(torch.sum(pre == test_label))

                test_acc = (pre_correct / float(10000)) * 100
            if test_acc > best_accuracy:
                best_accuracy = test_acc

            print('| Cifar10 ' + 'MLC type: '+ str(type) + ' ratio:' + str(ratio)+ '% ' + 'Test Best_Test_Acc: %.2f%% Test_Acc@1: %.2f%%' % (best_accuracy, test_acc))

if __name__=='__main__':
    result = {}
    sample_number = 50
    test_datas, test_lables = get_data(False, 0, 0)
    Val_choose = val_set_select(test_lables, sample_number)
    for noise_type in [2]:
        for noise_ratio in [10, 20, 30, 40]:
            train_datas, train_lables = get_data(True, noise_type,  noise_ratio)
            test_datas, test_lables = get_data(False, noise_type, noise_ratio)

            train_warm_up(Val_choose, train_datas, train_lables, test_datas, test_lables, type=noise_type, ratio=noise_ratio)
            acc = train_MLC(Val_choose, train_datas, train_lables, test_datas, test_lables, type=noise_type, ratio=noise_ratio)
            result[noise_ratio] = acc
    print(result)
