import sys
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
from model.model import DNN2, DNN5, DNN8
import argparse
import os
from math import log

if __name__ == '__main__':
    # arch_list = ['DNN-2', 'DNN-5', 'DNN-8']
    # color_list = ['r', 'g', 'b']
    # plt.figure()
    # for arch, color in zip(arch_list, color_list):
    #     checkpoint = torch.load('../../saved/{}/checkpoint-epoch10000.pth'.format(arch))
    #     logger = checkpoint['logger']
    #     print(arch, len(logger))
    #     x = [entry['epoch'] for _, entry in logger.entries.items()]
    #     y = [log(entry['loss'], 10) for _, entry in logger.entries.items()]
    #     plt.ylabel('log loss')
    #     plt.plot(x, y, color = color, label = arch)
    #     plt.legend(loc="best")
    
    # plt.savefig('plot_loss_acc_func_figure.png')

    arch_list = ['CNN1', 'CNN2', 'CNN3']
    color_list = ['r', 'g', 'b']
    plt.figure()
    for arch, color in zip(arch_list, color_list):
        checkpoint = torch.load('../../saved/{}/model_last.pth'.format(arch))
        logger = checkpoint['logger']
        x = [entry['epoch'] for _, entry in logger.entries.items()]
        y = [log(entry['loss'], 10) for _, entry in logger.entries.items()]
        
        plt.title('loss')
        plt.plot(x, y, color = color, label = arch)
        plt.legend(loc="best")
    plt.savefig('plot_loss_mnist_figure.png')

    plt.figure()
    for arch, color in zip(arch_list, color_list):
        checkpoint = torch.load('../../saved/{}/model_last.pth'.format(arch))
        logger = checkpoint['logger']
        x = [entry['epoch'] for _, entry in logger.entries.items()]
        y = [log(entry['my_metric'], 10) for _, entry in logger.entries.items()]
        
        plt.title('acc')
        plt.plot(x, y, color = color, label = arch)
        plt.legend(loc="best")
    plt.savefig('plot_acc_mnist_figure.png')
        
    
