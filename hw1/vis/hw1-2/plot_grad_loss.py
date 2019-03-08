import sys
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
import model.model
import argparse
import os

if __name__ == '__main__':

    arch_list = ['DNN8']
    color_list = ['b']
    plt.figure()
    for arch, color in zip(arch_list, color_list):
        checkpoint = torch.load('../../saved/{}/model_last.pth'.format(arch))
        logger = checkpoint['logger']
        x = [entry['epoch'] for _, entry in logger.entries.items()]
        loss = [entry['loss'] for _, entry in logger.entries.items()]
        grad_norm = [entry['grad_norm'] for _, entry in logger.entries.items()]

        plt.subplot(2, 1, 1)
        plt.title('loss')
        plt.plot(x, loss, color = color, label = arch)
        plt.legend(loc="best")

        plt.subplot(2, 1, 2)
        plt.title('grad_norm')
        plt.plot(x, grad_norm, color = color, label = arch)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('plot_grad_loss_func.png')

    arch_list = ['CNN2']
    color_list = ['b']
    plt.figure()
    for arch, color in zip(arch_list, color_list):
        checkpoint = torch.load('../../saved/{}/model_last.pth'.format(arch))
        logger = checkpoint['logger']
        x = [entry['epoch'] for _, entry in logger.entries.items()]
        loss = [entry['loss'] for _, entry in logger.entries.items()]
        grad_norm = [entry['grad_norm'] for _, entry in logger.entries.items()]

        plt.subplot(2, 1, 1)
        plt.title('loss')
        plt.plot(x, loss, color = color, label = arch)
        plt.legend(loc="best")

        plt.subplot(2, 1, 2)
        plt.title('grad_norm')
        plt.plot(x, grad_norm, color = color, label = arch)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('plot_grad_loss_mnist.png')
    
