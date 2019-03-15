import torch
import sys
sys.path.append('../../')
import os
import numpy as np
from model.model import CNNh 
import matplotlib.pyplot as plt
from collections import OrderedDict


if __name__ == '__main__':
    
    root_folder = os.path.join('../../saved/CNNh/')
    plt.figure()
    for base_folder in os.listdir(root_folder):
        # we need to iterate each epoch by order
        checkpoint = torch.load('../../saved/CNNh/{}/model_last.pth'.format(base_folder))
        logger = checkpoint['logger']
        x = [[entry['parameters'] for _, entry in logger.entries.items()][-1]]
        y1 = [[entry['loss'] for _, entry in logger.entries.items()][-1]]
        y2 = [[entry['val_loss'] for _, entry in logger.entries.items()][-1]]
        plt.scatter(x, y1, c='#ff8c00', label='train')
        plt.scatter(x, y2, c='b', label='test')
        plt.title('Training/testing loss')
        plt.xlabel('number of parameters')
        plt.ylabel('loss')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 'best')
    plt.savefig('parameters_loss.png')
    plt.figure()
    for base_folder in os.listdir(root_folder):
        # we need to iterate each epoch by order
        checkpoint = torch.load('../../saved/CNNh/{}/model_last.pth'.format(base_folder))
        logger = checkpoint['logger']
        x = [[entry['parameters'] for _, entry in logger.entries.items()][-1]]
        y1 = [[entry['my_metric'] for _, entry in logger.entries.items()][-1]]
        y2 = [[entry['val_my_metric'] for _, entry in logger.entries.items()][-1]]
        plt.scatter(x, y1, c='#ff8c00', label='train')
        plt.scatter(x, y2, c='b', label='test')
        plt.title('Training/testing accuracy')
        plt.xlabel('number of parameters')
        plt.ylabel('accuracy')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 'best')
    plt.savefig('parameters_acc.png')