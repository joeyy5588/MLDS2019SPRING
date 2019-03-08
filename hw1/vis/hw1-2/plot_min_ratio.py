import torch
import sys
sys.path.append('../../')
import os
import numpy as np
from model.model import CNN2 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    root_folder = os.path.join('saved/')
    weight_list = []
    loss, ratio = [], []
    for i in range(100):
        # we need to iterate each epoch by order
        checkpoint = torch.load(os.path.join(root_folder, str(i), 'model_last.pth'))
        # filter filenames so that it only contains checkpoint_x.pth
        logger = checkpoint['logger']
        ratio.append(logger[200]['min_ratio'])
        loss.append(logger[200]['loss'])

    x = ratio
    y = loss
    plt.xlabel("min_ratio")
    plt.ylabel("loss")
    plt.scatter(x, y, alpha=0.5)
    logger = checkpoint['logger']
    plt.savefig('min_ratio.png')