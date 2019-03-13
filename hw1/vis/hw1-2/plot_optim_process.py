import torch
import sys
sys.path.append('../../')
import os
import numpy as np
from model.model import CNN2 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    
    root_folder = os.path.join('../../saved/1-2-1/')
    w_list = []
    cw_list = [] # convolution weight
    acc = [] # record the acc

    for base_folder in os.listdir(root_folder):
        # we need to iterate each epoch by order
        file_list = os.listdir(os.path.join(root_folder, base_folder))
        # filter filenames so that it only contains checkpoint_x.pth
        file_list = list(filter(lambda x: x.startswith('checkpoint'), file_list))
        
        for file in file_list:
            checkpoint = torch.load(os.path.join(root_folder, base_folder, file))
            state = checkpoint['state_dict']

            logger = checkpoint['logger']
            acc.append(logger[-1]['my_metric']) # take the last one
            
            w_arr = [w.flatten() for key, w in state.items()]
            w_flat = torch.cat(w_arr, dim = 0)
            w_list.append(w_flat.data.cpu().numpy())
            # for layer 1
            cw_arr = [w.flatten() for key, w in state.items() if key.startswith('conv.3.')]
            cw_flat = torch.cat(cw_arr, dim = 0)
            cw_list.append(cw_flat.data.cpu().numpy())
    
    plt.figure()
    fig, ax = plt.subplots()
    w_list = np.array(w_list)
    pca = PCA(n_components = 2)
    w_list = pca.fit_transform(w_list)
    x = [x for (x, y) in w_list]
    y = [y for (x, y) in w_list]
    color = np.linspace(0, 1, len(w_list))
    plt.scatter(x, y, cmap = 'rainbow', c = color, alpha=0.5)

    cmap = cm.get_cmap('rainbow')
    for i, n in enumerate(acc):
        if i % 50 == 0 or i % 50 == 49:
            ax.annotate(round(n, 2), (x[i], y[i]))
    plt.savefig('weight_scatter.png')

    plt.figure()
    fig, ax = plt.subplots()
    cw_list = np.array(cw_list)
    pca = PCA(n_components = 2)
    cw_list = pca.fit_transform(cw_list)
    x = [x for (x, y) in cw_list]
    y = [y for (x, y) in cw_list]
    color = np.linspace(0, 1, len(cw_list))
    plt.scatter(x, y, cmap = 'rainbow', c = color, alpha=0.5)

    cmap = cm.get_cmap('rainbow')
    for i, n in enumerate(acc):
        if i % 50 == 0 or i % 50 == 49:
            ax.annotate(round(n, 2), (x[i], y[i]))
    plt.savefig('conv_weight_scatter.png')