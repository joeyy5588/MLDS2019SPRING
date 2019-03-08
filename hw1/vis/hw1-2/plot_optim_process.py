import torch
import sys
sys.path.append('../../')
import os
import numpy as np
from model.model import CNN2 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    root_folder = os.path.join('../../saved/1-2-1/')
    weight_list = []

    for base_folder in os.listdir(root_folder):
        # we need to iterate each epoch by order
        file_list = os.listdir(os.path.join(root_folder, base_folder))
        # filter filenames so that it only contains checkpoint_x.pth
        file_list = list(filter(lambda x: x.startswith('checkpoint'), file_list))
        for file in file_list:
            checkpoint = torch.load(os.path.join(root_folder, base_folder, file))
            state = checkpoint['state_dict']
            w_arr = [w.flatten() for key, w in state.items()]
            w_flat = torch.cat(w_arr, dim = 0)
            weight_list.append(w_flat.data.cpu().numpy())

    weight_list = np.array(weight_list)
    pca = PCA(n_components = 2)

    weight_list = pca.fit_transform(weight_list)

    x = [x for (x, y) in weight_list]
    y = [y for (x, y) in weight_list]
    color = np.linspace(0, 1, len(weight_list))
    plt.scatter(x, y, cmap = 'rainbow', c = color, alpha=0.5)
    logger = checkpoint['logger']
    plt.savefig('weight_scatter.png')