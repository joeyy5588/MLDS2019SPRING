import sys
sys.path.append("../../")
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
from model.model import DNN2, DNN5, DNN8

# checkpoint = torch.load('../../saved/DNN-8/checkpoint-epoch50.pth')
# print(checkpoint)

if __name__ == '__main__':
    func_name_list = ['sinc']
    func_lambda_list = [lambda x: np.sin(5 * np.pi * x) / (5 * np.pi * x + 1e-10)]
    arch_list = ['DNN2', 'DNN5', 'DNN8']
    color_list = ['r', 'g', 'b']

    plt.figure(figsize=(12, 9))
    for i, func_name in enumerate(func_name_list):
        func = func_lambda_list[i]
        x = np.array([i for i in np.linspace(0, 1, 10000)])
        y_target = np.array([func(i) for i in x])
        plt.title(func_name+' loss')
        plt.plot(x, y_target, 'k', label='Ground truth')
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load('../../saved/{}/model_last.pth'.format(arch))
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            y_pred = np.array([model(Variable(torch.FloatTensor(np.array([[i]])))).data.numpy() for i in x]).squeeze()
            plt.plot(x, y_pred, color, label=arch)
            plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig('plot_function_figure.png')
    plt.show()
