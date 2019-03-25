import numpy as np
import torch
import sys
sys.path.append('../../')
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from model.model import CNN2
from model.metric import my_metric
from model.loss import nll_loss
from data_loader.data_loaders import MnistDataLoader
from math import log
from torch.autograd import Variable

def main():
    #build model
    model = CNN2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    state = {}

    #load checkpoint
    checkpoint1 = torch.load('../../saved/CNN2/64_3/model_last.pth')
    checkpoint2 = torch.load('../../saved/CNN2/64_2/model_last.pth')
    
    loss_list = []
    val_loss_list = []
    acc_list = []
    val_acc_list = []
    a_list = []
    '''
    for i in range(2):
        if i == 0:
            logger = checkpoint1['logger']
        else:
            logger = checkpoint2['logger']
        a_list.append(i)
        loss_list.append([entry['loss'] for _, entry in logger.entries.items()][-1])
        val_loss_list.append([entry['val_loss'] for _, entry in logger.entries.items()][-1])
        acc_list.append([entry['my_metric'] for _, entry in logger.entries.items()][-1])
        val_acc_list.append([entry['val_my_metric'] for _, entry in logger.entries.items()][-1])
    '''
    #load image
    dataloader = MnistDataLoader("../../data/", batch_size = 512, shuffle = False, validation_split = 0.2, num_workers=2)
    valid_dataloader = dataloader.split_validation()
    #load feature
    alpha_list = np.linspace(-1.0, 2.0, num = 100)
    alpha_list = np.append(alpha_list, [0.0, 1.0])
    alpha_list = np.sort(alpha_list, axis = None)
    for alpha in alpha_list:
        if alpha == 1.0:
            logger = checkpoint2['logger']
            a_list.append(alpha)
            loss_list.append([entry['loss'] for _, entry in logger.entries.items()][-1])
            val_loss_list.append([entry['val_loss'] for _, entry in logger.entries.items()][-1])
            acc_list.append([entry['my_metric'] for _, entry in logger.entries.items()][-1])
            val_acc_list.append([entry['val_my_metric'] for _, entry in logger.entries.items()][-1])
            continue
        elif alpha == 0.0:
            logger = checkpoint1['logger']
            a_list.append(alpha)
            loss_list.append([entry['loss'] for _, entry in logger.entries.items()][-1])
            val_loss_list.append([entry['val_loss'] for _, entry in logger.entries.items()][-1])
            acc_list.append([entry['my_metric'] for _, entry in logger.entries.items()][-1])
            val_acc_list.append([entry['val_my_metric'] for _, entry in logger.entries.items()][-1])
            continue
        #Interpolation
        state_dict = OrderedDict()
        for i in checkpoint1['state_dict']:
            state_dict[i] = torch.from_numpy((1-alpha) * checkpoint1['state_dict'][i].cpu().numpy() + alpha * checkpoint2['state_dict'][i].cpu().numpy()).float().to(device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_val_loss = 0.0
        total_metrics = 0
        total_val_metrics = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # computing loss, metrics on test set
                loss = nll_loss(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                total_metrics += my_metric(output, target) * batch_size

        with torch.no_grad():
            for i, (data, target) in enumerate(valid_dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # computing loss, metrics on test set
                val_loss = nll_loss(output, target)
                batch_size = data.shape[0]
                total_val_loss += val_loss.item() * batch_size
                total_val_metrics += my_metric(output, target) * batch_size

        n_samples = len(dataloader.sampler)
        n_valid_samples = len(valid_dataloader.sampler)
        #prepared data for plot
        a_list.append(alpha)
        loss_list.append(total_loss / n_samples)
        acc_list.append(total_metrics / n_samples)
        val_loss_list.append(total_val_loss / n_valid_samples)
        val_acc_list.append(total_val_metrics / n_valid_samples)
        print("alpha: ", alpha)
        print("loss: ", total_loss / n_samples)
        print("acc: ", total_metrics / n_samples)
        print("val_loss: ", total_val_loss / n_valid_samples)
        print("val_acc: ", total_val_metrics / n_valid_samples)
    #plot figures
    loss_list = [log(a, 10) for a in loss_list]
    val_loss_list = [log(a, 10) for a in val_loss_list]
    print(a_list)
    print(acc_list)
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set(title='Flatness_lr: 1e-3 vs 2.5e-3')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('loss', color='b')
    ax1.plot(a_list, loss_list, color = 'b', label = 'train')
    ax1.plot(a_list, val_loss_list, color = 'b', label = 'test', linestyle = 'dotted')
    ax1.tick_params(axis='both', labelcolor='b')

    ax2 = ax1.twinx()
    #ax2.set(xlim=[-1.0, 2.0], title='Flatness')
    ax2.set_ylabel('acc', color='r')
    ax2.plot(a_list, acc_list, color = 'r', label = 'train')
    ax2.plot(a_list, val_acc_list, color = 'r', label = 'test', linestyle = 'dotted')
    ax2.tick_params(axis='both', labelcolor='r')
    ax1.legend(loc="best") 
    ax2.legend(loc="best")
    plt.savefig('plot_flatness_lr.png')

main()