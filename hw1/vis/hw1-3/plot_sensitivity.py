import torch
import sys
sys.path.append('../../')
import os
import numpy as np
import model.model 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

arr = [
    {'epoch': 10, 'loss': 0.7537376774966716, 'my_metric': 0.733275, 'grad_norm': 8.730851662029155, 'sensitivity': 0.5025375912988314, 'val_loss': 1.0961402046203614, 'val_my_metric': 0.6362, 'batch': 16},
    {'epoch': 10, 'loss': 0.7901091994047165, 'my_metric': 0.720875, 'grad_norm': 4.910462600683021, 'sensitivity': 0.3110100778120784, 'val_loss': 1.0763865587429498, 'val_my_metric': 0.6375798722044729, 'batch': 32},
    {'epoch': 10, 'loss': 0.8326365353107452, 'my_metric': 0.7065, 'grad_norm': 9.741373325849139, 'sensitivity': 0.1596503218329651, 'val_loss': 1.0719041421914557, 'val_my_metric': 0.631468949044586, 'batch': 64},
    {'epoch': 10, 'loss': 0.8724902456941696, 'my_metric': 0.6958117012779552, 'grad_norm': 5.802636014344815, 'sensitivity': 0.101199400905746, 'val_loss': 1.0632932812352724, 'val_my_metric': 0.6340981012658228, 'batch': 128},
    {'epoch': 10, 'loss': 0.9633027580892963, 'my_metric': 0.6611763535031847, 'grad_norm': 5.65396556670106, 'sensitivity': 0.056383777133655465, 'val_loss': 1.0707960784435273, 'val_my_metric': 0.62783203125, 'batch': 256},
    {'epoch': 10, 'loss': 1.0170151191421701, 'my_metric': 0.6478688686708861, 'grad_norm': 4.908861812692897, 'sensitivity': 0.0342986793838481, 'val_loss': 1.1189609229564668, 'val_my_metric': 0.6077090992647058, 'batch': 512},
    {'epoch': 10, 'loss': 1.1371045887470246, 'my_metric': 0.606396484375, 'grad_norm': 4.485698043739264, 'sensitivity': 0.018716294892537674, 'val_loss': 1.1842440009117126, 'val_my_metric': 0.5928212691326531, 'batch': 1024}
]

if __name__ == '__main__':
    loss_list = []
    val_loss_list = []
    acc_list = []
    val_acc_list = []
    sen_list = []
    batch_list = []
    for a in arr:
        loss_list.append(a['loss'])
        val_loss_list.append(a['val_loss'])
        acc_list.append(a['my_metric'])
        val_acc_list.append(a['val_my_metric'])
        sen_list.append(a['sensitivity'])
        batch_list.append(a['batch'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax1 = ax[0]
    ax1.set_title('loss & sensitivity v.s. batch_size')

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss', color = 'b')
    ax1.semilogx(batch_list, loss_list, color = 'b', label = 'train loss')
    ax1.semilogx(batch_list, val_loss_list, color = 'b', linestyle = '--', label = 'test loss')
    ax1.tick_params(axis='y', labelcolor= 'b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('sensitivity', color = 'r')
    ax2.semilogx(batch_list, sen_list, color = 'r', label = 'sensitivity')
    ax2.tick_params(axis='y', labelcolor= 'r')
    ax1.legend(loc="best") 
    ax2.legend(loc="best")

    ax3 = ax[1]
    ax3.set_title('acc & sensitivity v.s. batch_size')
    ax3.set_xlabel('batch size')
    ax3.set_ylabel('acc', color = 'b')
    ax3.semilogx(batch_list, acc_list, color = 'b', label = 'train acc')
    ax3.semilogx(batch_list, val_acc_list, color = 'b', linestyle = '--', label = 'test acc')
    ax3.tick_params(axis='y', labelcolor= 'b')
    ax4 = ax3.twinx()
    ax4.set_ylabel('sensitivity', color = 'r')
    ax4.semilogx(batch_list, sen_list, color = 'r', label = 'sensitivity')
    ax4.tick_params(axis='y', labelcolor= 'r')
    ax3.legend(loc="upper left") 
    ax4.legend(loc="best")

    plt.tight_layout()
    fig.savefig('sensitivity.png')
"""
# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
"""