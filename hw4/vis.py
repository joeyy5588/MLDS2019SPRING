import torch
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--check_path', type=str, default='saved/dqn/checkpoint_episode26200.pth.tar')
parser.add_argument('--save_path', type=str, default='plot/test_dqn.png')
opt = parser.parse_args()

def smooth(y, weight=0.95):
    smoothed = []
    last = y[0]
    for point in y:
        smoothed_val = weight * last + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def average(y, n_avg = 30):
    avg_list = []
    for i in range(len(y)):
        if i < 30:
            avg = sum(y[0:(i+1)])  / len(y[0:(i+1)])
        else:
            avg = sum(y[(i-30):(i+1)])  / len(y[(i-30):(i+1)])
        avg_list.append(avg)
    return avg_list

def average_dqn(y):
    avg_list = []
    for i in range(len(y)):
        if (i+1) * 10 < 30:
            avg = sum(y[-1][:(i+1) * 10]) / ((i+1) * 10)
        else:
            avg = sum(y[-1][((i-2)*10):((i+1) * 10)]) / 30.0
        avg_list.append(avg)
    return avg_list

if __name__ == '__main__':
    checkpoint = torch.load(opt.check_path)
    log = checkpoint['log']
    #print(log)
    x = [item['episode'] for item in log]
    loss = [item['latest_reward'] for item in log]
    #loss = [item['total_rewards'] for item in log]
    loss = average_dqn(loss)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, loss)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid()
    fig.tight_layout()
    fig.savefig(opt.save_path)
    print('Save {}'.format(opt.save_path))