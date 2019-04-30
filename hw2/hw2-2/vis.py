import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--check_path', type=str, default='saved/checkpoint.pth')
parser.add_argument('--save_path', type=str, default='saved/plot.png')
opt = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(opt.check_path)
    log = checkpoint['log']
    print(log)
    x = [item['epoch'] for item in log]
    loss = [item['loss'] for item in log]

    # Create figure
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, loss)
    ax1.set(title = 'Loss')
    ax1.grid()
    fig.tight_layout()
    fig.savefig(opt.save_path)
    print('Save {}'.format(opt.save_path))