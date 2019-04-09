import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--check_path', type=str, default='save/checkpoint.pth')
parser.add_argument('--save_path', type=str, default='save/plot.png')
opt = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(opt.check_path)
    log = checkpoint['log']
    loss = [item['loss'] for item in log]
    val_loss = [item['val_loss'] for item in log]
    score = [item['score'] for item in log]
    val_score = [item['val_score'] for item in log]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(loss)
    ax1.plot(val_loss)
    ax1.set(title = 'Loss')
    ax1.grid()
    ax2.plot(score)
    ax2.plot(val_score)
    ax2.set(title = 'Score')
    ax2.grid()
    fig.tight_layout()
    fig.savefig(opt.save_path)
    print('Save {}'.format(opt.save_path))