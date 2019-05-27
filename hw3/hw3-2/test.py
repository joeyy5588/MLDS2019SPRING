import logging
from model import CGAN, ACGAN
import torch
import argparse
import utils
import os
from torchvision.utils import save_image
from utils import embed, ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default='saved/checkpoint_75.pth', help='the path of checkpoint to be loaded')
parser.add_argument('-s', '--save_dir', type=str, default='results', help='the path to save result')
parser.add_argument('-m', '--method', type=str, default='acgan')
parser.add_argument('-e', '--nemb', type=int, default=120)
opt = parser.parse_args()
ensure_dir(opt.save_dir)
NOISE_DIM = 100
NF = 64
N_EMB = opt.nemb

if __name__ == '__main__':
    model = None
    if opt.method == 'cgan':
        model = CGAN
    elif opt.method == 'acgan':
        model = ACGAN
    
    G = model.Generator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    D = model.Discriminator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    g_state_dict = torch.load(opt.checkpoint)['gen_state_dict']
    d_state_dict = torch.load(opt.checkpoint)['dis_state_dict']
    G.load_state_dict(g_state_dict)
    D.load_state_dict(d_state_dict)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    G.to(device), G.eval()
    D.to(device), D.eval()
    noise_dim, n_emb = 100, N_EMB
    for r in range(10):
        fixed_noise = torch.randn(120, noise_dim, device = device)# .repeat(120, 1)
        fixed_condition = torch.zeros(120, N_EMB, device = device)

        for i in range(12):
            for j in range(10):
                fixed_condition[i * 10 + j] = embed(i, j, n_emb = n_emb).to(device)

        with torch.no_grad():
            fixed_image = G(fixed_noise, fixed_condition)
            save_image(fixed_image.data[:12*10], os.path.join(opt.save_dir, 'result%d.png' % r), nrow = 10, normalize = True)
    
    f = open(os.path.join(opt.save_dir, 'status.txt'), 'w')
    f.write(str(opt))