import logging
from loader import DataLoader, Dataset
from utils import build_dict
from models import Model
from trainer import Trainer
import torch
import argparse

handlers = [logging.FileHandler('logging.txt', mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--resume', type=str, default=None)

opt = parser.parse_args()

if __name__ == '__main__':
    w2i, i2w = build_dict('data')
    X = DataLoader(data_dir = 'data', word_to_index = w2i, batch_size = opt.batch, shuffle = True, validation_split = 0.2)
    V = X.split_validation()
    M = Model(n_embed = len(w2i), word_to_index = w2i)
    T = Trainer(model = M, index_to_word = i2w, dataloader = X, val_dataloader = V, opt = opt)
    T.train()
    # for x in X:
    #     y = M(x[0], x[1])
    #     input()