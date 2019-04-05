import logging
from loader import DataLoader, Dataset
from utils import build_dict
from models import Model
from trainer import Trainer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--save_dir', type=str, default='save/', help='the save path of checkpoint')
parser.add_argument('--batch', type=int, default=32, help='number of batchs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning')
parser.add_argument('--pad_len', type=int, default=15, help='the length of ')
parser.add_argument('--resume', type=str, default=None, help='the path of checkpoint to be loaded')

opt = parser.parse_args()

handlers = [logging.FileHandler('logging.txt', mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

if __name__ == '__main__':
    w2i, i2w = build_dict('data')
    dataloader = DataLoader(data_dir = 'data', word_to_index = w2i, batch_size = opt.batch, pad_len = opt.pad_len, shuffle = True, validation_split = 0.2)
    val_dataloader = dataloader.split_validation()
    model = Model(n_embed = len(w2i), word_to_index = w2i, batch_size = opt.batch, rnn_type = "GRU", attention_type = "general", bi = True)
    trainer = Trainer(model = model, index_to_word = i2w, dataloader = dataloader, val_dataloader = val_dataloader, opt = opt)
    trainer.train()