from loader import DataLoader
from models import Seq2Seq, Lang
from trainer import Trainer
from utils import ensure_dir
import logging
import argparse

handlers = [logging.FileHandler('output.log', mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default='saved/')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--resume', type=str, default=None)

opt = parser.parse_args()
ensure_dir(opt.save_dir)

if __name__ == '__main__':
    L = Lang('data', 'saved/lang.kv', True)
    D = DataLoader(data_dir = 'data', word_to_index = L.w2i, batch_size = opt.batch, shuffle = True, validation_split = 0.0, pad_len = 15)
    S = Seq2Seq(L.w2i, L.embed)
    T = Trainer(model = S, lang = L, dataloader = D, val_dataloader = None, opt = opt)
    T.train()