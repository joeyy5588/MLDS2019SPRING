from loader import DataLoader
from model import GAN, DCGAN, WGANGP
from trainer import EXPTrainer, DCGANTrainer, WGANGPTrainer
from utils import ensure_dir
import logging
import argparse

handlers = [logging.FileHandler('output.log', mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='saved/')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--checkpoint', type=str, default=None)

opt = parser.parse_args()
ensure_dir(opt.save_dir)
logger.info(opt)

if __name__ == '__main__':
    D = DataLoader(data_dir = 'data/faces', batch_size = opt.batch, shuffle = True, validation_split = 0.0)
    GEN = GAN.Generator()
    DIS = GAN.Discriminator()
    T = EXPTrainer(gen = GEN, dis = DIS, dataloader = D, opt = opt)
    T.train()