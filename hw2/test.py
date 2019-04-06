import logging
from loader import DataLoader, Dataset
from utils import build_dict
from models import Model
from trainer import Trainer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='save/checkpoint.pth', help='the path of checkpoint to be loaded')
parser.add_argument('--save_path', type=str, default='result.txt', help='the path to save result')
parser.add_argument('--pad_len', type=int, default=15, help='the length of ')
opt = parser.parse_args()

w2i, i2w = build_dict('data')

def batch_to_sentences(batch):
    sen_list = []
    for i in range(batch.shape[0]):
        sen_list.append(idxs_to_sentence(i2w, batch[i]))
    return sen_list

def idxs_to_sentence(i2w, sen_idxs):
    sen = []
    for i in sen_idxs:
        w = i2w[i.item()]
        if w == '<EOS>': break
        sen.append(w)
    return ' '.join(sen)

if __name__ == '__main__':
    dataloader = DataLoader(data_dir = 'data', word_to_index = w2i, batch_size = 1, pad_len = opt.pad_len, shuffle = False, validation_split = 0.0, training = False)
    model = Model(n_embed = len(w2i), word_to_index = w2i, batch_size = 32, rnn_type = "LSTM", attention_type = "concat", bi = True)
    state_dict = torch.load(opt.checkpoint)['state_dict']
    model.load_state_dict(state_dict)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    model.to(device)
    model.eval()
    
    f = open(opt.save_path, "w")

    s = ''
    for i, ((feat, idxs), id) in enumerate(dataloader):
        feat = feat.to(device)
        out = model(feat, idxs)
        pred_i = torch.argmax(out, dim = 2)
        pred_sen = batch_to_sentences(pred_i)
        s += '{},{}\n'.format(id[0], pred_sen[0])

    f.write(s)
    f.close()