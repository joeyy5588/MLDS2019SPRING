import logging
from loader import DataLoader, Dataset
from models import Seq2Seq, Lang
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='saved/checkpoint.pth', help='the path of checkpoint to be loaded')
parser.add_argument('--save_path', type=str, default='result.txt', help='the path to save result')
parser.add_argument('--pad_len', type=int, default=15, help='the length of input sentences')
opt = parser.parse_args()

L = Lang(model_path = 'saved/lang.kv', data_dir = 'data', min_count = 5, pretrain = True)
w2i, i2w = L.w2i, L.i2w

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
    dataloader = DataLoader(data_dir = 'data', word_to_index = w2i, batch_size = 64, pad_len = opt.pad_len, shuffle = False, validation_split = 0.0, training = False)
    model = Seq2Seq(L.w2i, L.embed)
    state_dict = torch.load(opt.checkpoint)['state_dict']
    model.load_state_dict(state_dict)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    model.to(device)
    model.eval()
    
    f = open(opt.save_path, "w")

    result = ''
    for i, q_idxs in enumerate(dataloader):
        q_idxs = q_idxs.to(device)
        out = model(q_idxs, None, 0)
        pred_i = torch.argmax(out, dim = 2)
        pred_sen = batch_to_sentences(pred_i)
        for s in pred_sen:
            print(s)
            result += '{}\n'.format(s)

    f.write(result)
    f.close()