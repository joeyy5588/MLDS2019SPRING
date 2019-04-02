import torch
import torch.nn as nn
import numpy as np
import random
from .base_model import BaseModel

class Model(BaseModel):
    def __init__(self, n_embed, word_to_index):
        super(Model, self).__init__()
        self.EN = nn.Embedding(num_embeddings =  n_embed, embedding_dim = 256, padding_idx = 0)
        self.feat_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
        )
        # input of shape (seq_len, batch, input_size)
        self.E = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2)
        self.D = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2)
        self.C = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_embed),
            nn.LogSoftmax(dim = 2)
        )
        self.w2i = word_to_index
        self.device = self._prapare_gpu()

    def forward(self, feat, idxs = None, p = 0):
        # built-in state in source code
        # True means training mode
        mode = self.training

        device, w2i = self.device, self.w2i
        # -------------------
        # Encoding stage
        # -------------------

        # switch batch_size and seq_len
        E_in = self.feat_fc(feat).transpose(0, 1) 
        # print('E_in', E_in.shape)
        # E_out contains h_t at each time: (seq_len, batch, num_directions * hidden_size):
        E_out, (_, _) = self.E(E_in)
        # print('E_out', E_out.shape)
        # pad embedding output
        EN_out_pad = torch.zeros(E_in.shape[0], E_in.shape[1], 256).to(device)
        # print('EN_out', EN_out_pad.shape)
        # concat on the third dim
        D_in = torch.cat((EN_out_pad, E_out), dim = 2)
        # print('D_in', D_in.shape)

        # -------------------
        # Decoding stage
        # -------------------

        seq_len = idxs.shape[1]
        all_D_out = torch.Tensor().to(device)
        D_out = None
        # from <BOS> to last word(before <EOS>)
        for i in range(-1, seq_len - 1):
            # print('iter', i)
            E_in_pad = torch.zeros(1, E_in.shape[1], 512).to(device)
            # print('E_in_pad', E_in_pad.shape)
            E_out, (_, _) = self.E(E_in_pad)
            # print('E_out', E_out.shape)
            use_gt = random.random() < p
            EN_out = None
            if i == -1:
                bos = torch.tensor(w2i['<BOS>']).expand_as(idxs[:, 0]).to(device)
                EN_out = self.EN(bos).unsqueeze(0)
            elif use_gt == True:
                in_idxs = idxs[:, i]
                EN_out = self.EN(in_idxs).unsqueeze(0)
            else:
                pred = self.C(D_out)
                in_idxs = torch.argmax(pred, dim = 2)
                EN_out = self.EN(in_idxs)

            # print('EN_out', EN_out.shape)
            D_in = torch.cat((EN_out, E_out), dim = 2)
            # print('D_in', D_in.shape)
            D_out, (_, _) = self.D(D_in)

            all_D_out = torch.cat((all_D_out, D_out), dim = 0)
        
        # print('all_D_out', all_D_out.shape)
        
        C_out = self.C(all_D_out).transpose(0, 1)
        # print('C_out', C_out.shape)
        return C_out

    def _prapare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device