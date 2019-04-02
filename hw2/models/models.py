import torch
import torch.nn as nn
import numpy as np
import random
from .base_model import BaseModel

class Model(BaseModel):
    def __init__(self, n_embed, word_to_index):
        """
        Usage:
            It is S2VT model.
            For visualized model structure, please refer to readme.md
            ref: https://arxiv.org/pdf/1505.00487.pdf
        """
        super(Model, self).__init__()
        self.EM = nn.Embedding(num_embeddings =  n_embed, embedding_dim = 256, padding_idx = 0)
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

    def forward(self, feat, idxs, p = 0):
        device, w2i = self.device, self.w2i
        """
        [Encoding stage]
        1. Feed feature in feat_fc
        2. EM_out is padded with all zeros.
        """
        # switch batch_size and seq_len
        # E_in dim: (batch, time_frame, feat_dim) -> (time_fram, batch, feat_dim)
        # we do transpose because in pytorch, rnn is defaultly batch-second.
        E_in = self.feat_fc(feat).transpose(0, 1) 

        # E_out contains hidden state at each time
        # E_out dim: (seq_len, batch, E_hidden_dim)
        E_out, (_, _) = self.E(E_in)
        
        # pad embedding output
        # EM_out_pad dim: (seq_len, batch, EM_hidden_dim)
        EM_out_pad = torch.zeros(E_in.shape[0], E_in.shape[1], 256).to(device)

        # concat on the third dim
        # D_in dim: (seq_len, batch, E_hidden_dim + EM_hidden_dim)
        D_in = torch.cat((EM_out_pad, E_out), dim = 2)

        """
        [Decoding stage]
        1. E_in is padded with zeros
        2. EM_out_pad is the ground truth if in training mode.
           However, we can determine whether we put gt or pred by schedule sampling.
        3. EM_out_pad is the prediction of model if in testing model.
        """

        seq_len = idxs.shape[1]
        all_D_out = torch.Tensor().to(device)
        D_out = None

        """
        We need to iterate the sequence rather than put them all into LSTM 
        because each time frame will require the hidden state of the last time frame.
        Order:
            <BOS> A man is playing <EOS>
            o   o  o   o    o      x
            We will iterate from <BOS> to last word.
            The reason why we do not include <EOS> is that we should use the last word to predict <EOS>.
        """
        for i in range(-1, seq_len - 1):
            E_in_pad = torch.zeros(1, E_in.shape[1], 512).to(device)
            E_out, (_, _) = self.E(E_in_pad)
            # schedule sampling
            use_gt = random.random() < p
            
            EM_out = None
            if i == -1:
                bos = torch.tensor(w2i['<BOS>']).expand_as(idxs[:, 0]).to(device)
                EM_out = self.EM(bos).unsqueeze(0)
            elif use_gt == True:
                in_idxs = idxs[:, i]
                EM_out = self.EM(in_idxs).unsqueeze(0)
            else:
                pred = self.C(D_out)
                in_idxs = torch.argmax(pred, dim = 2)
                EM_out = self.EM(in_idxs)

            D_in = torch.cat((EM_out, E_out), dim = 2)
            D_out, (_, _) = self.D(D_in)
            all_D_out = torch.cat((all_D_out, D_out), dim = 0)
        
        # Pass the distribution to classifier
        C_out = self.C(all_D_out).transpose(0, 1)
        return C_out

    def _prapare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device