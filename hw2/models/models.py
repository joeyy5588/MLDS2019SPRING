import torch
import torch.nn as nn
import numpy as np
import random
from .base_model import BaseModel
import torch.nn.functional as F


class Model(BaseModel):
    def __init__(self, n_embed, word_to_index, batch_size, rnn_type, attention_type, bi = False):
        """
        Usage:
            It is S2VT model.
            For visualized model structure, please refer to readme.md
            ref: https://arxiv.org/pdf/1505.00487.pdf
        """
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = 256
        self.max_length = 15
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        self.n_embed = n_embed
        self.bi = bi
        self.EM = nn.Embedding(num_embeddings =  n_embed, embedding_dim = 256, padding_idx = 0)
        self.feat_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
        )
        # input of shape (batch, seq_len, input_size)
        if rnn_type == "LSTM":
            self.E = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = self.bi)
            self.D = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = self.bi)
        elif rnn_type == "GRU":
            self.E = nn.GRU(input_size = 512, hidden_size = 256, num_layers = 1, dropout = 0, bidirectional = bi, batch_first = True)
            self.D = nn.GRU(input_size = 512, hidden_size = 256, num_layers = 1, dropout = 0, bidirectional = bi, batch_first = True)
        if attention_type is not None:
            self.A = Attention(self.batch_size, self.hidden_size, method = attention_type, mlp = False)
            self.C = nn.Sequential(
                nn.Linear(512, 2048),
                #nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, n_embed),
                #nn.BatchNorm1d(n_embed),
                #nn.ReLU(True),
                nn.LogSoftmax(dim = 1)
            )
        else:
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
        E_in = self.feat_fc(feat)#.transpose(0, 1) 

        # E_out contains hidden state at each time
        # E_out dim: (batch, seq_len, E_hidden_dim)
        if self.rnn_type == "LSTM":
            E_out, (E_h, _) = self.E(E_in)
        elif self.rnn_type == "GRU":
            E_out, E_h = self.E(E_in)
        if self.bi:
            E_out = E_out[:, :, :self.hidden_size] + E_out[:, :, self.hidden_size:]
        
        # pad embedding output
        # EM_out_pad dim: (batch, seq_len, EM_hidden_dim)
        EM_out_pad = torch.zeros(E_in.shape[0], E_in.shape[1], 256).to(device)

        # concat on the third dim
        # D_in dim: (batch, seq_len, E_hidden_dim + EM_hidden_dim)
        D_in = torch.cat((EM_out_pad, E_out), dim = 2)
        if self.rnn_type == "LSTM":
            D_out, (_, _) = self.D(D_in)
        elif self.rnn_type == "GRU":
            D_out, _ = self.D(D_in)
        if self.bi:
            D_out = D_out[:, :, :self.hidden_size] + D_out[:, :, self.hidden_size:]
        #print(E_out.size(), D_out.size(), E_h.size())
        """
        [Decoding stage]
        1. E_in is padded with zeros
        2. EM_out_pad is the ground truth if in training mode.
           However, we can determine whether we put gt or pred by schedule sampling.
        3. EM_out_pad is the prediction of model if in testing model.
        """

        seq_len = idxs.shape[1]
        all_D_out = torch.Tensor().to(device)
        D_c = E_out.transpose(1, 0)[-1].to(device)
        D_h = E_h
        alignments = torch.zeros(self.max_length, E_out.size(1), self.batch_size)
        logits = torch.zeros(self.max_length, self.batch_size, self.n_embed).to(device)

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
            #[batch, hidden]
            D_h = D_h.squeeze(0)
            hidden = None
            if self.bi:
                hidden = torch.sum(D_h, 0)
            else:
                hidden = D_h
            # schedule sampling
            use_gt = random.random() < p
            
            EM_out = None
            if i == -1:
                bos = torch.tensor(w2i['<BOS>']).expand_as(idxs[:, 0]).to(device)
                EM_out = self.EM(bos).unsqueeze(1)
                #print("FIRST: ", EM_out.size())
            elif use_gt == True:
                in_idxs = idxs[:, i]
                EM_out = self.EM(in_idxs).unsqueeze(1)
                #print("GT: ", EM_out.size())
            else:
                pred = None
                in_idxs = 0
                if self.attention_type is None:
                    pred = self.C(D_out)
                    in_idxs = torch.argmax(pred, dim = 2)
                    EM_out = self.EM(in_idxs)
                else:
                    pred = logits[i][:E_out.size()[0], :]
                    in_idxs = torch.argmax(pred, dim = 1)
                    EM_out = self.EM(in_idxs).unsqueeze(1)
                #print("PREV: ", EM_out.size())
            if self.attention_type is None:
                E_in_pad = torch.zeros(1, E_in.shape[0], 512).to(device)
                E_out, (_, _) = self.E(E_in_pad)
                D_in = torch.cat((EM_out, E_out), dim = 2)
                D_out = self.D(D_in)
                if self.bi:
                    D_out = D_out[:, :, :self.hidden_size] + D_out[:, :, self.hidden_size:]
                all_D_out = torch.cat((all_D_out, D_out), dim = 1)
                C_out = self.C(all_D_out)#.transpose(0, 1)
            else:
                weights = self.A.forward(hidden, E_out, seq_len)
                #[batch, 1, enc_hid_dim]
                context = weights.unsqueeze(1).bmm(E_out)
                #print(EM_out.size(), context.size(), D_h.size(), E_out.size())
                D_in = torch.cat((EM_out, context), dim = 2)
                #D_out: batch, 1, hidden
                #D_h: 1, batch, hidden
                if self.bi:
                    if self.rnn_type == "LSTM":
                        D_out, (D_h, _) = self.D(D_in, (D_h, _))
                    elif self.rnn_type == "GRU":
                        D_out, D_h = self.D(D_in, D_h)
                else:
                    if self.rnn_type == "LSTM":
                        D_out, (D_h, _) = self.D(D_in, (D_h.unsqueeze(0), _))
                    elif self.rnn_type == "GRU":
                        D_out, D_h = self.D(D_in, D_h.unsqueeze(0))
                if self.bi:
                    D_out = D_out[:, :, :self.hidden_size] + D_out[:, :, self.hidden_size:]
                #all_D_out: batch, hidden * 2
                all_D_out = torch.cat((D_out.squeeze(1), context.squeeze(1)), dim = 1)
                #C_out: batch, n_embed(6717)
                C_out = self.C(all_D_out)
                if C_out.size()[0] != self.batch_size:
                    C_pad = torch.zeros(self.batch_size - C_out.size()[0], C_out.size()[1]).to(device)
                    C_out = torch.cat((C_out, C_pad), dim = 0)
                    #print("C_out: ", C_out.shape)
                logits[i + 1] = C_out
        if self.attention_type is None:
            return C_out
        else:
            #print(logits)
            return logits.transpose(1, 0)            
        
        # Pass the distribution to classifier
        
        

    def _prapare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

class Attention(BaseModel):
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            #self.va = nn.Parameter(torch.ones(batch_size, hidden_size))
            self.va = nn.Parameter(torch.randn(batch_size, hidden_size, dtype=torch.float))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.randn(batch_size, hidden_size, dtype=torch.float))
        else:
            raise NotImplementedError
        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_len, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)
        # [32, 80]
        attention_energies = self._score(last_hidden, encoder_outputs, self.method)
        #print("attention: " , attention_energies.size(), attention_energies)

        #if seq_len is not None:
            #attention_energies = self.mask_3d(attention_energies, seq_len, -float('inf'))
        
        return F.softmax(attention_energies, -1)
    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`)
        :return:
        """
        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1).repeat(1, 80, 1)
            x = torch.tanh(self.Wa(torch.cat((x, encoder_outputs), 2)))#tanh(??)
            return x.bmm(self.va[:x.size()[0], :].unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)

    def mask_3d(self, inputs, seq_len, mask_value=0.):
        batches = inputs.size()[0]
        max_idx = seq_len
        for n, idx in enumerate(range(seq_len)):
            if idx < max_idx:
                if len(inputs.size()) == 3:
                    inputs[n, idx:, :] = mask_value
                else:
                    assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                    inputs[n, idx:] = mask_value
        return inputs
