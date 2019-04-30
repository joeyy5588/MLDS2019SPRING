import torch.nn as nn
import torch.nn.functional as F
import torch
from .base_model import BaseModel
import numpy as np

class EncoderRNN(BaseModel):
    def __init__(self, embedding, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True)
    def forward(self, Input):
        embedded = self.embedding(Input)
        output = embedded
        output = self.lstm(output)
        return output

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.output_size = output_size
        self.embedding = embedding
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True)
        self.out = nn.Linear(self.hidden_size, self.output_size) 

    def forward(self, Input, state, encoder_outputs):
        hidden = state[0].transpose(0, 1)
        embedded = self.embedding(Input)
        embedded = self.dropout(embedded)
        # print('embedded', embedded.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), dim = 2)), dim = 2)
        # print('attn_weights', attn_weights.shape)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # print('attn_applied', attn_applied.shape)
        output = torch.cat((embedded, attn_applied), dim = 2)
        # print('output', output.shape)
        output = self.attn_combine(output)
        # print('output', output.shape)
        output = F.relu(output)
        output, state = self.lstm(output, state)
        # print('output', output.shape)
        #output = F.softmax(self.out(output), dim = 2)
        output = F.log_softmax(self.out(output), dim = 2)
        # print('output', output.shape)
        return output, state

class Seq2Seq(BaseModel):
    def __init__(self, w2i, embed, hidden_size = 256, pad_len = 15):
        super(Seq2Seq, self).__init__()
        embedding = nn.Embedding(len(w2i), hidden_size, padding_idx = w2i['<PAD>'])
        embedding.weight = nn.Parameter(torch.tensor(embed))
        embedding.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.EncoderRNN = EncoderRNN(self.embedding, hidden_size)
        self.AttnDecoderRNN = AttnDecoderRNN(hidden_size, len(w2i), self.embedding, max_length = pad_len)
        self.w2i = w2i
        self.device = self._prepare_gpu()

    def forward(self, input_seq, output_seq, p = 1):
        # ==============
        # Encoding stage
        # ==============
        E_out, (_, _) = self.EncoderRNN(input_seq)
        
        # ==============
        # Dcoding stage
        # ==============

        # Firstly, put BOS into the decoder.
        batch_size = input_seq.shape[0]
        BOS = self.w2i['<BOS>']
        all_prob = torch.tensor([]).to(self.device)       
        first_input = torch.full((batch_size, 1), BOS, dtype = torch.long).to(self.device)
        default_h = torch.zeros((1, batch_size, self.hidden_size)).to(self.device)
        prob, state = self.AttnDecoderRNN(first_input, (default_h, default_h), E_out)
        all_prob = torch.cat((all_prob, prob), dim = 1)
        
        # Later, keep putting ground_truth or prediction into the decoder
        seq_len = output_seq.shape[1] if torch.is_tensor(output_seq) else 15
        for i in range(seq_len):
            pred_output = torch.argmax(prob, dim = 2)
            if p == 1:
                prob, state = self.AttnDecoderRNN(output_seq[:, i].unsqueeze(1), state, E_out)
            elif p == 0:
                prob, state = self.AttnDecoderRNN(pred_output, state, E_out)
            all_prob = torch.cat((all_prob, prob), dim = 1)

        return all_prob[:, :-1]

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

def beam_search(s2s, beam_size, input_seq, output_seq):
    # ==============
    # Encoding stage
    # ==============
    E_out, (_, _) = s2s.EncoderRNN(input_seq)
        
    # ==============
    # Dcoding stage
    # ==============

    # Firstly, put BOS into the decoder.

    batch_size = input_seq.shape[0]
    BOS = s2s.w2i['<BOS>']
    all_prob = torch.tensor([]).to(s2s.device)       
    first_input = torch.full((batch_size, 1), BOS, dtype = torch.long).to(s2s.device)
    default_h = torch.zeros((1, batch_size, s2s.hidden_size)).to(s2s.device)
    prob, state = s2s.AttnDecoderRNN(first_input, (default_h, default_h), E_out)
    #prob = torch.tensor(np.power(10, prob.data.cpu().numpy())).to(s2s.device)
    # Record Top k index
    top_n_prob = torch.topk(prob, beam_size)[0]
    top_n_idx = torch.topk(prob, beam_size)[1]
    top_n_path = torch.topk(prob, beam_size)[1].squeeze(1).unsqueeze(2)
    
    # Later, keep putting ground_truth or prediction into the decoder
    seq_len = output_seq.shape[1] if torch.is_tensor(output_seq) else 15
    for i in range(seq_len):
        #pred_output = torch.argmax(prob, dim = 2)
        temp_prob = torch.tensor([]).to(s2s.device)
        temp_idx = torch.tensor([]).long().to(s2s.device)
        temp_path = torch.tensor([]).long().to(s2s.device)
        for j in range(beam_size):
            prob, state = s2s.AttnDecoderRNN(top_n_idx[:, :, j], state, E_out)
            #prob = torch.tensor(np.power(10, prob.data.cpu().numpy())).to(s2s.device)
            temp_result = torch.topk(prob, beam_size)
            temp_n_prob = temp_result[0]

            temp_n_prob = top_n_prob[:, :, j].unsqueeze(2).repeat(1, 1, beam_size) + temp_n_prob
            temp_n_idx = temp_result[1]
            temp_n_path = temp_result[1].squeeze(1).unsqueeze(2)
            temp_n_path = torch.cat((top_n_path[:, j, :].unsqueeze(1).repeat(1, beam_size, 1), temp_n_path), dim = 2)
            temp_prob = torch.cat((temp_prob, temp_n_prob), dim = 2)
            temp_idx = torch.cat((temp_idx, temp_n_idx), dim = 2)
            temp_path = torch.cat((temp_path, temp_n_path), dim = 1)
        step = torch.topk(temp_prob, beam_size)
        mask = step[1]
        top_n_prob = step[0]
        top_n_idx = torch.gather(temp_idx, dim = 2, index = mask)
        top_n_path = torch.tensor([]).long().to(s2s.device)
        for j in range(mask.size()[0]):
            top_n_path = torch.cat((top_n_path, temp_path[j, mask[j, :, :], :]), dim = 0)

    return top_n_path[:, 0, :]