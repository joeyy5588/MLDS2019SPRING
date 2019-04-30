import torch.nn as nn
import torch.nn.functional as F
import torch
from .base_model import BaseModel

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