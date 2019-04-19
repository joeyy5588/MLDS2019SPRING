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
        # input()
        return output, state

H_SIZE = 256
EM_SIZE = 256
MAX_LENGTH = 15
 
class Seq2Seq(BaseModel):
    def __init__(self, w2i, embed):
        super(Seq2Seq, self).__init__()
        embedding = nn.Embedding(len(w2i), H_SIZE, padding_idx = w2i['<PAD>'])
        embedding.weight = nn.Parameter(torch.tensor(embed))
        embedding.weight.requires_grad = False
        self.embedding = embedding
        self.EncoderRNN = EncoderRNN(self.embedding, EM_SIZE)
        self.AttnDecoderRNN = AttnDecoderRNN(H_SIZE, len(w2i), self.embedding, max_length = MAX_LENGTH)
        self.w2i = w2i
        self.device = self._prepare_gpu()

    def forward(self, input_seq, output_seq):

        E_out, (_, _) = self.EncoderRNN(input_seq)
        # print('E_out', E_out.shape)
        batch_size = input_seq.shape[0]
        # fake_input = torch.zeros((batch_size, 1), dtype = torch.long)
        # fake_prevh = torch.zeros((batch_size, 1, H_SIZE))
        # fake_eout = torch.zeros((batch_size, MAX_LENGTH, H_SIZE))

        # <BOS>
        BOS = self.w2i['<BOS>']
        all_prob = torch.tensor([]).to(self.device)       
        
        first_input = torch.full((batch_size, 1), BOS, dtype = torch.long).to(self.device)
        default_h = torch.zeros((1, batch_size, H_SIZE)).to(self.device)
        prob, state = self.AttnDecoderRNN(first_input, (default_h, default_h), E_out)

        all_prob = torch.cat((all_prob, prob), dim = 1)
        
        seq_len = output_seq.shape[1]
        for i in range(seq_len):
            prob, state = self.AttnDecoderRNN(output_seq[:, i].unsqueeze(1), state, E_out)
            all_prob = torch.cat((all_prob, prob), dim = 1)
            # print(x)
            # input()
        # print('all_prob', all_prob.shape)
        # input()
        return all_prob[:, :-1]

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device