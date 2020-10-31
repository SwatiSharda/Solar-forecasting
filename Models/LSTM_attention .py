# -*- coding: utf-8 -*-
"""LSTM-attention.ipynb





import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class lstm_a(nn.Module) :
    def __init__(self, seq_len=256, ini_len=18, final_len=1) :
        super().__init__()
        self.d_model = ini_len 
        self.seq_len = seq_len
        self.hidden_size = 32
        self.num_layers = 1
        self.lstm = nn.LSTM(self.d_model,self.hidden_size,self.num_layers,return_sequences=True, batch_first=True, bidirectional=True)
        self.W_s1 = nn.Linear(self.hidden_size*self.seq_len,512)
	self.W_s2 = nn.Linear(512,final_len)
        self.final = nn.Sequential(nn.Linear(self.hidden_size*self.seq_len,512),nn.ReLU(),nn.Linear(512,final_len))
      def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix
    
    def forward(self,batch) :
        output, (_b,_a) = self.lstm_a( batch )
        output = output.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        out = self.final(hidden_matrix.reshape(-1,self.hidden_size*self.seq_len)                                                                                        
        return out
