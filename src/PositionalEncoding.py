import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        self.d_model = d_model
        self.seq_len = seq_len
        
        # dropout prevents co-adapation of neurons/feature-detectors 
        # probability that any neuron's output will be "dropped" i.e. val set to 0
        self.dropout = nn.Dropout(dropout)
        
        # create a 2d tensor (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # 1d vector of (seq_len, 1) representing position of word in sentence
        pos = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # term to multiply by pos and run through sin/cos value for pe(n, i)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float()*(-torch.log(10000.0)/d_model))
        
        # apply positional values
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        
        # "batchify" from [seq_len,d_model] -> [1,seq_len,d_model]
        pe.unsqueeze(0)
        
        # saves tensor as a file
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        
        # makes this tensor "unlearned", doesn't perform gradient descent
        x = x.requires_grad_(False)
        
        # pass through dropout for regularization 
        return self.dropout(x)