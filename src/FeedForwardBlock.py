from torch import nn

class FeedForwardBlock(nn.Module):
    def __init__(self, dropout, d_model=512, d_ff=2048):
        super.__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 
        self.relu = nn.ReLU()
    
    # x: (batch, seq_len, d_model)
    # -> (batch, seq_len, d_ff) layer 1 
    # -> (batch, seq_len, d_model) layer 2
    def forward(self, x):
        
        # # first layer
        # x = torch.max(0, self.linear_1(x))
        
        # # relu layer 
        # x = self.relu(x)a
        
        # # second linear layer
        # x = self.linear_2(x)
        
        # # dropout for regularization
        # return self.dropout(x)
        
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))