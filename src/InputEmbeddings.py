import torch
from torch import nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        
        # dimensions of the representational vector for 1 sample
        self.d_model = d_model
        
        # size of the total number of words in the model's dictionary
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    # creates an input embedding "lookup table" 
    # x: 1d tensor representing input tokens ->
    # returns [vocab_size, d_model] tensor 
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)