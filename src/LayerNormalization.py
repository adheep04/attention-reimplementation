import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, eps=(10**-6)):
        super().__init__()
        self.eps = eps
        
        # after normalization, each element of the input vector is
        # scaled by an alpha and has its own bias which are learnable 
        # parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.mean(x, dim=-1, keepdim=True)
        
        return (
            self.alpha * 
            (x - mean)/(std + self.eps)
            + self.bias
            )