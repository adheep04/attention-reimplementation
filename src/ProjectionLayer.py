from torch import nn
import torch

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x,apply_softmax=False):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # projects the rich high-dimensional representation into vectors that identify
        # specific tokens
        x = self.proj(x)
        
        # Apply softmax during inference if requested
        # only use if loss is not cross entropy loss
        if apply_softmax:
            x = torch.softmax(x, dim=-1)
        
        return x