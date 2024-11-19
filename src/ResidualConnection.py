from torch import nn
import LayerNormalization

# the "Add and Norm" block
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        
        # original paper passes x through the sublayer before normalizing it
        # but most implementations do it the other way around as below
        sublayer_processed = self.dropout(sublayer(self.norm(x)))
        
        # raw input is combined with input that is processed
        residual_connection = x + sublayer_processed
        
        return residual_connection