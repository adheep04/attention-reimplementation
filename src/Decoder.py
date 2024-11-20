from torch import nn
import ResidualConnection
import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(self, masked_self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.masked_self_attention_block = masked_self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
    # in_mask -> source mask applied encoder encoder
    # out_mask -> target mask applied in decoder     
    # - ensures each token attends to itself and the ones before and not ahead
    # - essentially "pads" future tokens
    def forward(self, y, encoder_output, in_mask, out_mask):
        masked_self_attention_output = self.masked_self_attention_block(y, y, y, out_mask)
        y = self.residual_connection[0](y, masked_self_attention_output)
        
        # encoder output becomes the Key and Value matrices and the decoder provides the Query
        # "This allows every position in the decoder to attend over all positions in the input sequence"
        cross_attention_output = self.cross_attention_block(y, encoder_output, encoder_output, in_mask)
        y = self.residual_connection[1](y, cross_attention_output)
        
        feed_forward_output = self.feed_forward_block(y)
        y = self.residual_connection[2](y, feed_forward_output)
        return y
    
class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, y, encoder_output, in_mask, out_mask):
        
        for layer in self.layers:
            y = layer(y, encoder_output, in_mask, out_mask)
        
        y = self.norm(y)
        return y