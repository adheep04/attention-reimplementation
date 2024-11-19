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
        
    # enc_mask -> source mask applied encoder encoder
    # dec_mask -> target mask applied in decoder     
    # - ensures each token attends to itself and the ones before and not ahead
    # - essentially "pads" future tokens
    def forward(self, x, encoder_output, enc_mask, dec_mask):
        masked_self_attention_output = self.masked_self_attention_block(x, x, x, dec_mask)
        x = self.residual_connection[0](x, masked_self_attention_output)
        
        # encoder output becomes the Key and Value matrices and the decoder provides the Query
        # "This allows every position in the decoder to attend over all positions in the input sequence"
        cross_attention_output = self.cross_attention_block(x, encoder_output, encoder_output, enc_mask)
        x = self.residual_connection[1](x, cross_attention_output)
        
        feed_forward_output = self.feed_forward_block(x)
        x = self.residual_connection[2](x, feed_forward_output)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, enc_mask, dec_mask):
        
        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask, dec_mask)
        
        x = self.norm(x)
        return x