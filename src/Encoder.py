from torch import nn
import ResidualConnection, LayerNormalization

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        # 1) residual connection for Attention Output = LayerNorm(Input + MultiHeadAttention(Input))
        # 2) residual connection for FFN Output = LayerNorm(Input + FeedForwardNetwork(Input)) 
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    # src_mask ensures "padding" words don't interact with actual words 
    def forward(self, x, src_mask):
        
        attention_output = self.self_attention_block(x, x, x, src_mask)
        x = self.residual_connection[0](x, attention_output)

        feed_forward_output = self.feed_forward_block(x)
        x = self.residual_connection[1](x, feed_forward_output)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        # represents the N encoder block layers
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        # each successive layer consumes the previous layer recursively
        for layer in self.layers:
            x = layer(x, mask)
            
        # normalize
        x = self.norm(x)
        
        return x 
        