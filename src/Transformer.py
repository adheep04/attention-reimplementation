from torch import nn
import TokenEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadedAttentionBlock
import ResidualConnection, Encoder, Decoder, ProjectionLayer

class Transformer(nn.Module):
    def __init__(self,      
                 encoder,   # encoder
                 decoder,   # decoder
                 in_embed,     # embeddings of input language
                 out_embed,     # embeddings of output language
                 in_pos_enc,       # positional encoding layer input
                 out_pos_enc,      # positional encoding layer output         
                 proj_layer):   # projection into output word-space
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_embed = in_embed
        self.out_embed = out_embed
        self.in_pos_enc = in_pos_enc
        self.out_pos_enc = out_pos_enc
        self.proj_layer = proj_layer
        
    def encode(self, x, enc_mask):
        x = self.in_pos_enc(self.in_embed(x))
        return self.encoder(x, enc_mask)
        
    
    def decode(self, y, encoder_output, enc_mask, dec_mask):
        y = self.out_pos_enc(self.out_embed(y))
        y = self.decoder(y, encoder_output, enc_mask, dec_mask)
        return y
         
    def project(self, x, apply_softmax=False):
        return self.proj_layer(x, apply_softmax)
    

def build_transformer(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    
    # Create embedding layers
    input_embed = TokenEmbeddings(d_model, in_vocab_size)
    output_embed = TokenEmbeddings(d_model, out_vocab_size)
    
    # Create the positional encoding layers
    # Both positional encoding layers are essentially the same but for code readability:
    input_pos_enc = PositionalEncoding(d_model, in_seq_len, dropout)
    output_pos_enc = PositionalEncoding(d_model, out_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for n in range(N):
        self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dropout, d_model, d_ff)
        encoder_block = Encoder.EncoderBlock(self_attention_block, feed_forward_block)
        encoder_blocks.append(encoder_block)
        
    # Create decoder blocks
    decoder_blocks = []
    for n in range(N):
        masked_self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(dropout, d_model, d_ff)
        decoder_block = Decoder.DecoderBlock(masked_self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create Encoder
    encoder = Encoder(encoder_blocks)
    
    # Create Decoder
    decoder = Decoder(decoder_blocks)
    
    # Create projection layer
    proj_layer = ProjectionLayer(d_model, out_vocab_size)
    
    # Finally. Build. Transformer.... 
    transformer = Transformer(encoder, 
                              decoder, 
                              input_embed,
                              output_embed,
                              input_pos_enc,
                              output_pos_enc,
                              proj_layer)
    
    # initialize parameters to speed training process so they're not just random values
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    
    