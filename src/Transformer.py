from torch import nn

class Transformer(nn.Module):
    def __init__(self,      
                 encoder,   # encoder
                 decoder,   # decoder
                 src_embed,     # embeddings of input language
                 tgt_embed,     # embeddings of output language
                 pos_enc,        # positional encoding layer
                 proj_layer):   # projection into output word-space
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos_enc = pos_enc
        self.proj_layer = proj_layer
        
    def encode(self, x):
        return #TODO
    
    def decode(self, x):
        return #TODO
    
    def project(self, x):
        return #TODO
    
    def forward(self, x):
        return #TODO