import torch
from torch import nn

class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super.__init__()
        self.d_model = d_model
        
        # number of heads that Q, K, and V will be split into
        # heads are split so each head contains the entire input sequence
        self.h = h
        assert d_model % h == 0, "d_model needs to be divisible by h"
        self.dk = d_model // h
        self.dropout = dropout
        
        # below are the 3 weights that the input embeddings are cloned and 
        # multiplied by to begin self-attention
        self.wQ = nn.Linear(d_model, d_model) # Query
        self.wK = nn.Linear(d_model, d_model) # Key
        self.wV = nn.Linear(d_model, d_model) # Value
        
        # the final weight matrix 
        self.wO = nn.Linear(d_model, d_model, bias=False) # Value
    
    @staticmethod
    # method is non-vectorized intentionally because it helped me understand :D
    def attention_simple(query, key, value, mask, dk, dropout):
        
        # intiialize list of heads
        heads = []
        scores = []
        
        # Assume q, k, v have been reshaped to: (batch, h, seq_len, d_k)
        # Split q, k, v into individual heads
        q_heads = torch.unbind(query, dim=1)  # Creates a tuple of `h` tensors, each of shape (batch, seq_len, d_k)
        k_heads = torch.unbind(key, dim=1)  # Similarly for k
        v_heads = torch.unbind(value, dim=1)  # Similarly for v
        
        # for each head in Q, K, V respectively, calculate attention
        # scores for each head using scaled dot-product and add it to list
        for q, k, v in zip(q_heads, k_heads, v_heads):
            
            # calculate attention score
            # transpose k
            # (batch, seq, d_model) * (batch, d_model, seq) -> (batch, seq, seq)
            attention_scores = q @ k.transpose(-2, -1)
            
            # attention score scaled (ASS) by dividing by sqrt(d_model/h)
            # indicates how much attention each element should pay to other elements
            # A[i, j] represents the importance between the i-th input "query" and 
            # j-th input "key"
            attention_scores_scaled = attention_scores/torch.sqrt(dk)
            
            # Apply mask (if provided)
            # used to ignore/pay no attention to certain positions in the sequence for padding (fillter to ensure sequences are aligned) 
            # and future tokens by setting their attention scores to -1e9
            if mask is not None:
                # Assuming mask has shape (batch, 1, seq_len, seq_len)
                attention_scores = attention_scores_scaled.masked_fill(mask == 0, -1e9)
            
            # softmaxing the head makes it so each row sums to 1
            attention_scores_scaled = torch.softmax(attention_scores_scaled, dim=-1)
            
            # adding the scaled attention vector to a seperate list
            scores.append(attention_scores_scaled)
            
            # multiply attention scores by v
            # since the head gives us attention weights, multiplying by v will
            # propogate the attention information by weighting and adding the 
            # features of each element according to their importance to the current element
            head_i = attention_scores_scaled @ value
            heads.append(head_i)
        
        
        # applies dropout regularization
        if dropout is not None:
            heads = dropout(heads)
            
        # returns a concatenation of the h heads
        # Q, K, V: (batch, seq, d_model) -> (batch, h, seq, d_model/h) -> (batch, seq, d_model)
        concatenated_heads = torch.cat(heads, dim=-1)
        concatenated_scores = torch.cat(scores, dim=-1)
        
        # returns a concatenation of the heads and the scores
        # Q, K, V: (batch, seq, d_model) -> (batch, h, seq, d_model/h) -> (batch, seq, d_model)
        return concatenated_heads, concatenated_scores
            
    @staticmethod
    # vectorized approach, more time efficient but less clear
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        h = query.shape[1]

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # (Batch, h, Seq_Len, Seq_Len) x (Batch, h, Seq_Len, d_k) -> (Batch, h, Seq_Len, d_k)
        x = attention_scores @ value
        
        # (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k) -> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, h * d_k)
        
        return x, attention_scores

    
    def forward(self, q, k, v, mask):
        
        # multiplied by respective weights
        # 3 x (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.wQ(q)
        key = self.wK(k)
        value = self.wV(v)
        
        
        # Assume q, k, v have the shape: (batch, seq, d_model)
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # Reshape q, k, v to have separate heads and transpose to 
        # bring the heads dimension before the sequence dimension
        # New shape: (batch, seq, h, d_model // h)
        query = query.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        
        # 
        x, self.attention_scores = MultiHeadedAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # run this through the final weight matrix in this block
        return self.wO(x)
    