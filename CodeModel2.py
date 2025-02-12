import torch
import torch.nn as nn
import torch.nn.functional as F
from Background1 import *  # Import our constants

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Calculates attention scores between queries and keys, then applies them to values.
    """
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / torch.sqrt(torch.tensor(D_K, dtype=torch.float32))
    
    def forward(self, queries, keys, values, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            queries: (batch_size, num_queries, d_k) # d_k is used for both Q and K
            keys: (batch_size, num_keys, d_k)
            values: (batch_size, num_values, d_v)
            mask: Optional mask to prevent attention to certain positions
        
        Returns:
            attention_output: (batch_size, num_queries, d_v)
            attention_weights: (batch_size, num_queries, num_keys)
        """
        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch_size, num_queries, num_keys)
        attention_scores = attention_scores * self.scale_factor
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attention_output = torch.bmm(attention_weights, values)
        
        return attention_output, attention_weights 

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model=D_MODEL, num_heads=N_HEADS):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model          # 512
        self.num_heads = num_heads      # 8
        self.d_k = d_model // num_heads # 64
        self.d_v = self.d_k            # 64
        
        # Linear layers for projecting input into Q, K, V for each head
        self.W_q = nn.Linear(d_model, d_model)  # Projects to (num_heads * d_k)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention() 

    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention.
        
        Args:
            query: (batch_size, num_q, d_model)   # e.g., (32, 196, 512) for image patches
            key: (batch_size, num_k, d_model)     # e.g., (32, 196, 512)
            value: (batch_size, num_v, d_model)   # e.g., (32, 196, 512)
            mask: Optional mask to prevent attention to certain positions
        
        Returns:
            output: (batch_size, num_q, d_model)  # Same shape as query
            attention_weights: Average attention weights over all heads
        """
        batch_size = query.size(0)
        
        # 1. Linear projections and reshape for multi-head attention
        # Process:
        # a) Project with W_q/W_k/W_v: (batch_size, num_q/k/v, d_model)
        # b) Reshape: (batch_size, num_q/k/v, num_heads, d_k)
        # c) Transpose: (batch_size, num_heads, num_q/k/v, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # 2. Apply attention to each head
        # Each head processes its own d_k-dimensional subspace
        # Q, K dimensions: (batch_size, num_heads, num_q/k, d_k)
        # V dimensions: (batch_size, num_heads, num_v, d_v)
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. Reshape and project back to d_model size
        # a) Transpose back: (batch_size, num_q, num_heads, d_k)
        # b) Concatenate heads: (batch_size, num_q, d_model)
        # c) Project to final output: (batch_size, num_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)  # Final linear projection
        
        return output, attention_weights 