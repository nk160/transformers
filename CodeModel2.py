import torch
import torch.nn as nn
import torch.nn.functional as F
from Background1 import *  # Import our constants
import math

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

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    def __init__(self, d_model=D_MODEL, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # First linear layer (512 -> 2048)
        self.linear2 = nn.Linear(d_ff, d_model)   # Second linear layer (2048 -> 512)
        self.relu = nn.ReLU()                     # ReLU activation between layers
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
                e.g., (32, 196, 512) for image patches
        
        Returns:
            Output tensor of same shape (batch_size, seq_len, d_model)
        """
        # First linear layer + ReLU
        x = self.relu(self.linear1(x))  # (batch_size, seq_len, d_ff)
        
        # Second linear layer
        x = self.linear2(x)             # (batch_size, seq_len, d_model)
        
        return x 

class LayerNorm(nn.Module):
    """
    Layer Normalization.
    Normalizes the inputs across the features for stability.
    """
    def __init__(self, d_model=D_MODEL, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Learnable bias parameter
        self.eps = eps  # Small constant for numerical stability
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
                e.g., (32, 196, 512) for image patches
        
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift with learnable parameters
        return self.gamma * x_norm + self.beta 

class EncoderLayer(nn.Module):
    """
    Single layer of the encoder.
    Consists of Multi-Head Attention followed by Feed-Forward Network,
    with Layer Normalization and residual connections.
    """
    def __init__(self, d_model=D_MODEL, num_heads=N_HEADS, d_ff=2048, dropout=0.1):
        super().__init__()
        # Main components
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
                e.g., (32, 196, 512) for image patches
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape
        """
        # Self attention block
        norm_x = self.norm1(x)
        attn_output, attention_weights = self.self_attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed forward block
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        if return_attention:
            return x, attention_weights
        return x 

class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings.
    Uses fixed sinusoidal encodings from the original transformer paper.
    """
    def __init__(self, d_model=D_MODEL, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (won't be trained)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Output tensor with added positional encodings
        """
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    """
    Complete Encoder with multiple layers.
    Processes input sequence through multiple encoder layers with positional encoding.
    """
    def __init__(self, num_layers=6, d_model=D_MODEL, num_heads=N_HEADS, 
                 d_ff=2048, dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Optional, attention weights from all layers
        """
        attention_weights = []
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through each encoder layer
        for layer in self.layers:
            x, attn = layer(x, mask, return_attention=return_attention)
            if return_attention:
                attention_weights.append(attn)
        
        # Final layer normalization
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights
        return x 

class DecoderLayer(nn.Module):
    """
    Single layer of the decoder.
    Contains three sub-layers:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention with Encoder output
    3. Position-wise Feed-Forward Network
    """
    def __init__(self, d_model=D_MODEL, num_heads=N_HEADS, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Main components
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence (batch_size, target_seq_len, d_model)
            encoder_output: Output from encoder (batch_size, source_seq_len, d_model)
            src_mask: Mask for encoder output
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
        
        Returns:
            Output tensor of shape (batch_size, target_seq_len, d_model)
        """
        # 1. Masked Self-Attention
        # First normalization
        norm_x = self.norm1(x)
        # Apply masked self-attention
        attn_output, _ = self.self_attention(norm_x, norm_x, norm_x, tgt_mask)
        # Residual connection and dropout
        x = x + self.dropout(attn_output)
        
        # 2. Cross-Attention with Encoder output
        # Second normalization
        norm_x = self.norm2(x)
        # Apply cross-attention
        cross_attn_output, _ = self.cross_attention(
            norm_x,                  # Query from decoder
            encoder_output,          # Key from encoder
            encoder_output,          # Value from encoder
            src_mask                 # Mask for encoder output
        )
        # Residual connection and dropout
        x = x + self.dropout(cross_attn_output)
        
        # 3. Feed Forward
        # Third normalization
        norm_x = self.norm3(x)
        # Apply feed forward
        ff_output = self.feed_forward(norm_x)
        # Final residual connection and dropout
        x = x + self.dropout(ff_output)
        
        return x

class Decoder(nn.Module):
    """
    Complete Decoder with multiple layers.
    Processes target sequence using multiple decoder layers with positional encoding.
    """
    def __init__(self, num_layers=6, d_model=D_MODEL, num_heads=N_HEADS,
                 d_ff=2048, dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence (batch_size, target_seq_len, d_model)
            encoder_output: Output from encoder (batch_size, source_seq_len, d_model)
            src_mask: Mask for encoder output
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
        
        Returns:
            Output tensor of shape (batch_size, target_seq_len, d_model)
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final layer normalization
        return self.norm(x)

class Transformer(nn.Module):
    """
    Complete Transformer model for image captioning.
    Combines Encoder, Decoder, and final prediction layers.
    """
    def __init__(self, vocab_size, d_model=D_MODEL, num_heads=N_HEADS,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        # Main components
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Final linear layer for prediction
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence (image features)
            tgt: Target sequence (partial caption)
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence
        
        Returns:
            Output probabilities over vocabulary
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode with encoder output
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size and apply softmax
        output = self.linear(decoder_output)
        return F.log_softmax(output, dim=-1)