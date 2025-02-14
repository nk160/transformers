import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.config import load_config

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Calculates attention scores between queries and keys, then applies them to values.
    """
    def __init__(self, d_k=64):  # d_k is dimension per head
        super().__init__()
        self.scale_factor = 1 / math.sqrt(d_k)
    
    def forward(self, queries, keys, values, mask=None):
        """
        Args:
            queries: (batch_size, num_heads, seq_len, d_k)
            keys: (batch_size, num_heads, seq_len, d_k)
            values: (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask
        """
        batch_size = queries.size(0)
        num_heads = queries.size(1)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores * self.scale_factor
        
        # Handle mask properly
        if mask is not None:
            # Add head dimension if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dim
            
            # Expand mask to match batch size and heads
            if mask.size(0) == 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            if mask.size(1) == 1:
                mask = mask.expand(-1, num_heads, -1, -1)
            
            # Apply mask
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, values)
        
        return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(d_k=self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Flatten the input tensors if needed
        if query.dim() == 4:
            query = query.reshape(batch_size, -1, self.d_model)
        if key.dim() == 4:
            key = key.reshape(batch_size, -1, self.d_model)
        if value.dim() == 4:
            value = value.reshape(batch_size, -1, self.d_model)
        
        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Handle mask
        if mask is not None:
            # Add head dimension if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            
            # Expand mask for all heads
            mask = mask.expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Calculate attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, d_k)
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    def __init__(self, d_model=512, d_ff=2048):
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
    def __init__(self, d_model=512, eps=1e-12):
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
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        # Self attention block with residual connection
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        attn_output, _ = self.self_attention(x, x, x, mask)
        
        # Ensure attn_output has same shape as residual before adding
        if attn_output.shape != residual.shape:
            attn_output = attn_output.view(residual.shape)
            
        x = residual + self.dropout(attn_output)
        
        # Feed forward block with residual connection
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings.
    Uses fixed sinusoidal encodings from the original transformer paper.
    """
    def __init__(self, d_model=512, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(1, max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Apply positional encoding
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])  # Make sure we don't exceed d_model
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]  # Match input dimensions

class Encoder(nn.Module):
    """
    Complete Encoder with multiple layers.
    Processes input sequence through multiple encoder layers with positional encoding.
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, 
                 d_ff=2048, dropout=0.1, max_seq_length=5000):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through each encoder layer
        attentions = []
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                x = layer(x, mask)  # Don't request attention weights
        
        # Final layer normalization
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    """
    Single layer of the decoder.
    Contains three sub-layers:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention with Encoder output
    3. Position-wise Feed-Forward Network
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence (batch_size, target_seq_len, d_model)
            encoder_output: Output from encoder (batch_size, source_seq_len, d_model)
            src_mask: Mask for encoder output
            tgt_mask: Mask for target sequence
        """
        # Self attention block
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = residual + self.dropout(self_attn_output)
        
        # Cross attention block
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = residual + self.dropout(cross_attn_output)
        
        # Feed forward block
        residual = x
        x = self.norm3(x)  # Pre-norm architecture
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        return x

class Decoder(nn.Module):
    """
    Complete Decoder with multiple layers.
    Processes target sequence using multiple decoder layers with positional encoding.
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8,
                 d_ff=2048, dropout=0.1, max_seq_length=5000, vocab_size=None):
        super().__init__()
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence (batch_size, seq_len, d_model) or (batch_size, num_captions, seq_len, d_model)
        """
        # Handle case where input has extra dimension for multiple captions
        if x.dim() == 4:
            B, num_caps, seq_len, d_model = x.shape
            x = x.reshape(B * num_caps, seq_len, d_model)  # Use reshape instead of view
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Restore shape if needed
        if x.dim() == 3 and 'num_caps' in locals():
            x = x.reshape(B, num_caps, seq_len, -1)  # Use reshape here too
            
        return self.norm(x)

class Transformer(nn.Module):
    """
    Complete Transformer model for image captioning.
    Combines Encoder, Decoder, and final prediction layers.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = load_config('config.yaml')
        
        self.vocab_size = config['model']['vocab_size']
        self.d_model = config['model']['d_model']
        
        # Add embedding layer at transformer level
        self.caption_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Initialize components
        self.encoder = Encoder(
            num_layers=config['model']['num_encoder_layers'],
            d_model=self.d_model,
            num_heads=config['model']['num_heads'],
            d_ff=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        )
        
        self.decoder = Decoder(
            num_layers=config['model']['num_decoder_layers'],
            d_model=self.d_model,
            num_heads=config['model']['num_heads'],
            d_ff=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            vocab_size=self.vocab_size
        )
        
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Image features (batch_size, n_patches, d_model)
            tgt: Caption tokens (batch_size, num_captions, seq_len)
        """
        print(f"Input shapes - src: {src.shape}, tgt: {tgt.shape}")
        print(f"Mask shapes - src_mask: {src_mask.shape if src_mask is not None else None}, tgt_mask: {tgt_mask.shape if tgt_mask is not None else None}")
        
        # Encode image features
        encoder_output = self.encoder(src, src_mask)
        print(f"Encoder output shape: {encoder_output.shape}")
        
        # Process encoder output
        encoder_output = encoder_output.squeeze()
        print(f"After squeeze shape: {encoder_output.shape}")
        
        # Get dimensions
        B, num_caps, seq_len = tgt.shape
        print(f"Target dimensions - B: {B}, num_caps: {num_caps}, seq_len: {seq_len}")
        
        # Reshape target for embedding
        tgt = tgt.reshape(B * num_caps, seq_len)
        tgt_embeddings = self.caption_embedding(tgt)
        
        # Prepare encoder output for each caption
        encoder_output = encoder_output.unsqueeze(1)  # [B, 1, N, D]
        encoder_output = encoder_output.expand(-1, num_caps, -1, -1)  # [B, num_caps, N, D]
        encoder_output = encoder_output.reshape(B * num_caps, -1, self.d_model)  # [B*num_caps, N, D]
        
        # Adjust masks for multiple captions
        if src_mask is not None:
            src_mask = src_mask.repeat_interleave(num_caps, dim=0)  # Repeat for each caption
        if tgt_mask is not None:
            # Add batch dimension and expand
            if tgt_mask.dim() == 2:
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
            elif tgt_mask.dim() == 3:
                tgt_mask = tgt_mask.unsqueeze(0)  # Add head dim
            # Now expand to match batch size
            tgt_mask = tgt_mask.expand(B * num_caps, -1, seq_len, seq_len)
        
        # Decode
        decoder_output = self.decoder(tgt_embeddings, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.linear(decoder_output)
        
        # Restore batch dimension
        output = output.reshape(B, num_caps, seq_len, -1)
        
        return output