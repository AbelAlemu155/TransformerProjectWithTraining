import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

'''
TODO: Implement this Module.

This file contains the encoder layer implementation used in transformer architectures:

SelfAttentionEncoderLayer: Used in encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Unlike decoder, does not use causal masking (can attend to all positions)
- Used for tasks like encoding input sequences where bidirectional context is needed

The layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for self-attention (no causal mask needed)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate padding masks (no causal mask needed)
   - Return both outputs and attention weights
'''

# class SelfAttentionEncoderLayer(nn.Module):
#     '''
#     Pre-LN Encoder Layer with self-attention mechanism.
#     Used in the encoder part of transformer architectures.
#     '''
#     def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
#         '''
#         Initialize the SelfAttentionEncoderLayer. 
#         Args:
#             d_model   (int): The dimension of the model.
#             num_heads (int): The number of attention heads.
#             d_ff      (int): The dimension of the feedforward network.
#             dropout (float): The dropout rate.
#         '''
#         super().__init__()
#         # TODO: Implement __init__

#         # TODO: Initialize the sublayers      
#         self.self_attn = NotImplementedError # Self-attention layer
#         self.ffn = NotImplementedError # Feed-forward network
#         raise NotImplementedError # Remove once implemented

#     def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         '''
#         Forward pass for the EncoderLayer.
#         Args:
#             x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
#             key_padding_mask (torch.Tensor): The padding mask for the input. shape: (batch_size, seq_len)

#         Returns:
#             x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
#             mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
#         '''
#         # TODO: Implement forward: Follow the figure in the writeup

#         # What will be different from decoder self-attention layer?
#         x, mha_attn_weights = NotImplementedError, NotImplementedError
        
#         # TODO: Return the output tensor and attention weights
#         raise NotImplementedError # Remove once implemented

import torch
from torch import nn
from typing import Optional, Tuple

class SelfAttentionEncoderLayer(nn.Module):
    """
    Pre-LN Encoder Layer with self-attention and feed-forward sublayers.
    Used in the encoder part of transformer architectures.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the SelfAttentionEncoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        """
        super().__init__()
        
        # --- LayerNorms for Pre-LN ---
        self.norm1 = nn.LayerNorm(d_model)  # before self-attention
        self.norm2 = nn.LayerNorm(d_model)  # before feed-forward

        # --- Multi-Head Self-Attention (no causal masking) ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # --- Feed-Forward Network ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the EncoderLayer.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            key_padding_mask: Padding mask for input (batch_size, seq_len)

        Returns:
            x: Output tensor (batch_size, seq_len, d_model)
            mha_attn_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        # --- Self-Attention Sublayer ---
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask
        )
        x = x + attn_output  # Residual connection

        # --- Feed-Forward Sublayer ---
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn  # Residual connection

        return x, attn_weights
