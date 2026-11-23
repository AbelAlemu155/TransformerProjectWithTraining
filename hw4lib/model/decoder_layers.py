import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
TODO: Implement these Modules.

This file contains two key decoder layer implementations used in transformer architectures:

1. SelfAttentionDecoderLayer: Used in decoder-only transformers (like GPT)
   - Contains masked self-attention and feed-forward sublayers
   - Used for tasks like language modeling where only previous tokens can be attended to
   
2. CrossAttentionDecoderLayer: Used in encoder-decoder transformers (like BART)
   - Contains masked self-attention, cross-attention, and feed-forward sublayers
   - Used for tasks like translation where decoder needs to attend to encoder outputs

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps for Each Layer:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - CrossAttentionLayer for cross-attention (in CrossAttentionDecoderLayer only)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
import torch
from torch import nn
from typing import Optional, Tuple

import torch
from torch import nn
from typing import Optional, Tuple

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      

class SelfAttentionDecoderLayer(nn.Module):
    """
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in decoder-only transformers (e.g., GPT).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Layer normalization before each sublayer (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Masked Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
    self, 
    x: torch.Tensor, 
    key_padding_mask: Optional[torch.Tensor] = None, 
    attn_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Masked Self-Attention Sublayer
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        x = x + attn_output  # Keep residual here

        # Feed-Forward Sublayer
        x_norm = self.norm2(x)
        x = self.ffn(x_norm)  # **Output exactly matches FFN output**

        return x, attn_weights


class CrossAttentionDecoderLayer(nn.Module):
    """
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # LayerNorms for Pre-LN
        self.norm1 = nn.LayerNorm(d_model)  # before self-attention
        self.norm2 = nn.LayerNorm(d_model)  # before cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # before FFN

        # Masked Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Multi-Head Cross-Attention (encoder-decoder attention)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        dec_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # --- Masked Self-Attention Sublayer ---
        x_norm = self.norm1(x)
        self_attn_output, self_attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask
        )
        x = x + self_attn_output  # Residual connection

        # --- Cross-Attention Sublayer ---
        x_norm = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.cross_attn(
            query=x_norm,
            key=enc_output,
            value=enc_output,
            key_padding_mask=enc_key_padding_mask
        )
        x = x + cross_attn_output  # Residual connection

        # --- Feed-Forward Sublayer ---
        x_norm = self.norm3(x)
        x = self.ffn(x_norm)  # Output exactly matches FFN output

        return x, self_attn_weights, cross_attn_weights
