"""S4 Decoder Model for Speech Classification.

Based on the paper:
"Structured State Space Decoder for Speech Recognition and Synthesis"
Authors: Koichi Miyazaki, Masato Murata, Tomoki Koriyama

Key components:
- S4-based decoder blocks (replacing masked multi-head self-attention)
- Feed-forward networks with GELU activation
- Linear & GLU gating mechanism
- LayerNorm and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.sequence.modules.s4block import S4Block
from src.models.nn import Normalization, DropoutNd


class ConformerConvModule(nn.Module):
    """Conformer-style convolution module with gating."""
    
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Pointwise conv expansion
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.norm_conv = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        # Pointwise conv projection
        self.pw2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, L, D)
        x = self.norm(x)
        # GLU gating
        x = self.pw1(x)  # (B, L, 2D)
        x, gate = x.chunk(2, dim=-1)
        x = x * gate
        # Depthwise conv
        x = rearrange(x, 'b l d -> b d l')
        x = self.dw_conv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.norm_conv(x)
        x = self.activation(x)
        # Projection
        x = self.pw2(x)
        x = self.dropout(x)
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with GLU activation (Conformer-style)."""
    
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class S4DecoderBlock(nn.Module):
    """S4-based decoder block from Miyazaki et al. 2022.
    
    Paper: "Structured State Space Decoder for Speech Recognition and Synthesis"
    Authors: Koichi Miyazaki, Masato Murata, Tomoki Koriyama
    
    Architecture from Figure 1(b) - Proposed S4 decoder:
    - Feed-Forward Block (with residual + positional encoding)
    - Multi-Head Attention Block (with residual + positional encoding)  
    - S4 Block (replacing masked self-attention):
      * S4 layer
      * Linear & GLU
      * GELU & Dropout
      * S4 layer (output)
      * LayerNorm
      * Residual connection
    """
    
    def __init__(
        self,
        d_model,
        d_state=64,
        d_ffn=2048,
        kernel_size=31,
        dropout=0.1,
        layer_dropout=0.0,
    ):
        super().__init__()
        
        # Feed-Forward Block (top)
        self.ffn = FeedForwardModule(d_model, d_ffn, dropout)
        
        # S4 Block (replaces masked multi-head self-attention)
        self.norm_s4 = nn.LayerNorm(d_model)
        
        # S4 layer
        self.s4 = S4Block(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            transposed=False,
            lr=None,
        )
        
        # Linear & GLU
        self.linear = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
        # GELU & Dropout
        self.gelu = nn.GELU()
        self.dropout_gelu = nn.Dropout(dropout)
        
        # LayerNorm (final)
        self.norm_out = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, L, D)
        
        # 1. Feed-Forward Block with residual
        residual = x
        x = residual + self.ffn(x)
        
        # 2. S4 Block (replaces masked multi-head self-attention)
        residual = x
        
        # Pre-LayerNorm
        x = self.norm_s4(x)
        
        # S4 layer
        x, _ = self.s4(x)
        
        # Linear & GLU
        x_linear = self.linear(x)
        x_gate = torch.sigmoid(self.gate(x))
        x = x_linear * x_gate
        
        # GELU & Dropout
        x = self.gelu(x)
        x = self.dropout_gelu(x)
        
        # Final LayerNorm
        x = self.norm_out(x)
        
        # Residual connection
        x = residual + self.dropout(x)
        
        return x


class ConformerS4Encoder(nn.Module):
    """S4 Decoder Stack from Miyazaki et al. 2022.
    
    Stacks multiple S4DecoderBlocks for speech tasks.
    """
    
    def __init__(
        self,
        d_model=256,
        n_layers=6,
        d_state=64,
        d_ffn=1024,
        kernel_size=31,
        dropout=0.1,
        layer_dropout=0.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            S4DecoderBlock(
                d_model=d_model,
                d_state=d_state,
                d_ffn=d_ffn,
                kernel_size=kernel_size,
                dropout=dropout,
                layer_dropout=layer_dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class ConformerS4Model(nn.Module):
    """S4 Decoder Model from Miyazaki et al. 2022 for sequence classification.
    
    Paper: "Structured State Space Decoder for Speech Recognition and Synthesis"
    Adapted for FSC intent classification task.
    """
    
    def __init__(
        self,
        d_input=1,
        d_output=31,  # FSC has 31 action classes
        d_model=256,
        n_layers=6,
        d_state=64,
        d_ffn=1024,
        kernel_size=31,
        dropout=0.1,
        layer_dropout=0.0,
        prenorm=True,
        **kwargs,  # Ignore extra arguments from config
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_output = d_output
        self.prenorm = prenorm
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # S4 Decoder Stack
        self.encoder = ConformerS4Encoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_ffn=d_ffn,
            kernel_size=kernel_size,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )
        
        # Pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_output)
        
    def forward(self, x, state=None, **kwargs):
        # x: (B, H, W, C) from fsc_image dataset - melspectrogram format
        # Ignore extra kwargs like 'rate' from dataloader
        
        # Flatten to sequence: (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b (h w) c')  # (B, L, D_in) where L=H*W
        
        # Project to model dimension
        x = self.input_proj(x)  # (B, L, d_model)
        
        # S4 Decoder
        x = self.encoder(x)  # (B, L, d_model)
        
        # Pool over sequence dimension
        x = rearrange(x, 'b l d -> b d l')
        x = self.pool(x)  # (B, d_model, 1)
        x = rearrange(x, 'b d 1 -> b d')
        
        # Output projection
        x = self.dropout(x)
        x = self.output_proj(x)
        
        return x, None  # Return None for state to match S4 interface
    
    def default_state(self, *args, **kwargs):
        """Compatibility with S4 interface."""
        return None
    
    def step(self, x, state=None):
        """Single step forward for autoregressive generation."""
        return self.forward(x.unsqueeze(1), state)
