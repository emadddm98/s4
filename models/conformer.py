import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, d_model, ff_dim, num_heads, conv_kernel_size, dropout):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.GLU(dim=1),
            nn.BatchNorm1d(d_model//2),
            nn.Conv1d(d_model//2, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Feedforward 1
        x = x + 0.5 * self.ffn1(x)
        # Multi-head attention
        attn_out, _ = self.mha(x, x, x)
        x = x + attn_out
        # Convolutional module
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        # Feedforward 2
        x = x + 0.5 * self.ffn2(x)
        # Final norm
        x = self.norm(x)
        return x

class Conformer(nn.Module):
    def __init__(self, d_model=144, n_layers=6, ff_dim=288, num_heads=4, conv_kernel_size=31, dropout=0.2, num_classes=31):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, ff_dim, num_heads, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])
        # Remove pooling and classifier to let decoder handle it
        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, **kwargs):
        # x: (B, L, D)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # Return sequence features, let decoder handle pooling/classification
        return x, None
