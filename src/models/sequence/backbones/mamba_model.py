import torch
import torch.nn as nn
from src.models.sequence.base import SequenceModule

from mamba_ssm.modules.mamba_simple import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, use_residual=True, **kwargs):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, **kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = self.mamba(x)      # uses simple Mamba if tri­ton one fails
        x = self.norm(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x

class MambaSequenceModel(SequenceModule):
    def __init__(self, d_model=128, n_layers=2, dropout=0.1, use_residual=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.layers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout, use_residual=use_residual, **kwargs)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        x = x
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return None