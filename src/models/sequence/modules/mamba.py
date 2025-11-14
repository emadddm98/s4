import torch
import torch.nn as nn
from src.models.sequence.base import SequenceModule
from mamba_ssm.modules.mamba_simple import Mamba #Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, use_residual=True, **kwargs):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, **kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x):
        # x: [batch, length, d_model]
        residual = x
        x = self.mamba(x)
        x = self.norm(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x