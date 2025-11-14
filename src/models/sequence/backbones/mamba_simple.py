import torch
import torch.nn as nn
from src.models.sequence.base import SequenceModule

from mamba_ssm.modules.mamba_simple import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, **kwargs)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, length, d_model]
        x = self.mamba(x)
        x = self.norm(x)
        return x

class MambaSequenceModel(SequenceModule):
    def __init__(self, d_model=128, n_layers=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.layers = nn.ModuleList([
            MambaBlock(d_model, **kwargs) for _ in range(n_layers)
        ])

    def forward(self, x, **kwargs):
        # x: [batch, length, d_model] (float)
        for layer in self.layers:
            x = layer(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return None