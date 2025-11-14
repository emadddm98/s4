import torch.nn as nn
import torch.nn.functional as F
from src.models.sequence.base import SequenceModule
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba
from mamba_ssm.modules.mamba_simple import Mamba  # Fallback to simple Mamba if Triton fails

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, use_residual=True, **kwargs):
        super().__init__()
        self.mamba        = Mamba(d_model=d_model, **kwargs)
        self.norm         = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x):
        res = x
        x = self.mamba(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x + res if self.use_residual else x

class MambaSequenceModel(SequenceModule):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
        ffn_dim: int = None,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_model

        # 1) conv front-end
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        # 2) stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout, use_residual=use_residual, **kwargs)
            for _ in range(n_layers)
        ])
        # 3) final norm + small FFN
        self.final_norm = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.d_output = d_model

    def forward(self, x, **kwargs):
        # x: [B, L, d_model] already projected by upstream encoder
        # conv over time:
        x = x.permute(0, 2, 1)              # → [B, d_model, L]
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)              # → [B, L, d_model]

        # Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # final norm + FFN residual
        x = self.final_norm(x)
        x = x + self.ffn(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return None