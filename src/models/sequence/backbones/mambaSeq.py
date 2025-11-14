import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.modules.mamba_simple import Mamba as MambaSimple

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, use_residual=True, **kwargs):
        super().__init__()
        # only register the Triton‐accelerated block
        self.mamba2       = Mamba2Simple(d_model=d_model, **kwargs)
        self.norm         = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)
        self.use_residual = use_residual
        # we’ll lazy‐make a simple Mamba only if needed
        self._mamba_simple = None

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        residual = x

        # pad only the feature dim to multiple of 8
        D_p = ((D + 7) // 8) * 8
        if D_p != D:
            x = F.pad(x, (0, D_p - D))
            residual = F.pad(residual, (0, D_p - D))

        # enforce channels-last so Mamba2Simple sees the right strides
        x = x.contiguous(memory_format=torch.channels_last)
        residual = residual.contiguous(memory_format=torch.channels_last)

        # try the Triton kernel
        try:
            out = self.mamba2(x)
        except RuntimeError as e:
            if "requires strides" in str(e):
                # lazy‐instantiate the pure‐Python fallback
                if self._mamba_simple is None:
                    self._mamba_simple = MambaSimple(d_model=self.norm.normalized_shape[0])
                out = self._mamba_simple(x)
            else:
                raise

        # undo padding
        out = out[:, :, :D]

        # final norm / dropout / residual
        out = self.norm(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + residual

        return out