"""CI+Decomp Transformer with lightweight cross-channel mixing (Exp 21).

Adds a residual Linear(D→D) after the CI transformer output to capture
cross-channel dynamics that pure channel-independence misses.
"""

import torch
import torch.nn as nn
from .ci_decomp_transformer import CIDecompTransformer


class CIChannelMixTransformer(nn.Module):
    """CI+Decomp Transformer with cross-channel mixing residual."""

    def __init__(self, input_dim, forecast_length, lookback_length,
                 patch_size=16, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3, trend_kernel=25):
        super().__init__()
        self.ci_transformer = CIDecompTransformer(
            input_dim=input_dim, forecast_length=forecast_length,
            lookback_length=lookback_length, patch_size=patch_size,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            trend_kernel=trend_kernel)

        # Lightweight cross-channel mixing: Linear(D→D) with zero-init for safe residual
        self.channel_mix = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.zeros_(self.channel_mix.weight)  # Start as identity (no mixing)

    def forward(self, lookback):
        """lookback: [B, H, D] -> [B, T, D]"""
        forecast = self.ci_transformer(lookback)  # [B, T, D]
        # Residual cross-channel correction
        forecast = forecast + self.channel_mix(forecast)  # +49 params for D=7
        return forecast
