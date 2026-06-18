"""CI+Decomp Transformer with frequency-enhanced dual branch (Exp 25).

Adds a parallel FFT branch that captures exact periodicities.
Time branch (CI transformer) + Freq branch (linear on FFT coefficients).
"""

import torch
import torch.nn as nn
from .ci_decomp_transformer import CIDecompTransformer


class CIFreqTransformer(nn.Module):
    """CI+Decomp Transformer with parallel frequency branch."""

    def __init__(self, input_dim, forecast_length, lookback_length,
                 patch_size=16, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3, trend_kernel=25):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length

        # Time-domain branch (existing CI+Decomp transformer)
        self.time_branch = CIDecompTransformer(
            input_dim=input_dim, forecast_length=forecast_length,
            lookback_length=lookback_length, patch_size=patch_size,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            trend_kernel=trend_kernel)

        # Frequency-domain branch: lightweight linear on FFT coefficients
        # rfft on lookback_length → lookback_length//2 + 1 complex values
        # We use real+imag → 2 * (L//2+1) real features per channel
        n_freq = lookback_length // 2 + 1
        # CI: process each channel independently via shared freq projection
        self.freq_proj = nn.Sequential(
            nn.Linear(n_freq * 2, forecast_length),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forecast_length, forecast_length),
        )

        # Learned blend weight (initialized small so time branch dominates)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._init_freq_weights()

    def _init_freq_weights(self):
        for m in self.freq_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lookback):
        """lookback: [B, H, D] -> [B, T, D]"""
        B, H, D = lookback.shape

        # Time-domain forecast
        time_forecast = self.time_branch(lookback)  # [B, T, D]

        # Frequency-domain forecast (CI: each channel independent)
        # [B, H, D] → [B, D, H] → FFT → [B, D, F] complex
        x = lookback.transpose(1, 2)  # [B, D, H]
        freq = torch.fft.rfft(x, dim=2)  # [B, D, F] complex
        # Stack real and imag: [B, D, 2F]
        freq_features = torch.cat([freq.real, freq.imag], dim=2)
        # CI projection: [B*D, 2F] → [B*D, T]
        freq_flat = freq_features.reshape(B * D, -1)
        freq_out = self.freq_proj(freq_flat)  # [B*D, T]
        freq_forecast = freq_out.reshape(B, D, self.forecast_length).transpose(1, 2)  # [B, T, D]

        # Blend: time + α * freq
        return time_forecast + self.alpha * freq_forecast
