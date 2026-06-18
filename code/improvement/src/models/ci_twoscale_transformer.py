"""CI Decomposed Transformer with two-scale decomposition.

Three frequency bands instead of two:
  - coarse_trend: avg-pool with large kernel (e.g. k=96 = 1 day for ETTm1)
  - mid_band:     fine_trend - coarse_trend
  - fine_residual: input - fine_trend

Each processed by the shared transformer with its own output head, then summed.
"""

import torch
import torch.nn as nn
from typing import Optional


class CITwoScaleTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        forecast_length: int,
        lookback_length: int,
        patch_size: int = 16,
        patch_stride: Optional[int] = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        trend_kernel_fine: int = 15,
        trend_kernel_coarse: int = 96,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.trend_kernel_fine = trend_kernel_fine
        self.trend_kernel_coarse = trend_kernel_coarse

        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_norm = nn.LayerNorm(d_model)

        # Three output heads: coarse, mid, fine
        self.coarse_temporal = nn.Linear(self.n_patches, forecast_length)
        self.coarse_channel = nn.Linear(d_model, 1)
        self.mid_temporal = nn.Linear(self.n_patches, forecast_length)
        self.mid_channel = nn.Linear(d_model, 1)
        self.fine_temporal = nn.Linear(self.n_patches, forecast_length)
        self.fine_channel = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _avg_pool(self, x, kernel):
        pad_left = (kernel - 1) // 2
        pad_right = kernel // 2
        return nn.functional.avg_pool1d(
            nn.functional.pad(x, (pad_left, pad_right), mode='replicate'),
            kernel_size=kernel, stride=1,
        )

    def _encode_and_project(self, x_1d, temporal_proj, channel_proj, B, D):
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)
        x = self.patch_embed(x) + self.pos_embed
        x = self.transformer(x)
        x = self.head_norm(x)
        x = temporal_proj(x.transpose(1, 2))
        x = channel_proj(x.transpose(1, 2))
        return x.squeeze(-1).reshape(B, D, self.forecast_length).transpose(1, 2)

    def forward(self, lookback):
        """[B, H, D] -> [B, T, D]"""
        B, H, D = lookback.shape
        x = lookback.transpose(1, 2)

        fine_trend = self._avg_pool(x, self.trend_kernel_fine)
        coarse_trend = self._avg_pool(x, self.trend_kernel_coarse)
        mid = fine_trend - coarse_trend
        fine_resid = x - fine_trend

        coarse_flat = coarse_trend.reshape(B * D, H)
        mid_flat = mid.reshape(B * D, H)
        resid_flat = fine_resid.reshape(B * D, H)

        return (
            self._encode_and_project(coarse_flat, self.coarse_temporal, self.coarse_channel, B, D)
            + self._encode_and_project(mid_flat, self.mid_temporal, self.mid_channel, B, D)
            + self._encode_and_project(resid_flat, self.fine_temporal, self.fine_channel, B, D)
        )


def create_model(config: dict) -> CITwoScaleTransformer:
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)
    patch_size = max(4, min(16, lookback_length // 8))

    # k_coarse aligned to daily cycle
    if lookback_length >= 1440:
        trend_kernel_coarse = 96   # 96 * 15min = 24h
    elif lookback_length >= 336:
        trend_kernel_coarse = 24   # 24h for hourly data
    else:
        trend_kernel_coarse = 60

    return CITwoScaleTransformer(
        input_dim=input_dim, forecast_length=forecast_length,
        lookback_length=lookback_length, patch_size=patch_size,
        trend_kernel_fine=15, trend_kernel_coarse=trend_kernel_coarse,
    )
