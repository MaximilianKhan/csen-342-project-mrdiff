"""Channel-Independent Decomposed Transformer (Exp 17 + overlapping patches).

Adds DLinear's trend/residual decomposition before CI patching.
Same shared transformer processes both trend and residual independently.
Outputs are summed. This combines DLinear's inductive bias with transformer attention.

Overlapping patches: patch_stride < patch_size creates 50% overlap between
adjacent patches, doubling the token count for short lookbacks (ETTh1: 42→83).
Implemented via torch.Tensor.unfold — no extra dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional


class CIDecompTransformer(nn.Module):
    """Channel-Independent Decomposed Transformer.

    Architecture:
        lookback → trend/residual decomposition
        → CI-patch each (trend, residual) × each channel independently
        → shared transformer
        → CI-head (temporal + channel projection)
        → sum trend + residual forecasts
    """

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
        trend_kernel: int = 25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.trend_kernel = trend_kernel

        # patch_stride=None → no overlap (original behaviour, stride == patch_size)
        # patch_stride < patch_size → overlapping patches
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
        self.d_model = d_model

        # Shared patch embedding (same for trend and residual)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional embedding (shared)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Shared transformer encoder (processes trend AND residual)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate heads for trend and residual (small — just N→T + d→1)
        self.head_norm = nn.LayerNorm(d_model)

        self.trend_temporal = nn.Linear(self.n_patches, forecast_length)
        self.trend_channel = nn.Linear(d_model, 1)

        self.resid_temporal = nn.Linear(self.n_patches, forecast_length)
        self.resid_channel = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _encode_and_project(self, x_1d: torch.Tensor, temporal_proj, channel_proj, B: int, D: int) -> torch.Tensor:
        """Shared encode + project pipeline for one component (trend or residual).

        Args:
            x_1d: [B*D, H] — flattened channel-independent input
            temporal_proj: Linear(N → T)
            channel_proj: Linear(d_model → 1)
            B: batch size
            D: number of channels

        Returns:
            [B, T, D]
        """
        # Overlapping patch extraction via unfold: [B*D, H] → [B*D, n_patches, patch_size]
        # unfold(dimension, size, step) slides a window of `size` with step `patch_stride`
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)  # [B*D, n_patches, patch_size]

        # Embed + position: [B*D, n_patches, d_model]
        x = self.patch_embed(x) + self.pos_embed

        # Transformer: [B*D, n_patches, d_model]
        x = self.transformer(x)
        x = self.head_norm(x)

        # Temporal projection: [B*D, d_model, n_patches] → [B*D, d_model, T]
        x = temporal_proj(x.transpose(1, 2))

        # Channel projection: [B*D, T, d_model] → [B*D, T, 1]
        x = channel_proj(x.transpose(1, 2))

        # Reshape: [B*D, T, 1] → [B, D, T] → [B, T, D]
        x = x.squeeze(-1).reshape(B, D, self.forecast_length).transpose(1, 2)
        return x

    def forward(self, lookback: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lookback: [B, H, D]
        Returns:
            forecast: [B, T, D]
        """
        B, H, D = lookback.shape

        # Trend/residual decomposition (DLinear-style)
        x = lookback.transpose(1, 2)
        ks = self.trend_kernel
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)  # [B, D, H]
        resid = x - trend  # [B, D, H]

        # Flatten to channel-independent: [B, D, H] → [B*D, H]
        trend_flat = trend.reshape(B * D, H)
        resid_flat = resid.reshape(B * D, H)

        # Process each through shared transformer with separate heads
        trend_forecast = self._encode_and_project(
            trend_flat, self.trend_temporal, self.trend_channel, B, D
        )
        resid_forecast = self._encode_and_project(
            resid_flat, self.resid_temporal, self.resid_channel, B, D
        )

        return trend_forecast + resid_forecast


def create_model(config: dict) -> CIDecompTransformer:
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)

    patch_size = min(16, lookback_length // 8)
    patch_size = max(4, patch_size)

    return CIDecompTransformer(
        input_dim=input_dim,
        forecast_length=forecast_length,
        lookback_length=lookback_length,
        patch_size=patch_size,
        patch_stride=None,  # set explicitly per-config in train_single/train_ensemble
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
        trend_kernel=25,
    )
