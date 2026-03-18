"""Channel-Independent Decomposed Transformer (Exp 17 + overlapping patches + split transformer).

Adds DLinear's trend/residual decomposition before CI patching.
Outputs are summed. This combines DLinear's inductive bias with transformer attention.

Overlapping patches: patch_stride < patch_size creates 50% overlap between
adjacent patches, doubling the token count for short lookbacks (ETTh1: 42→83).
Implemented via torch.Tensor.unfold — no extra dependencies.

Split transformer (split_transformer=True): trend and residual get separate
transformer encoders instead of sharing one. Trend signals are smooth and
low-frequency; residual signals are noisy and high-frequency. Forcing both
through the same weights constrains the transformer to find a compromise
representation. Separate encoders let each specialize. Adds ~50% more params
to the transformer portion, but CI keeps total params reasonable.
"""

import torch
import torch.nn as nn
from typing import Optional


class CIDecompTransformer(nn.Module):
    """Channel-Independent Decomposed Transformer.

    Architecture:
        lookback → trend/residual decomposition
        → CI-patch each (trend, residual) × each channel independently
        → transformer (shared or separate depending on split_transformer)
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
        split_transformer: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.trend_kernel = trend_kernel
        self.split_transformer = split_transformer

        # patch_stride=None → no overlap (original behaviour, stride == patch_size)
        # patch_stride < patch_size → overlapping patches
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
        self.d_model = d_model

        # Shared patch embedding (same for trend and residual)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional embedding (shared)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        def _make_transformer():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_layers,
            )

        if split_transformer:
            # Separate encoders: trend gets its own, residual gets its own.
            # Each specialises on its signal type without sharing capacity.
            self.trend_transformer = _make_transformer()
            self.resid_transformer = _make_transformer()
            self.trend_norm = nn.LayerNorm(d_model)
            self.resid_norm = nn.LayerNorm(d_model)
        else:
            # Original: one shared transformer for both components
            self.transformer = _make_transformer()
            self.head_norm = nn.LayerNorm(d_model)

        # Separate output heads for trend and residual (unchanged either way)
        self.trend_temporal = nn.Linear(self.n_patches, forecast_length)
        self.trend_channel = nn.Linear(d_model, 1)

        self.resid_temporal = nn.Linear(self.n_patches, forecast_length)
        self.resid_channel = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _patch_and_embed(self, x_1d: torch.Tensor) -> torch.Tensor:
        """Patch, embed, and add positional encoding.

        Args:
            x_1d: [B*D, H]
        Returns:
            [B*D, n_patches, d_model]
        """
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)  # [B*D, n_patches, patch_size]
        return self.patch_embed(x) + self.pos_embed                # [B*D, n_patches, d_model]

    def _project(self, x: torch.Tensor, temporal_proj, channel_proj, B: int, D: int) -> torch.Tensor:
        """Apply CI output head and reshape.

        Args:
            x: [B*D, n_patches, d_model] — after transformer + norm
        Returns:
            [B, T, D]
        """
        x = temporal_proj(x.transpose(1, 2))   # [B*D, d_model, T]
        x = channel_proj(x.transpose(1, 2))    # [B*D, T, 1]
        return x.squeeze(-1).reshape(B, D, self.forecast_length).transpose(1, 2)

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
        resid = x - trend                                           # [B, D, H]

        # Flatten to channel-independent: [B, D, H] → [B*D, H]
        trend_flat = trend.reshape(B * D, H)
        resid_flat = resid.reshape(B * D, H)

        # Patch and embed (shared embedding either way)
        trend_x = self._patch_and_embed(trend_flat)  # [B*D, N, d]
        resid_x = self._patch_and_embed(resid_flat)  # [B*D, N, d]

        if self.split_transformer:
            trend_x = self.trend_norm(self.trend_transformer(trend_x))
            resid_x = self.resid_norm(self.resid_transformer(resid_x))
        else:
            trend_x = self.head_norm(self.transformer(trend_x))
            resid_x = self.head_norm(self.transformer(resid_x))

        trend_forecast = self._project(trend_x, self.trend_temporal, self.trend_channel, B, D)
        resid_forecast = self._project(resid_x, self.resid_temporal, self.resid_channel, B, D)

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
        patch_stride=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
        trend_kernel=25,
        split_transformer=False,
    )
