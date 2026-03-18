"""CI Decomposed Transformer with two-scale decomposition (Exp 31).

Replaces the single trend/residual split with a two-scale decomposition:
  - fine_trend:   avg-pool with small kernel (k_fine, default 15)
  - coarse_trend: avg-pool with large kernel (k_coarse, default 96)
  - mid:          fine_trend - coarse_trend
  - residual:     input - fine_trend

For ETTm1 (15-min data, L=1440), k_coarse=96 is exactly one day.
The dominant pattern in electricity data is the daily cycle — giving the model
an explicit daily-scale component means the transformer can specialize each
head rather than trying to learn all scales from a single decomposition.

Three components are processed by the shared transformer with three separate
output heads, then summed. Parameter overhead vs baseline: ~2 * (n_patches +
d_model) extra weights per head — ~6K for the ETTm1 multi config.

No change to the transformer itself. Drop-in replacement for
ci_decomp_transformer.py.
"""

import torch
import torch.nn as nn
from typing import Optional


class CITwoScaleTransformer(nn.Module):
    """CI Decomposed Transformer with two-scale trend decomposition.

    Three components: coarse trend, mid-band, fine residual.
    Each processed by shared transformer with its own output head.
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

        # Shared patch embedding and positional embedding
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Shared transformer encoder
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
        self.head_norm = nn.LayerNorm(d_model)

        # Three separate output heads: coarse trend, mid-band, fine residual
        self.coarse_temporal = nn.Linear(self.n_patches, forecast_length)
        self.coarse_channel  = nn.Linear(d_model, 1)

        self.mid_temporal    = nn.Linear(self.n_patches, forecast_length)
        self.mid_channel     = nn.Linear(d_model, 1)

        self.fine_temporal   = nn.Linear(self.n_patches, forecast_length)
        self.fine_channel    = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _avg_pool(self, x: torch.Tensor, kernel: int) -> torch.Tensor:
        """Apply avg pool with correct padding for both odd and even kernels."""
        pad_left  = (kernel - 1) // 2
        pad_right = kernel // 2
        return nn.functional.avg_pool1d(
            nn.functional.pad(x, (pad_left, pad_right), mode='replicate'),
            kernel_size=kernel, stride=1,
        )

    def _encode_and_project(
        self,
        x_1d: torch.Tensor,
        temporal_proj: nn.Linear,
        channel_proj: nn.Linear,
        B: int,
        D: int,
    ) -> torch.Tensor:
        """Patch, embed, transform, project one component. [B*D, H] -> [B, T, D]."""
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)  # [B*D, N, P]
        x = self.patch_embed(x) + self.pos_embed                 # [B*D, N, d]
        x = self.transformer(x)
        x = self.head_norm(x)
        x = temporal_proj(x.transpose(1, 2))   # [B*D, d, T]
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
        x = lookback.transpose(1, 2)  # [B, D, H]

        # Two-scale decomposition
        fine_trend   = self._avg_pool(x, self.trend_kernel_fine)    # [B, D, H]
        coarse_trend = self._avg_pool(x, self.trend_kernel_coarse)  # [B, D, H]
        mid          = fine_trend - coarse_trend                     # [B, D, H]
        fine_resid   = x - fine_trend                                # [B, D, H]

        # Flatten to CI: [B, D, H] -> [B*D, H]
        coarse_flat = coarse_trend.reshape(B * D, H)
        mid_flat    = mid.reshape(B * D, H)
        resid_flat  = fine_resid.reshape(B * D, H)

        # Each component through shared transformer, separate heads
        coarse_fc = self._encode_and_project(
            coarse_flat, self.coarse_temporal, self.coarse_channel, B, D)
        mid_fc    = self._encode_and_project(
            mid_flat, self.mid_temporal, self.mid_channel, B, D)
        resid_fc  = self._encode_and_project(
            resid_flat, self.fine_temporal, self.fine_channel, B, D)

        return coarse_fc + mid_fc + resid_fc


def create_model(config: dict) -> CITwoScaleTransformer:
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)

    patch_size = min(16, lookback_length // 8)
    patch_size = max(4, patch_size)

    # k_coarse=96 = 1 day for ETTm1 (15-min data, 96 * 15min = 24h)
    # k_coarse=24 for ETTh1 (hourly data, 24h = 1 day)
    # Falls back to k_fine * 4 for other lookback lengths
    if lookback_length >= 1440:
        trend_kernel_coarse = 96
    elif lookback_length >= 336:
        trend_kernel_coarse = 24
    else:
        trend_kernel_coarse = 15 * 4

    return CITwoScaleTransformer(
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
        trend_kernel_fine=15,
        trend_kernel_coarse=trend_kernel_coarse,
    )