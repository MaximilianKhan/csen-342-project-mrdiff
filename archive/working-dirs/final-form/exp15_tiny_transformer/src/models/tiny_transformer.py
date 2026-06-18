"""Tiny PatchTST-style transformer for direct time series forecasting.

No diffusion. No conditioning bottleneck. Just patches, self-attention, and a linear head.
Deliberately right-sized to ~100-300K params to match the DLinear data regime.
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens."""

    def __init__(self, patch_size: int, input_dim: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, D] -> [B, N_patches, d_model]"""
        B, H, D = x.shape
        # Truncate to exact multiple of patch_size
        n_patches = H // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]  # [B, N*P, D]
        # Reshape into patches
        x = x.reshape(B, n_patches, self.patch_size * D)  # [B, N, P*D]
        return self.proj(x)  # [B, N, d_model]


class TinyTransformer(nn.Module):
    """Tiny PatchTST-style direct forecaster.

    Architecture:
        lookback → patch_embed → pos_embed → transformer_encoder → flatten → linear → forecast

    No diffusion, no multi-resolution decomposition, no conditioning network.
    Pure end-to-end transformer forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        forecast_length: int,
        lookback_length: int,
        patch_size: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size

        n_patches = lookback_length // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, input_dim, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output head: flatten all patch tokens → forecast
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(n_patches * d_model, forecast_length * input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, lookback: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lookback: [B, H, D]

        Returns:
            forecast: [B, T, D]
        """
        B = lookback.size(0)

        # Patch + position embedding
        x = self.patch_embed(lookback)  # [B, N, d_model]
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)  # [B, N, d_model]

        # Output projection
        x = self.head_norm(x)
        x = x.reshape(B, -1)  # [B, N * d_model]
        x = self.head(x)      # [B, T * D]
        x = x.reshape(B, self.forecast_length, self.input_dim)  # [B, T, D]

        return x


def create_model(config: dict) -> TinyTransformer:
    """Create TinyTransformer from configuration."""
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)

    # Adaptive patch size: ensure at least 8 patches
    patch_size = min(16, lookback_length // 8)
    # Round down to power of 2 for clean division
    patch_size = max(4, patch_size)

    model = TinyTransformer(
        input_dim=input_dim,
        forecast_length=forecast_length,
        lookback_length=lookback_length,
        patch_size=patch_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
    )

    return model
