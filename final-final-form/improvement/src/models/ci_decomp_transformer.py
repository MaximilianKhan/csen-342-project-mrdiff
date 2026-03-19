"""Channel-Independent Decomposed Transformer.

Trend/residual decomposition (DLinear-style) followed by CI patching through
a shared TransformerEncoder with separate output heads per component.
"""

import torch
import torch.nn as nn


class CIDecompTransformer(nn.Module):
    """lookback -> trend/residual decomp -> CI patch -> shared transformer
    -> dual heads -> sum forecasts."""

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
        trend_kernel: int = 25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.n_patches = lookback_length // patch_size
        self.d_model = d_model
        self.trend_kernel = trend_kernel

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

    def _encode_and_project(self, x_1d, temporal_proj, channel_proj, B, D):
        """Patch, embed, transform, project. [B*D, H] -> [B, T, D]"""
        x = x_1d[:, :self.n_patches * self.patch_size]
        x = x.reshape(-1, self.n_patches, self.patch_size)
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
        ks = self.trend_kernel
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)
        resid = x - trend

        trend_flat = trend.reshape(B * D, H)
        resid_flat = resid.reshape(B * D, H)

        return (
            self._encode_and_project(trend_flat, self.trend_temporal, self.trend_channel, B, D)
            + self._encode_and_project(resid_flat, self.resid_temporal, self.resid_channel, B, D)
        )


def create_model(config: dict) -> CIDecompTransformer:
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)
    patch_size = max(4, min(16, lookback_length // 8))

    return CIDecompTransformer(
        input_dim=input_dim, forecast_length=forecast_length,
        lookback_length=lookback_length, patch_size=patch_size,
    )
