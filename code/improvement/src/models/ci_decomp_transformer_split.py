"""CI Decomposed Transformer with optional split encoders.

When split_transformer=True, trend and residual get separate transformer
encoders instead of sharing one. Lets each specialize on its signal type.
"""

import torch
import torch.nn as nn
from typing import Optional


class CIDecompTransformer(nn.Module):
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

        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        def _make_transformer():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
                ),
                num_layers=num_layers,
            )

        if split_transformer:
            self.trend_transformer = _make_transformer()
            self.resid_transformer = _make_transformer()
            self.trend_norm = nn.LayerNorm(d_model)
            self.resid_norm = nn.LayerNorm(d_model)
        else:
            self.transformer = _make_transformer()
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

    def _patch_and_embed(self, x_1d):
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)
        return self.patch_embed(x) + self.pos_embed

    def _project(self, x, temporal_proj, channel_proj, B, D):
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

        trend_x = self._patch_and_embed(trend.reshape(B * D, H))
        resid_x = self._patch_and_embed(resid.reshape(B * D, H))

        if self.split_transformer:
            trend_x = self.trend_norm(self.trend_transformer(trend_x))
            resid_x = self.resid_norm(self.resid_transformer(resid_x))
        else:
            trend_x = self.head_norm(self.transformer(trend_x))
            resid_x = self.head_norm(self.transformer(resid_x))

        return (
            self._project(trend_x, self.trend_temporal, self.trend_channel, B, D)
            + self._project(resid_x, self.resid_temporal, self.resid_channel, B, D)
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
        split_transformer=False,
    )
