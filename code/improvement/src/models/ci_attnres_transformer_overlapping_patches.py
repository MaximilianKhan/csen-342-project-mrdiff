"""CI+Decomp Transformer with Attention Residuals and overlapping patches.

patch_stride < patch_size creates 50% overlap, doubling the token count
for short lookbacks (e.g. ETTh1: 42 -> 83 tokens).
"""

from typing import Optional
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class AttnResTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model)

    def _attnres_aggregate(self, layer_outputs):
        sources = torch.stack(layer_outputs, dim=0)
        keys = self.key_norm(sources)
        logits = torch.einsum('d, s b n d -> s b n', self.attn_query, keys)
        attn = torch.softmax(logits, dim=0)
        return (attn.unsqueeze(-1) * sources).sum(dim=0)

    def forward(self, layer_outputs):
        x = self._attnres_aggregate(layer_outputs)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class CIAttnResDecompTransformer(nn.Module):
    """CI+Decomp+AttnRes with optional overlapping patches via unfold."""

    def __init__(
        self,
        input_dim, forecast_length, lookback_length,
        patch_size=16, patch_stride: Optional[int] = None,
        d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.3, trend_kernel=25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.trend_kernel = trend_kernel

        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.layers = nn.ModuleList([
            AttnResTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
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
        for layer in self.layers:
            nn.init.zeros_(layer.attn_query)

    def _encode_and_project(self, x_1d, temporal_proj, channel_proj, B, D):
        x = x_1d.unfold(-1, self.patch_size, self.patch_stride)
        x = self.patch_embed(x) + self.pos_embed

        layer_outputs = [x]
        for layer in self.layers:
            layer_outputs.append(layer(layer_outputs))

        x = self.head_norm(layer_outputs[-1])
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

        trend_out = self._encode_and_project(
            trend_flat, self.trend_temporal, self.trend_channel, B, D)
        resid_out = self._encode_and_project(
            resid_flat, self.resid_temporal, self.resid_channel, B, D)

        return trend_out + resid_out
