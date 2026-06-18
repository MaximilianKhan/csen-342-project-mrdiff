"""Conditioning network for mr-Diff.

Creates conditioning signals from historical data and coarser trends,
with optional future mixup during training.
"""

from typing import Optional

import torch
import torch.nn as nn


class ConditioningNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, forecast_length=168,
                 lookback_length=336, num_conv_layers=3, conv_kernel_size=7, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_length = forecast_length

        self.history_input = nn.Linear(input_dim, hidden_dim)
        conv_layers = []
        for _ in range(num_conv_layers):
            padding = conv_kernel_size // 2
            conv_layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel_size, padding=padding),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
        self.history_convs = nn.Sequential(*conv_layers)
        self.length_proj = nn.Linear(lookback_length, forecast_length)
        self._target_proj_dim = (input_dim, hidden_dim)

        self.coarse_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, history, coarse_trend=None, target=None, mixup_prob=0.5, training=True):
        """Compute conditioning signal. [B, H, D] -> [B, T, hidden_dim]"""
        batch_size = history.size(0)

        z_history = self.history_input(history)
        z_history = z_history.transpose(1, 2)
        z_history = self.history_convs(z_history) + z_history
        z_history = self.length_proj(z_history)
        z_history = z_history.transpose(1, 2)

        if training and target is not None and torch.rand(1).item() < mixup_prob:
            z_mix = self._apply_mixup(z_history, target)
        else:
            z_mix = z_history

        if coarse_trend is not None:
            coarse_projected = self.coarse_proj(coarse_trend)
            fused = torch.cat([z_mix, coarse_projected], dim=-1)
        else:
            zero_trend = torch.zeros(
                batch_size, self.forecast_length, self.hidden_dim,
                device=history.device, dtype=history.dtype
            )
            fused = torch.cat([z_mix, zero_trend], dim=-1)

        return self.fusion(fused)

    def _apply_mixup(self, z_history, target):
        """Future mixup with random (not learned) projection weights."""
        batch_size, seq_len, _ = z_history.shape
        in_dim, out_dim = self._target_proj_dim
        random_weight = torch.randn(out_dim, in_dim, device=target.device) * 0.02
        target_projected = nn.functional.linear(target, random_weight)
        mask = torch.rand(batch_size, seq_len, 1, device=z_history.device)
        return mask * z_history + (1 - mask) * target_projected


class MultiStageConditioningNetwork(nn.Module):
    """One conditioning network per resolution stage."""

    def __init__(self, num_stages, input_dim, hidden_dim=256, forecast_length=168,
                 lookback_length=336, num_conv_layers=3, conv_kernel_size=7, dropout=0.1):
        super().__init__()
        self.networks = nn.ModuleList([
            ConditioningNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim,
                forecast_length=forecast_length, lookback_length=lookback_length,
                num_conv_layers=num_conv_layers, conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])

    def forward(self, stage, history, coarse_trend=None, target=None,
                mixup_prob=0.5, training=True):
        return self.networks[stage](
            history=history, coarse_trend=coarse_trend, target=target,
            mixup_prob=mixup_prob, training=training,
        )
