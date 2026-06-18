"""Denoising network for mr-Diff.

Encoder-decoder with residual convolutions and skip connections.
"""

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> LeakyReLU -> Dropout, with residual skip."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=stride)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        residual = x
        x = self.dropout(self.activation(self.norm(self.conv(x))))
        if self.use_residual:
            x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, step_embed_dim=256,
                 num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + step_embed_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, y_noisy, step_embed):
        """[B, T, D] + [B, embed_dim] -> ([B, T, hidden], list of [B, hidden, T])"""
        seq_len = y_noisy.size(1)
        step_embed_expanded = step_embed.unsqueeze(1).expand(-1, seq_len, -1)
        x = self.input_proj(torch.cat([y_noisy, step_embed_expanded], dim=-1))
        x = x.transpose(1, 2)

        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)

        x = x.transpose(1, 2)
        return self.output_proj(x), skips


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=256, cond_dim=256,
                 num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.cond_fusion = nn.Linear(hidden_dim + cond_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_projs.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.layers.append(ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout))
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, conditioning, encoder_skips=None):
        x = self.cond_fusion(torch.cat([z, conditioning], dim=-1))
        x = x.transpose(1, 2)

        reversed_skips = list(reversed(encoder_skips)) if encoder_skips else [None] * len(self.layers)

        for layer, skip_proj, skip in zip(self.layers, self.skip_projs, reversed_skips):
            if skip is not None:
                combined = torch.cat([x, skip], dim=1).transpose(1, 2)
                x = skip_proj(combined).transpose(1, 2)
            x = layer(x)

        return self.output_proj(x.transpose(1, 2))


class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, step_embed_dim=256, cond_dim=256,
                 num_encoder_layers=3, num_decoder_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, step_embed_dim,
                               num_encoder_layers, kernel_size, dropout)
        self.decoder = Decoder(input_dim, hidden_dim, cond_dim,
                               num_decoder_layers, kernel_size, dropout)

    def forward(self, y_noisy, step_embed, conditioning):
        z, skips = self.encoder(y_noisy, step_embed)
        return self.decoder(z, conditioning, skips)


class MultiStageDenoisingNetwork(nn.Module):
    """One denoising network per resolution stage."""

    def __init__(self, num_stages, input_dim, hidden_dim=256, step_embed_dim=256,
                 cond_dim=256, num_encoder_layers=3, num_decoder_layers=3,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.networks = nn.ModuleList([
            DenoisingNetwork(input_dim, hidden_dim, step_embed_dim, cond_dim,
                             num_encoder_layers, num_decoder_layers, kernel_size, dropout)
            for _ in range(num_stages)
        ])

    def forward(self, stage, y_noisy, step_embed, conditioning):
        return self.networks[stage](y_noisy, step_embed, conditioning)
