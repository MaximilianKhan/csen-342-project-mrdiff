"""Denoising network for mr-Diff.

Experiment 10: Direct x0-prediction with explicit trend/seasonality decomposition.
The denoiser outputs separate trend and seasonality components that sum to x0.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block: Conv1d -> GroupNorm -> LeakyReLU -> Dropout + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class Encoder(nn.Module):
    """Encoder with skip connections."""

    def __init__(self, input_dim, hidden_dim=256, step_embed_dim=256,
                 num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim + step_embed_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, y_noisy, step_embed):
        batch_size, seq_len, _ = y_noisy.shape
        step_embed_expanded = step_embed.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([y_noisy, step_embed_expanded], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        x = x.transpose(1, 2)
        z = self.output_proj(x)
        return z, skips


class TrendHead(nn.Module):
    """Trend extraction head using learned moving average.

    Applies a learned 1D convolution with constrained positive weights
    to extract smooth trend from the decoder's hidden representation.
    """

    def __init__(self, hidden_dim: int, output_dim: int, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size

        # Project hidden to output dim first
        self.proj = nn.Linear(hidden_dim, output_dim)

        # Learned moving average kernel (initialized to uniform)
        self.ma_weight = nn.Parameter(torch.ones(1, 1, kernel_size) / kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract trend from hidden representation.

        Args:
            x: Hidden features [B, T, hidden_dim].

        Returns:
            Trend component [B, T, D].
        """
        # Project to output dim: [B, T, D]
        x = self.proj(x)

        # Apply learned moving average per-channel
        B, T, D = x.shape
        x = x.transpose(1, 2)  # [B, D, T]
        x = x.reshape(B * D, 1, T)  # [B*D, 1, T]

        # Softmax to ensure weights are positive and sum to 1
        weights = F.softmax(self.ma_weight, dim=-1)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad), mode='replicate')
        trend = F.conv1d(x, weights)

        trend = trend.reshape(B, D, T).transpose(1, 2)  # [B, T, D]
        return trend


class SeasonalityHead(nn.Module):
    """Seasonality extraction head using top-K Fourier basis.

    Projects the decoder output to frequency domain, keeps top-K
    components, and reconstructs the seasonal pattern.
    """

    def __init__(self, hidden_dim: int, output_dim: int, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract seasonality using top-K Fourier components.

        Args:
            x: Hidden features [B, T, hidden_dim].

        Returns:
            Seasonality component [B, T, D].
        """
        # Project to output dim: [B, T, D]
        x = self.proj(x)

        # FFT along time dimension
        freq = torch.fft.rfft(x, dim=1)

        # Keep only top-K frequency components by magnitude
        magnitudes = freq.abs()
        # Find top-K frequencies per channel
        _, topk_indices = magnitudes.topk(min(self.top_k, magnitudes.size(1)), dim=1)

        # Create mask
        mask = torch.zeros_like(magnitudes)
        mask.scatter_(1, topk_indices, 1.0)

        # Apply mask and reconstruct
        filtered = freq * mask
        seasonality = torch.fft.irfft(filtered, n=x.size(1), dim=1)

        return seasonality


class DecompositionDecoder(nn.Module):
    """Decoder with explicit trend + seasonality decomposition heads."""

    def __init__(self, output_dim, hidden_dim=256, cond_dim=256,
                 num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Conditioning fusion
        self.cond_fusion = nn.Linear(hidden_dim + cond_dim, hidden_dim)

        # Conv layers with skip connections
        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_projs.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.layers.append(ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout))

        # Decomposition heads
        self.trend_head = TrendHead(hidden_dim, output_dim, kernel_size=25)
        self.seasonality_head = SeasonalityHead(hidden_dim, output_dim, top_k=5)

    def forward(self, z, conditioning, encoder_skips=None):
        """Decode with decomposition.

        Returns x0 = trend + seasonality.
        """
        x = torch.cat([z, conditioning], dim=-1)
        x = self.cond_fusion(x)
        x = x.transpose(1, 2)

        if encoder_skips is not None:
            reversed_skips = list(reversed(encoder_skips))
        else:
            reversed_skips = [None] * len(self.layers)

        for i, (layer, skip_proj) in enumerate(zip(self.layers, self.skip_projs)):
            if reversed_skips[i] is not None:
                combined = torch.cat([x, reversed_skips[i]], dim=1)
                combined = combined.transpose(1, 2)
                x = skip_proj(combined).transpose(1, 2)
            x = layer(x)

        x = x.transpose(1, 2)  # [B, T, hidden_dim]

        # Decomposed prediction
        trend = self.trend_head(x)
        seasonality = self.seasonality_head(x)

        # x0 = trend + seasonality
        x0 = trend + seasonality

        return x0


class DenoisingNetwork(nn.Module):
    """Denoising network with decomposition: directly predicts x0 = trend + seasonality."""

    def __init__(self, input_dim, hidden_dim=256, step_embed_dim=256,
                 cond_dim=256, num_encoder_layers=3, num_decoder_layers=3,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = Encoder(
            input_dim=input_dim * 2,
            hidden_dim=hidden_dim,
            step_embed_dim=step_embed_dim,
            num_layers=num_encoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.decoder = DecompositionDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_layers=num_decoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, y_noisy, step_embed, conditioning, x0_prev=None):
        """Directly predict x0 (not epsilon) with decomposition."""
        if x0_prev is None:
            x0_prev = torch.zeros_like(y_noisy)
        encoder_input = torch.cat([y_noisy, x0_prev], dim=-1)

        z, encoder_skips = self.encoder(encoder_input, step_embed)
        x0_pred = self.decoder(z, conditioning, encoder_skips)

        return x0_pred


class MultiStageDenoisingNetwork(nn.Module):
    """Collection of denoising networks for all stages."""

    def __init__(self, num_stages, input_dim, hidden_dim=256, step_embed_dim=256,
                 cond_dim=256, num_encoder_layers=3, num_decoder_layers=3,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.num_stages = num_stages
        self.networks = nn.ModuleList([
            DenoisingNetwork(input_dim, hidden_dim, step_embed_dim, cond_dim,
                           num_encoder_layers, num_decoder_layers, kernel_size, dropout)
            for _ in range(num_stages)
        ])

    def forward(self, stage, y_noisy, step_embed, conditioning, x0_prev=None):
        return self.networks[stage](y_noisy, step_embed, conditioning, x0_prev)
