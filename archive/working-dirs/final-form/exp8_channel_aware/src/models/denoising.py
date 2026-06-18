"""Denoising network for mr-Diff.

Implements the encoder-decoder architecture with residual and skip connections.
Experiment 8: Channel-Aware Denoising — per-channel temporal encoders + cross-channel attention.
"""

from typing import List

import torch
import torch.nn as nn
import math


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

        padding = kernel_size // 2  # Same padding

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C_out, T].
        """
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_residual:
            x = x + residual

        return x


class ChannelIndependentEncoder(nn.Module):
    """Per-channel temporal encoder.

    Processes each variable independently through shared Conv1d layers,
    then applies cross-channel multi-head attention for channel mixing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 256,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_heads: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Per-channel input projection: 2 (y_noisy_d + x0_prev_d) + step_embed_dim -> hidden_dim
        self.channel_input_proj = nn.Linear(2 + step_embed_dim, hidden_dim)

        # Shared temporal Conv1d layers (applied per-channel)
        self.temporal_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.temporal_layers.append(
                ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            )

        # Cross-channel attention for channel mixing
        # Only useful when input_dim > 1
        self.cross_channel_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, y_noisy: torch.Tensor, step_embed: torch.Tensor) -> tuple:
        """Encode noisy input with channel-aware processing.

        Args:
            y_noisy: Noisy data [B, T, 2D] (y_noisy concat x0_prev).
            step_embed: Step embedding [B, step_embed_dim].

        Returns:
            Tuple of (z [B, T, hidden_dim], skips list).
        """
        B, T, D2 = y_noisy.shape
        D = D2 // 2  # Original input_dim (self-conditioning doubles it)

        # Split into per-channel pairs: [B, T, D, 2]
        y_split = y_noisy.view(B, T, D, 2)

        # Expand step embedding: [B, step_embed_dim] -> [B, T, D, step_embed_dim]
        step_exp = step_embed.unsqueeze(1).unsqueeze(2).expand(-1, T, D, -1)

        # Concatenate per-channel: [B, T, D, 2 + step_embed_dim]
        x = torch.cat([y_split, step_exp], dim=-1)

        # Project per-channel: [B, T, D, hidden_dim]
        x = self.channel_input_proj(x)

        # Process each channel through temporal convolutions
        # Reshape: [B*D, hidden_dim, T]
        x = x.permute(0, 2, 3, 1).reshape(B * D, self.hidden_dim, T)

        skips = []
        for layer in self.temporal_layers:
            x = layer(x)
            skips.append(x.view(B, D, self.hidden_dim, T).mean(dim=1))  # Aggregate skips across channels

        # Reshape back: [B, D, hidden_dim, T] -> [B, T, D, hidden_dim]
        x = x.view(B, D, self.hidden_dim, T).permute(0, 3, 1, 2)

        # Cross-channel attention: treat D as sequence dimension
        if D > 1:
            # Reshape for attention: [B*T, D, hidden_dim]
            x_attn = x.reshape(B * T, D, self.hidden_dim)
            attn_out, _ = self.cross_channel_attn(x_attn, x_attn, x_attn)
            x_attn = self.attn_norm(x_attn + attn_out)
            x = x_attn.view(B, T, D, self.hidden_dim)

        # Average across channels: [B, T, hidden_dim]
        z = x.mean(dim=2)
        z = self.output_proj(z)

        return z, skips


class Encoder(nn.Module):
    """Encoder with skip connections for U-Net style architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection: D + step_embed_dim -> hidden_dim
        self.input_proj = nn.Linear(input_dim + step_embed_dim, hidden_dim)

        # Stack of conv blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        y_noisy: torch.Tensor,
        step_embed: torch.Tensor,
    ) -> tuple:
        """Encode noisy input, returning latent + skip features."""
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


class Decoder(nn.Module):
    """Decoder with skip connections from encoder."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Conditioning fusion: hidden_dim + cond_dim -> hidden_dim
        self.cond_fusion = nn.Linear(hidden_dim + cond_dim, hidden_dim)

        # Stack of conv blocks — input is 2*hidden_dim due to skip connections
        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i in range(num_layers):
            self.skip_projs.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.layers.append(
                ConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # Output projection: hidden_dim -> D
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_skips: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode with skip connections from encoder."""
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

        x = x.transpose(1, 2)
        y_pred = self.output_proj(x)
        return y_pred


class DenoisingNetwork(nn.Module):
    """Channel-aware denoising network.

    Uses per-channel temporal encoders with cross-channel attention for
    multivariate data, falling back to standard processing for univariate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 256,
        cond_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim

        if input_dim > 1:
            # Channel-aware encoder for multivariate data
            self.encoder = ChannelIndependentEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                step_embed_dim=step_embed_dim,
                num_layers=num_encoder_layers,
                kernel_size=kernel_size,
                dropout=dropout,
                num_heads=min(2, hidden_dim // 32),
            )
        else:
            # Standard encoder for univariate data
            self.encoder = Encoder(
                input_dim=input_dim * 2,
                hidden_dim=hidden_dim,
                step_embed_dim=step_embed_dim,
                num_layers=num_encoder_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_layers=num_decoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        y_noisy: torch.Tensor,
        step_embed: torch.Tensor,
        conditioning: torch.Tensor,
        x0_prev: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict clean data from noisy input with optional self-conditioning."""
        if x0_prev is None:
            x0_prev = torch.zeros_like(y_noisy)
        encoder_input = torch.cat([y_noisy, x0_prev], dim=-1)  # [B, T, 2D]

        z, encoder_skips = self.encoder(encoder_input, step_embed)
        y_pred = self.decoder(z, conditioning, encoder_skips)

        return y_pred


class MultiStageDenoisingNetwork(nn.Module):
    """Collection of denoising networks for all stages."""

    def __init__(
        self,
        num_stages: int,
        input_dim: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 256,
        cond_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = num_stages

        self.networks = nn.ModuleList([
            DenoisingNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                step_embed_dim=step_embed_dim,
                cond_dim=cond_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])

    def forward(
        self,
        stage: int,
        y_noisy: torch.Tensor,
        step_embed: torch.Tensor,
        conditioning: torch.Tensor,
        x0_prev: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply denoising for a specific stage."""
        return self.networks[stage](y_noisy, step_embed, conditioning, x0_prev)
