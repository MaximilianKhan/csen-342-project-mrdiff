"""Denoising network for mr-Diff.

Implements the encoder-decoder architecture with residual and skip connections.
"""

from typing import List, Optional

import torch
import torch.nn as nn


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
        """Encode noisy input, returning latent + skip features.

        Args:
            y_noisy: Noisy data Y^k [B, T, D].
            step_embed: Diffusion step embedding p^k [B, step_embed_dim].

        Returns:
            Tuple of (z [B, T, hidden_dim], skips list of [B, hidden_dim, T]).
        """
        batch_size, seq_len, _ = y_noisy.shape

        # Expand step embedding to sequence length
        step_embed_expanded = step_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate: [B, T, D + step_embed_dim]
        x = torch.cat([y_noisy, step_embed_expanded], dim=-1)

        # Input projection: [B, T, hidden_dim]
        x = self.input_proj(x)

        # Convert to conv format: [B, hidden_dim, T]
        x = x.transpose(1, 2)

        # Apply conv blocks, saving intermediates for skip connections
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)  # Save after each layer

        # Convert back: [B, T, hidden_dim]
        x = x.transpose(1, 2)

        # Output projection
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
            # Project concatenated [x, skip] back to hidden_dim
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
        """Decode with skip connections from encoder.

        Args:
            z: Latent representation z^k [B, T, hidden_dim].
            conditioning: Conditioning signal c_s [B, T, cond_dim].
            encoder_skips: List of encoder features [B, hidden_dim, T] (reversed order).

        Returns:
            Predicted clean data Y^θ_s [B, T, D].
        """
        # Fuse with conditioning: [B, T, hidden_dim + cond_dim]
        x = torch.cat([z, conditioning], dim=-1)
        x = self.cond_fusion(x)

        # Convert to conv format: [B, hidden_dim, T]
        x = x.transpose(1, 2)

        # Apply conv blocks with skip connections
        # Reverse encoder skips so deepest encoder layer matches first decoder layer
        if encoder_skips is not None:
            reversed_skips = list(reversed(encoder_skips))
        else:
            reversed_skips = [None] * len(self.layers)

        for i, (layer, skip_proj) in enumerate(zip(self.layers, self.skip_projs)):
            if reversed_skips[i] is not None:
                # Concatenate skip and project: [B, 2*hidden_dim, T] -> [B, hidden_dim, T]
                combined = torch.cat([x, reversed_skips[i]], dim=1)  # Channel dim
                # Transpose to [B, T, 2*hidden_dim] for linear, then back
                combined = combined.transpose(1, 2)
                x = skip_proj(combined).transpose(1, 2)
            x = layer(x)

        # Convert back: [B, T, hidden_dim]
        x = x.transpose(1, 2)

        # Output projection: [B, T, D]
        y_pred = self.output_proj(x)

        return y_pred


class DenoisingNetwork(nn.Module):
    """Full denoising network with encoder-decoder skip connections."""

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

        self.encoder = Encoder(
            input_dim=input_dim,
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
    ) -> torch.Tensor:
        """Predict clean data from noisy input.

        Args:
            y_noisy: Noisy data Y^k [B, T, D].
            step_embed: Diffusion step embedding p^k [B, step_embed_dim].
            conditioning: Conditioning signal c_s [B, T, cond_dim].

        Returns:
            Predicted clean data Y^θ_s [B, T, D].
        """
        # Encode — get latent + skip features
        z, encoder_skips = self.encoder(y_noisy, step_embed)

        # Decode with skip connections
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

        # Create denoising network for each stage
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
    ) -> torch.Tensor:
        """Apply denoising for a specific stage."""
        return self.networks[stage](y_noisy, step_embed, conditioning)
