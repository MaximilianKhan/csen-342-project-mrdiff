"""Denoising network for mr-Diff.

Implements the encoder-decoder architecture for noise prediction.
"""

from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv1d -> BatchNorm -> LeakyReLU -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        stride: int = 1,
    ):
        """Initialize the conv block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
            stride: Convolution stride.
        """
        super().__init__()

        padding = kernel_size // 2  # Same padding

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C_out, T].
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """Encoder network for denoising.

    Takes noisy input and diffusion step embedding to produce latent representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        step_embed_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize the encoder.

        Args:
            input_dim: Number of input features D.
            hidden_dim: Hidden dimension.
            step_embed_dim: Dimension of diffusion step embedding.
            num_layers: Number of conv layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
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
    ) -> torch.Tensor:
        """Encode noisy input.

        Args:
            y_noisy: Noisy data Y^k [B, T, D].
            step_embed: Diffusion step embedding p^k [B, step_embed_dim].

        Returns:
            Latent representation z^k [B, T, hidden_dim].
        """
        batch_size, seq_len, _ = y_noisy.shape

        # Expand step embedding to sequence length: [B, step_embed_dim] -> [B, T, step_embed_dim]
        step_embed_expanded = step_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate: [B, T, D + step_embed_dim]
        x = torch.cat([y_noisy, step_embed_expanded], dim=-1)

        # Input projection: [B, T, D + step_embed_dim] -> [B, T, hidden_dim]
        x = self.input_proj(x)

        # Convert to conv format: [B, T, hidden_dim] -> [B, hidden_dim, T]
        x = x.transpose(1, 2)

        # Apply conv blocks
        for layer in self.layers:
            x = layer(x)

        # Convert back: [B, hidden_dim, T] -> [B, T, hidden_dim]
        x = x.transpose(1, 2)

        # Output projection
        z = self.output_proj(x)

        return z


class Decoder(nn.Module):
    """Decoder network for denoising.

    Takes latent representation and conditioning to predict clean data.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize the decoder.

        Args:
            output_dim: Number of output features D.
            hidden_dim: Hidden dimension.
            cond_dim: Dimension of conditioning signal.
            num_layers: Number of conv layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Conditioning fusion: hidden_dim + cond_dim -> hidden_dim
        self.cond_fusion = nn.Linear(hidden_dim + cond_dim, hidden_dim)

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

        # Output projection: hidden_dim -> D
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent representation.

        Args:
            z: Latent representation z^k [B, T, hidden_dim].
            conditioning: Conditioning signal c_s [B, T, cond_dim].

        Returns:
            Predicted clean data Y^θ_s [B, T, D].
        """
        # Fuse with conditioning: [B, T, hidden_dim + cond_dim]
        x = torch.cat([z, conditioning], dim=-1)
        x = self.cond_fusion(x)

        # Convert to conv format: [B, T, hidden_dim] -> [B, hidden_dim, T]
        x = x.transpose(1, 2)

        # Apply conv blocks
        for layer in self.layers:
            x = layer(x)

        # Convert back: [B, hidden_dim, T] -> [B, T, hidden_dim]
        x = x.transpose(1, 2)

        # Output projection: [B, T, hidden_dim] -> [B, T, D]
        y_pred = self.output_proj(x)

        return y_pred


class DenoisingNetwork(nn.Module):
    """Full denoising network combining encoder and decoder.

    Predicts clean data from noisy input given conditioning.
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
        """Initialize the denoising network.

        Args:
            input_dim: Number of input/output features D.
            hidden_dim: Hidden dimension.
            step_embed_dim: Dimension of diffusion step embedding.
            cond_dim: Dimension of conditioning signal.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
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
        # Encode
        z = self.encoder(y_noisy, step_embed)

        # Decode with conditioning
        y_pred = self.decoder(z, conditioning)

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
        """Initialize denoising networks for all stages.

        Args:
            num_stages: Number of resolution stages S.
            input_dim: Number of input features.
            hidden_dim: Hidden dimension.
            step_embed_dim: Step embedding dimension.
            cond_dim: Conditioning dimension.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
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
        """Apply denoising for a specific stage.

        Args:
            stage: Stage index (0 to S-1).
            y_noisy: Noisy data.
            step_embed: Step embedding.
            conditioning: Conditioning signal.

        Returns:
            Predicted clean data for the stage.
        """
        return self.networks[stage](y_noisy, step_embed, conditioning)
