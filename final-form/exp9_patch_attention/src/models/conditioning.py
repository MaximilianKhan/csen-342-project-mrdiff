"""Conditioning network for mr-Diff.

Implements the conditioning mechanism with history encoding and future mixup.
Experiment 9: Patch + Attention History Encoder (PatchTST-style).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchAttentionEncoder(nn.Module):
    """PatchTST-style history encoder.

    Segments the lookback window into non-overlapping patches, projects
    each to hidden_dim, and applies self-attention over patches for
    global receptive field.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        lookback_length: int = 336,
        patch_size: int = 24,
        num_attention_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.lookback_length = lookback_length

        # Number of patches
        self.num_patches = lookback_length // patch_size
        # Handle remainder by adjusting effective length
        self.effective_length = self.num_patches * patch_size

        # Patch embedding: project each patch to hidden_dim
        self.patch_proj = nn.Linear(patch_size * input_dim, hidden_dim)

        # Learnable positional embedding for patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Transformer encoder layers for patch-level self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode history using patch + attention.

        Args:
            history: Historical data [B, H, D].

        Returns:
            Encoded history [B, num_patches, hidden_dim].
        """
        B, H, D = history.shape

        # Truncate to effective length (drop earliest timesteps if needed)
        if H > self.effective_length:
            history = history[:, H - self.effective_length:, :]

        # Reshape into patches: [B, num_patches, patch_size * D]
        patches = history.reshape(B, self.num_patches, self.patch_size * D)

        # Project patches: [B, num_patches, hidden_dim]
        x = self.patch_proj(patches)

        # Add positional embedding
        x = x + self.pos_embed

        # Self-attention over patches
        x = self.transformer(x)

        # Output projection
        x = self.output_proj(x)

        return x


class ConditioningNetwork(nn.Module):
    """Conditioning network with PatchTST-style history encoder.

    Replaces Conv1d history encoder with patch embedding + self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        forecast_length: int = 168,
        lookback_length: int = 336,
        num_conv_layers: int = 3,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length

        # Patch size: 24 works for both ETTh1 (24h daily) and ETTm1 (6h at 15min)
        patch_size = 24

        # PatchTST-style history encoder
        self.patch_encoder = PatchAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lookback_length=lookback_length,
            patch_size=patch_size,
            num_attention_layers=2,
            num_heads=min(4, hidden_dim // 16),
            dropout=dropout,
        )

        # Project from num_patches to forecast_length
        num_patches = lookback_length // patch_size
        self.length_proj = nn.Linear(num_patches, forecast_length)

        # Target projection dimension for future mixup
        self._target_proj_dim = (input_dim, hidden_dim)

        # Project coarse_trend from D -> hidden_dim for equal representation
        self.coarse_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

        # Conditioning fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        history: torch.Tensor,
        coarse_trend: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        mixup_prob: float = 0.5,
        training: bool = True,
    ) -> torch.Tensor:
        """Compute conditioning signal."""
        batch_size = history.size(0)

        # Encode history with patch + attention
        # [B, H, D] -> [B, num_patches, hidden_dim]
        z_history = self.patch_encoder(history)

        # Project to forecast length: [B, hidden_dim, num_patches] -> [B, hidden_dim, T]
        z_history = z_history.transpose(1, 2)  # [B, hidden_dim, num_patches]
        z_history = self.length_proj(z_history)  # [B, hidden_dim, T]
        z_history = z_history.transpose(1, 2)    # [B, T, hidden_dim]

        # Future mixup during training
        if training and target is not None and torch.rand(1).item() < mixup_prob:
            z_mix = self._apply_mixup(z_history, target)
        else:
            z_mix = z_history

        # Fuse with coarser trend if provided
        if coarse_trend is not None:
            coarse_projected = self.coarse_proj(coarse_trend)
            fused = torch.cat([z_mix, coarse_projected], dim=-1)
            conditioning = self.fusion(fused)
        else:
            zero_trend = torch.zeros(
                batch_size, self.forecast_length, self.hidden_dim,
                device=history.device, dtype=history.dtype
            )
            fused = torch.cat([z_mix, zero_trend], dim=-1)
            conditioning = self.fusion(fused)

        return conditioning

    def _apply_mixup(
        self,
        z_history: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Apply future mixup for training regularization."""
        batch_size, seq_len, _ = z_history.shape
        in_dim, out_dim = self._target_proj_dim

        random_weight = torch.randn(out_dim, in_dim, device=target.device) * 0.02
        target_projected = nn.functional.linear(target, random_weight)

        mask = torch.rand(batch_size, seq_len, 1, device=z_history.device)
        z_mix = mask * z_history + (1 - mask) * target_projected

        return z_mix


class MultiStageConditioningNetwork(nn.Module):
    """Collection of conditioning networks for all stages."""

    def __init__(
        self,
        num_stages: int,
        input_dim: int,
        hidden_dim: int = 256,
        forecast_length: int = 168,
        lookback_length: int = 336,
        num_conv_layers: int = 3,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = num_stages

        self.networks = nn.ModuleList([
            ConditioningNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                forecast_length=forecast_length,
                lookback_length=lookback_length,
                num_conv_layers=num_conv_layers,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])

    def forward(
        self,
        stage: int,
        history: torch.Tensor,
        coarse_trend: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        mixup_prob: float = 0.5,
        training: bool = True,
    ) -> torch.Tensor:
        """Get conditioning for a specific stage."""
        return self.networks[stage](
            history=history,
            coarse_trend=coarse_trend,
            target=target,
            mixup_prob=mixup_prob,
            training=training,
        )
