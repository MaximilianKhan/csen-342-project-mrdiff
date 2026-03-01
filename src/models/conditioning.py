"""Conditioning network for mr-Diff.

Implements the conditioning mechanism with history encoding and future mixup.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConditioningNetwork(nn.Module):
    """Conditioning network for multi-resolution diffusion.

    Creates conditioning signals from historical data and coarser trends.
    Includes future mixup during training.
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
        """Initialize the conditioning network.

        Args:
            input_dim: Number of input features D.
            hidden_dim: Hidden dimension for embeddings.
            forecast_length: Length of forecast window T.
            lookback_length: Length of lookback window H.
            num_conv_layers: Number of conv layers in history encoder.
            conv_kernel_size: Kernel size for history convolutions.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length

        # Temporal history encoder: captures sequential patterns via Conv1d
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

        # Projection to match forecast length
        self.length_proj = nn.Linear(lookback_length, forecast_length)

        # Target projection dimension for future mixup
        self._target_proj_dim = (input_dim, hidden_dim)

        # Project coarse_trend from D -> hidden_dim for equal representation
        self.coarse_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

        # Conditioning fusion: combines z_mix with projected coarse trend
        # Input: hidden_dim (from mixup) + hidden_dim (from projected coarse trend)
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
        """Compute conditioning signal.

        Args:
            history: Historical data Xs [B, H, D].
            coarse_trend: Coarser trend Y^0_{s+1} [B, T, D]. None for coarsest stage.
            target: Ground truth Y^0_s [B, T, D] for mixup during training.
            mixup_prob: Probability of applying future mixup.
            training: Whether in training mode.

        Returns:
            Conditioning signal cs [B, T, hidden_dim].
        """
        batch_size = history.size(0)

        # Encode history with temporal convolutions
        # [B, H, D] -> [B, H, hidden_dim]
        z_history = self.history_input(history)
        # Conv1d needs [B, C, T]: [B, hidden_dim, H]
        z_history = z_history.transpose(1, 2)
        z_history = self.history_convs(z_history) + z_history  # Residual
        # Project to forecast length: [B, hidden_dim, H] -> [B, hidden_dim, T]
        z_history = self.length_proj(z_history)
        z_history = z_history.transpose(1, 2)  # [B, T, hidden_dim]

        # Future mixup during training
        if training and target is not None and torch.rand(1).item() < mixup_prob:
            z_mix = self._apply_mixup(z_history, target)
        else:
            z_mix = z_history

        # Fuse with coarser trend if provided
        if coarse_trend is not None:
            # Project coarse_trend: [B, T, D] -> [B, T, hidden_dim]
            coarse_projected = self.coarse_proj(coarse_trend)
            fused = torch.cat([z_mix, coarse_projected], dim=-1)
            conditioning = self.fusion(fused)
        else:
            # For coarsest stage, use zeros at hidden_dim
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
        """Apply future mixup for training regularization (paper Eq. 9).

        z_mix = m ⊙ z_history + (1-m) ⊙ Proj(Y^0_s)

        Uses fresh random projection weights each call so the model cannot
        learn to extract predictive features from the ground-truth target.
        This keeps train-time conditioning close to inference (history-only),
        preventing a train-test gap.

        Args:
            z_history: Encoded history [B, T, hidden_dim].
            target: Ground truth target Y^0_s [B, T, D].

        Returns:
            Mixed representation [B, T, hidden_dim].
        """
        batch_size, seq_len, _ = z_history.shape
        in_dim, out_dim = self._target_proj_dim

        # Project target with fresh random weights (not learned)
        # This acts as noise injection rather than information extraction
        random_weight = torch.randn(out_dim, in_dim, device=target.device) * 0.02
        target_projected = nn.functional.linear(target, random_weight)

        # Generate continuous mixing mask from U(0, 1) as in paper Eq. 9
        mask = torch.rand(batch_size, seq_len, 1, device=z_history.device)

        # Apply mixup
        z_mix = mask * z_history + (1 - mask) * target_projected

        return z_mix


class MultiStageConditioningNetwork(nn.Module):
    """Collection of conditioning networks for all stages.

    Manages conditioning for the multi-resolution hierarchy.
    """

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
        """Initialize conditioning networks for all stages.

        Args:
            num_stages: Number of resolution stages S.
            input_dim: Number of input features.
            hidden_dim: Hidden dimension.
            forecast_length: Length of forecast window.
            lookback_length: Length of lookback window.
            num_conv_layers: Number of conv layers in history encoder.
            conv_kernel_size: Kernel size for history convolutions.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_stages = num_stages

        # Create conditioning network for each stage
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
        """Get conditioning for a specific stage.

        Args:
            stage: Stage index (0 to S-1).
            history: Historical data.
            coarse_trend: Coarser trend (None for coarsest stage).
            target: Ground truth for mixup.
            mixup_prob: Mixup probability.
            training: Whether in training mode.

        Returns:
            Conditioning signal for the stage.
        """
        return self.networks[stage](
            history=history,
            coarse_trend=coarse_trend,
            target=target,
            mixup_prob=mixup_prob,
            training=training,
        )
