"""Main mr-Diff model implementation.

Multi-Resolution Diffusion Model for time series forecasting.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .conditioning import MultiStageConditioningNetwork
from .denoising import MultiStageDenoisingNetwork
from .diffusion import DiffusionSchedule, DiffusionStepEmbedding, forward_diffusion
from ..data.preprocessing import TrendExtraction, InstanceNormalization


class MRDiff(nn.Module):
    """Multi-Resolution Diffusion Model for time series forecasting.

    Implements the mr-Diff architecture with:
    - Multi-resolution trend decomposition
    - Stage-wise diffusion and denoising
    - Conditioning on history and coarser trends
    """

    def __init__(
        self,
        input_dim: int,
        num_stages: int = 5,
        diffusion_steps: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        kernel_sizes: List[int] = None,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        forecast_length: int = 168,
        lookback_length: int = 336,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize the mr-Diff model.

        Args:
            input_dim: Number of input features D.
            num_stages: Number of resolution stages S.
            diffusion_steps: Number of diffusion steps K.
            embedding_dim: Dimension of sinusoidal embeddings.
            hidden_dim: Hidden dimension for networks.
            kernel_sizes: Kernel sizes [τ1, ..., τS-1] for trend extraction.
            beta_start: Starting variance β1.
            beta_end: Ending variance βK.
            forecast_length: Length of forecast window T.
            lookback_length: Length of lookback window H.
            num_encoder_layers: Number of encoder conv layers.
            num_decoder_layers: Number of decoder conv layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_stages = num_stages
        self.diffusion_steps = diffusion_steps
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length

        # Default kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = [5, 25, 51, 201]
        # Ensure we have the right number of kernel sizes
        if len(kernel_sizes) != num_stages - 1:
            kernel_sizes = kernel_sizes[:num_stages - 1]
            while len(kernel_sizes) < num_stages - 1:
                kernel_sizes.append(kernel_sizes[-1] * 2 + 1)

        # Trend extraction module
        self.trend_extraction = TrendExtraction(kernel_sizes)

        # Instance normalization
        self.instance_norm = InstanceNormalization()

        # Diffusion schedule
        self.schedule = DiffusionSchedule(
            num_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Diffusion step embedding
        self.step_embedding = DiffusionStepEmbedding(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        # Conditioning networks (one per stage)
        self.conditioning = MultiStageConditioningNetwork(
            num_stages=num_stages,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            forecast_length=forecast_length,
            lookback_length=lookback_length,
        )

        # Denoising networks (one per stage)
        self.denoising = MultiStageDenoisingNetwork(
            num_stages=num_stages,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            step_embed_dim=hidden_dim,
            cond_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )

    def to(self, device: torch.device) -> "MRDiff":
        """Move model to device."""
        super().to(device)
        self.schedule = self.schedule.to(device)
        return self

    def decompose_target(self, target: torch.Tensor) -> List[torch.Tensor]:
        """Decompose target into multi-resolution components.

        Args:
            target: Target forecast [B, T, D].

        Returns:
            List of components [Y0, Y1, ..., YS-1] from finest to coarsest.
        """
        return self.trend_extraction(target)

    def training_step(
        self,
        lookback: torch.Tensor,
        forecast: torch.Tensor,
        mixup_prob: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Perform a single training step (Algorithm 1).

        Args:
            lookback: Historical data [B, H, D].
            forecast: Target forecast [B, T, D].
            mixup_prob: Probability of applying future mixup.

        Returns:
            Tuple of (total_loss, stage_losses_dict).
        """
        batch_size = lookback.size(0)
        device = lookback.device

        # Instance normalize
        forecast_norm, mean, std = self.instance_norm(forecast, return_stats=True)

        # Decompose into multi-resolution components
        components = self.decompose_target(forecast_norm)

        total_loss = torch.tensor(0.0, device=device)
        stage_losses = {}

        # Process each stage from coarsest to finest
        for s in range(self.num_stages - 1, -1, -1):
            y0_s = components[s]  # Clean component at stage s

            # Sample random diffusion step
            k = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

            # Apply forward diffusion
            yk_s, noise = forward_diffusion(y0_s, k, self.schedule)

            # Get diffusion step embedding
            step_embed = self.step_embedding(k)

            # Get coarser trend (None for coarsest stage)
            coarse_trend = components[s + 1] if s < self.num_stages - 1 else None

            # Get conditioning
            conditioning = self.conditioning(
                stage=s,
                history=lookback,
                coarse_trend=coarse_trend,
                target=y0_s,
                mixup_prob=mixup_prob,
                training=True,
            )

            # Predict clean data
            y_pred = self.denoising(
                stage=s,
                y_noisy=yk_s,
                step_embed=step_embed,
                conditioning=conditioning,
            )

            # Compute loss: L_s = E[||Y^0_s - Y^θ_s||²]
            loss_s = nn.functional.mse_loss(y_pred, y0_s)
            stage_losses[s] = loss_s

            total_loss = total_loss + loss_s

        return total_loss, stage_losses

    @torch.no_grad()
    def sample(
        self,
        lookback: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate forecasts using reverse diffusion (Algorithm 2).

        Args:
            lookback: Historical data [B, H, D].
            num_samples: Number of forecast samples to generate.

        Returns:
            Forecasts [B, num_samples, T, D] or [B, T, D] if num_samples=1.
        """
        batch_size = lookback.size(0)
        device = lookback.device

        all_samples = []

        for _ in range(num_samples):
            # Initialize predictions for all stages
            predictions = [None] * self.num_stages

            # Generate from coarsest to finest stage
            for s in range(self.num_stages - 1, -1, -1):
                # Start from pure noise for the coarsest level
                yk = torch.randn(
                    batch_size, self.forecast_length, self.input_dim,
                    device=device,
                )

                # Get coarser trend (use prediction from previous stage)
                coarse_trend = predictions[s + 1] if s < self.num_stages - 1 else None

                # Reverse diffusion process
                for k in range(self.diffusion_steps - 1, -1, -1):
                    k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.long)

                    # Get step embedding
                    step_embed = self.step_embedding(k_tensor)

                    # Get conditioning (no mixup during inference)
                    conditioning = self.conditioning(
                        stage=s,
                        history=lookback,
                        coarse_trend=coarse_trend,
                        target=None,
                        mixup_prob=0.0,
                        training=False,
                    )

                    # Predict clean data
                    y_pred = self.denoising(
                        stage=s,
                        y_noisy=yk,
                        step_embed=step_embed,
                        conditioning=conditioning,
                    )

                    # DDPM update step
                    if k > 0:
                        alpha = self.schedule.alphas[k]
                        alpha_bar = self.schedule.alpha_bars[k]
                        alpha_bar_prev = self.schedule.alpha_bars[k - 1]
                        beta = self.schedule.betas[k]

                        # Posterior mean
                        coef1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                        coef2 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                        posterior_mean = coef1 * y_pred + coef2 * yk

                        # Add noise
                        noise = torch.randn_like(yk)
                        posterior_var = self.schedule.posterior_variance[k]
                        yk = posterior_mean + torch.sqrt(posterior_var) * noise
                    else:
                        yk = y_pred

                predictions[s] = yk

            # Return finest-stage prediction only.
            # sum(predictions) is theoretically correct but produces worse
            # results under DDPM sampling due to cascading coarse-stage errors.
            # See Run 173919 analysis in README.
            forecast = predictions[0]
            all_samples.append(forecast)

        # Stack samples
        samples = torch.stack(all_samples, dim=1)  # [B, num_samples, T, D]

        if num_samples == 1:
            samples = samples.squeeze(1)  # [B, T, D]

        return samples

    def forward(
        self,
        lookback: torch.Tensor,
        forecast: Optional[torch.Tensor] = None,
        mixup_prob: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        During training: compute loss.
        During inference: generate forecasts.

        Args:
            lookback: Historical data [B, H, D].
            forecast: Target forecast [B, T, D] (required for training).
            mixup_prob: Probability of future mixup.

        Returns:
            Dictionary with 'loss' and 'stage_losses' during training,
            or 'predictions' during inference.
        """
        if self.training and forecast is not None:
            total_loss, stage_losses = self.training_step(lookback, forecast, mixup_prob)
            return {
                "loss": total_loss,
                "stage_losses": stage_losses,
            }
        else:
            predictions = self.sample(lookback)
            return {
                "predictions": predictions,
            }


def create_model(config: dict) -> MRDiff:
    """Create mr-Diff model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized MRDiff model.
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Determine input dimension
    input_dim = 1 if data_config.get("univariate", False) else 7

    model = MRDiff(
        input_dim=input_dim,
        num_stages=model_config.get("num_stages", 5),
        diffusion_steps=model_config.get("diffusion_steps", 100),
        embedding_dim=model_config.get("embedding_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        kernel_sizes=model_config.get("kernel_sizes", [5, 25, 51, 201]),
        beta_start=training_config.get("beta_start", 1e-4),
        beta_end=training_config.get("beta_end", 0.1),
        forecast_length=data_config.get("forecast_length", 168),
        lookback_length=data_config.get("lookback_length", 336),
        num_encoder_layers=model_config.get("num_encoder_layers", 3),
        num_decoder_layers=model_config.get("num_decoder_layers", 3),
        dropout=model_config.get("dropout", 0.1),
    )

    return model
