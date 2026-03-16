"""Main mr-Diff model implementation.

Multi-Resolution Diffusion Model for time series forecasting.
Experiment 7: Multi-Granularity Guided Diffusion (MG-TSD).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import MultiStageConditioningNetwork
from .denoising import MultiStageDenoisingNetwork
from .diffusion import DiffusionSchedule, DiffusionStepEmbedding, forward_diffusion
from ..data.preprocessing import TrendExtraction

from dpm_solver_pp import DPMSolverPP


def compute_multi_granularity_targets(x: torch.Tensor, num_granularities: int = 3) -> List[torch.Tensor]:
    """Compute multi-granularity versions of a signal.

    Creates progressively smoothed versions of the input using average pooling
    at different temporal resolutions.

    Args:
        x: Input tensor [B, T, D].
        num_granularities: Number of granularity levels (including original).

    Returns:
        List of tensors from finest (original) to coarsest (most smoothed).
    """
    granularities = [x]  # Level 0: original
    T = x.size(1)

    # Smoothing kernel sizes: progressively larger
    kernel_sizes = [max(3, T // (2 ** (num_granularities - 1 - i))) | 1 for i in range(1, num_granularities)]
    # Ensure odd kernel sizes
    kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]

    for ks in kernel_sizes:
        # [B, T, D] -> [B, D, T] for avg_pool1d
        x_t = x.transpose(1, 2)
        pad = ks // 2
        x_padded = F.pad(x_t, (pad, pad), mode='replicate')
        smoothed = F.avg_pool1d(x_padded, ks, stride=1)
        granularities.append(smoothed.transpose(1, 2))

    return granularities


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
        schedule_type: str = "linear",
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

        # Diffusion schedule
        self.schedule = DiffusionSchedule(
            num_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
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
            dropout=dropout,
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

        # Direct prediction backbone (DLinear-style)
        # Provides a strong initial forecast; diffusion refines the residual
        self.direct_trend_proj = nn.Linear(lookback_length, forecast_length)
        self.direct_resid_proj = nn.Linear(lookback_length, forecast_length)
        self.direct_kernel_size = 25

    def to(self, device: torch.device) -> "MRDiff":
        """Move model to device."""
        super().to(device)
        self.schedule = self.schedule.to(device)
        return self

    def direct_predict(self, lookback: torch.Tensor) -> torch.Tensor:
        """Direct DLinear-style forecast from lookback (no diffusion).

        Args:
            lookback: Historical data [B, H, D].

        Returns:
            Direct forecast [B, T, D].
        """
        # [B, H, D] -> [B, D, H]
        x = lookback.transpose(1, 2)
        # Trend extraction
        ks = self.direct_kernel_size
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)
        resid = x - trend
        # Project each to forecast length
        out = self.direct_trend_proj(trend) + self.direct_resid_proj(resid)
        return out.transpose(1, 2)  # [B, T, D]

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

        # Direct prediction + residual diffusion (joint training)
        direct_pred = self.direct_predict(lookback)
        direct_loss = nn.functional.mse_loss(direct_pred, forecast)
        residual = forecast - direct_pred  # No detach: diffusion gradients flow through backbone

        # Decompose RESIDUAL into multi-resolution components
        components = self.decompose_target(residual)

        # Scale diffusion loss relative to direct loss to prevent gradient dominance
        diffusion_loss_scale = 0.3
        total_loss = direct_loss
        stage_losses = {}
        stage_predictions = {}  # For scheduled sampling

        # Scheduled sampling probability — ramp from 0 to 0.5 over 50 epochs
        epoch = getattr(self, 'current_epoch', 0)
        ss_prob = min(epoch / 50.0, 0.5)

        # Process each stage from coarsest to finest
        for s in range(self.num_stages - 1, -1, -1):
            y0_s = components[s]  # Clean component at stage s

            # Sample random diffusion step
            k = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

            # Apply forward diffusion
            yk_s, noise = forward_diffusion(y0_s, k, self.schedule)

            # Get diffusion step embedding
            step_embed = self.step_embedding(k)

            # Get coarser trend with scheduled sampling
            if s < self.num_stages - 1:
                use_model_pred = (
                    ss_prob > 0
                    and torch.rand(1).item() < ss_prob
                    and (s + 1) in stage_predictions
                )
                if use_model_pred:
                    coarse_trend = stage_predictions[s + 1].detach()
                else:
                    coarse_trend = components[s + 1]
            else:
                coarse_trend = None

            # Get conditioning
            conditioning = self.conditioning(
                stage=s,
                history=lookback,
                coarse_trend=coarse_trend,
                target=y0_s,
                mixup_prob=mixup_prob,
                training=True,
            )

            # Self-conditioning: 50% of the time, get a preliminary x0 estimate first
            x0_prev = None
            if torch.rand(1).item() < 0.5:
                with torch.no_grad():
                    eps_prelim = self.denoising(
                        stage=s, y_noisy=yk_s, step_embed=step_embed,
                        conditioning=conditioning, x0_prev=None,
                    )
                    alpha_bar_k = self.schedule.alpha_bars[k].view(-1, 1, 1)
                    x0_prev = (yk_s - torch.sqrt(1 - alpha_bar_k) * eps_prelim) / torch.sqrt(alpha_bar_k).clamp(min=1e-5)
                    x0_prev = x0_prev.detach()

            # Predict noise (epsilon prediction) with self-conditioning
            eps_pred = self.denoising(
                stage=s,
                y_noisy=yk_s,
                step_embed=step_embed,
                conditioning=conditioning,
                x0_prev=x0_prev,
            )

            # Recover x0 from epsilon prediction for scheduled sampling
            alpha_bar_k = self.schedule.alpha_bars[k].view(-1, 1, 1)
            x0_pred = (yk_s - torch.sqrt(1 - alpha_bar_k) * eps_pred) / torch.sqrt(alpha_bar_k).clamp(min=1e-3)
            x0_pred = x0_pred.clamp(-10, 10)  # Prevent explosion at high noise levels
            stage_predictions[s] = x0_pred.detach()

            # Compute loss: predict noise (uniform difficulty across k)
            loss_s = nn.functional.mse_loss(eps_pred, noise)

            # Frequency-domain auxiliary loss — match spectral structure
            # Use detached x0_pred to prevent FFT loss gradient explosion
            fft_pred = torch.fft.rfft(x0_pred.detach(), dim=1).abs()
            fft_target = torch.fft.rfft(y0_s, dim=1).abs()
            freq_loss = nn.functional.mse_loss(fft_pred, fft_target)
            loss_s = loss_s + 0.1 * freq_loss

            # Multi-Granularity Guidance Loss (MG-TSD)
            # At high noise levels (large k), x0_pred should match coarse targets
            # At low noise levels (small k), x0_pred should match fine targets
            mg_targets = compute_multi_granularity_targets(y0_s, num_granularities=3)
            # Map k to granularity level: high k -> coarse, low k -> fine
            k_normalized = k.float() / self.diffusion_steps  # [0, 1]
            # Compute weighted guidance loss
            mg_loss = torch.tensor(0.0, device=device)
            for g_idx, mg_target in enumerate(mg_targets):
                # Weight: coarser targets weighted more at high noise
                # granularity 0 = finest, granularity N = coarsest
                if g_idx == 0:
                    # Fine: weight high when k is low
                    w = (1.0 - k_normalized).view(-1, 1, 1)
                else:
                    # Coarser: weight high when k is high
                    w = (k_normalized * g_idx / (len(mg_targets) - 1)).view(-1, 1, 1)
                mg_loss = mg_loss + (w * (x0_pred.detach() - mg_target) ** 2).mean()
            loss_s = loss_s + 0.05 * mg_loss

            # Stage weighting — coarser stages matter more (errors cascade)
            stage_weight = (s + 1) / self.num_stages
            loss_s = loss_s * stage_weight

            stage_losses[s] = loss_s
            total_loss = total_loss + diffusion_loss_scale * loss_s

        return total_loss, stage_losses

    @torch.no_grad()
    def sample(
        self,
        lookback: torch.Tensor,
        num_samples: int = 1,
        solver: str = "dpm_solver_pp",
        solver_steps: int = 20,
        aggregation: str = "sum",
        epsilon_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate forecasts using reverse diffusion (Algorithm 2).

        Args:
            lookback: Historical data [B, H, D].
            num_samples: Number of forecast samples to generate.
            solver: Sampling solver — "ddpm" (100-step) or "dpm_solver_pp" (fast ODE).
            solver_steps: Number of DPM-Solver++ steps (ignored for DDPM).
            aggregation: How to combine multi-resolution predictions:
                "first" — use predictions[0] only (safe default for DDPM),
                "sum" — sum all stages (paper's method, needs DPM-Solver++).
            epsilon_scale: Scale factor for x0 predictions (default 1.0).
                Values < 1.0 (e.g. 0.98) reduce exposure bias drift.

        Returns:
            Forecasts [B, num_samples, T, D] or [B, T, D] if num_samples=1.
        """
        batch_size = lookback.size(0)
        device = lookback.device

        all_samples = []

        for _ in range(num_samples):
            predictions = [None] * self.num_stages

            # Generate from coarsest to finest stage
            for s in range(self.num_stages - 1, -1, -1):
                coarse_trend = predictions[s + 1] if s < self.num_stages - 1 else None

                if solver == "dpm_solver_pp":
                    predictions[s] = self._sample_stage_dpm(
                        s, lookback, coarse_trend, batch_size, device, solver_steps,
                        epsilon_scale,
                    )
                else:
                    predictions[s] = self._sample_stage_ddpm(
                        s, lookback, coarse_trend, batch_size, device,
                        epsilon_scale,
                    )

            # Aggregate multi-resolution predictions
            if aggregation == "sum":
                forecast = sum(predictions)
            else:
                forecast = predictions[0]

            all_samples.append(forecast)

        samples = torch.stack(all_samples, dim=1)  # [B, num_samples, T, D]

        if num_samples == 1:
            samples = samples.squeeze(1)  # [B, T, D]

        # Add direct prediction baseline to diffusion residual
        direct_pred = self.direct_predict(lookback)
        if samples.dim() == 3:
            samples = direct_pred + samples
        else:
            samples = direct_pred.unsqueeze(1) + samples

        return samples

    def _sample_stage_ddpm(self, s, lookback, coarse_trend, batch_size, device,
                           epsilon_scale=1.0):
        """DDPM 100-step reverse diffusion for a single stage (epsilon prediction)."""
        yk = torch.randn(
            batch_size, self.forecast_length, self.input_dim, device=device,
        )

        x0_prev = None  # Self-conditioning: track previous x0 estimate

        for k in range(self.diffusion_steps - 1, -1, -1):
            k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.long)
            step_embed = self.step_embedding(k_tensor)

            conditioning = self.conditioning(
                stage=s, history=lookback, coarse_trend=coarse_trend,
                target=None, mixup_prob=0.0, training=False,
            )

            # Model predicts epsilon (noise) with self-conditioning
            eps_pred = self.denoising(
                stage=s, y_noisy=yk, step_embed=step_embed,
                conditioning=conditioning, x0_prev=x0_prev,
            )

            # Convert epsilon to x0
            alpha_bar = self.schedule.alpha_bars[k]
            x0_pred = (yk - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar).clamp(min=1e-5)
            x0_pred = x0_pred * epsilon_scale
            x0_prev = x0_pred  # Feed x0 estimate to next step for self-conditioning

            if k > 0:
                alpha = self.schedule.alphas[k]
                alpha_bar_prev = self.schedule.alpha_bars[k - 1]
                beta = self.schedule.betas[k]

                coef1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                coef2 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                posterior_mean = coef1 * x0_pred + coef2 * yk

                noise = torch.randn_like(yk)
                posterior_var = self.schedule.posterior_variance[k]
                yk = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                yk = x0_pred

        return yk

    def _sample_stage_dpm(self, s, lookback, coarse_trend, batch_size, device,
                          solver_steps, epsilon_scale=1.0):
        """DPM-Solver++ sampling for a single stage (epsilon prediction)."""
        # Precompute conditioning (fixed across all diffusion steps)
        conditioning = self.conditioning(
            stage=s, history=lookback, coarse_trend=coarse_trend,
            target=None, mixup_prob=0.0, training=False,
        )

        # Self-conditioning state: track x0 from previous solver call
        x0_state = [None]  # Use list for closure mutability

        # Build model_fn closure: (y_noisy, k_tensor) -> x0_pred
        def model_fn(yk, k_tensor):
            step_embed = self.step_embedding(k_tensor)
            eps_pred = self.denoising(
                stage=s, y_noisy=yk, step_embed=step_embed,
                conditioning=conditioning, x0_prev=x0_state[0],
            )
            # Convert epsilon to x0
            alpha_bar = self.schedule.alpha_bars[k_tensor[0]]
            x0_pred = (yk - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar).clamp(min=1e-5)
            x0_pred = x0_pred * epsilon_scale
            x0_state[0] = x0_pred  # Update for next call
            return x0_pred

        dpm_solver = DPMSolverPP(self.schedule, num_solver_steps=solver_steps)
        shape = (batch_size, self.forecast_length, self.input_dim)
        return dpm_solver.sample(model_fn, shape, device)

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
        schedule_type=training_config.get("schedule_type", "linear"),
        forecast_length=data_config.get("forecast_length", 168),
        lookback_length=data_config.get("lookback_length", 336),
        num_encoder_layers=model_config.get("num_encoder_layers", 3),
        num_decoder_layers=model_config.get("num_decoder_layers", 3),
        dropout=model_config.get("dropout", 0.1),
    )

    return model
