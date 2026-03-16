"""Main mr-Diff model implementation.

Multi-Resolution Diffusion Model for time series forecasting.
Experiment 10: Direct x0-prediction with trend/seasonality decomposition.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .conditioning import MultiStageConditioningNetwork
from .denoising import MultiStageDenoisingNetwork
from .diffusion import DiffusionSchedule, DiffusionStepEmbedding, forward_diffusion
from ..data.preprocessing import TrendExtraction

from dpm_solver_pp import DPMSolverPP


class MRDiff(nn.Module):
    """Multi-Resolution Diffusion Model with direct x0-prediction + decomposition."""

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
        super().__init__()

        self.input_dim = input_dim
        self.num_stages = num_stages
        self.diffusion_steps = diffusion_steps
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length

        if kernel_sizes is None:
            kernel_sizes = [5, 25, 51, 201]
        if len(kernel_sizes) != num_stages - 1:
            kernel_sizes = kernel_sizes[:num_stages - 1]
            while len(kernel_sizes) < num_stages - 1:
                kernel_sizes.append(kernel_sizes[-1] * 2 + 1)

        self.trend_extraction = TrendExtraction(kernel_sizes)

        self.schedule = DiffusionSchedule(
            num_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
        )

        self.step_embedding = DiffusionStepEmbedding(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        self.conditioning = MultiStageConditioningNetwork(
            num_stages=num_stages,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            forecast_length=forecast_length,
            lookback_length=lookback_length,
            dropout=dropout,
        )

        # Denoising networks — now predict x0 directly (with decomposition heads)
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
        self.direct_trend_proj = nn.Linear(lookback_length, forecast_length)
        self.direct_resid_proj = nn.Linear(lookback_length, forecast_length)
        self.direct_kernel_size = 25

    def to(self, device: torch.device) -> "MRDiff":
        super().to(device)
        self.schedule = self.schedule.to(device)
        return self

    def direct_predict(self, lookback: torch.Tensor) -> torch.Tensor:
        x = lookback.transpose(1, 2)
        ks = self.direct_kernel_size
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)
        resid = x - trend
        out = self.direct_trend_proj(trend) + self.direct_resid_proj(resid)
        return out.transpose(1, 2)

    def decompose_target(self, target: torch.Tensor) -> List[torch.Tensor]:
        return self.trend_extraction(target)

    def training_step(
        self,
        lookback: torch.Tensor,
        forecast: torch.Tensor,
        mixup_prob: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Training step with direct x0-prediction loss."""
        batch_size = lookback.size(0)
        device = lookback.device

        # Direct prediction + residual diffusion (joint training)
        direct_pred = self.direct_predict(lookback)
        direct_loss = nn.functional.mse_loss(direct_pred, forecast)
        residual = forecast - direct_pred

        # Decompose RESIDUAL into multi-resolution components
        components = self.decompose_target(residual)

        diffusion_loss_scale = 0.3
        total_loss = direct_loss
        stage_losses = {}
        stage_predictions = {}

        epoch = getattr(self, 'current_epoch', 0)
        ss_prob = min(epoch / 50.0, 0.5)

        for s in range(self.num_stages - 1, -1, -1):
            y0_s = components[s]

            k = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
            yk_s, noise = forward_diffusion(y0_s, k, self.schedule)
            step_embed = self.step_embedding(k)

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

            conditioning = self.conditioning(
                stage=s, history=lookback, coarse_trend=coarse_trend,
                target=y0_s, mixup_prob=mixup_prob, training=True,
            )

            # Self-conditioning
            x0_prev = None
            if torch.rand(1).item() < 0.5:
                with torch.no_grad():
                    x0_prelim = self.denoising(
                        stage=s, y_noisy=yk_s, step_embed=step_embed,
                        conditioning=conditioning, x0_prev=None,
                    )
                    x0_prev = x0_prelim.detach()

            # Model directly predicts x0 (with decomposition heads)
            x0_pred = self.denoising(
                stage=s, y_noisy=yk_s, step_embed=step_embed,
                conditioning=conditioning, x0_prev=x0_prev,
            )

            x0_pred_clamped = x0_pred.clamp(-10, 10)
            stage_predictions[s] = x0_pred_clamped.detach()

            # Direct x0 MSE loss (instead of epsilon MSE)
            loss_s = nn.functional.mse_loss(x0_pred, y0_s)

            # Frequency-domain auxiliary loss
            fft_pred = torch.fft.rfft(x0_pred.detach(), dim=1).abs()
            fft_target = torch.fft.rfft(y0_s, dim=1).abs()
            freq_loss = nn.functional.mse_loss(fft_pred, fft_target)
            loss_s = loss_s + 0.1 * freq_loss

            # Stage weighting
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
        """Generate forecasts using reverse diffusion with x0-prediction."""
        batch_size = lookback.size(0)
        device = lookback.device

        all_samples = []

        for _ in range(num_samples):
            predictions = [None] * self.num_stages

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

            if aggregation == "sum":
                forecast = sum(predictions)
            else:
                forecast = predictions[0]

            all_samples.append(forecast)

        samples = torch.stack(all_samples, dim=1)
        if num_samples == 1:
            samples = samples.squeeze(1)

        direct_pred = self.direct_predict(lookback)
        if samples.dim() == 3:
            samples = direct_pred + samples
        else:
            samples = direct_pred.unsqueeze(1) + samples

        return samples

    def _sample_stage_ddpm(self, s, lookback, coarse_trend, batch_size, device,
                           epsilon_scale=1.0):
        """DDPM reverse diffusion with direct x0-prediction."""
        yk = torch.randn(batch_size, self.forecast_length, self.input_dim, device=device)
        x0_prev = None

        for k in range(self.diffusion_steps - 1, -1, -1):
            k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.long)
            step_embed = self.step_embedding(k_tensor)

            conditioning = self.conditioning(
                stage=s, history=lookback, coarse_trend=coarse_trend,
                target=None, mixup_prob=0.0, training=False,
            )

            # Model directly predicts x0
            x0_pred = self.denoising(
                stage=s, y_noisy=yk, step_embed=step_embed,
                conditioning=conditioning, x0_prev=x0_prev,
            )
            x0_pred = x0_pred * epsilon_scale
            x0_prev = x0_pred

            if k > 0:
                alpha_bar = self.schedule.alpha_bars[k]
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
        """DPM-Solver++ with direct x0-prediction model."""
        conditioning = self.conditioning(
            stage=s, history=lookback, coarse_trend=coarse_trend,
            target=None, mixup_prob=0.0, training=False,
        )

        x0_state = [None]

        def model_fn(yk, k_tensor):
            step_embed = self.step_embedding(k_tensor)
            # Model directly predicts x0
            x0_pred = self.denoising(
                stage=s, y_noisy=yk, step_embed=step_embed,
                conditioning=conditioning, x0_prev=x0_state[0],
            )
            x0_pred = x0_pred * epsilon_scale
            x0_state[0] = x0_pred
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
        if self.training and forecast is not None:
            total_loss, stage_losses = self.training_step(lookback, forecast, mixup_prob)
            return {"loss": total_loss, "stage_losses": stage_losses}
        else:
            predictions = self.sample(lookback)
            return {"predictions": predictions}


def create_model(config: dict) -> MRDiff:
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
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
