"""Diffusion process implementation for mr-Diff.

Implements forward diffusion, variance schedules, and step embeddings.
Includes ANT (Adaptive Noise Time) schedule based on IAAT.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


def compute_iaat(data: np.ndarray, max_lag: int = 50) -> float:
    """Compute Integrated Absolute Autocorrelation Time (IAAT).

    IAAT measures temporal non-stationarity: higher IAAT means stronger
    temporal correlations that the noise schedule should account for.

    Args:
        data: Time series data [N, D] or [N].
        max_lag: Maximum lag for autocorrelation computation.

    Returns:
        IAAT value (scalar).
    """
    if data.ndim == 1:
        data = data[:, None]

    n_samples, n_dims = data.shape
    max_lag = min(max_lag, n_samples // 4)

    # Compute autocorrelation for each dimension
    total_iaat = 0.0
    for d in range(n_dims):
        x = data[:, d]
        x = (x - x.mean()) / (x.std() + 1e-8)
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(x[:-lag] * x[lag:])
        # Integrated absolute autocorrelation
        total_iaat += np.sum(np.abs(acf))

    return total_iaat / n_dims


def create_ant_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
    iaat: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Create ANT (Adaptive Noise Time) beta schedule based on IAAT.

    The idea: datasets with higher temporal autocorrelation need a schedule
    that spends more steps at intermediate noise levels (where temporal
    structure is being destroyed/reconstructed). The IAAT controls the
    curvature of the beta schedule.

    For high IAAT (strong temporal correlation): concave schedule that
    ramps up slowly then accelerates — spending more capacity at low noise
    where temporal structure matters.

    For low IAAT (weak temporal correlation): closer to linear schedule.

    Args:
        num_steps: Number of diffusion steps.
        beta_start: Starting beta.
        beta_end: Ending beta.
        iaat: Integrated Absolute Autocorrelation Time.
        device: Device for tensors.

    Returns:
        Beta schedule tensor [num_steps].
    """
    device = device or torch.device("cpu")

    # Map IAAT to curvature parameter gamma
    # IAAT typically ranges from ~5 (low correlation) to ~40 (high correlation)
    # gamma < 1: concave (slow start, fast end) — for high IAAT
    # gamma > 1: convex (fast start, slow end)
    # gamma = 1: linear
    gamma = np.clip(1.0 / (1.0 + 0.05 * iaat), 0.3, 1.0)

    # Create non-linear spacing using power law
    t = torch.linspace(0, 1, num_steps, device=device)
    t_warped = t ** gamma

    # Map to beta range
    betas = beta_start + (beta_end - beta_start) * t_warped

    return betas


class DiffusionSchedule:
    """Variance schedule for diffusion process.

    Supports linear, cosine, and ANT (adaptive) schedules.
    """

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        schedule_type: str = "linear",
        device: torch.device = None,
        iaat: float = None,
    ):
        """Initialize the diffusion schedule.

        Args:
            num_steps: Number of diffusion steps K.
            beta_start: Starting variance β₁ (linear only).
            beta_end: Ending variance βK (linear only).
            schedule_type: "linear", "cosine", or "ant".
            device: Device to store tensors on.
            iaat: IAAT value for ANT schedule (required if schedule_type="ant").
        """
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")

        if schedule_type == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = torch.linspace(0, num_steps, num_steps + 1, device=self.device)
            f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = f / f[0]
            alpha_bars = alpha_bars[1:]
            alpha_bars = alpha_bars.clamp(min=1e-4)
            alpha_bars_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bars[:-1]])
            self.betas = (1 - alpha_bars / alpha_bars_prev).clamp(max=0.999)
            self.alpha_bars = alpha_bars
        elif schedule_type == "ant":
            # ANT adaptive schedule based on IAAT
            if iaat is None:
                raise ValueError("IAAT value required for ANT schedule")
            self.betas = create_ant_schedule(num_steps, beta_start, beta_end, iaat, self.device)
        else:
            # Linear schedule: β_k from beta_start to beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)

        # α_k = 1 - β_k
        self.alphas = 1.0 - self.betas

        # ᾱ_k = ∏_{s=1}^{k} α_s (cumulative product)
        # For cosine schedule, alpha_bars is already set above
        if not hasattr(self, 'alpha_bars') or schedule_type != "cosine":
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute useful quantities
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance: β̃_k = β_k * (1 - ᾱ_{k-1}) / (1 - ᾱ_k)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        """Move schedule tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def get_alpha_bar(self, k: torch.Tensor) -> torch.Tensor:
        """Get ᾱ_k for given timesteps.

        Args:
            k: Timestep indices [B] or scalar.

        Returns:
            ᾱ_k values.
        """
        return self.alpha_bars[k]

    def get_sqrt_alpha_bar(self, k: torch.Tensor) -> torch.Tensor:
        """Get √ᾱ_k for given timesteps."""
        return self.sqrt_alpha_bars[k]

    def get_sqrt_one_minus_alpha_bar(self, k: torch.Tensor) -> torch.Tensor:
        """Get √(1-ᾱ_k) for given timesteps."""
        return self.sqrt_one_minus_alpha_bars[k]


def forward_diffusion(
    y0: torch.Tensor,
    k: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply forward diffusion process (Eq. 8).

    Y^k = √ᾱ_k · Y^0 + √(1-ᾱ_k) · ε

    Args:
        y0: Clean data Y^0 [B, T, D].
        k: Diffusion timesteps [B].
        schedule: Diffusion schedule with precomputed values.
        noise: Optional pre-sampled noise ε.

    Returns:
        Tuple of (noisy_data Y^k, noise ε).
    """
    if noise is None:
        noise = torch.randn_like(y0)

    # Get schedule values for batch [B] -> [B, 1, 1]
    sqrt_alpha_bar = schedule.get_sqrt_alpha_bar(k)
    sqrt_one_minus_alpha_bar = schedule.get_sqrt_one_minus_alpha_bar(k)

    # Reshape for broadcasting
    while sqrt_alpha_bar.dim() < y0.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

    # Apply forward diffusion
    yk = sqrt_alpha_bar * y0 + sqrt_one_minus_alpha_bar * noise

    return yk, noise


class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps (Eq. 7).

    Uses sinusoidal positional encoding followed by two FC layers.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        """Initialize the step embedding.

        Args:
            embedding_dim: Dimension of the sinusoidal embedding.
            hidden_dim: Hidden dimension of FC layers.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Two FC layers to transform sinusoidal embedding
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Compute diffusion step embedding.

        Args:
            k: Timestep indices [B].

        Returns:
            Step embeddings [B, hidden_dim].
        """
        # Sinusoidal embedding (similar to transformer positional encoding)
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=k.device, dtype=torch.float32) * -emb_scale)

        # [B] x [half_dim] -> [B, half_dim]
        emb = k.float().unsqueeze(1) * emb.unsqueeze(0)

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Pad if embedding_dim is odd
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        # Apply FC layers
        return self.fc(emb)



