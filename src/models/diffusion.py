"""Diffusion process implementation for mr-Diff.

Implements forward diffusion, variance schedules, and step embeddings.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class DiffusionSchedule:
    """Variance schedule for diffusion process.

    Supports linear and cosine schedules with precomputed α values.
    """

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        schedule_type: str = "linear",
        device: torch.device = None,
    ):
        """Initialize the diffusion schedule.

        Args:
            num_steps: Number of diffusion steps K.
            beta_start: Starting variance β₁ (linear only).
            beta_end: Ending variance βK (linear only).
            schedule_type: "linear" or "cosine".
            device: Device to store tensors on.
        """
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")

        if schedule_type == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            # Gives smoother noise progression, better for fewer steps
            s = 0.008
            steps = torch.linspace(0, num_steps, num_steps + 1, device=self.device)
            f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = f / f[0]
            # Clip and compute betas from alpha_bars
            alpha_bars = alpha_bars[1:]  # Remove t=0
            alpha_bars_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bars[:-1]])
            self.betas = (1 - alpha_bars / alpha_bars_prev).clamp(max=0.999)
            self.alpha_bars = alpha_bars
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


def get_diffusion_step_embedding(
    k: torch.Tensor,
    embedding_dim: int = 128,
) -> torch.Tensor:
    """Get sinusoidal embedding for diffusion timesteps.

    Standalone function for simple embedding without learned parameters.

    Args:
        k: Timestep indices [B].
        embedding_dim: Dimension of the embedding.

    Returns:
        Sinusoidal embeddings [B, embedding_dim].
    """
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=k.device, dtype=torch.float32) * -emb_scale)

    emb = k.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

    return emb


class LinearBetaSchedule:
    """Simple linear beta schedule wrapper for trainer compatibility."""

    def __init__(self, num_steps: int, beta_start: float, beta_end: float):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_schedule(self, device: torch.device = None) -> DiffusionSchedule:
        """Create a DiffusionSchedule from this configuration."""
        return DiffusionSchedule(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device,
        )


def get_alpha_bar(schedule: DiffusionSchedule, k: torch.Tensor) -> torch.Tensor:
    """Get cumulative product of alphas up to step k.

    Wrapper function for compatibility.

    Args:
        schedule: Diffusion schedule.
        k: Timestep indices.

    Returns:
        ᾱ_k values.
    """
    return schedule.get_alpha_bar(k)
