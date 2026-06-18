"""Diffusion process: forward diffusion, variance schedules, step embeddings."""

import math
from typing import Tuple

import torch
import torch.nn as nn


class DiffusionSchedule:
    """Variance schedule with precomputed alpha values for forward/reverse diffusion."""

    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.1,
                 schedule_type="linear", device=None):
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")

        if schedule_type == "cosine":
            s = 0.008
            steps = torch.linspace(0, num_steps, num_steps + 1, device=self.device)
            f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = f / f[0]
            alpha_bars = alpha_bars[1:]
            alpha_bars_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bars[:-1]])
            self.betas = (1 - alpha_bars / alpha_bars_prev).clamp(max=0.999)
            self.alpha_bars = alpha_bars
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)

        self.alphas = 1.0 - self.betas

        if not hasattr(self, 'alpha_bars') or schedule_type != "cosine":
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def to(self, device):
        self.device = device
        for attr in ['betas', 'alphas', 'alpha_bars', 'sqrt_alpha_bars',
                      'sqrt_one_minus_alpha_bars', 'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def get_alpha_bar(self, k):
        return self.alpha_bars[k]

    def get_sqrt_alpha_bar(self, k):
        return self.sqrt_alpha_bars[k]

    def get_sqrt_one_minus_alpha_bar(self, k):
        return self.sqrt_one_minus_alpha_bars[k]


def forward_diffusion(y0, k, schedule, noise=None):
    """Y^k = sqrt(alpha_bar_k) * Y^0 + sqrt(1-alpha_bar_k) * eps"""
    if noise is None:
        noise = torch.randn_like(y0)

    sqrt_alpha_bar = schedule.get_sqrt_alpha_bar(k)
    sqrt_one_minus_alpha_bar = schedule.get_sqrt_one_minus_alpha_bar(k)

    while sqrt_alpha_bar.dim() < y0.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

    yk = sqrt_alpha_bar * y0 + sqrt_one_minus_alpha_bar * noise
    return yk, noise


class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps, followed by two FC layers."""

    def __init__(self, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, k):
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=k.device, dtype=torch.float32) * -emb_scale)
        emb = k.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.fc(emb)


def get_diffusion_step_embedding(k, embedding_dim=128):
    """Standalone sinusoidal embedding (no learned parameters)."""
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=k.device, dtype=torch.float32) * -emb_scale)
    emb = k.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class LinearBetaSchedule:
    """Simple wrapper for trainer compatibility."""

    def __init__(self, num_steps, beta_start, beta_end):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_schedule(self, device=None):
        return DiffusionSchedule(
            num_steps=self.num_steps, beta_start=self.beta_start,
            beta_end=self.beta_end, device=device,
        )


def get_alpha_bar(schedule, k):
    return schedule.get_alpha_bar(k)
