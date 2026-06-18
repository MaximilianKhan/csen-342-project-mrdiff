"""Learning rate and variance schedulers for mr-Diff training."""

from typing import Optional

import torch


class LinearBetaSchedule:
    """Linear variance schedule for diffusion.

    Provides beta values from beta_start to beta_end.
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
    ):
        """Initialize linear beta schedule.

        Args:
            num_steps: Number of diffusion steps K.
            beta_start: Starting variance β₁.
            beta_end: Ending variance βK.
        """
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_betas(self, device: torch.device = None) -> torch.Tensor:
        """Get beta values.

        Args:
            device: Device to place tensor on.

        Returns:
            Tensor of beta values [K].
        """
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_steps,
            device=device,
        )

    def get_alphas(self, device: torch.device = None) -> torch.Tensor:
        """Get alpha values (1 - beta).

        Args:
            device: Device to place tensor on.

        Returns:
            Tensor of alpha values [K].
        """
        return 1.0 - self.get_betas(device)


def get_alpha_bar(
    alphas: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Get cumulative product of alphas up to step k.

    ᾱ_k = ∏_{s=1}^{k} α_s

    Args:
        alphas: Alpha values [K].
        k: Timestep indices [B].

    Returns:
        ᾱ_k values [B].
    """
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars[k]


class CosineAnnealingSchedule:
    """Cosine annealing learning rate schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ):
        """Initialize cosine annealing schedule.

        Args:
            optimizer: Optimizer to schedule.
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )

    def step(self):
        """Take a scheduler step."""
        self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]


class WarmupCosineSchedule:
    """Cosine schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
    ):
        """Initialize warmup + cosine schedule.

        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of steps.
            eta_min: Minimum learning rate after decay.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Take a scheduler step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            import math
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' for optimization direction.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if should stop.

        Args:
            score: Current metric value.

        Returns:
            True if should stop training.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
