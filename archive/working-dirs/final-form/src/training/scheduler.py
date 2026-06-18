"""Learning rate schedulers and early stopping for mr-Diff training."""

import math

import torch


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
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
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
