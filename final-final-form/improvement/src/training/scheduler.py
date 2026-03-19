"""Learning rate schedulers and early stopping."""

import math
import torch


class CosineAnnealingSchedule:
    """Thin wrapper around PyTorch's CosineAnnealingLR."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min)

    def step(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]


class WarmupCosineSchedule:
    """Cosine decay with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
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
        self.counter = 0
        self.best_score = None
        self.should_stop = False
