"""Training loop for mr-Diff."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_mae, compute_mse, evaluate_model
from ..utils.logging import TrainingLogger
from .scheduler import EarlyStopping


class Trainer:
    """Trainer for mr-Diff model.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """Initialize the trainer.

        Args:
            model: The mr-Diff model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Configuration dictionary.
            device: Device to train on.
            checkpoint_dir: Directory for saving checkpoints.
            log_dir: Directory for logs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)

        # Training config
        train_config = config.get("training", {})
        self.max_epochs = train_config.get("max_epochs", 100)
        self.min_epochs = train_config.get("min_epochs", 30)
        self.learning_rate = train_config.get("learning_rate", 1e-3)
        self.weight_decay = train_config.get("weight_decay", 1e-4)
        self.grad_clip_norm = train_config.get("grad_clip_norm", 1.0)
        self.mixup_prob = train_config.get("mixup_prob", 0.5)

        # Logging config
        log_config = config.get("logging", {})
        self.log_every_n_steps = log_config.get("log_every_n_steps", 100)
        self.save_every_n_epochs = log_config.get("save_every_n_epochs", 5)

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer — AdamW decouples weight decay from gradient updates
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing LR schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs,
        )

        # Mixed precision training -- DISABLED
        # AMP causes precision issues in diffusion schedule computations
        # (float16 corrupts alpha_bar values and MSE loss gradients)
        self.use_amp = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        patience = train_config.get("early_stopping_patience", 10)
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        # Logger
        exp_name = config.get("experiment", {}).get("name", "mr_diff")
        self.logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=exp_name,
            config=config,
            use_tensorboard=log_config.get("tensorboard", True),
            use_csv=log_config.get("csv_logging", True),
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Dictionary with final metrics.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Set epoch on model for scheduled sampling
            self.model.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation (fast mode skips expensive sampling during training)
            val_metrics = self._validate(fast_mode=True)

            # Log epoch metrics
            self.logger.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "val_mse": val_metrics["mse"],
                },
                epoch=epoch,
            )

            # Log stage losses
            if "stage_losses" in train_metrics:
                self.logger.log_stage_losses(train_metrics["stage_losses"])

            # Print progress
            print(
                f"Epoch {epoch + 1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )

            # Checkpointing
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best.pt", val_metrics)

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt", val_metrics)

            # Step LR scheduler
            self.scheduler.step()

            # Early stopping (only after min_epochs)
            if epoch >= self.min_epochs - 1:
                if self.early_stopping(val_metrics["loss"]):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Save final checkpoint
        self._save_checkpoint("final.pt", val_metrics)
        self.logger.close()

        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.current_epoch,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        epoch_stage_losses = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            lookback = batch["lookback"].to(self.device, non_blocking=True)
            forecast = batch["forecast"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(lookback, forecast, mixup_prob=self.mixup_prob)
                loss = outputs["loss"]

            stage_losses = outputs.get("stage_losses", {})

            # Backward pass with grad scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            for s, s_loss in stage_losses.items():
                if s not in epoch_stage_losses:
                    epoch_stage_losses[s] = 0.0
                epoch_stage_losses[s] += s_loss.item()

            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                self.logger.log_metrics(
                    {"batch_loss": loss.item()},
                    step=self.global_step,
                    prefix="train",
                )
                self.logger.log_learning_rate(
                    self.optimizer.param_groups[0]["lr"],
                    step=self.global_step,
                )
                self.logger.log_gradient_norms(self.model, step=self.global_step)
                self.logger.log_gpu_memory(step=self.global_step)

            self.global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_stage_losses = {s: l / num_batches for s, l in epoch_stage_losses.items()}

        return {
            "loss": avg_loss,
            "stage_losses": avg_stage_losses,
        }

    @torch.no_grad()
    def _validate(self, fast_mode: bool = True) -> Dict[str, float]:
        """Run validation.

        Args:
            fast_mode: If True, skip expensive sampling and only compute loss.

        Returns:
            Dictionary with validation metrics.
        """
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            lookback = batch["lookback"].to(self.device, non_blocking=True)
            forecast = batch["forecast"].to(self.device, non_blocking=True)

            # Compute loss (need model in train mode for loss computation)
            self.model.train()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(lookback, forecast, mixup_prob=0.0)
                loss = outputs["loss"]
            total_loss += loss.item()

            if not fast_mode:
                # Full sampling is expensive (100 diffusion steps per batch)
                # Only do this for final evaluation, not during training
                self.model.eval()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self.model.sample(lookback, num_samples=1)
                mae = compute_mae(predictions, forecast)
                mse = compute_mse(predictions, forecast)
                total_mae += mae.item()
                total_mse += mse.item()

            num_batches += 1

        result = {"loss": total_loss / num_batches}
        if not fast_mode:
            result["mae"] = total_mae / num_batches
            result["mse"] = total_mse / num_batches
        else:
            result["mae"] = 0.0  # Placeholder in fast mode
            result["mse"] = 0.0

        return result

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """Save a checkpoint.

        Args:
            filename: Checkpoint filename.
            metrics: Current metrics to save.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.save_checkpoint_metadata(str(checkpoint_path), metrics, self.current_epoch)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def resume_training(self, checkpoint_path: str) -> Dict[str, float]:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.

        Returns:
            Final training metrics.
        """
        self.load_checkpoint(checkpoint_path)
        self.current_epoch += 1  # Start from next epoch
        return self.train()
