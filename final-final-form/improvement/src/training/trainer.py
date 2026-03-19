"""Training loop for mr-Diff baseline model."""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_mae, compute_mse
from ..utils.logging import TrainingLogger
from .scheduler import EarlyStopping


class Trainer:
    """Handles training, validation, checkpointing, and logging."""

    def __init__(self, model, train_loader, val_loader, config,
                 device=None, checkpoint_dir="checkpoints", log_dir="logs"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)

        tc = config.get("training", {})
        self.max_epochs = tc.get("max_epochs", 100)
        self.min_epochs = tc.get("min_epochs", 30)
        self.learning_rate = tc.get("learning_rate", 1e-3)
        self.weight_decay = tc.get("weight_decay", 1e-4)
        self.grad_clip_norm = tc.get("grad_clip_norm", 1.0)
        self.mixup_prob = tc.get("mixup_prob", 0.5)

        lc = config.get("logging", {})
        self.log_every_n_steps = lc.get("log_every_n_steps", 100)
        self.save_every_n_epochs = lc.get("save_every_n_epochs", 5)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs)

        self.early_stopping = EarlyStopping(
            patience=tc.get("early_stopping_patience", 10), mode="min")

        exp_name = config.get("experiment", {}).get("name", "mr_diff")
        self.logger = TrainingLogger(
            log_dir=log_dir, experiment_name=exp_name, config=config,
            use_tensorboard=lc.get("tensorboard", True),
            use_csv=lc.get("csv_logging", True))

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train(self) -> Dict[str, float]:
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self.model.current_epoch = epoch

            train_metrics = self._train_epoch()
            val_metrics = self._validate(fast_mode=True)

            self.logger.log_metrics({
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
            }, epoch=epoch)

            if "stage_losses" in train_metrics:
                self.logger.log_stage_losses(train_metrics["stage_losses"])

            print(f"Epoch {epoch + 1}/{self.max_epochs} | "
                  f"Train: {train_metrics['loss']:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best.pt", val_metrics)

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt", val_metrics)

            self.scheduler.step()

            if epoch >= self.min_epochs - 1:
                if self.early_stopping(val_metrics["loss"]):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self._save_checkpoint("final.pt", val_metrics)
        self.logger.close()
        return {"best_val_loss": self.best_val_loss, "final_epoch": self.current_epoch}

    def _train_epoch(self):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        epoch_stage_losses = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch in pbar:
            lb = batch["lookback"].to(self.device, non_blocking=True)
            fc = batch["forecast"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(lb, fc, mixup_prob=self.mixup_prob)
            loss = outputs["loss"]
            loss.backward()

            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            for s, sl in outputs.get("stage_losses", {}).items():
                epoch_stage_losses[s] = epoch_stage_losses.get(s, 0.0) + sl.item()

            if self.global_step % self.log_every_n_steps == 0:
                self.logger.log_metrics({"batch_loss": loss.item()},
                                        step=self.global_step, prefix="train")
                self.logger.log_learning_rate(
                    self.optimizer.param_groups[0]["lr"], step=self.global_step)
                self.logger.log_gradient_norms(self.model, step=self.global_step)
                self.logger.log_gpu_memory(step=self.global_step)

            self.global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        return {
            "loss": total_loss / num_batches,
            "stage_losses": {s: l / num_batches for s, l in epoch_stage_losses.items()},
        }

    @torch.no_grad()
    def _validate(self, fast_mode=True):
        total_loss, total_mae, total_mse, num_batches = 0.0, 0.0, 0.0, 0

        for batch in self.val_loader:
            lb = batch["lookback"].to(self.device, non_blocking=True)
            fc = batch["forecast"].to(self.device, non_blocking=True)

            self.model.train()
            outputs = self.model(lb, fc, mixup_prob=0.0)
            total_loss += outputs["loss"].item()

            if not fast_mode:
                self.model.eval()
                preds = self.model.sample(lb, num_samples=1)
                total_mae += compute_mae(preds, fc).item()
                total_mse += compute_mse(preds, fc).item()

            num_batches += 1

        result = {"loss": total_loss / num_batches}
        result["mae"] = (total_mae / num_batches) if not fast_mode else 0.0
        result["mse"] = (total_mse / num_batches) if not fast_mode else 0.0
        return result

    def _save_checkpoint(self, filename, metrics):
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)
        self.logger.save_checkpoint_metadata(str(path), metrics, self.current_epoch)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def resume_training(self, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        self.current_epoch += 1
        return self.train()
