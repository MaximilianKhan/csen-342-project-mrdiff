"""Training logging — TensorBoard and JSONL backends."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Logs metrics to TensorBoard and/or JSONL files."""

    def __init__(self, log_dir, experiment_name, config,
                 use_tensorboard=True, use_csv=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_csv = use_csv

        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. pip install tensorboard")

        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.metrics_file = None
        self._save_config(config)
        self.global_step = 0
        self.epoch = 0

    def _save_config(self, config):
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _write_jsonl(self, data):
        if self.metrics_file is None:
            self.metrics_file = open(self.metrics_path, "w")
        self.metrics_file.write(json.dumps(data, default=str) + "\n")
        self.metrics_file.flush()

    def log_metrics(self, metrics, step=None, epoch=None, prefix=""):
        if step is not None: self.global_step = step
        if epoch is not None: self.epoch = epoch
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if self.use_tensorboard and self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.global_step)

        if self.use_csv:
            self._write_jsonl({
                "epoch": self.epoch, "global_step": self.global_step,
                "timestamp": datetime.now().isoformat(), **metrics})

    def log_stage_losses(self, stage_losses, step=None):
        metrics = {f"stage_{s}_loss": loss for s, loss in stage_losses.items()}
        self.log_metrics(metrics, step=step, prefix="train")

    def log_gradient_norms(self, model, step=None):
        total_norm = sum(p.grad.data.norm(2).item() ** 2
                         for p in model.parameters() if p.grad is not None) ** 0.5
        self.log_metrics({"gradient_norm": total_norm}, step=step, prefix="train")

    def log_learning_rate(self, lr, step=None):
        self.log_metrics({"learning_rate": lr}, step=step, prefix="train")

    def log_gpu_memory(self, step=None):
        if torch.cuda.is_available():
            self.log_metrics({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }, step=step, prefix="system")

    def log_histogram(self, name, values, step=None):
        if self.use_tensorboard and self.writer:
            self.writer.add_histogram(name, values, step or self.global_step)

    def log_figure(self, name, figure, step=None):
        if self.use_tensorboard and self.writer:
            self.writer.add_figure(name, figure, step or self.global_step)

    def save_checkpoint_metadata(self, checkpoint_path, metrics, epoch):
        metadata = {
            "checkpoint_path": checkpoint_path, "epoch": epoch,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat(), "metrics": metrics}
        with open(Path(checkpoint_path).with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def close(self):
        if self.writer: self.writer.close()
        if self.metrics_file: self.metrics_file.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close(); return False


def load_training_logs(log_dir):
    """Load config and metrics from a log directory."""
    log_path = Path(log_dir)
    config = {}
    config_path = log_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    import pandas as pd
    metrics_df = None
    jsonl_path = log_path / "metrics.jsonl"
    csv_path = log_path / "metrics.csv"

    if jsonl_path.exists():
        records = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        if records:
            metrics_df = pd.DataFrame(records)
    elif csv_path.exists():
        metrics_df = pd.read_csv(csv_path)

    return {"config": config, "metrics": metrics_df, "log_dir": str(log_path)}
