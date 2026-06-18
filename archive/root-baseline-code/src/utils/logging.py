"""Training logging utilities for mr-Diff.

Provides TensorBoard, CSV, and JSON logging for training metrics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Try to import TensorBoard, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Comprehensive logging for training metrics.

    Supports TensorBoard, CSV, and JSON logging formats.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ):
        """Initialize the training logger.

        Args:
            log_dir: Base directory for logs.
            experiment_name: Name of the experiment.
            config: Configuration dictionary to save.
            use_tensorboard: Whether to use TensorBoard logging.
            use_csv: Whether to use CSV logging.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_csv = use_csv

        # TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")

        # Metrics file setup (JSONL format)
        self.csv_path = self.log_dir / "metrics.jsonl"
        self.csv_file = None

        # Save configuration
        self._save_config(config)

        # Track global step
        self.global_step = 0
        self.epoch = 0

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to JSON file."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _write_jsonl(self, data: Dict[str, Any]) -> None:
        """Write a line to the JSONL metrics file."""
        if self.csv_file is None:
            # Use JSONL format for flexibility with dynamic metrics
            self.csv_path = self.log_dir / "metrics.jsonl"
            self.csv_file = open(self.csv_path, "w")

        self.csv_file.write(json.dumps(data, default=str) + "\n")
        self.csv_file.flush()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names and values.
            step: Global step (uses internal counter if None).
            epoch: Current epoch number.
            prefix: Prefix to add to metric names.
        """
        if step is not None:
            self.global_step = step
        if epoch is not None:
            self.epoch = epoch

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # TensorBoard logging
        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.global_step)

        # JSONL logging (more flexible than CSV for dynamic metrics)
        if self.use_csv:
            row = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }
            self._write_jsonl(row)

    def log_stage_losses(
        self,
        stage_losses: Dict[int, float],
        step: Optional[int] = None,
    ) -> None:
        """Log per-stage losses.

        Args:
            stage_losses: Dictionary mapping stage index to loss value.
            step: Global step.
        """
        metrics = {f"stage_{s}_loss": loss for s, loss in stage_losses.items()}
        self.log_metrics(metrics, step=step, prefix="train")

    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
    ) -> None:
        """Log gradient norms for model parameters.

        Args:
            model: PyTorch model to log gradients for.
            step: Global step.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.log_metrics({"gradient_norm": total_norm}, step=step, prefix="train")

    def log_learning_rate(self, lr: float, step: Optional[int] = None) -> None:
        """Log current learning rate.

        Args:
            lr: Current learning rate.
            step: Global step.
        """
        self.log_metrics({"learning_rate": lr}, step=step, prefix="train")

    def log_gpu_memory(self, step: Optional[int] = None) -> None:
        """Log GPU memory usage if available.

        Args:
            step: Global step.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9  # GB
            reserved = torch.cuda.memory_reserved() / 1e9  # GB
            self.log_metrics(
                {"gpu_memory_allocated_gb": allocated, "gpu_memory_reserved_gb": reserved},
                step=step,
                prefix="system",
            )

    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """Log histogram to TensorBoard.

        Args:
            name: Name of the histogram.
            values: Tensor of values.
            step: Global step.
        """
        if self.use_tensorboard and self.writer is not None:
            step = step if step is not None else self.global_step
            self.writer.add_histogram(name, values, step)

    def log_figure(
        self,
        name: str,
        figure,
        step: Optional[int] = None,
    ) -> None:
        """Log matplotlib figure to TensorBoard.

        Args:
            name: Name of the figure.
            figure: Matplotlib figure object.
            step: Global step.
        """
        if self.use_tensorboard and self.writer is not None:
            step = step if step is not None else self.global_step
            self.writer.add_figure(name, figure, step)

    def save_checkpoint_metadata(
        self,
        checkpoint_path: str,
        metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Save metadata alongside a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            metrics: Metrics at the time of saving.
            epoch: Current epoch.
        """
        metadata = {
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        metadata_path = Path(checkpoint_path).with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def close(self) -> None:
        """Close all logging backends."""
        if self.writer is not None:
            self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_training_logs(log_dir: str) -> Dict[str, Any]:
    """Load training logs from a directory.

    Args:
        log_dir: Path to the log directory.

    Returns:
        Dictionary containing config and metrics.
    """
    log_path = Path(log_dir)

    # Load config
    config_path = log_path / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Load metrics (try JSONL first, then CSV for backwards compatibility)
    import pandas as pd
    metrics_df = None

    jsonl_path = log_path / "metrics.jsonl"
    csv_path = log_path / "metrics.csv"

    if jsonl_path.exists():
        # Load JSONL format
        records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if records:
            metrics_df = pd.DataFrame(records)
    elif csv_path.exists():
        # Fallback to CSV
        metrics_df = pd.read_csv(csv_path)

    return {
        "config": config,
        "metrics": metrics_df,
        "log_dir": str(log_path),
    }
