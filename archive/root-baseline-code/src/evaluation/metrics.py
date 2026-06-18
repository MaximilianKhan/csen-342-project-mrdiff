"""Evaluation metrics for mr-Diff.

Implements MAE, MSE, and full model evaluation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def compute_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Mean Absolute Error.

    Args:
        pred: Predictions [B, T, D] or [B, T].
        target: Ground truth [B, T, D] or [B, T].
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        MAE value(s).
    """
    error = torch.abs(pred - target)

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    else:
        return error


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Mean Squared Error.

    Args:
        pred: Predictions [B, T, D] or [B, T].
        target: Ground truth [B, T, D] or [B, T].
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        MSE value(s).
    """
    error = (pred - target) ** 2

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    else:
        return error


def compute_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute Root Mean Squared Error.

    Args:
        pred: Predictions.
        target: Ground truth.

    Returns:
        RMSE value.
    """
    return torch.sqrt(compute_mse(pred, target, reduction="mean"))


def compute_mape(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Mean Absolute Percentage Error.

    Args:
        pred: Predictions.
        target: Ground truth.
        eps: Small constant to avoid division by zero.

    Returns:
        MAPE value.
    """
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean() * 100


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler=None,
    num_samples: int = 10,
    device: torch.device = None,
    return_predictions: bool = False,
    **sample_kwargs,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader for evaluation data.
        scaler: Optional scaler for inverse transformation.
        num_samples: Number of random trajectories to average over.
        device: Device to run on.
        return_predictions: Whether to return all predictions.
        **sample_kwargs: Extra kwargs passed to model.sample()
            (e.g. solver, solver_steps, aggregation).

    Returns:
        Dictionary with evaluation metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_preds = []
    all_targets = []
    all_mae = []
    all_mse = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        lookback = batch["lookback"].to(device, non_blocking=True)
        target = batch["forecast"].to(device, non_blocking=True)
        norm_mean = batch["norm_mean"].to(device, non_blocking=True)  # [B, D]
        norm_std = batch["norm_std"].to(device, non_blocking=True)    # [B, D]

        # Generate multiple samples and average
        samples = []
        for _ in range(num_samples):
            pred = model.sample(lookback, num_samples=1, **sample_kwargs)
            samples.append(pred)

        # Average predictions
        pred_mean = torch.stack(samples).mean(dim=0)

        # Inverse RevIN: convert from per-window normalized to original scale
        pred_orig = pred_mean * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
        target_orig = target * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)

        # Apply global standardization (training set mean/std) for comparable metrics
        if scaler is not None:
            pred_eval = scaler.transform(pred_orig)
            target_eval = scaler.transform(target_orig)
        else:
            pred_eval = pred_orig
            target_eval = target_orig

        # Compute metrics in globally-standardized space (matches paper convention)
        mae = compute_mae(pred_eval, target_eval, reduction="none")
        mse = compute_mse(pred_eval, target_eval, reduction="none")

        all_mae.append(mae.mean(dim=(1, 2)).cpu())  # Per-sample MAE
        all_mse.append(mse.mean(dim=(1, 2)).cpu())  # Per-sample MSE

        if return_predictions:
            all_preds.append(pred_eval.cpu())
            all_targets.append(target_eval.cpu())

    # Concatenate all batches
    all_mae = torch.cat(all_mae)
    all_mse = torch.cat(all_mse)

    # Compute statistics
    mae_mean = all_mae.mean().item()
    mae_std = all_mae.std().item()
    mse_mean = all_mse.mean().item()
    mse_std = all_mse.std().item()

    results = {
        "mae": mae_mean,
        "mae_std": mae_std,
        "mse": mse_mean,
        "mse_std": mse_std,
        "rmse": np.sqrt(mse_mean),
        "num_samples": len(all_mae),
    }

    if return_predictions:
        results["predictions"] = torch.cat(all_preds)
        results["targets"] = torch.cat(all_targets)

    return results


def compute_confidence_interval(
    values: torch.Tensor,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval for a set of values.

    Args:
        values: Tensor of values.
        confidence: Confidence level (default 95%).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    mean = values.mean().item()
    std = values.std().item()
    n = len(values)

    # Z-score for confidence level
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    margin = z * (std / np.sqrt(n))

    return mean, mean - margin, mean + margin


def format_metrics(
    metrics: Dict[str, float],
    precision: int = 4,
) -> str:
    """Format metrics dictionary as a string.

    Args:
        metrics: Dictionary of metrics.
        precision: Decimal precision.

    Returns:
        Formatted string.
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"  {key}: {value:.{precision}f}")
    return "\n".join(lines)


def compare_metrics(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
) -> str:
    """Compare metrics across multiple experiments.

    Args:
        results: Dictionary mapping experiment name to metrics.
        metric_names: Metrics to compare (default: all).

    Returns:
        Formatted comparison table.
    """
    if metric_names is None:
        metric_names = ["mae", "mse", "rmse"]

    # Header
    header = f"{'Experiment':<30}" + "".join(f"{m:<15}" for m in metric_names)
    lines = [header, "-" * len(header)]

    # Rows
    for exp_name, metrics in results.items():
        row = f"{exp_name:<30}"
        for m in metric_names:
            value = metrics.get(m, float("nan"))
            row += f"{value:<15.4f}"
        lines.append(row)

    return "\n".join(lines)
