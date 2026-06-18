"""Evaluation metrics for time series forecasting."""

from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def compute_mae(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    error = torch.abs(pred - target)
    if reduction == "mean": return error.mean()
    elif reduction == "sum": return error.sum()
    return error


def compute_mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    error = (pred - target) ** 2
    if reduction == "mean": return error.mean()
    elif reduction == "sum": return error.sum()
    return error


def compute_rmse(pred: torch.Tensor, target: torch.Tensor):
    return torch.sqrt(compute_mse(pred, target, reduction="mean"))


def compute_mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean() * 100


@torch.no_grad()
def evaluate_model(model, dataloader, scaler=None, num_samples=10,
                   return_predictions=False, **sample_kwargs) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Computes metrics in globally-standardized space when a scaler is provided
    (matching the paper's evaluation convention).
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_targets = [], []
    all_mae, all_mse = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        lookback = batch["lookback"].to(device, non_blocking=True)
        target = batch["forecast"].to(device, non_blocking=True)
        norm_mean = batch["norm_mean"].to(device, non_blocking=True)
        norm_std = batch["norm_std"].to(device, non_blocking=True)

        # Average over multiple stochastic samples
        samples = [model.sample(lookback, num_samples=1, **sample_kwargs)
                   for _ in range(num_samples)]
        pred_mean = torch.stack(samples).mean(dim=0)

        # Inverse RevIN then global standardization
        pred_orig = pred_mean * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
        target_orig = target * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)

        if scaler is not None:
            pred_eval = scaler.transform(pred_orig)
            target_eval = scaler.transform(target_orig)
        else:
            pred_eval, target_eval = pred_orig, target_orig

        all_mae.append(compute_mae(pred_eval, target_eval, "none").mean(dim=(1, 2)).cpu())
        all_mse.append(compute_mse(pred_eval, target_eval, "none").mean(dim=(1, 2)).cpu())

        if return_predictions:
            all_preds.append(pred_eval.cpu())
            all_targets.append(target_eval.cpu())

    all_mae = torch.cat(all_mae)
    all_mse = torch.cat(all_mse)

    results = {
        "mae": all_mae.mean().item(),
        "mae_std": all_mae.std().item(),
        "mse": all_mse.mean().item(),
        "mse_std": all_mse.std().item(),
        "rmse": np.sqrt(all_mse.mean().item()),
        "num_samples": len(all_mae),
    }
    if return_predictions:
        results["predictions"] = torch.cat(all_preds)
        results["targets"] = torch.cat(all_targets)

    return results


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    return "\n".join(f"  {k}: {v:.{precision}f}"
                     for k, v in metrics.items() if isinstance(v, (int, float)))


def compare_metrics(results: Dict[str, Dict[str, float]],
                    metric_names: List[str] = None) -> str:
    if metric_names is None:
        metric_names = ["mae", "mse", "rmse"]
    header = f"{'Experiment':<30}" + "".join(f"{m:<15}" for m in metric_names)
    lines = [header, "-" * len(header)]
    for exp_name, metrics in results.items():
        row = f"{exp_name:<30}"
        for m in metric_names:
            row += f"{metrics.get(m, float('nan')):<15.4f}"
        lines.append(row)
    return "\n".join(lines)
