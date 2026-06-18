"""Visualization utilities for forecasting results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6), "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 14,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def plot_predictions(ground_truth, predictions, lookback=None,
                     title="Forecast vs Ground Truth", save_path=None, feature_idx=0):
    set_style()
    fig, ax = plt.subplots()

    if ground_truth.ndim > 1:
        ground_truth = ground_truth[:, feature_idx]
        predictions = predictions[:, feature_idx]
        if lookback is not None: lookback = lookback[:, feature_idx]

    if lookback is not None:
        L = len(lookback)
        ax.plot(range(L), lookback, label="History", color="gray", alpha=0.7)
        fx = range(L, L + len(ground_truth))
        ax.plot(fx, ground_truth, label="Ground Truth", color="blue")
        ax.plot(fx, predictions, label="Prediction", color="red", linestyle="--")
        ax.axvline(x=L, color="black", linestyle=":", alpha=0.5)
    else:
        ax.plot(ground_truth, label="Ground Truth", color="blue")
        ax.plot(predictions, label="Prediction", color="red", linestyle="--")

    ax.set_xlabel("Time Step"); ax.set_ylabel("Value")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curves(train_losses, val_losses=None, title="Training Loss",
                     save_path=None, smoothing=0.0):
    set_style()
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)

    def smooth(vals, f):
        if f <= 0: return vals
        out, last = [], vals[0]
        for v in vals:
            last = last * f + v * (1 - f)
            out.append(last)
        return out

    if smoothing > 0:
        ax.plot(epochs, train_losses, alpha=0.3, color="blue")
        ax.plot(epochs, smooth(train_losses, smoothing), label="Train (smoothed)", color="blue")
        if val_losses:
            ax.plot(epochs, val_losses, alpha=0.3, color="orange")
            ax.plot(epochs, smooth(val_losses, smoothing), label="Val (smoothed)", color="orange")
    else:
        ax.plot(epochs, train_losses, label="Train Loss", color="blue")
        if val_losses:
            ax.plot(epochs, val_losses, label="Val Loss", color="orange")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stage_losses(stage_losses, title="Per-Stage Losses", save_path=None):
    set_style()
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(stage_losses)))
    for (stage, losses), color in zip(sorted(stage_losses.items()), colors):
        ax.plot(range(1, len(losses) + 1), losses, label=f"Stage {stage}", color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diffusion_process(diffusion_steps, step_indices,
                           title="Diffusion Process", save_path=None, feature_idx=0):
    """Visualize the denoising process across diffusion steps."""
    set_style()
    n = len(diffusion_steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, data, step in zip(axes, diffusion_steps, step_indices):
        if data.ndim > 1: data = data[:, feature_idx]
        ax.plot(data, color="blue")
        ax.set_title(f"Step k={step}"); ax.set_xlabel("Time"); ax.set_ylabel("Value")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trend_decomposition(original, trends, residuals,
                             title="Multi-Resolution Trend Decomposition",
                             save_path=None, feature_idx=0):
    set_style()
    n_stages = len(trends)
    fig, axes = plt.subplots(n_stages + 1, 1, figsize=(12, 3 * (n_stages + 1)))

    if original.ndim > 1:
        original = original[:, feature_idx]
        trends = [t[:, feature_idx] if t.ndim > 1 else t for t in trends]
        residuals = [r[:, feature_idx] if r.ndim > 1 else r for r in residuals]

    axes[0].plot(original, color="black", label="Original")
    axes[0].set_title("Original Signal"); axes[0].legend()
    for i, (trend, res) in enumerate(zip(trends, residuals)):
        axes[i + 1].plot(trend, label=f"Trend {i}", color="blue", alpha=0.7)
        axes[i + 1].plot(res, label=f"Residual {i}", color="red", alpha=0.7)
        axes[i + 1].set_title(f"Stage {i}"); axes[i + 1].legend()
    for ax in axes:
        ax.set_xlabel("Time"); ax.set_ylabel("Value")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]],
                            title="Model Comparison", save_path=None):
    set_style()
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [metrics[model][m] for m in metric_names]
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, vals, width, label=model, color=color)
    ax.set_xlabel("Metric"); ax.set_ylabel("Value")
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(metric_names); ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_figure(fig, path, formats=("png", "pdf")):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(path.with_suffix(f".{fmt}")), dpi=150, bbox_inches="tight")
