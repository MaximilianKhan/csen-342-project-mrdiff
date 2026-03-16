"""Visualization utilities for mr-Diff.

Provides plotting functions for predictions, losses, and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def set_style():
    """Set consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_predictions(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    lookback: Optional[np.ndarray] = None,
    title: str = "Forecast vs Ground Truth",
    save_path: Optional[str] = None,
    feature_idx: int = 0,
) -> plt.Figure:
    """Plot ground truth vs predicted values.

    Args:
        ground_truth: Ground truth values [T] or [T, D].
        predictions: Predicted values [T] or [T, D].
        lookback: Optional lookback window to show context.
        title: Plot title.
        save_path: Path to save the figure.
        feature_idx: Feature index to plot for multivariate data.

    Returns:
        Matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots()

    # Handle multivariate data
    if ground_truth.ndim > 1:
        ground_truth = ground_truth[:, feature_idx]
        predictions = predictions[:, feature_idx]
        if lookback is not None:
            lookback = lookback[:, feature_idx]

    forecast_len = len(ground_truth)

    if lookback is not None:
        lookback_len = len(lookback)
        # Plot lookback
        ax.plot(
            range(lookback_len),
            lookback,
            label="History",
            color="gray",
            alpha=0.7,
        )
        # Plot forecast
        forecast_x = range(lookback_len, lookback_len + forecast_len)
        ax.plot(forecast_x, ground_truth, label="Ground Truth", color="blue")
        ax.plot(forecast_x, predictions, label="Prediction", color="red", linestyle="--")
        ax.axvline(x=lookback_len, color="black", linestyle=":", alpha=0.5)
    else:
        ax.plot(ground_truth, label="Ground Truth", color="blue")
        ax.plot(predictions, label="Prediction", color="red", linestyle="--")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    smoothing: float = 0.0,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses.
        val_losses: Optional list of validation losses.
        title: Plot title.
        save_path: Path to save the figure.
        smoothing: Exponential moving average smoothing factor (0-1).

    Returns:
        Matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots()

    def smooth(values, factor):
        if factor <= 0:
            return values
        smoothed = []
        last = values[0]
        for v in values:
            smoothed_val = last * factor + v * (1 - factor)
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    epochs = range(1, len(train_losses) + 1)

    if smoothing > 0:
        train_smooth = smooth(train_losses, smoothing)
        ax.plot(epochs, train_losses, alpha=0.3, color="blue")
        ax.plot(epochs, train_smooth, label="Train Loss (smoothed)", color="blue")

        if val_losses:
            val_smooth = smooth(val_losses, smoothing)
            ax.plot(epochs, val_losses, alpha=0.3, color="orange")
            ax.plot(epochs, val_smooth, label="Val Loss (smoothed)", color="orange")
    else:
        ax.plot(epochs, train_losses, label="Train Loss", color="blue")
        if val_losses:
            ax.plot(epochs, val_losses, label="Val Loss", color="orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stage_losses(
    stage_losses: Dict[int, List[float]],
    title: str = "Per-Stage Losses",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot loss curves for each stage.

    Args:
        stage_losses: Dictionary mapping stage index to list of losses.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots()

    colors = plt.cm.viridis(np.linspace(0, 1, len(stage_losses)))

    for (stage, losses), color in zip(sorted(stage_losses.items()), colors):
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label=f"Stage {stage}", color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_diffusion_process(
    diffusion_steps: List[np.ndarray],
    step_indices: List[int],
    title: str = "Diffusion Process",
    save_path: Optional[str] = None,
    feature_idx: int = 0,
) -> plt.Figure:
    """Visualize the denoising process across diffusion steps.

    Args:
        diffusion_steps: List of arrays at different diffusion steps.
        step_indices: Corresponding diffusion step indices.
        title: Plot title.
        save_path: Path to save the figure.
        feature_idx: Feature index for multivariate data.

    Returns:
        Matplotlib figure.
    """
    set_style()

    n_steps = len(diffusion_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))

    if n_steps == 1:
        axes = [axes]

    for ax, data, step in zip(axes, diffusion_steps, step_indices):
        if data.ndim > 1:
            data = data[:, feature_idx]

        ax.plot(data, color="blue")
        ax.set_title(f"Step k={step}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_trend_decomposition(
    original: np.ndarray,
    trends: List[np.ndarray],
    residuals: List[np.ndarray],
    title: str = "Multi-Resolution Trend Decomposition",
    save_path: Optional[str] = None,
    feature_idx: int = 0,
) -> plt.Figure:
    """Plot trend decomposition at each stage.

    Args:
        original: Original time series.
        trends: List of trend components at each stage.
        residuals: List of residual components at each stage.
        title: Plot title.
        save_path: Path to save the figure.
        feature_idx: Feature index for multivariate data.

    Returns:
        Matplotlib figure.
    """
    set_style()

    n_stages = len(trends)
    fig, axes = plt.subplots(n_stages + 1, 1, figsize=(12, 3 * (n_stages + 1)))

    # Handle multivariate
    if original.ndim > 1:
        original = original[:, feature_idx]
        trends = [t[:, feature_idx] if t.ndim > 1 else t for t in trends]
        residuals = [r[:, feature_idx] if r.ndim > 1 else r for r in residuals]

    # Plot original
    axes[0].plot(original, color="black", label="Original")
    axes[0].set_title("Original Signal")
    axes[0].legend()

    # Plot each stage
    for i, (trend, residual) in enumerate(zip(trends, residuals)):
        ax = axes[i + 1]
        ax.plot(trend, label=f"Trend (Stage {i})", color="blue", alpha=0.7)
        ax.plot(residual, label=f"Residual (Stage {i})", color="red", alpha=0.7)
        ax.set_title(f"Stage {i} Decomposition")
        ax.legend()

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart comparing metrics across models/experiments.

    Args:
        metrics: Nested dict {model_name: {metric_name: value}}.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    set_style()

    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [metrics[model][m] for m in metric_names]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=color)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_figure(fig: plt.Figure, path: str, formats: List[str] = ["png", "pdf"]) -> None:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure.
        path: Base path without extension.
        formats: List of file formats to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(str(path.with_suffix(f".{fmt}")), dpi=150, bbox_inches="tight")
