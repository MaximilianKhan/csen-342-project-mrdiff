"""Generate report figures from experiment results.

Usage in Jupyter notebook:
    from generate_figures import figure1_loss_curves, figure2_forecasts, figure3_mae_comparison
    figure1_loss_curves()
    figure2_forecasts()
    figure3_mae_comparison()
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

try:
    EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
except NameError:
    EXPERIMENTS_DIR = Path.cwd() / "experiments"

BEST_RUN = "run_20260212_082858"

EXPERIMENTS = ["ETTh1_multi", "ETTh1_uni", "ETTm1_multi", "ETTm1_uni"]

EXPERIMENT_LABELS = {
    "ETTh1_multi": "ETTh1 Multivariate",
    "ETTh1_uni": "ETTh1 Univariate",
    "ETTm1_multi": "ETTm1 Multivariate",
    "ETTm1_uni": "ETTm1 Univariate",
}

PAPER_VALUES = {
    "ETTh1_multi": {"mae": 0.422, "mse": 0.411},
    "ETTh1_uni":   {"mae": 0.196, "mse": 0.066},
    "ETTm1_multi": {"mae": 0.373, "mse": 0.340},
    "ETTm1_uni":   {"mae": 0.149, "mse": 0.039},
}


def _load_loss_curve(run_dir: str, experiment: str) -> list:
    """Load validation loss history from checkpoint JSONs.

    Returns list of (epoch, val_loss) tuples sorted by epoch.
    """
    ckpt_dir = EXPERIMENTS_DIR / run_dir / experiment / "checkpoints"
    points = []

    for json_path in sorted(ckpt_dir.glob("*.json")):
        name = json_path.stem
        # Skip best.json (duplicate of an epoch) to avoid overlapping points
        if name == "best":
            continue
        with open(json_path) as f:
            data = json.load(f)
        epoch = data["epoch"]
        loss = data["metrics"]["loss"]
        points.append((epoch, loss))

    # Sort by epoch and deduplicate
    points.sort(key=lambda x: x[0])
    seen = set()
    unique = []
    for ep, loss in points:
        if ep not in seen:
            seen.add(ep)
            unique.append((ep, loss))
    return unique


# ---------------------------------------------------------------------------
# Figure 1: Validation Loss Curves
# ---------------------------------------------------------------------------

def figure1_loss_curves():
    """Figure 1: Validation loss curves for all 4 experiments (best run).

    Shows loss progression over training epochs with best checkpoint marked.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Figure 1: Validation Loss During Training (Best Run)",
                 fontsize=14, fontweight="bold", y=0.98)

    colors = ["#2c3e50", "#e74c3c", "#2980b9", "#27ae60"]

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx // 2][idx % 2]
        curve = _load_loss_curve(BEST_RUN, exp)

        if not curve:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(EXPERIMENT_LABELS[exp])
            continue

        epochs = [p[0] + 1 for p in curve]  # 1-indexed for display
        losses = [p[1] for p in curve]

        # Plot loss curve
        ax.plot(epochs, losses, color=colors[idx], linewidth=1.8,
                marker="o", markersize=4, label="Val Loss")

        # Mark best checkpoint
        best_idx = np.argmin(losses)
        ax.plot(epochs[best_idx], losses[best_idx], marker="*",
                markersize=15, color="#f39c12", zorder=5,
                label=f"Best (ep {epochs[best_idx]}, {losses[best_idx]:.4f})")

        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Sample Forecast Visualizations
# ---------------------------------------------------------------------------

def figure2_forecasts(sample_num: int = 1):
    """Figure 2: Sample forecast visualizations (2x2 grid).

    Loads existing forecast plots from the best run's evaluation output.

    Args:
        sample_num: Which sample to display (1-5).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Figure 2: Forecast Predictions vs Ground Truth (Sample {sample_num})",
                 fontsize=14, fontweight="bold", y=0.98)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx // 2][idx % 2]
        img_path = (EXPERIMENTS_DIR / BEST_RUN / exp /
                    "evaluation" / "plots" / f"sample_{sample_num}.png")

        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(EXPERIMENT_LABELS[exp], fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Plot not found:\n{img_path.name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(EXPERIMENT_LABELS[exp], fontsize=11)

        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: MAE Comparison (Ours vs Paper)
# ---------------------------------------------------------------------------

def figure3_mae_comparison():
    """Figure 3: Bar chart comparing our MAE vs the paper's reported MAE."""
    # Load best run results
    results_path = EXPERIMENTS_DIR / BEST_RUN / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    labels = []
    ours_mae = []
    paper_mae = []

    for exp in EXPERIMENTS:
        labels.append(EXPERIMENT_LABELS[exp].replace(" ", "\n"))
        ours_mae.append(results["experiments"][exp]["evaluation"]["mae"])
        paper_mae.append(PAPER_VALUES[exp]["mae"])

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width / 2, paper_mae, width, label="Paper (mr-Diff)",
                   color="#2ecc71", edgecolor="#27ae60", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, ours_mae, width, label="Ours (Baseline)",
                   color="#e74c3c", edgecolor="#c0392b", linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

    # Add gap annotations
    for i in range(len(labels)):
        gap = ours_mae[i] / paper_mae[i]
        ax.annotate(f"{gap:.1f}x", xy=(x[i], max(ours_mae[i], paper_mae[i]) + 0.08),
                    ha="center", fontsize=9, fontweight="bold", color="#7f8c8d")

    ax.set_ylabel("MAE (lower is better)", fontsize=11)
    ax.set_title("Figure 3: MAE Comparison -- Our Baseline vs. Paper",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(ours_mae) * 1.2)

    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def all_figures():
    """Generate and display all report figures."""
    figure1_loss_curves()
    figure2_forecasts()
    figure3_mae_comparison()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    all_figures()
    print("Figures generated.")
