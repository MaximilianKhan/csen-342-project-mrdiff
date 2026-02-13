#!/usr/bin/env python3
"""Analysis CLI tool for mr-Diff experiments."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.utils.logging import load_training_logs
from src.utils.visualization import (
    plot_loss_curves,
    plot_stage_losses,
    plot_predictions,
    plot_metrics_comparison,
    save_figure,
)


def plot_loss(log_dir: str, output_dir: str, smoothing: float = 0.6):
    """Generate loss curve plots from training logs."""
    logs = load_training_logs(log_dir)
    metrics_df = logs.get("metrics")

    if metrics_df is None or len(metrics_df) == 0:
        print(f"No metrics found in {log_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot training/validation loss
    if "train_loss" in metrics_df.columns:
        train_losses = metrics_df["train_loss"].dropna().tolist()
        val_losses = (
            metrics_df["val_loss"].dropna().tolist()
            if "val_loss" in metrics_df.columns
            else None
        )

        fig = plot_loss_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            title="Training Loss",
            smoothing=smoothing,
        )
        save_figure(fig, str(output_path / "loss_curves"))
        print(f"Saved: {output_path / 'loss_curves.png'}")

    # Plot per-stage losses if available
    stage_cols = [c for c in metrics_df.columns if c.startswith("train/stage_")]
    if stage_cols:
        stage_losses = {}
        for col in stage_cols:
            stage_num = int(col.split("_")[1])
            stage_losses[stage_num] = metrics_df[col].dropna().tolist()

        if stage_losses:
            fig = plot_stage_losses(
                stage_losses=stage_losses,
                title="Per-Stage Losses",
            )
            save_figure(fig, str(output_path / "stage_losses"))
            print(f"Saved: {output_path / 'stage_losses.png'}")


def plot_preds(predictions_path: str, output_dir: str, num_samples: int = 10):
    """Generate prediction visualizations."""
    data = torch.load(predictions_path, map_location="cpu")
    predictions = data["predictions"]
    targets = data["targets"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, len(predictions))

    for i in range(num_samples):
        fig = plot_predictions(
            ground_truth=targets[i].numpy(),
            predictions=predictions[i].numpy(),
            title=f"Sample {i + 1}",
        )
        save_figure(fig, str(output_path / f"prediction_{i + 1}"))

    print(f"Saved {num_samples} prediction plots to {output_path}")


def compare_runs(run_dirs: List[str], output_dir: str):
    """Compare metrics across multiple training runs."""
    results = {}

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        run_name = run_path.name

        # Try to load metrics
        metrics_path = run_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            results[run_name] = metrics
        else:
            # Try loading from training logs
            logs = load_training_logs(run_dir)
            metrics_df = logs.get("metrics")
            if metrics_df is not None and len(metrics_df) > 0:
                # Get final metrics
                final_row = metrics_df.iloc[-1]
                results[run_name] = {
                    "train_loss": final_row.get("train_loss", float("nan")),
                    "val_loss": final_row.get("val_loss", float("nan")),
                    "val_mae": final_row.get("val_mae", float("nan")),
                }

    if not results:
        print("No metrics found in any of the provided directories")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create comparison plot
    fig = plot_metrics_comparison(
        metrics=results,
        title="Run Comparison",
    )
    save_figure(fig, str(output_path / "comparison"))
    print(f"Saved: {output_path / 'comparison.png'}")

    # Save comparison table
    df = pd.DataFrame(results).T
    df.to_csv(output_path / "comparison.csv")
    print(f"Saved: {output_path / 'comparison.csv'}")

    # Print table
    print("\nComparison:")
    print(df.to_string())


def export_latex(metrics_path: str, output_path: str):
    """Export metrics to LaTeX table format."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Format as LaTeX table
    latex = r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Metric & Value & Std \\
\midrule
"""

    for key in ["mae", "mse", "rmse"]:
        value = metrics.get(key, float("nan"))
        std = metrics.get(f"{key}_std", 0.0)
        latex += f"{key.upper()} & {value:.4f} & {std:.4f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\caption{Model Performance}
\label{tab:results}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze mr-Diff experiments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plot loss command
    loss_parser = subparsers.add_parser("plot-loss", help="Plot loss curves")
    loss_parser.add_argument("--log-dir", type=str, required=True, help="Log directory")
    loss_parser.add_argument(
        "--output-dir", type=str, default="analysis", help="Output directory"
    )
    loss_parser.add_argument(
        "--smoothing", type=float, default=0.6, help="EMA smoothing factor"
    )

    # Plot predictions command
    pred_parser = subparsers.add_parser(
        "plot-predictions", help="Plot prediction visualizations"
    )
    pred_parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions.pt"
    )
    pred_parser.add_argument(
        "--output-dir", type=str, default="analysis", help="Output directory"
    )
    pred_parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to plot"
    )

    # Compare runs command
    compare_parser = subparsers.add_parser(
        "compare-runs", help="Compare multiple training runs"
    )
    compare_parser.add_argument(
        "--runs", type=str, nargs="+", required=True, help="List of run directories"
    )
    compare_parser.add_argument(
        "--output-dir", type=str, default="analysis", help="Output directory"
    )

    # Export LaTeX command
    latex_parser = subparsers.add_parser(
        "export-metrics", help="Export metrics to LaTeX"
    )
    latex_parser.add_argument(
        "--metrics", type=str, required=True, help="Path to metrics.json"
    )
    latex_parser.add_argument(
        "--output", type=str, default="results.tex", help="Output .tex file"
    )

    args = parser.parse_args()

    if args.command == "plot-loss":
        plot_loss(args.log_dir, args.output_dir, args.smoothing)
    elif args.command == "plot-predictions":
        plot_preds(args.predictions, args.output_dir, args.num_samples)
    elif args.command == "compare-runs":
        compare_runs(args.runs, args.output_dir)
    elif args.command == "export-metrics":
        export_latex(args.metrics, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
