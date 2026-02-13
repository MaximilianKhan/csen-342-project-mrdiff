#!/usr/bin/env python3
"""Evaluation script for mr-Diff."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.dataset import create_dataloaders
from src.evaluation.metrics import evaluate_model, format_metrics, compare_metrics
from src.models.mr_diff import create_model
from src.utils.visualization import plot_predictions, save_figure


def load_checkpoint(checkpoint_path: str, model, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    config = checkpoint.get("config", {})
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate mr-Diff model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (if not in checkpoint)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--univariate",
        action="store_true",
        help="Use univariate mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to average over",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=5,
        help="Number of samples to plot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--lookback-length",
        type=int,
        default=None,
        help="Lookback window length (overrides config)",
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("config", {})
        if not config:
            raise ValueError(
                "No config found in checkpoint. Please provide --config."
            )

    # Override with args
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.univariate:
        config["data"]["univariate"] = True
    if args.lookback_length:
        config["data"]["lookback_length"] = args.lookback_length

    print(f"Evaluating on {config['data'].get('dataset', 'ETTh1')}")
    print(f"Univariate: {config['data'].get('univariate', False)}")

    # Create dataloaders
    data_config = config["data"]
    _, _, test_loader, scaler = create_dataloaders(
        data_path=data_config.get("data_path", "data/ETDataset"),
        dataset_name=data_config.get("dataset", "ETTh1"),
        lookback_length=data_config.get("lookback_length", 336),
        forecast_length=data_config.get("forecast_length", 168),
        batch_size=data_config.get("batch_size", 64),
        num_workers=data_config.get("num_workers", 4),
        univariate=data_config.get("univariate", False),
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Create and load model
    model = create_model(config)
    model, _ = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")

    # Evaluate
    print(f"\nEvaluating with {args.num_samples} samples...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        scaler=scaler,
        num_samples=args.num_samples,
        device=device,
        return_predictions=args.save_predictions or args.plot_samples > 0,
    )

    # Print results
    print("\nResults:")
    print(format_metrics(results))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "dataset": data_config.get("dataset", "ETTh1"),
        "univariate": data_config.get("univariate", False),
        "num_samples": args.num_samples,
        "checkpoint": args.checkpoint,
        "mae": results["mae"],
        "mae_std": results["mae_std"],
        "mse": results["mse"],
        "mse_std": results["mse_std"],
        "rmse": results["rmse"],
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_dir / 'metrics.json'}")

    # Save predictions
    if args.save_predictions and "predictions" in results:
        torch.save(
            {
                "predictions": results["predictions"],
                "targets": results["targets"],
            },
            output_dir / "predictions.pt",
        )
        print(f"Predictions saved to {output_dir / 'predictions.pt'}")

    # Plot samples
    if args.plot_samples > 0 and "predictions" in results:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        predictions = results["predictions"]
        targets = results["targets"]

        num_plots = min(args.plot_samples, len(predictions))

        for i in range(num_plots):
            pred = predictions[i].numpy()
            target = targets[i].numpy()

            fig = plot_predictions(
                ground_truth=target,
                predictions=pred,
                title=f"Sample {i + 1}",
            )

            save_figure(fig, str(plots_dir / f"sample_{i + 1}"))
            print(f"Saved plot: {plots_dir / f'sample_{i + 1}.png'}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
