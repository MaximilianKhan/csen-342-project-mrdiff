#!/usr/bin/env python3
"""Main training script for mr-Diff."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import create_dataloaders
from src.models.mr_diff import create_model
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train mr-Diff model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
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
        help="Use univariate mode (overrides config)",
    )
    parser.add_argument(
        "--multivariate",
        action="store_true",
        help="Use multivariate mode (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints (overrides config)",
    )
    parser.add_argument(
        "--lookback-length",
        type=int,
        default=None,
        help="Lookback window length (overrides config)",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=None,
        help="Forecast horizon length (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.dataset:
        config["data"]["dataset"] = args.dataset

    if args.univariate:
        config["data"]["univariate"] = True
    elif args.multivariate:
        config["data"]["univariate"] = False

    if args.epochs:
        config["training"]["max_epochs"] = args.epochs

    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    if args.lr:
        config["training"]["learning_rate"] = args.lr

    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name

    if args.lookback_length:
        config["data"]["lookback_length"] = args.lookback_length

    if args.forecast_length:
        config["data"]["forecast_length"] = args.forecast_length

    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Configuration: {config}")
    print(f"Device: {device}")

    # Create dataloaders
    data_config = config["data"]
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_path=data_config.get("data_path", "data/ETDataset"),
        dataset_name=data_config.get("dataset", "ETTh1"),
        lookback_length=data_config.get("lookback_length", 336),
        forecast_length=data_config.get("forecast_length", 168),
        batch_size=data_config.get("batch_size", 64),
        num_workers=data_config.get("num_workers", 4),
        univariate=data_config.get("univariate", False),
        train_ratio=data_config.get("train_ratio", 0.6),
        val_ratio=data_config.get("val_ratio", 0.2),
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Determine checkpoint directory
    checkpoint_dir = args.checkpoint_dir or config.get("logging", {}).get("checkpoint_dir", "checkpoints")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=config.get("logging", {}).get("log_dir", "logs"),
    )

    # Train
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        results = trainer.resume_training(args.checkpoint)
    else:
        results = trainer.train()

    print(f"\nTraining completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final epoch: {results['final_epoch'] + 1}")


if __name__ == "__main__":
    main()
