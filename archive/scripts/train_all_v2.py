#!/usr/bin/env python3
"""Train all 4 experiments with architectural v2 fixes."""

import sys
import time
import torch
from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.mr_diff import create_model
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model

EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "univariate": False,
     "lookback_length": 336, "forecast_length": 168},
    {"name": "ETTh1_uni", "dataset": "ETTh1", "univariate": True,
     "lookback_length": 336, "forecast_length": 168},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "univariate": False,
     "lookback_length": 1440, "forecast_length": 192},
    {"name": "ETTm1_uni", "dataset": "ETTm1", "univariate": True,
     "lookback_length": 1440, "forecast_length": 192},
]

PAPER_MAE = {
    "ETTh1_multi": 0.42, "ETTh1_uni": 0.34,
    "ETTm1_multi": 0.37, "ETTm1_uni": 0.15,
}


def run_experiment(exp, base_dir, max_epochs=100):
    print(f"\n{'='*60}")
    print(f"  {exp['name']} — {exp['dataset']} {'Uni' if exp['univariate'] else 'Multi'}")
    print(f"{'='*60}")

    config = {
        "data": {
            "data_path": "data/ETDataset", "dataset": exp["dataset"],
            "lookback_length": exp["lookback_length"],
            "forecast_length": exp["forecast_length"],
            "batch_size": 64, "num_workers": 4,
            "univariate": exp["univariate"],
        },
        "model": {
            "num_stages": 5, "diffusion_steps": 100, "embedding_dim": 128,
            "hidden_dim": 256, "kernel_sizes": [5, 25, 51, 201],
            "num_encoder_layers": 3, "num_decoder_layers": 3, "dropout": 0.1,
        },
        "training": {
            "max_epochs": max_epochs, "min_epochs": 30,
            "learning_rate": 1e-3, "weight_decay": 1e-4,
            "grad_clip_norm": 1.0, "mixup_prob": 0.5,
            "early_stopping_patience": 10,
        },
        "logging": {
            "log_every_n_steps": 100, "save_every_n_epochs": 10,
            "tensorboard": False, "csv_logging": True,
        },
        "experiment": {"name": exp["name"]},
    }

    exp_dir = base_dir / exp["name"]
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_path=config["data"]["data_path"],
        dataset_name=config["data"]["dataset"],
        lookback_length=config["data"]["lookback_length"],
        forecast_length=config["data"]["forecast_length"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        univariate=config["data"]["univariate"],
    )

    # Create model
    model = create_model(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Train
    start = time.time()
    trainer = Trainer(
        model, train_loader, val_loader, config,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )
    result = trainer.train()
    train_time = time.time() - start
    print(f"  Training time: {train_time/60:.1f} min")

    # Evaluate with DPM-Solver++ sum aggregation
    print(f"\n  Evaluating with DPM-Solver++ (sum aggregation)...")
    device = next(model.parameters()).device
    model.eval()

    # Load best checkpoint
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']+1})")

    eval_results = evaluate_model(
        model=model, dataloader=test_loader, scaler=scaler,
        num_samples=10, device=device,
        solver="dpm_solver_pp", solver_steps=20,
        aggregation="sum",
    )

    mae = eval_results["mae"]
    mse = eval_results["mse"]
    paper = PAPER_MAE[exp["name"]]
    gap = mae / paper

    print(f"\n  Results:")
    print(f"    MAE:  {mae:.4f} (paper: {paper}, gap: {gap:.2f}x)")
    print(f"    MSE:  {mse:.4f}")
    print(f"    RMSE: {eval_results['rmse']:.4f}")

    # Also evaluate with aggregation="first" for comparison
    eval_first = evaluate_model(
        model=model, dataloader=test_loader, scaler=scaler,
        num_samples=10, device=device,
        solver="dpm_solver_pp", solver_steps=20,
        aggregation="first",
    )
    print(f"    MAE (first): {eval_first['mae']:.4f}")

    return {
        "mae_sum": mae, "mse_sum": mse,
        "mae_first": eval_first["mae"],
        "train_time": train_time,
        "final_epoch": result["final_epoch"],
    }


def main():
    max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    base_dir = Path(f"experiments/arch_v2")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  mr-Diff Arch V2 Training")
    print("  Fixes: residual convs, skip connections, strong conditioning,")
    print("         scheduled sampling, sum aggregation")
    print(f"  Epochs: {max_epochs}")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> Experiment {i}/4")
        try:
            results = run_experiment(exp, base_dir, max_epochs)
            all_results[exp["name"]] = results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp["name"]] = {"error": str(e)}

    total_time = time.time() - total_start

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY — Arch V2")
    print(f"{'='*60}")
    print(f"  Total time: {total_time/60:.1f} min")
    print()
    print(f"  {'Experiment':<20} {'MAE(sum)':<12} {'MAE(first)':<12} {'Paper':<10} {'Gap':<8}")
    print(f"  {'-'*62}")
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<20} FAILED: {res['error']}")
        else:
            paper = PAPER_MAE[name]
            gap = res["mae_sum"] / paper
            print(f"  {name:<20} {res['mae_sum']:<12.4f} {res['mae_first']:<12.4f} {paper:<10} {gap:<8.2f}x")


if __name__ == "__main__":
    main()
