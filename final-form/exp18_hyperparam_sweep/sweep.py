#!/usr/bin/env python3
"""Exp 18: Hyperparameter sweep on CI+Decomp Transformer.

30 configs × 4 benchmarks. Logs all results to a single CSV for analysis.
"""

import torch
import torch.nn as nn
import yaml
import time
import csv
import itertools
import random
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer


EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168},
    {"name": "ETTh1_uni",   "dataset": "ETTh1", "uni": True,  "L": 336, "H": 168},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192},
    {"name": "ETTm1_uni",   "dataset": "ETTm1", "uni": True,  "L": 1440, "H": 192},
]

# Sweep space — informed by 17 experiments of learning
SWEEP_CONFIGS = []

# Key parameters to sweep
patch_sizes = [8, 12, 16, 24]
d_models = [32, 48, 64, 96]
num_layers_list = [1, 2, 3]
dropouts = [0.2, 0.3, 0.4, 0.5]
trend_kernels = [15, 25, 49]
learning_rates = [0.0005, 0.001, 0.002]
dim_ffs = [64, 128, 256]

# Generate smart random sample of 30 configs
# Seed for reproducibility
random.seed(42)

# Always include the Exp 17 baseline config
SWEEP_CONFIGS.append({
    "id": 0,
    "patch_size": 16, "d_model": 64, "num_layers": 2,
    "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 25,
    "lr": 0.0005, "weight_decay": 0.01,
    "label": "exp17_baseline",
})

# Generate 29 random configs
for i in range(1, 30):
    cfg = {
        "id": i,
        "patch_size": random.choice(patch_sizes),
        "d_model": random.choice(d_models),
        "num_layers": random.choice(num_layers_list),
        "dim_feedforward": random.choice(dim_ffs),
        "dropout": random.choice(dropouts),
        "trend_kernel": random.choice(trend_kernels),
        "lr": random.choice(learning_rates),
        "weight_decay": random.choice([0.005, 0.01, 0.05]),
        "label": f"config_{i:02d}",
    }
    # Ensure nhead divides d_model
    cfg["nhead"] = 4 if cfg["d_model"] >= 64 else (2 if cfg["d_model"] >= 32 else 1)
    SWEEP_CONFIGS.append(cfg)


def create_model_from_sweep(cfg, input_dim, forecast_length, lookback_length):
    """Create model from sweep config."""
    patch_size = cfg["patch_size"]
    # Ensure patch_size divides lookback cleanly (at least 4 patches)
    while lookback_length // patch_size < 4:
        patch_size = patch_size // 2
    patch_size = max(4, patch_size)

    return CIDecompTransformer(
        input_dim=input_dim,
        forecast_length=forecast_length,
        lookback_length=lookback_length,
        patch_size=patch_size,
        d_model=cfg["d_model"],
        nhead=cfg.get("nhead", 4),
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        trend_kernel=cfg["trend_kernel"],
    )


def train_and_eval(model, train_loader, val_loader, test_loader, scaler, cfg, exp):
    """Train model and evaluate. Returns metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    max_epochs = 100
    min_epochs = 30
    patience = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            lookback = batch["lookback"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            pred = model(lookback)
            loss = nn.functional.mse_loss(pred, forecast)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        # Validate
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                lookback = batch["lookback"].to(device, non_blocking=True)
                forecast = batch["forecast"].to(device, non_blocking=True)
                pred = model(lookback)
                val_loss += nn.functional.mse_loss(pred, forecast).item()
                vn += 1
        avg_val = val_loss / vn

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        scheduler.step()

        if epoch >= min_epochs - 1 and epochs_no_improve >= patience:
            break

    final_epoch = epoch + 1

    # Evaluate best model
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    all_mae = []
    all_mse = []
    with torch.no_grad():
        for batch in test_loader:
            lookback = batch["lookback"].to(device)
            target = batch["forecast"].to(device)
            norm_mean = batch["norm_mean"].to(device)
            norm_std = batch["norm_std"].to(device)

            target_orig = target * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            target_g = scaler.transform(target_orig)

            pred = model(lookback)
            pred_orig = pred * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            pred_g = scaler.transform(pred_orig)

            all_mae.append((pred_g - target_g).abs().mean(dim=(1, 2)).cpu())
            all_mse.append(((pred_g - target_g) ** 2).mean(dim=(1, 2)).cpu())

    mae = torch.cat(all_mae).mean().item()
    mse = torch.cat(all_mse).mean().item()

    return {"mae": mae, "mse": mse, "epochs": final_epoch, "best_val": best_val_loss}


def main():
    print("=" * 70)
    print("EXP 18: Hyperparameter Sweep — CI+Decomp Transformer")
    print(f"  {len(SWEEP_CONFIGS)} configs × {len(EXPERIMENTS)} benchmarks")
    print("=" * 70)

    with open("configs/small.yaml") as f:
        base_config = yaml.safe_load(f)

    # Preload all dataloaders (reuse across configs)
    dataloaders = {}
    for exp in EXPERIMENTS:
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            data_path=base_config["data"]["data_path"],
            dataset_name=exp["dataset"],
            lookback_length=exp["L"],
            forecast_length=exp["H"],
            batch_size=64, num_workers=4,
            univariate=exp["uni"],
        )
        dataloaders[exp["name"]] = (train_loader, val_loader, test_loader, scaler)
        print(f"  Loaded {exp['name']}: {len(train_loader.dataset)} train samples")

    # Results CSV
    csv_path = Path("sweep_results.csv")
    fieldnames = [
        "config_id", "label", "patch_size", "d_model", "num_layers",
        "dim_feedforward", "dropout", "trend_kernel", "lr", "weight_decay",
        "benchmark", "params", "mae", "mse", "epochs", "time_sec",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Track bests
    best_per_benchmark = {exp["name"]: {"mae": float("inf"), "config": None} for exp in EXPERIMENTS}

    total_start = time.time()

    for ci, cfg in enumerate(SWEEP_CONFIGS):
        print(f"\n{'='*60}")
        print(f"  Config {ci+1}/{len(SWEEP_CONFIGS)}: {cfg['label']}")
        print(f"  patch={cfg['patch_size']} d={cfg['d_model']} layers={cfg['num_layers']} "
              f"ff={cfg['dim_feedforward']} drop={cfg['dropout']} tk={cfg['trend_kernel']} "
              f"lr={cfg['lr']} wd={cfg['weight_decay']}")
        print(f"{'='*60}")

        for exp in EXPERIMENTS:
            input_dim = 1 if exp["uni"] else 7
            train_loader, val_loader, test_loader, scaler = dataloaders[exp["name"]]

            model = create_model_from_sweep(cfg, input_dim, exp["H"], exp["L"])
            params = sum(p.numel() for p in model.parameters())

            start = time.time()
            result = train_and_eval(
                model, train_loader, val_loader, test_loader, scaler, cfg, exp
            )
            elapsed = time.time() - start

            # Log
            row = {
                "config_id": cfg["id"],
                "label": cfg["label"],
                "patch_size": cfg["patch_size"],
                "d_model": cfg["d_model"],
                "num_layers": cfg["num_layers"],
                "dim_feedforward": cfg["dim_feedforward"],
                "dropout": cfg["dropout"],
                "trend_kernel": cfg["trend_kernel"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "benchmark": exp["name"],
                "params": params,
                "mae": f"{result['mae']:.4f}",
                "mse": f"{result['mse']:.4f}",
                "epochs": result["epochs"],
                "time_sec": f"{elapsed:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()

            # Track best
            if result["mae"] < best_per_benchmark[exp["name"]]["mae"]:
                best_per_benchmark[exp["name"]]["mae"] = result["mae"]
                best_per_benchmark[exp["name"]]["config"] = cfg["label"]

            print(f"    {exp['name']}: MAE={result['mae']:.4f} | {params:,} params | "
                  f"{result['epochs']} epochs | {elapsed:.1f}s")

    csv_file.close()
    total_time = time.time() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE — {len(SWEEP_CONFIGS)} configs × {len(EXPERIMENTS)} benchmarks")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}")

    print(f"\nBest per benchmark:")
    print(f"{'Benchmark':<18} {'Best MAE':>10} {'Config':>20} {'Baseline':>10} {'Delta':>10}")
    print("-" * 70)
    baselines = {
        "ETTh1_multi": 0.4744, "ETTh1_uni": 0.2535,
        "ETTm1_multi": 0.4204, "ETTm1_uni": 0.2011,
    }
    for name, best in best_per_benchmark.items():
        bl = baselines[name]
        delta = (best["mae"] - bl) / bl * 100
        print(f"{name:<18} {best['mae']:>10.4f} {best['config']:>20} {bl:>10.4f} {delta:>+9.1f}%")

    # Also print all-time bests for reference
    print(f"\nAll-time bests (prior to sweep):")
    print(f"  ETTh1 Multi: 0.4719 (Exp 2)")
    print(f"  ETTh1 Uni:   0.2508 (Exp 10)")
    print(f"  ETTm1 Multi: 0.4159 (Exp 17)")
    print(f"  ETTm1 Uni:   0.1885 (Exp 16)")


if __name__ == "__main__":
    main()
