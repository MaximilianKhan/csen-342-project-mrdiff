#!/usr/bin/env python3
"""Parallel shard worker for hyperparameter sweep.

Usage: python sweep_shard.py <start_idx> <end_idx> <shard_id>
"""

import sys
import torch
import torch.nn as nn
import yaml
import time
import csv
import random
from pathlib import Path
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer

EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168},
    {"name": "ETTh1_uni",   "dataset": "ETTh1", "uni": True,  "L": 336, "H": 168},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192},
    {"name": "ETTm1_uni",   "dataset": "ETTm1", "uni": True,  "L": 1440, "H": 192},
]

# Must match sweep.py exactly (same seed)
patch_sizes = [8, 12, 16, 24]
d_models = [32, 48, 64, 96]
num_layers_list = [1, 2, 3]
dropouts = [0.2, 0.3, 0.4, 0.5]
trend_kernels = [15, 25, 49]
learning_rates = [0.0005, 0.001, 0.002]
dim_ffs = [64, 128, 256]

random.seed(42)
SWEEP_CONFIGS = [{"id": 0, "patch_size": 16, "d_model": 64, "num_layers": 2,
    "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 25,
    "lr": 0.0005, "weight_decay": 0.01, "label": "exp17_baseline"}]
for i in range(1, 30):
    cfg = {"id": i, "patch_size": random.choice(patch_sizes),
        "d_model": random.choice(d_models), "num_layers": random.choice(num_layers_list),
        "dim_feedforward": random.choice(dim_ffs), "dropout": random.choice(dropouts),
        "trend_kernel": random.choice(trend_kernels), "lr": random.choice(learning_rates),
        "weight_decay": random.choice([0.005, 0.01, 0.05]), "label": f"config_{i:02d}"}
    cfg["nhead"] = 4 if cfg["d_model"] >= 64 else (2 if cfg["d_model"] >= 32 else 1)
    SWEEP_CONFIGS.append(cfg)


def create_model_from_sweep(cfg, input_dim, forecast_length, lookback_length):
    patch_size = cfg["patch_size"]
    while lookback_length // patch_size < 4:
        patch_size = patch_size // 2
    patch_size = max(4, patch_size)
    return CIDecompTransformer(
        input_dim=input_dim, forecast_length=forecast_length,
        lookback_length=lookback_length, patch_size=patch_size,
        d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])


def train_and_eval(model, train_loader, val_loader, test_loader, scaler, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(100):
        model.train()
        for batch in train_loader:
            lookback = batch["lookback"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)
            loss = nn.functional.mse_loss(model(lookback), forecast)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += nn.functional.mse_loss(
                    model(batch["lookback"].to(device)), batch["forecast"].to(device)).item()
                vn += 1
        avg_val = val_loss / vn

        if avg_val < best_val_loss:
            best_val_loss = avg_val; epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        scheduler.step()
        if epoch >= 29 and epochs_no_improve >= 20: break
    final_epoch = epoch + 1

    model.load_state_dict(best_state); model = model.to(device); model.eval()
    all_mae, all_mse = [], []
    with torch.no_grad():
        for batch in test_loader:
            lookback = batch["lookback"].to(device)
            target = batch["forecast"].to(device)
            norm_mean, norm_std = batch["norm_mean"].to(device), batch["norm_std"].to(device)
            target_g = scaler.transform(target * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1))
            pred_g = scaler.transform(model(lookback) * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1))
            all_mae.append((pred_g - target_g).abs().mean(dim=(1,2)).cpu())
            all_mse.append(((pred_g - target_g)**2).mean(dim=(1,2)).cpu())
    return {"mae": torch.cat(all_mae).mean().item(), "mse": torch.cat(all_mse).mean().item(),
            "epochs": final_epoch, "best_val": best_val_loss}


def main():
    start_idx, end_idx, shard_id = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    configs = SWEEP_CONFIGS[start_idx:end_idx]
    print(f"Shard {shard_id}: configs {start_idx}-{end_idx-1} ({len(configs)} configs)")

    with open("configs/small.yaml") as f:
        base_config = yaml.safe_load(f)

    dataloaders = {}
    for exp in EXPERIMENTS:
        dataloaders[exp["name"]] = create_dataloaders(
            data_path=base_config["data"]["data_path"], dataset_name=exp["dataset"],
            lookback_length=exp["L"], forecast_length=exp["H"],
            batch_size=64, num_workers=2, univariate=exp["uni"])

    csv_path = Path(f"sweep_results_shard_{shard_id}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "config_id","label","patch_size","d_model","num_layers","dim_feedforward",
        "dropout","trend_kernel","lr","weight_decay","benchmark","params","mae","mse","epochs","time_sec"])
    writer.writeheader()

    for ci, cfg in enumerate(configs):
        print(f"\n  Config {start_idx+ci+1}/30: {cfg['label']} | "
              f"patch={cfg['patch_size']} d={cfg['d_model']} layers={cfg['num_layers']} "
              f"drop={cfg['dropout']} tk={cfg['trend_kernel']} lr={cfg['lr']}")

        for exp in EXPERIMENTS:
            input_dim = 1 if exp["uni"] else 7
            train_loader, val_loader, test_loader, scaler = dataloaders[exp["name"]]
            model = create_model_from_sweep(cfg, input_dim, exp["H"], exp["L"])
            params = sum(p.numel() for p in model.parameters())
            start = time.time()
            result = train_and_eval(model, train_loader, val_loader, test_loader, scaler, cfg)
            elapsed = time.time() - start
            writer.writerow({**{k: cfg[k] for k in ["patch_size","d_model","num_layers",
                "dim_feedforward","dropout","trend_kernel","lr","weight_decay"]},
                "config_id": cfg["id"], "label": cfg["label"], "benchmark": exp["name"],
                "params": params, "mae": f"{result['mae']:.4f}", "mse": f"{result['mse']:.4f}",
                "epochs": result["epochs"], "time_sec": f"{elapsed:.1f}"})
            csv_file.flush()
            print(f"    {exp['name']}: MAE={result['mae']:.4f} | {params:,} params | {elapsed:.0f}s")

    csv_file.close()
    print(f"\nShard {shard_id} complete. Results: {csv_path}")

if __name__ == "__main__":
    main()
