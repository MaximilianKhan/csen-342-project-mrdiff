#!/usr/bin/env python3
"""Train CI Decomp Transformer (Exp 17) — direct forecasting, no diffusion.

Evaluates in globally-standardized space for fair comparison.
"""

import torch
import torch.nn as nn
import yaml
import time
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import create_model


EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168},
    {"name": "ETTh1_uni",   "dataset": "ETTh1", "uni": True,  "L": 336, "H": 168},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192},
    {"name": "ETTm1_uni",   "dataset": "ETTm1", "uni": True,  "L": 1440, "H": 192},
]


def train_model(model, train_loader, val_loader, config, checkpoint_dir):
    """Simple training loop for direct forecasting model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_cfg = config.get("training", {})
    max_epochs = train_cfg.get("max_epochs", 100)
    min_epochs = train_cfg.get("min_epochs", 30)
    lr = train_cfg.get("learning_rate", 0.0005)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    patience = train_cfg.get("early_stopping_patience", 20)
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = 0

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            lookback = batch["lookback"].to(device, non_blocking=True)
            forecast = batch["forecast"].to(device, non_blocking=True)

            pred = model(lookback)
            loss = nn.functional.mse_loss(pred, forecast)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train = total_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                lookback = batch["lookback"].to(device, non_blocking=True)
                forecast = batch["forecast"].to(device, non_blocking=True)
                pred = model(lookback)
                val_loss += nn.functional.mse_loss(pred, forecast).item()
                val_n += 1
        avg_val = val_loss / val_n

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, f"{checkpoint_dir}/best.pt")
            print(f"Saved checkpoint: {checkpoint_dir}/best.pt")
        else:
            epochs_without_improvement += 1

        scheduler.step()
        final_epoch = epoch

        # Early stopping
        if epoch >= min_epochs - 1 and epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return {"best_val_loss": best_val_loss, "final_epoch": final_epoch}


def evaluate_model(model, test_loader, scaler):
    """Evaluate in globally-standardized space."""
    device = next(model.parameters()).device
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

    return {
        "mae": torch.cat(all_mae).mean().item(),
        "mse": torch.cat(all_mse).mean().item(),
    }


def main():
    print("=" * 70)
    print("EXP 17: CI Decomp Transformer (channel-independent + decomposition)")
    print("=" * 70)

    with open("configs/small.yaml") as f:
        base_config = yaml.safe_load(f)

    results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp['name']} ({exp['dataset']}, {'Uni' if exp['uni'] else 'Multi'}, L={exp['L']}, H={exp['H']})")
        print(f"{'='*60}")

        config = yaml.safe_load(yaml.dump(base_config))
        config["data"]["dataset"] = exp["dataset"]
        config["data"]["univariate"] = exp["uni"]
        config["data"]["lookback_length"] = exp["L"]
        config["data"]["forecast_length"] = exp["H"]
        config["experiment"]["name"] = exp["name"]

        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            data_path=config["data"]["data_path"],
            dataset_name=exp["dataset"],
            lookback_length=exp["L"],
            forecast_length=exp["H"],
            batch_size=64, num_workers=4,
            univariate=exp["uni"],
        )
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        model = create_model(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

        ckpt_dir = f"checkpoints/final/{exp['name']}"
        start = time.time()
        train_result = train_model(model, train_loader, val_loader, config, ckpt_dir)
        elapsed = time.time() - start
        print(f"  Training time: {elapsed/60:.1f}min")

        # Evaluate best checkpoint
        print(f"  Evaluating...", flush=True)
        model.load_state_dict(
            torch.load(f"{ckpt_dir}/best.pt", map_location="cpu")["model_state_dict"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        eval_result = evaluate_model(model, test_loader, scaler)

        results[exp["name"]] = {
            "train_time_min": elapsed / 60,
            "best_val_loss": train_result["best_val_loss"],
            "final_epoch": train_result["final_epoch"] + 1,
            **eval_result,
        }

        print(f"  MAE (global-std): {eval_result['mae']:.4f}")
        print(f"  MSE (global-std): {eval_result['mse']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS (globally-standardized metrics)")
    print(f"{'='*70}")
    print(f"{'Experiment':<18} {'MAE':>11} {'MSE':>11} {'Epochs':>8} {'Time':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<18} {r['mae']:>11.4f} {r['mse']:>11.4f} {r['final_epoch']:>8} {r['train_time_min']:>7.1f}m")

    print(f"\nPaper reference (mr-Diff):")
    print(f"  ETTh1 Multi: MAE=0.42")
    print(f"  ETTh1 Uni:   MAE=0.34")
    print(f"  ETTm1 Multi: MAE=0.37")
    print(f"  ETTm1 Uni:   MAE=0.15")


if __name__ == "__main__":
    main()
