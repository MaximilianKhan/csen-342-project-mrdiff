#!/usr/bin/env python3
"""Experiment 6: Contrastive Conditioning Loss (CCDM).

Adds InfoNCE contrastive term to training loss.
"""

import torch
import yaml
import time
from src.data.dataset import create_dataloaders
from src.models.mr_diff import create_model
from src.training.trainer import Trainer


EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168},
    {"name": "ETTh1_uni",   "dataset": "ETTh1", "uni": True,  "L": 336, "H": 168},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192},
    {"name": "ETTm1_uni",   "dataset": "ETTm1", "uni": True,  "L": 1440, "H": 192},
]


def evaluate_model(model, test_loader, scaler, num_samples=3):
    """Evaluate model in globally-standardized space."""
    device = next(model.parameters()).device
    model.eval()
    all_mae_direct = []
    all_mae_full = []
    all_mse_full = []

    with torch.no_grad():
        for batch in test_loader:
            lookback = batch["lookback"].to(device)
            target = batch["forecast"].to(device)
            norm_mean = batch["norm_mean"].to(device)
            norm_std = batch["norm_std"].to(device)

            target_orig = target * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            target_g = scaler.transform(target_orig)

            # Direct prediction only
            direct = model.direct_predict(lookback)
            direct_orig = direct * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            direct_g = scaler.transform(direct_orig)
            all_mae_direct.append((direct_g - target_g).abs().mean(dim=(1, 2)).cpu())

            # Full model (direct + diffusion)
            samples = []
            for _ in range(num_samples):
                pred = model.sample(lookback, num_samples=1,
                                    solver="dpm_solver_pp", solver_steps=20)
                samples.append(pred)
            pred = torch.stack(samples).mean(dim=0)

            pred_orig = pred * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            pred_g = scaler.transform(pred_orig)
            all_mae_full.append((pred_g - target_g).abs().mean(dim=(1, 2)).cpu())
            all_mse_full.append(((pred_g - target_g) ** 2).mean(dim=(1, 2)).cpu())

    return {
        "mae_direct": torch.cat(all_mae_direct).mean().item(),
        "mae_full": torch.cat(all_mae_full).mean().item(),
        "mse_full": torch.cat(all_mse_full).mean().item(),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 6: Contrastive Conditioning Loss (CCDM)")
    print("=" * 70)

    with open("configs/small.yaml") as f:
        base_config = yaml.safe_load(f)

    results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp['name']} ({exp['dataset']}, {'Uni' if exp['uni'] else 'Multi'}, L={exp['L']}, H={exp['H']})")
        print(f"{'='*60}")

        # Update config for this experiment
        config = yaml.safe_load(yaml.dump(base_config))  # deep copy
        config["data"]["dataset"] = exp["dataset"]
        config["data"]["univariate"] = exp["uni"]
        config["data"]["lookback_length"] = exp["L"]
        config["data"]["forecast_length"] = exp["H"]
        config["experiment"]["name"] = exp["name"]

        # Create dataloaders
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            data_path=config["data"]["data_path"],
            dataset_name=exp["dataset"],
            lookback_length=exp["L"],
            forecast_length=exp["H"],
            batch_size=64, num_workers=4,
            univariate=exp["uni"],
        )
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Create model
        model = create_model(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

        # Train
        ckpt_dir = f"checkpoints/final/{exp['name']}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=ckpt_dir,
        )

        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start
        print(f"  Training time: {elapsed/60:.1f}min")
        print(f"  Best val loss: {train_result['best_val_loss']:.4f}")

        # Evaluate
        print(f"  Evaluating...", flush=True)
        model.load_state_dict(
            torch.load(f"{ckpt_dir}/best.pt", map_location="cpu")["model_state_dict"]
        )
        eval_result = evaluate_model(model, test_loader, scaler, num_samples=3)

        results[exp["name"]] = {
            "train_time_min": elapsed / 60,
            "best_val_loss": train_result["best_val_loss"],
            "final_epoch": train_result["final_epoch"] + 1,
            **eval_result,
        }

        print(f"  Direct MAE (global-std): {eval_result['mae_direct']:.4f}")
        print(f"  Full MAE (global-std):   {eval_result['mae_full']:.4f}")
        print(f"  Full MSE (global-std):   {eval_result['mse_full']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS (globally-standardized metrics)")
    print(f"{'='*70}")
    print(f"{'Experiment':<18} {'Direct MAE':>11} {'Full MAE':>11} {'Full MSE':>11} {'Epochs':>8} {'Time':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<18} {r['mae_direct']:>11.4f} {r['mae_full']:>11.4f} "
              f"{r['mse_full']:>11.4f} {r['final_epoch']:>8} {r['train_time_min']:>7.1f}m")

    print(f"\nPaper reference (mr-Diff):")
    print(f"  ETTh1 Multi: MAE=0.42")
    print(f"  ETTh1 Uni:   MAE=0.34")
    print(f"  ETTm1 Multi: MAE=0.37")
    print(f"  ETTm1 Uni:   MAE=0.15")


if __name__ == "__main__":
    main()
