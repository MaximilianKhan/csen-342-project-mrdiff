#!/usr/bin/env python3
"""Check if our MAE scale matches the paper's by computing metrics in both spaces."""

import torch
import torch.nn as nn
from src.data.dataset import create_dataloaders, ETTDataset
import numpy as np
import pandas as pd


class DLinearModel(nn.Module):
    """DLinear-style: trend + residual, channel-independent."""
    def __init__(self, lookback, forecast, dim, kernel_size=25):
        super().__init__()
        self.trend_proj = nn.Linear(lookback, forecast)
        self.resid_proj = nn.Linear(lookback, forecast)
        self.ks = kernel_size

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, L]
        pad = self.ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, self.ks, stride=1)
        resid = x - trend
        out = self.trend_proj(trend) + self.resid_proj(resid)
        return out.transpose(1, 2)


def main():
    print("=" * 70)
    print("SCALE DIAGNOSTIC: Are we comparing apples to oranges?")
    print("=" * 70)

    # Load test data
    _, _, test_loader, _ = create_dataloaders(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168,
        batch_size=64, num_workers=0, univariate=False,
    )

    # Also load with GLOBAL standardization for comparison
    # First, compute global stats from training data
    train_ds = ETTDataset(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168, split="train"
    )
    global_mean = torch.tensor(train_ds.split_data.mean(axis=0), dtype=torch.float32)
    global_std = torch.tensor(train_ds.split_data.std(axis=0), dtype=torch.float32) + 1e-5

    print(f"\nGlobal training stats:")
    print(f"  Mean: {global_mean.numpy()}")
    print(f"  Std:  {global_std.numpy()}")

    # Compute metrics in different spaces
    print(f"\n--- ZERO PREDICTION METRICS ---")

    norm_maes = []
    orig_maes = []
    global_maes = []
    norm_stds_all = []

    for batch in test_loader:
        forecast_norm = batch["forecast"]  # RevIN-normalized
        norm_mean = batch["norm_mean"]     # [B, D]
        norm_std = batch["norm_std"]       # [B, D]

        B, T, D = forecast_norm.shape

        # Inverse RevIN: get original-scale forecast
        forecast_orig = forecast_norm * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)

        # Zero prediction in normalized space → norm_mean in original space
        zero_pred_orig = norm_mean.unsqueeze(1).expand_as(forecast_orig)

        # Global standardized forecast
        forecast_global = (forecast_orig - global_mean.unsqueeze(0).unsqueeze(0)) / \
                          global_std.unsqueeze(0).unsqueeze(0)
        zero_pred_global = (zero_pred_orig - global_mean.unsqueeze(0).unsqueeze(0)) / \
                           global_std.unsqueeze(0).unsqueeze(0)

        # MAE in normalized (RevIN) space
        mae_norm = forecast_norm.abs().mean(dim=(1, 2))  # zero pred = 0
        norm_maes.append(mae_norm)

        # MAE in original space
        mae_orig = (zero_pred_orig - forecast_orig).abs().mean(dim=(1, 2))
        orig_maes.append(mae_orig)

        # MAE in global-standardized space
        mae_global = (zero_pred_global - forecast_global).abs().mean(dim=(1, 2))
        global_maes.append(mae_global)

        norm_stds_all.append(norm_std)

    norm_maes = torch.cat(norm_maes)
    orig_maes = torch.cat(orig_maes)
    global_maes = torch.cat(global_maes)
    norm_stds_all = torch.cat(norm_stds_all)

    print(f"  RevIN-normalized MAE:     {norm_maes.mean():.4f}")
    print(f"  Original-scale MAE:       {orig_maes.mean():.4f}")
    print(f"  Global-standardized MAE:  {global_maes.mean():.4f}")
    print(f"  Average norm_std:         {norm_stds_all.mean():.4f}")

    # Now compute for DLinear model
    print(f"\n--- DLINEAR METRICS ---")

    # Train a quick DLinear
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168,
        batch_size=64, num_workers=4, univariate=False,
    )

    model = DLinearModel(336, 168, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val = float("inf")
    best_state = None
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            pred = model(batch["lookback"])
            loss = nn.functional.mse_loss(pred, batch["forecast"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["lookback"])
                val_loss += nn.functional.mse_loss(pred, batch["forecast"]).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0:
                print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f} *", flush=True)

    model.load_state_dict(best_state)
    model.eval()

    norm_maes = []
    orig_maes = []
    global_maes = []
    norm_mses = []
    global_mses = []

    with torch.no_grad():
        for batch in test_loader:
            lookback = batch["lookback"]
            forecast_norm = batch["forecast"]
            norm_mean = batch["norm_mean"]
            norm_std = batch["norm_std"]

            pred_norm = model(lookback)

            B, T, D = forecast_norm.shape

            # Inverse RevIN
            forecast_orig = forecast_norm * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)
            pred_orig = pred_norm * norm_std.unsqueeze(1) + norm_mean.unsqueeze(1)

            # Global standardized
            forecast_global = (forecast_orig - global_mean) / global_std
            pred_global = (pred_orig - global_mean) / global_std

            # Metrics
            norm_maes.append((pred_norm - forecast_norm).abs().mean(dim=(1, 2)))
            orig_maes.append((pred_orig - forecast_orig).abs().mean(dim=(1, 2)))
            global_maes.append((pred_global - forecast_global).abs().mean(dim=(1, 2)))

            norm_mses.append(((pred_norm - forecast_norm)**2).mean(dim=(1, 2)))
            global_mses.append(((pred_global - forecast_global)**2).mean(dim=(1, 2)))

    print(f"\nDLinear Results:")
    print(f"  RevIN-normalized MAE:     {torch.cat(norm_maes).mean():.4f}")
    print(f"  RevIN-normalized MSE:     {torch.cat(norm_mses).mean():.4f}")
    print(f"  Original-scale MAE:       {torch.cat(orig_maes).mean():.4f}")
    print(f"  Global-standardized MAE:  {torch.cat(global_maes).mean():.4f}")
    print(f"  Global-standardized MSE:  {torch.cat(global_mses).mean():.4f}")

    print(f"\n--- PAPER REFERENCE (ETTh1 multivariate) ---")
    print(f"  H=96:  MSE≈0.375, MAE≈0.400 (DLinear paper)")
    print(f"  H=192: MSE≈0.405, MAE≈0.429")
    print(f"  H=336: MSE≈0.439, MAE≈0.452")
    print(f"  H=720: MSE≈0.472, MAE≈0.479")
    print(f"  Our H=168, so expect between H=96 and H=192 values")
    print(f"\n  mr-Diff paper (Table 2, ETTh1 multi, H=168): MAE=0.42")
    print(f"\n  If our global-std MAE ≈ 0.4, the data pipeline is correct!")
    print(f"  If our global-std MAE >> 0.4, there's a data issue.")


if __name__ == "__main__":
    main()
