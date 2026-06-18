#!/usr/bin/env python3
"""Deep diagnostic of data pipeline — find why no model can beat zero prediction."""

import torch
import torch.nn as nn
import numpy as np
from src.data.dataset import create_dataloaders, ETTDataset


def main():
    print("=" * 70)
    print("DATA PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # Load data
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168,
        batch_size=64, num_workers=0, univariate=False,
    )

    # ---- 1. Basic statistics of normalized data ----
    print("\n--- 1. DATA STATISTICS (normalized) ---")
    all_lookback = []
    all_forecast = []
    for batch in test_loader:
        all_lookback.append(batch["lookback"])
        all_forecast.append(batch["forecast"])
    all_lookback = torch.cat(all_lookback)  # [N, 336, 7]
    all_forecast = torch.cat(all_forecast)  # [N, 168, 7]

    print(f"Test set: {len(all_lookback)} windows")
    print(f"Lookback shape: {all_lookback.shape}")
    print(f"Forecast shape: {all_forecast.shape}")
    print(f"\nLookback: mean={all_lookback.mean():.4f}, std={all_lookback.std():.4f}")
    print(f"Forecast: mean={all_forecast.mean():.4f}, std={all_forecast.std():.4f}")
    print(f"Forecast abs mean (= zero prediction MAE): {all_forecast.abs().mean():.4f}")

    # Per-feature statistics
    print("\nPer-feature forecast stats:")
    for d in range(all_forecast.shape[2]):
        feat = all_forecast[:, :, d]
        print(f"  Feature {d}: mean={feat.mean():.4f}, std={feat.std():.4f}, "
              f"|mean|={feat.abs().mean():.4f}")

    # ---- 2. Simple baselines ----
    print("\n--- 2. BASELINES ---")

    # Zero prediction
    zero_mae = all_forecast.abs().mean().item()
    print(f"Zero prediction MAE:       {zero_mae:.4f}")

    # Last value prediction (repeat last lookback value)
    last_val = all_lookback[:, -1:, :].expand_as(all_forecast)
    last_mae = (last_val - all_forecast).abs().mean().item()
    print(f"Last-value prediction MAE: {last_mae:.4f}")

    # Lookback mean prediction (= 0 in normalized space, same as zero)
    # But let's verify
    lb_mean = all_lookback.mean(dim=1, keepdim=True).expand_as(all_forecast)
    lbmean_mae = (lb_mean - all_forecast).abs().mean().item()
    print(f"Lookback-mean pred MAE:    {lbmean_mae:.4f}")

    # Linear extrapolation: fit line to last 24 steps, extrapolate
    # Slope from last 24 lookback steps
    last_n = 24
    last_chunk = all_lookback[:, -last_n:, :]  # [N, 24, 7]
    t_lb = torch.arange(last_n, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    t_lb_mean = t_lb.mean()
    slope = ((t_lb - t_lb_mean) * (last_chunk - last_chunk.mean(dim=1, keepdim=True))).sum(dim=1, keepdim=True) / \
            ((t_lb - t_lb_mean) ** 2).sum(dim=1, keepdim=True)
    intercept = last_chunk.mean(dim=1, keepdim=True) - slope * t_lb_mean
    t_fc = torch.arange(last_n, last_n + 168, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    linear_pred = slope * t_fc + intercept
    linear_mae = (linear_pred - all_forecast).abs().mean().item()
    print(f"Linear extrap (24-step) MAE: {linear_mae:.4f}")

    # ---- 3. Correlation between lookback and forecast ----
    print("\n--- 3. LOOKBACK-FORECAST CORRELATION ---")
    # Flatten time dimension and compute correlation
    lb_flat = all_lookback.mean(dim=1)  # [N, 7] - avg lookback features
    fc_flat = all_forecast.mean(dim=1)  # [N, 7] - avg forecast features
    for d in range(7):
        corr = torch.corrcoef(torch.stack([lb_flat[:, d], fc_flat[:, d]]))[0, 1]
        print(f"  Feature {d}: corr(lookback_mean, forecast_mean) = {corr:.4f}")

    # ---- 4. Linear model (OLS) ----
    print("\n--- 4. LINEAR MODEL (lookback → forecast) ---")

    # Collect train data
    train_X, train_Y = [], []
    for batch in train_loader:
        train_X.append(batch["lookback"])
        train_Y.append(batch["forecast"])
    train_X = torch.cat(train_X)  # [N_train, 336, 7]
    train_Y = torch.cat(train_Y)  # [N_train, 168, 7]

    # Flatten: X=[N, 336*7], Y=[N, 168*7]
    N_train = len(train_X)
    X_flat = train_X.reshape(N_train, -1)  # [N, 2352]
    Y_flat = train_Y.reshape(N_train, -1)  # [N, 1176]

    print(f"Train: X={X_flat.shape}, Y={Y_flat.shape}")

    # OLS solution per output feature (use first feature only for speed)
    # Y = X @ W + b
    # Use ridge regression (regularized) to avoid overfitting
    X_test = all_lookback.reshape(len(all_lookback), -1)
    Y_test = all_forecast.reshape(len(all_forecast), -1)

    # Ridge with different lambdas
    for lam in [1e-1, 1.0, 10.0, 100.0, 1000.0]:
        # W = (X^T X + lambda*I)^(-1) X^T Y
        XtX = X_flat.T @ X_flat
        reg = lam * torch.eye(X_flat.shape[1])
        W = torch.linalg.solve(XtX + reg, X_flat.T @ Y_flat)
        pred_test = X_test @ W
        ridge_mae = (pred_test - Y_test).abs().mean().item()
        pred_train = X_flat @ W
        ridge_train_mae = (pred_train - Y_flat).abs().mean().item()
        print(f"  Ridge(λ={lam:.0e}): train MAE={ridge_train_mae:.4f}, "
              f"test MAE={ridge_mae:.4f}")

    # ---- 5. Check for data leakage / overlap ----
    print("\n--- 5. DATA SPLIT CHECK ---")
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    print(f"Train: start_idx={train_ds.start_idx}, end_idx={train_ds.end_idx}, "
          f"windows={len(train_ds)}")
    print(f"Val:   start_idx={val_ds.start_idx}, end_idx={val_ds.end_idx}, "
          f"windows={len(val_ds)}")
    print(f"Test:  start_idx={test_ds.start_idx}, end_idx={test_ds.end_idx}, "
          f"windows={len(test_ds)}")
    print(f"Total data points: {len(train_ds.data)}")

    # Standard ETT split
    n = len(train_ds.data)
    print(f"\nOur split (60/20/20): train={int(n*0.6)}, val={int(n*0.8)-int(n*0.6)}, "
          f"test={n-int(n*0.8)}")
    print(f"Standard ETT split: train=8640, val=2880, test=2880 (total=14400)")
    print(f"  (or for full data: train={int(n*12/20)}, val={int(n*4/20)}, test={n-int(n*16/20)})")

    # ---- 6. Check if forecast is just noise relative to lookback ----
    print("\n--- 6. FORECAST PREDICTABILITY ---")
    # Autocorrelation: how similar is lookback end to forecast start?
    last_lb = all_lookback[:, -1, :]  # [N, 7]
    first_fc = all_forecast[:, 0, :]  # [N, 7]
    continuity_error = (last_lb - first_fc).abs().mean().item()
    print(f"Continuity error (|lookback[-1] - forecast[0]|): {continuity_error:.4f}")

    # Check if forecast is just mean-reverting noise
    fc_variance_explained = []
    for d in range(7):
        total_var = all_forecast[:, :, d].var().item()
        # Variance explained by per-window mean
        window_means = all_forecast[:, :, d].mean(dim=1, keepdim=True)
        residual_var = (all_forecast[:, :, d] - window_means).var().item()
        explained = 1 - residual_var / total_var if total_var > 0 else 0
        fc_variance_explained.append(explained)
        print(f"  Feature {d}: total_var={total_var:.4f}, "
              f"variance explained by mean={explained:.2%}")

    # ---- 7. Raw data check ----
    print("\n--- 7. RAW DATA SANITY CHECK ---")
    import pandas as pd
    df = pd.read_csv("data/ETDataset/ETTh1.csv")
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nData stats:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
