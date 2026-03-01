#!/usr/bin/env python3
"""Minimal models to establish neural network baselines on our data pipeline.

Tests whether the gap is regularization vs architecture vs diffusion framework.
"""

import torch
import torch.nn as nn
import math
from src.data.dataset import create_dataloaders


class LinearBaseline(nn.Module):
    """Pure linear: forecast = W @ flatten(lookback) + b, with dropout."""
    def __init__(self, lookback_length, forecast_length, input_dim, dropout=0.5):
        super().__init__()
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lookback_length * input_dim, forecast_length * input_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x.reshape(B, self.forecast_length, self.input_dim)


class ChannelIndependentMLP(nn.Module):
    """Process each feature independently with a small MLP.
    This is similar to DLinear but with nonlinearity."""
    def __init__(self, lookback_length, forecast_length, input_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(lookback_length, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, forecast_length),
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        out = self.net(x)      # [B, D, H]
        return out.transpose(1, 2)  # [B, H, D]


class PatchLinear(nn.Module):
    """DLinear-style: decompose into trend + remainder, predict each linearly."""
    def __init__(self, lookback_length, forecast_length, input_dim, kernel_size=25):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.trend_proj = nn.Linear(lookback_length, forecast_length)
        self.resid_proj = nn.Linear(lookback_length, forecast_length)

    def forward(self, x):
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)
        # Moving average for trend
        pad = self.kernel_size // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, self.kernel_size, stride=1)
        resid = x - trend
        # Project each independently
        trend_out = self.trend_proj(trend)
        resid_out = self.resid_proj(resid)
        out = trend_out + resid_out
        return out.transpose(1, 2)


def train_and_eval(model, train_loader, val_loader, test_loader,
                   lr=1e-3, weight_decay=1e-2, epochs=50, name="Model"):
    print(f"\n{'='*60}")
    print(f"{name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"LR={lr}, WD={weight_decay}, Epochs={epochs}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            pred = model(batch["lookback"])
            loss = nn.functional.mse_loss(pred, batch["forecast"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["lookback"])
                val_loss += nn.functional.mse_loss(pred, batch["forecast"]).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == epochs - 1 or no_improve == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}{marker}",
                  flush=True)

        if no_improve > 20:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break

    # Evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    all_mae = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["lookback"])
            mae = (pred - batch["forecast"]).abs().mean(dim=(1, 2))
            all_mae.append(mae)
    all_mae = torch.cat(all_mae)
    test_mae = all_mae.mean().item()
    print(f"\n  TEST MAE: {test_mae:.4f}")
    print(f"  (Zero baseline: 0.9465, Ridge: 0.7898, Paper: 0.42)")
    return test_mae


def main():
    print("Loading data...", flush=True)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168,
        batch_size=64, num_workers=4, univariate=False,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    L, H, D = 336, 168, 7
    results = {}

    # 1. Pure linear with heavy dropout
    model = LinearBaseline(L, H, D, dropout=0.5)
    results["Linear+Dropout"] = train_and_eval(
        model, train_loader, val_loader, test_loader,
        lr=1e-3, weight_decay=1e-1, epochs=50, name="Linear + Dropout(0.5) + WD(0.1)"
    )

    # 2. Channel-independent MLP (like DLinear with nonlinearity)
    model = ChannelIndependentMLP(L, H, D, hidden=64, dropout=0.3)
    results["CI-MLP"] = train_and_eval(
        model, train_loader, val_loader, test_loader,
        lr=1e-3, weight_decay=1e-2, epochs=80, name="Channel-Independent MLP (h=64)"
    )

    # 3. DLinear-style (trend + residual linear)
    model = PatchLinear(L, H, D, kernel_size=25)
    results["DLinear"] = train_and_eval(
        model, train_loader, val_loader, test_loader,
        lr=1e-3, weight_decay=1e-2, epochs=80, name="DLinear-style (trend+residual)"
    )

    # 4. Channel-independent MLP - smaller
    model = ChannelIndependentMLP(L, H, D, hidden=32, dropout=0.3)
    results["CI-MLP-small"] = train_and_eval(
        model, train_loader, val_loader, test_loader,
        lr=5e-4, weight_decay=1e-1, epochs=80, name="Channel-Independent MLP (h=32, heavy WD)"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Test MAE':>10}")
    print("-" * 42)
    print(f"{'Zero prediction':<30} {'0.9465':>10}")
    print(f"{'Ridge regression (λ=1000)':<30} {'0.7898':>10}")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:<30} {mae:>10.4f}")
    print(f"{'Paper target':<30} {'0.42':>10}")


if __name__ == "__main__":
    main()
