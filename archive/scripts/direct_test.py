#!/usr/bin/env python3
"""Direct predictor baseline (no diffusion) to test architecture ceiling."""

import sys
import torch
import torch.nn as nn
from src.data.dataset import create_dataloaders


class DirectPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, lookback_length, forecast_length):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.GroupNorm(32, hidden_dim), nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.GroupNorm(32, hidden_dim), nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.GroupNorm(32, hidden_dim), nn.LeakyReLU(0.1),
        )
        self.length_proj = nn.Linear(lookback_length, forecast_length)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, history):
        x = self.input_proj(history)
        x = x.transpose(1, 2)
        x = self.convs(x) + x
        x = self.length_proj(x)
        x = x.transpose(1, 2)
        return self.output_proj(x)


def main():
    print("Direct Predictor Baseline (no diffusion)", flush=True)

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_path="data/ETDataset", dataset_name="ETTh1",
        lookback_length=336, forecast_length=168,
        batch_size=64, num_workers=4, univariate=False,
    )

    model = DirectPredictor(7, 256, 336, 168)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(80):
        model.train()
        total_loss = 0
        for batch in train_loader:
            pred = model(batch["lookback"])
            loss = nn.functional.mse_loss(pred, batch["forecast"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

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
            no_improve = 0
        else:
            no_improve += 1

        train_avg = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}: train={train_avg:.4f}, val={val_loss:.4f}" +
              (f" (best)" if no_improve == 0 else ""), flush=True)

        if no_improve > 15 and epoch > 30:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    all_mae = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["lookback"])
            all_mae.append((pred - batch["forecast"]).abs().mean(dim=(1, 2)))
    all_mae = torch.cat(all_mae)
    print(f"\nDIRECT PREDICTOR MAE: {all_mae.mean():.4f}", flush=True)
    print(f"Paper target: 0.42, diffusion model: ~0.95", flush=True)


if __name__ == "__main__":
    main()
