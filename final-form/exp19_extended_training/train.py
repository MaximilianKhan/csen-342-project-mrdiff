#!/usr/bin/env python3
"""Exp 19: Extended training + LR warmup on best sweep configs."""

import torch, torch.nn as nn, yaml, time
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer

EXPERIMENTS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168,
     "cfg": {"patch_size": 8, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
             "dropout": 0.3, "trend_kernel": 15, "lr": 0.0005, "wd": 0.05}},
    {"name": "ETTh1_uni", "dataset": "ETTh1", "uni": True, "L": 336, "H": 168,
     "cfg": {"patch_size": 8, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
             "dropout": 0.3, "trend_kernel": 15, "lr": 0.0005, "wd": 0.05}},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192,
     "cfg": {"patch_size": 8, "d_model": 48, "num_layers": 3, "dim_feedforward": 256,
             "dropout": 0.3, "trend_kernel": 15, "lr": 0.001, "wd": 0.01}},
    {"name": "ETTm1_uni", "dataset": "ETTm1", "uni": True, "L": 1440, "H": 192,
     "cfg": {"patch_size": 16, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
             "dropout": 0.2, "trend_kernel": 25, "lr": 0.0005, "wd": 0.05}},
]

def create_model(cfg, input_dim, H, L):
    ps = cfg["patch_size"]
    while L // ps < 4: ps //= 2
    return CIDecompTransformer(input_dim=input_dim, forecast_length=H, lookback_length=L,
        patch_size=max(4,ps), d_model=cfg["d_model"], nhead=4 if cfg["d_model"]>=64 else 2,
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])

def train_and_eval(model, train_loader, val_loader, test_loader, scaler, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    # Extended training: 200 epochs, warmup + cosine
    max_epochs, warmup_epochs, patience = 200, 10, 30

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val, no_improve, best_state = float("inf"), 0, None

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            lb = batch["lookback"].to(device, non_blocking=True)
            fc = batch["forecast"].to(device, non_blocking=True)
            loss = nn.functional.mse_loss(model(lb), fc)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        vl = sum(nn.functional.mse_loss(model(b["lookback"].to(device)),
            b["forecast"].to(device)).item() for b in val_loader) / len(val_loader)
        if vl < best_val:
            best_val = vl; no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: no_improve += 1
        scheduler.step()
        if epoch >= 29 and no_improve >= patience: break
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train={loss.item():.4f} val={vl:.4f} best={best_val:.4f}")
    model.load_state_dict(best_state); model = model.to(device); model.eval()
    all_mae = []
    with torch.no_grad():
        for batch in test_loader:
            lb, tgt = batch["lookback"].to(device), batch["forecast"].to(device)
            nm, ns = batch["norm_mean"].to(device), batch["norm_std"].to(device)
            tg = scaler.transform(tgt * ns.unsqueeze(1) + nm.unsqueeze(1))
            pg = scaler.transform(model(lb) * ns.unsqueeze(1) + nm.unsqueeze(1))
            all_mae.append((pg - tg).abs().mean(dim=(1,2)).cpu())
    return torch.cat(all_mae).mean().item(), epoch+1

def main():
    print("EXP 19: Extended Training + LR Warmup")
    with open("configs/small.yaml") as f: base = yaml.safe_load(f)
    for exp in EXPERIMENTS:
        tl, vl, te, sc = create_dataloaders(data_path=base["data"]["data_path"],
            dataset_name=exp["dataset"], lookback_length=exp["L"], forecast_length=exp["H"],
            batch_size=64, num_workers=2, univariate=exp["uni"])
        D = 1 if exp["uni"] else 7
        m = create_model(exp["cfg"], D, exp["H"], exp["L"])
        p = sum(x.numel() for x in m.parameters())
        t0 = time.time()
        mae, epochs = train_and_eval(m, tl, vl, te, sc, exp["cfg"])
        print(f"  {exp['name']}: MAE={mae:.4f} | {p:,} params | {epochs} epochs | {time.time()-t0:.0f}s")
    print("Done.")

if __name__ == "__main__": main()
