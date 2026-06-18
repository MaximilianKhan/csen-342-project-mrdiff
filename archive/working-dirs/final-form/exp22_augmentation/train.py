#!/usr/bin/env python3
"""Exp 22: CI+Decomp Transformer with temporal data augmentation."""
import torch, torch.nn as nn, yaml, time, random
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

def augment(lookback, forecast):
    """Apply temporal augmentations with 50% probability each."""
    B, H, D = lookback.shape
    # 1. Gaussian jitter (σ=0.03)
    if random.random() < 0.5:
        lookback = lookback + torch.randn_like(lookback) * 0.03
    # 2. Random scaling (0.9-1.1 per sample)
    if random.random() < 0.5:
        scale = 0.9 + 0.2 * torch.rand(B, 1, 1, device=lookback.device)
        lookback = lookback * scale
        forecast = forecast * scale
    # 3. Temporal shift (±2 timesteps via roll)
    if random.random() < 0.5:
        shift = random.randint(-2, 2)
        lookback = torch.roll(lookback, shift, dims=1)
    return lookback, forecast

def create_model(cfg, D, H, L):
    ps = cfg["patch_size"]
    while L // ps < 4: ps //= 2
    return CIDecompTransformer(input_dim=D, forecast_length=H, lookback_length=L,
        patch_size=max(4,ps), d_model=cfg["d_model"], nhead=4 if cfg["d_model"]>=64 else 2,
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])

def train_and_eval(model, tl, vl, te, sc, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    best_val, no_imp, best_st = float("inf"), 0, None
    for epoch in range(100):
        model.train()
        for b in tl:
            lb = b["lookback"].to(device, non_blocking=True)
            fc = b["forecast"].to(device, non_blocking=True)
            lb, fc = augment(lb, fc)  # Apply augmentation
            loss = nn.functional.mse_loss(model(lb), fc)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        v = sum(nn.functional.mse_loss(model(b["lookback"].to(device)), b["forecast"].to(device)).item() for b in vl)/len(vl)
        if v < best_val: best_val=v; no_imp=0; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: no_imp += 1
        sched.step()
        if epoch >= 29 and no_imp >= 20: break
    model.load_state_dict(best_st); model=model.to(device); model.eval()
    all_mae=[]
    with torch.no_grad():
        for b in te:
            lb,tgt=b["lookback"].to(device),b["forecast"].to(device)
            nm,ns=b["norm_mean"].to(device),b["norm_std"].to(device)
            tg=sc.transform(tgt*ns.unsqueeze(1)+nm.unsqueeze(1))
            pg=sc.transform(model(lb)*ns.unsqueeze(1)+nm.unsqueeze(1))
            all_mae.append((pg-tg).abs().mean(dim=(1,2)).cpu())
    return torch.cat(all_mae).mean().item(), epoch+1

def main():
    print("EXP 22: CI+Decomp + Temporal Augmentation")
    with open("configs/small.yaml") as f: base=yaml.safe_load(f)
    for exp in EXPERIMENTS:
        tl,vl,te,sc=create_dataloaders(data_path=base["data"]["data_path"],
            dataset_name=exp["dataset"],lookback_length=exp["L"],forecast_length=exp["H"],
            batch_size=64,num_workers=2,univariate=exp["uni"])
        D=1 if exp["uni"] else 7
        m=create_model(exp["cfg"],D,exp["H"],exp["L"])
        p=sum(x.numel() for x in m.parameters())
        t0=time.time()
        mae,epochs=train_and_eval(m,tl,vl,te,sc,exp["cfg"])
        print(f"  {exp['name']}: MAE={mae:.4f} | {p:,} params | {epochs} ep | {time.time()-t0:.0f}s")
    print("Done.")

if __name__=="__main__": main()
