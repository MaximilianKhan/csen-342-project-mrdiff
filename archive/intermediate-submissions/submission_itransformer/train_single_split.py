#!/usr/bin/env python3
"""Exp 30: Split transformer — separate encoders for trend and residual.

The shared transformer must represent both smooth trend signals and noisy
residual signals with the same weights. Separate encoders let each specialise.
split_transformer=True adds ~50% more transformer params but keeps total low
because CI keeps the effective dataset large (D*N sequences, not N).
"""
import torch, torch.nn as nn, yaml, time, random, json, os
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer
from src.models.ci_attnres_transformer import CIAttnResDecompTransformer

BEST_CONFIGS = [
    {"name": "ETTh1_multi", "dataset": "ETTh1", "uni": False, "L": 336, "H": 168,
     "arch": "split", "augment": True,
     "patch_size": 8, "patch_stride": 4, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
     "dropout": 0.3, "trend_kernel": 15, "lr": 0.0005, "wd": 0.05},
    {"name": "ETTh1_uni", "dataset": "ETTh1", "uni": True, "L": 336, "H": 168,
     "arch": "split", "augment": False,
     "patch_size": 8, "patch_stride": 4, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
     "dropout": 0.3, "trend_kernel": 15, "lr": 0.0005, "wd": 0.05},
    {"name": "ETTm1_multi", "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192,
     "arch": "split", "augment": False,
     "patch_size": 8, "d_model": 48, "num_layers": 3, "dim_feedforward": 256,
     "dropout": 0.3, "trend_kernel": 15, "lr": 0.001, "wd": 0.01},
    {"name": "ETTm1_uni", "dataset": "ETTm1", "uni": True, "L": 1440, "H": 192,
     "arch": "split", "augment": False,
     "patch_size": 16, "d_model": 32, "num_layers": 3, "dim_feedforward": 128,
     "dropout": 0.2, "trend_kernel": 25, "lr": 0.0005, "wd": 0.05},
]


def augment(lookback, forecast):
    B, H, D = lookback.shape
    if random.random() < 0.3:
        lookback = lookback + torch.randn_like(lookback) * 0.01
    if random.random() < 0.3:
        scale = 0.95 + 0.1 * torch.rand(B, 1, 1, device=lookback.device)
        lookback = lookback * scale; forecast = forecast * scale
    if random.random() < 0.3:
        mask_len = random.randint(H // 20, H // 10)
        start = random.randint(0, H - mask_len - 1)
        lookback = lookback.clone(); lookback[:, start:start+mask_len, :] = 0.0
    return lookback, forecast


def create_model(cfg, D, H, L):
    ps = cfg.get("patch_size", 8)
    while L // ps < 4: ps //= 2
    ps = max(4, ps)
    nhead = 4 if cfg["d_model"] >= 64 else 2
    stride = cfg.get("patch_stride", None)

    if cfg["arch"] == "split":
        return CIDecompTransformer(D, H, L, ps, stride, cfg["d_model"], nhead,
            cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"],
            cfg["trend_kernel"], split_transformer=True)
    if cfg["arch"] == "attnres":
        return CIAttnResDecompTransformer(D, H, L, ps, stride, cfg["d_model"], nhead,
            cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"], cfg["trend_kernel"])
    return CIDecompTransformer(D, H, L, ps, stride, cfg["d_model"], nhead,
        cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"], cfg["trend_kernel"])


def train_and_eval(model, tl, vl, te, sc, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    best_val, no_imp, best_st = float("inf"), 0, None
    do_aug = cfg.get("augment", False)
    for epoch in range(100):
        model.train()
        for b in tl:
            lb = b["lookback"].to(device, non_blocking=True)
            fc = b["forecast"].to(device, non_blocking=True)
            if do_aug: lb, fc = augment(lb, fc)
            loss = nn.functional.mse_loss(model(lb), fc)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        vl_sum = sum(nn.functional.mse_loss(model(b["lookback"].to(device)),
            b["forecast"].to(device)).item() for b in vl)
        avg_val = vl_sum / len(vl)
        if avg_val < best_val:
            best_val = avg_val; no_imp = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: no_imp += 1
        sched.step()
        if epoch >= 29 and no_imp >= 20: break
    model.load_state_dict(best_st); model = model.to(device); model.eval()
    all_mae, all_mse = [], []
    with torch.no_grad():
        for b in te:
            lb, tgt = b["lookback"].to(device), b["forecast"].to(device)
            nm, ns = b["norm_mean"].to(device), b["norm_std"].to(device)
            tg = sc.transform(tgt * ns.unsqueeze(1) + nm.unsqueeze(1))
            pg = sc.transform(model(lb) * ns.unsqueeze(1) + nm.unsqueeze(1))
            all_mae.append((pg - tg).abs().mean(dim=(1, 2)).cpu())
            all_mse.append(((pg - tg) ** 2).mean(dim=(1, 2)).cpu())
    mae = torch.cat(all_mae).mean().item()
    mse = torch.cat(all_mse).mean().item()
    os.makedirs(f"results/{cfg['name']}", exist_ok=True)
    with open(f"results/{cfg['name']}/metrics.json", "w") as f:
        json.dump({"mae": mae, "mse": mse}, f, indent=2)
    return mae, epoch + 1


def main():
    print("Exp 30: Split transformer (separate encoders for trend + residual)", flush=True)
    print("=" * 60, flush=True)
    with open("configs/small.yaml") as f: base = yaml.safe_load(f)
    for cfg in BEST_CONFIGS:
        tl, vl, te, sc = create_dataloaders(data_path=base["data"]["data_path"],
            dataset_name=cfg["dataset"], lookback_length=cfg["L"],
            forecast_length=cfg["H"], batch_size=64, num_workers=2, univariate=cfg["uni"])
        D = 1 if cfg["uni"] else 7
        m = create_model(cfg, D, cfg["H"], cfg["L"])
        p = sum(x.numel() for x in m.parameters())
        t0 = time.time()
        mae, ep = train_and_eval(m, tl, vl, te, sc, cfg)
        print(f"  {cfg['name']}: MAE={mae:.4f} | {p:,} params | {ep} ep | {time.time()-t0:.0f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__": main()
