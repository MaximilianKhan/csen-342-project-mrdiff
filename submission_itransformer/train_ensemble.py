#!/usr/bin/env python3
"""Exp 29: iTransformer ensemble.

For each multivariate benchmark, ensemble of:
  - iTransformer (cross-variate attention)
  - CI+AttnRes (temporal attention, complementary inductive bias)
  - CI+Decomp base (temporal attention, different hyperparams)

For univariate benchmarks, ensemble of CI models only (iTransformer
is degenerate with D=1 — single token, no meaningful attention).
"""
import torch, torch.nn as nn, yaml, time, random, json, os, copy
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer
from src.models.ci_attnres_transformer import CIAttnResDecompTransformer
from src.models.itransformer import ITransformerDecomp

BENCHMARKS = {
    "ETTh1_multi": {
        "dataset": "ETTh1", "uni": False, "L": 336, "H": 168,
        "models": [
            # 1. iTransformer — cross-variate attention (new)
            {"arch": "itransformer", "d_model": 64, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.0005, "wd": 0.05, "augment": True, "label": "itransformer"},
            # 2. CI+AttnRes+Aug (Exp 26/28 champion, temporal attention)
            {"arch": "attnres", "patch_size": 8, "patch_stride": 4, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.0005, "wd": 0.05, "augment": True, "label": "attnres_aug"},
            # 3. CI base cfg07 (Exp 18 sweep, different capacity)
            {"arch": "base", "patch_size": 16, "patch_stride": 8, "d_model": 64, "num_layers": 3,
             "dim_feedforward": 64, "dropout": 0.2, "trend_kernel": 15,
             "lr": 0.002, "wd": 0.005, "augment": False, "label": "cfg07"},
        ],
    },
    "ETTh1_uni": {
        "dataset": "ETTh1", "uni": True, "L": 336, "H": 168,
        "models": [
            # CI only for univariate — iTransformer is degenerate at D=1
            {"arch": "base", "patch_size": 8, "patch_stride": 4, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.0005, "wd": 0.05, "augment": False, "label": "cfg01"},
            {"arch": "base", "patch_size": 12, "patch_stride": 6, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 64, "dropout": 0.5, "trend_kernel": 15,
             "lr": 0.001, "wd": 0.05, "augment": False, "label": "cfg03"},
            {"arch": "attnres", "patch_size": 8, "patch_stride": 4, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.0005, "wd": 0.05, "augment": True, "label": "attnres_aug"},
        ],
    },
    "ETTm1_multi": {
        "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192,
        "models": [
            # 1. iTransformer — cross-variate attention (new)
            {"arch": "itransformer", "d_model": 64, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.001, "wd": 0.01, "augment": False, "label": "itransformer"},
            # 2. CI base sweep champion cfg10
            {"arch": "base", "patch_size": 8, "d_model": 48, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.001, "wd": 0.01, "augment": False, "label": "cfg10"},
            # 3. CI base cfg02 (different size)
            {"arch": "base", "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2, "trend_kernel": 15,
             "lr": 0.0005, "wd": 0.005, "augment": False, "label": "cfg02"},
        ],
    },
    "ETTm1_uni": {
        "dataset": "ETTm1", "uni": True, "L": 1440, "H": 192,
        "models": [
            # CI only for univariate
            {"arch": "base", "patch_size": 16, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2, "trend_kernel": 25,
             "lr": 0.0005, "wd": 0.05, "augment": False, "label": "cfg06"},
            {"arch": "base", "patch_size": 24, "d_model": 32, "num_layers": 2,
             "dim_feedforward": 128, "dropout": 0.5, "trend_kernel": 49,
             "lr": 0.001, "wd": 0.05, "augment": False, "label": "cfg16"},
            {"arch": "base", "patch_size": 12, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3, "trend_kernel": 15,
             "lr": 0.001, "wd": 0.005, "augment": False, "label": "cfg19"},
        ],
    },
}


def augment(lookback, forecast):
    B, H, D = lookback.shape
    if random.random() < 0.3:
        lookback = lookback + torch.randn_like(lookback) * 0.01
    if random.random() < 0.3:
        scale = 0.95 + 0.1 * torch.rand(B, 1, 1, device=lookback.device)
        lookback = lookback * scale
        forecast = forecast * scale
    if random.random() < 0.3:
        mask_len = random.randint(H // 20, H // 10)
        start = random.randint(0, H - mask_len - 1)
        lookback = lookback.clone()
        lookback[:, start:start+mask_len, :] = 0.0
    return lookback, forecast


def create_model(cfg, D, H, L):
    nhead = 4 if cfg["d_model"] >= 64 else 2

    if cfg["arch"] == "itransformer":
        return ITransformerDecomp(
            input_dim=D, forecast_length=H, lookback_length=L,
            d_model=cfg["d_model"], nhead=nhead,
            num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])

    ps = cfg["patch_size"]
    while L // ps < 4: ps //= 2
    ps = max(4, ps)
    stride = cfg.get("patch_stride", None)
    if cfg["arch"] == "attnres":
        return CIAttnResDecompTransformer(
            input_dim=D, forecast_length=H, lookback_length=L,
            patch_size=ps, patch_stride=stride, d_model=cfg["d_model"], nhead=nhead,
            num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])
    return CIDecompTransformer(
        input_dim=D, forecast_length=H, lookback_length=L,
        patch_size=ps, patch_stride=stride, d_model=cfg["d_model"], nhead=nhead,
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], trend_kernel=cfg["trend_kernel"])


def train_model(model, train_loader, val_loader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    best_val, no_imp, best_st = float("inf"), 0, None
    do_aug = cfg.get("augment", False)

    for epoch in range(100):
        model.train()
        for b in train_loader:
            lb = b["lookback"].to(device, non_blocking=True)
            fc = b["forecast"].to(device, non_blocking=True)
            if do_aug: lb, fc = augment(lb, fc)
            loss = nn.functional.mse_loss(model(lb), fc)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        vl = sum(nn.functional.mse_loss(model(b["lookback"].to(device)),
            b["forecast"].to(device)).item() for b in val_loader) / len(val_loader)
        if vl < best_val:
            best_val = vl; no_imp = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: no_imp += 1
        sched.step()
        if epoch >= 29 and no_imp >= 20: break

    return best_st, epoch + 1


def evaluate_ensemble(models, test_loader, scaler, device):
    all_mae_ensemble = []
    all_mse_ensemble = []
    all_mae_individual = [[] for _ in models]

    with torch.no_grad():
        for b in test_loader:
            lb = b["lookback"].to(device)
            tgt = b["forecast"].to(device)
            nm, ns = b["norm_mean"].to(device), b["norm_std"].to(device)
            tg = scaler.transform(tgt * ns.unsqueeze(1) + nm.unsqueeze(1))

            preds_g = []
            for i, m in enumerate(models):
                pred = m(lb)
                pg = scaler.transform(pred * ns.unsqueeze(1) + nm.unsqueeze(1))
                preds_g.append(pg)
                all_mae_individual[i].append((pg - tg).abs().mean(dim=(1, 2)).cpu())

            ensemble_pg = torch.stack(preds_g).mean(dim=0)
            all_mae_ensemble.append((ensemble_pg - tg).abs().mean(dim=(1, 2)).cpu())
            all_mse_ensemble.append(((ensemble_pg - tg) ** 2).mean(dim=(1, 2)).cpu())

    ensemble_mae = torch.cat(all_mae_ensemble).mean().item()
    ensemble_mse = torch.cat(all_mse_ensemble).mean().item()
    individual_maes = [torch.cat(m).mean().item() for m in all_mae_individual]
    return ensemble_mae, ensemble_mse, individual_maes


def main():
    print("=" * 70, flush=True)
    print("EXP 29: iTransformer + CI Heterogeneous Ensemble", flush=True)
    print("  iTransformer for multivariate, CI for univariate", flush=True)
    print("=" * 70, flush=True)

    with open("configs/small.yaml") as f:
        base = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bench_name, bench in BENCHMARKS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"  {bench_name}", flush=True)
        print(f"{'='*60}", flush=True)

        D = 1 if bench["uni"] else 7
        tl, vl, te, sc = create_dataloaders(
            data_path=base["data"]["data_path"], dataset_name=bench["dataset"],
            lookback_length=bench["L"], forecast_length=bench["H"],
            batch_size=64, num_workers=2, univariate=bench["uni"])

        trained_models = []
        t0_total = time.time()

        for i, cfg in enumerate(bench["models"]):
            model = create_model(cfg, D, bench["H"], bench["L"])
            params = sum(p.numel() for p in model.parameters())
            t0 = time.time()
            best_st, epochs = train_model(model, tl, vl, cfg)
            elapsed = time.time() - t0

            model.load_state_dict(best_st)
            model = model.to(device)
            model.eval()
            trained_models.append(model)

            print(f"    Model {i+1}/3 [{cfg['label']}]: {params:,} params | "
                  f"{epochs} ep | {elapsed:.0f}s", flush=True)

        ensemble_mae, ensemble_mse, individual_maes = evaluate_ensemble(
            trained_models, te, sc, device)
        total_time = time.time() - t0_total

        print(f"\n  Individual MAEs: {' | '.join(f'{m:.4f}' for m in individual_maes)}", flush=True)
        print(f"  ** ENSEMBLE MAE: {ensemble_mae:.4f} | MSE: {ensemble_mse:.4f} **", flush=True)
        print(f"  Total time: {total_time:.0f}s", flush=True)

        os.makedirs(f"results/{bench_name}", exist_ok=True)
        with open(f"results/{bench_name}/metrics.json", "w") as f:
            json.dump({"mae": ensemble_mae, "mse": ensemble_mse}, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__": main()
