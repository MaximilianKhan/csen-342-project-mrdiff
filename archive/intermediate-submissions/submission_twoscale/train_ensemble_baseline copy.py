#!/usr/bin/env python3
"""Exp 31 ensemble — two-scale + CI base for ETTm1.

For ETTm1 multi and uni, ensemble the two-scale model with the best
baseline CI config. Two-scale attends over coarse/mid/fine components
explicitly; CI base attends over a single trend/residual split. Their
decomposition strategies differ so their errors are partially uncorrelated.

ETTh1 configs unchanged from Exp 30 ensemble.
"""
import torch, torch.nn as nn, yaml, time, random, json, os
from src.data.dataset import create_dataloaders
from src.models.ci_decomp_transformer import CIDecompTransformer
from src.models.ci_attnres_transformer import CIAttnResDecompTransformer
from src.models.ci_twoscale_transformer import CITwoScaleTransformer

ENSEMBLE_CONFIGS = [
    {
        "name": "ETTh1_multi",
        "dataset": "ETTh1", "uni": False, "L": 336, "H": 168,
        "members": [
            {"arch": "attnres", "augment": True,
             "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3,
             "trend_kernel": 15, "lr": 0.0005, "wd": 0.05},
            {"arch": "base", "augment": False,
             "patch_size": 16, "d_model": 64, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel": 15, "lr": 0.002, "wd": 0.05},
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel": 15, "lr": 0.0005, "wd": 0.05},
        ],
    },
    {
        "name": "ETTh1_uni",
        "dataset": "ETTh1", "uni": True, "L": 336, "H": 168,
        "members": [
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3,
             "trend_kernel": 15, "lr": 0.0005, "wd": 0.05},
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.3,
             "trend_kernel": 15, "lr": 0.001, "wd": 0.05},
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel": 15, "lr": 0.0005, "wd": 0.01},
        ],
    },
    {
        "name": "ETTm1_multi",
        "dataset": "ETTm1", "uni": False, "L": 1440, "H": 192,
        "members": [
            # Two-scale: explicit daily cycle separation
            {"arch": "twoscale", "augment": False,
             "patch_size": 8, "d_model": 48, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.3,
             "trend_kernel_fine": 15, "trend_kernel_coarse": 96,
             "lr": 0.001, "wd": 0.01},
            # CI base: best config from Exp 18 sweep
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 48, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.3,
             "trend_kernel": 15, "lr": 0.001, "wd": 0.01},
            # CI base: second-best config from Exp 18
            {"arch": "base", "augment": False,
             "patch_size": 8, "d_model": 48, "num_layers": 3,
             "dim_feedforward": 256, "dropout": 0.2,
             "trend_kernel": 15, "lr": 0.001, "wd": 0.01},
        ],
    },
    {
        "name": "ETTm1_uni",
        "dataset": "ETTm1", "uni": True, "L": 1440, "H": 192,
        "members": [
            # Two-scale
            {"arch": "twoscale", "augment": False,
             "patch_size": 16, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel_fine": 25, "trend_kernel_coarse": 96,
             "lr": 0.0005, "wd": 0.05},
            # CI base: best config from Exp 30
            {"arch": "base", "augment": False,
             "patch_size": 16, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel": 25, "lr": 0.0005, "wd": 0.05},
            # CI base: slight lr variation
            {"arch": "base", "augment": False,
             "patch_size": 16, "d_model": 32, "num_layers": 3,
             "dim_feedforward": 128, "dropout": 0.2,
             "trend_kernel": 25, "lr": 0.001, "wd": 0.01},
        ],
    },
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


def build_model(cfg, D, H, L):
    ps = cfg.get("patch_size", 8)
    while L // ps < 4: ps //= 2
    ps = max(4, ps)
    nhead = 4 if cfg["d_model"] >= 64 else 2

    if cfg["arch"] == "twoscale":
        return CITwoScaleTransformer(
            input_dim=D, forecast_length=H, lookback_length=L,
            patch_size=ps, patch_stride=None,
            d_model=cfg["d_model"], nhead=nhead,
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            trend_kernel_fine=cfg["trend_kernel_fine"],
            trend_kernel_coarse=cfg["trend_kernel_coarse"],
        )
    if cfg["arch"] == "attnres":
        return CIAttnResDecompTransformer(
            D, H, L, ps, cfg["d_model"], nhead,
            cfg["num_layers"], cfg["dim_feedforward"],
            cfg["dropout"], cfg["trend_kernel"])
    return CIDecompTransformer(
        D, H, L, ps, cfg["d_model"], nhead,
        cfg["num_layers"], cfg["dim_feedforward"],
        cfg["dropout"], cfg["trend_kernel"])


def train_model(model, tl, vl, cfg):
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
        with torch.no_grad():
            vl_sum = sum(
                nn.functional.mse_loss(
                    model(b["lookback"].to(device)),
                    b["forecast"].to(device)
                ).item() for b in vl
            )
        avg_val = vl_sum / len(vl)
        if avg_val < best_val:
            best_val = avg_val; no_imp = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        sched.step()
        if epoch >= 29 and no_imp >= 20:
            break
    model.load_state_dict(best_st)
    return model.to(device), epoch + 1


def evaluate_ensemble(models, te, sc, device, forecast_length):
    all_mae, all_mse = [], []
    with torch.no_grad():
        for b in te:
            lb  = b["lookback"].to(device)
            tgt = b["forecast"].to(device)
            nm  = b["norm_mean"].to(device)
            ns  = b["norm_std"].to(device)
            preds = torch.stack([m(lb) for m in models], dim=0).mean(dim=0)
            tg = sc.transform(tgt   * ns.unsqueeze(1) + nm.unsqueeze(1))
            pg = sc.transform(preds * ns.unsqueeze(1) + nm.unsqueeze(1))
            all_mae.append((pg - tg).abs().mean(dim=(1, 2)).cpu())
            all_mse.append(((pg - tg) ** 2).mean(dim=(1, 2)).cpu())
    return (torch.cat(all_mae).mean().item(),
            torch.cat(all_mse).mean().item())


def main():
    print("Exp 31 — Two-Scale + CI Ensemble", flush=True)
    print("=" * 70, flush=True)

    with open("configs/small.yaml") as f:
        base = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bench in ENSEMBLE_CONFIGS:
        name = bench["name"]
        print(f"\n{'=' * 60}", flush=True)
        print(f"  {name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        tl, vl, te, sc = create_dataloaders(
            data_path=base["data"]["data_path"],
            dataset_name=bench["dataset"],
            lookback_length=bench["L"],
            forecast_length=bench["H"],
            batch_size=64, num_workers=2,
            univariate=bench["uni"],
        )
        D = 1 if bench["uni"] else 7
        t0 = time.time()
        trained_models = []
        individual_maes = []

        for i, mcfg in enumerate(bench["members"]):
            mt0 = time.time()
            model = build_model(mcfg, D, bench["H"], bench["L"])
            p = sum(x.numel() for x in model.parameters())
            model, ep = train_model(model, tl, vl, mcfg)
            model.eval()

            # Individual MAE
            ind_mae, _ = evaluate_ensemble([model], te, sc, device, bench["H"])
            individual_maes.append(ind_mae)
            trained_models.append(model)

            label = mcfg["arch"]
            if mcfg.get("augment"): label += "_aug"
            print(f"    Model {i+1}/{len(bench['members'])} [{label}]: "
                  f"{p:,} params | {ep} ep | {time.time()-mt0:.0f}s", flush=True)

        ind_str = " | ".join(f"{m:.4f}" for m in individual_maes)
        print(f"\n  Individual MAEs: {ind_str}", flush=True)

        ens_mae, ens_mse = evaluate_ensemble(trained_models, te, sc, device, bench["H"])
        print(f"  ** ENSEMBLE MAE: {ens_mae:.4f} | MSE: {ens_mse:.4f} **", flush=True)
        print(f"  Total time: {time.time()-t0:.0f}s", flush=True)

        os.makedirs(f"results/{name}", exist_ok=True)
        with open(f"results/{name}/metrics.json", "w") as f:
            json.dump({
                "mae": ens_mae, "mse": ens_mse,
                "individual_maes": individual_maes,
            }, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()