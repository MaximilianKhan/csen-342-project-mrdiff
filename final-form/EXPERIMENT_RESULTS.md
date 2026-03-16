# Experiment Results — mr-Diff Improvement Campaign

> **Started:** 2026-03-15 | **Baseline:** 843K params, ~34 min full training
> **Method:** Cumulative improvements — each experiment builds on all previous ones.

## Baseline (Reference)

| Experiment | MAE | Paper MAE | Gap |
|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.42 | +13.0% |
| ETTh1 Uni | 0.2535 | 0.34 | -25.4% |
| ETTm1 Multi | 0.4204 | 0.37 | +13.6% |
| ETTm1 Uni | 0.2011 | 0.15 | +34.1% |

---

## Experiment 1: Remove the Detach — Joint End-to-End Training

**Change:** Removed `.detach()` from `residual = forecast - direct_pred.detach()` so diffusion gradients flow back through the DLinear backbone. Added `diffusion_loss_scale = 0.3` to prevent diffusion gradients from overwhelming the backbone.

**Training time:** 31.1 min (53-58 epochs across experiments)

| Experiment | Baseline MAE | Exp 1 MAE | Delta | vs Paper |
|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4765 | +0.4% | +13.5% |
| ETTh1 Uni | 0.2535 | 0.2543 | +0.3% | -25.2% |
| ETTm1 Multi | 0.4204 | 0.4224 | +0.5% | +14.2% |
| ETTm1 Uni | 0.2011 | 0.1988 | **-1.1%** | +32.5% |

**What worked:** ETTm1 Uni showed a small improvement (0.2011 → 0.1988), and the diffusion component is now slightly active (Direct MAE ≠ Full MAE, though the gap is tiny and sometimes negative).

**What didn't work:** ETTh1 Multi/Uni and ETTm1 Multi slightly regressed. The joint training with 0.3 loss scaling wasn't enough to make diffusion meaningfully contribute — the backbone still dominates, and allowing diffusion gradients through it introduced slight instability without corresponding benefit.

**Why:** The fundamental problem remains: the residuals after DLinear prediction are small and noise-like. Simply allowing gradient flow doesn't change what the diffusion sees — it still faces near-random residuals. The loss scaling may need tuning, or the architecture needs deeper changes to benefit from joint training.

**Potential next steps:** Self-conditioning (Exp 2) may help because it gives diffusion iterative refinement ability, making it better at handling small residuals.

---

## Experiment 2: Self-Conditioning

**Change:** Added self-conditioning to the denoising network. During training, with 50% probability, the denoiser runs twice per step — first to get a preliminary x0 estimate (stop-gradient), then concatenates that estimate alongside the noisy input for the real forward pass. The encoder input dimension doubled (input_dim * 2) to accommodate the concatenated x0_prev. During sampling, the previous step's x0_pred is passed as x0_prev for iterative refinement.

**Training time:** 32.7 min (52-65 epochs across experiments)

| Experiment | Baseline MAE | Exp 1 MAE | Exp 2 MAE | vs Baseline | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4765 | **0.4719** | **-0.5%** | +12.4% |
| ETTh1 Uni | 0.2535 | 0.2543 | **0.2523** | **-0.5%** | -25.8% |
| ETTm1 Multi | 0.4204 | 0.4224 | 0.4218 | +0.3% | +14.0% |
| ETTm1 Uni | 0.2011 | 0.1988 | 0.1999 | -0.6% | +33.3% |

**What worked:** ETTh1 results improved over both baseline and Exp 1. ETTh1 Multi recovered from Exp 1's regression and slightly beat the original baseline (0.4744 → 0.4719). ETTh1 Uni also improved (0.2535 → 0.2523). The Direct MAE vs Full MAE gap shows diffusion is now slightly contributing (Full < Direct for ETTh1 Uni: 0.2523 vs 0.2530).

**What didn't work:** ETTm1 Multi showed no meaningful change. ETTm1 Uni slightly regressed from Exp 1's best (0.1988 → 0.1999), though still better than baseline. The self-conditioning doubles forward passes 50% of the time during training, increasing compute per epoch, but the iterative refinement doesn't seem to help much when residuals are already near-noise.

**Why:** Self-conditioning helps most when the denoiser can meaningfully refine its own estimates. On ETTh1 (shorter sequences, more structured residuals), it provides a small benefit. On ETTm1 (longer sequences, noisier residuals), the initial x0 estimate from the denoiser is too poor to be useful as conditioning — garbage-in, garbage-out. The fundamental issue remains: the backbone leaves too little structured signal for diffusion.

**Potential next steps:** Cosine schedule + fewer diffusion steps (Exp 3) to improve training efficiency and noise distribution. Better noise scheduling may help the denoiser learn more from each gradient step.

---

## Experiment 3: Cosine Schedule

**Change:** Switched from linear beta schedule (`beta_start=1e-4, beta_end=0.1`) to cosine schedule (Nichol & Dhariwal 2021). The cosine schedule provides smoother noise progression with less wasted capacity at the endpoints. Config-only change: `schedule_type: cosine` in `small.yaml`.

**Training time:** ~25 min (46-98 epochs across experiments, early stopping triggered earlier for ETTm1 Multi)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 3 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.6709 | **+42.2%** | +59.7% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2813 | **+11.5%** | -17.3% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.6433 | **+52.5%** | +73.9% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2141 | **+7.1%** | +42.7% |

**Direct MAE vs Full MAE:** Diffusion still cosmetic (Direct ≈ Full across all experiments).

**What went wrong:** The cosine schedule catastrophically hurt multivariate performance (+42-53% regression) and also degraded univariate (+7-12%). The cosine schedule concentrates noise at lower levels (more steps with small noise), which may starve the denoiser of high-noise training signal on these small datasets. With only ~8K-10K samples, the model needs exposure to a wide range of noise levels per epoch — the linear schedule's uniform spread is actually better for our data regime.

**Why multivariate is hit hardest:** Multivariate (D=7) has 7x more variance to handle. The cosine schedule's slow noise ramp means the model sees mostly low-noise inputs during training, but at inference it must denoise from pure noise. The distribution mismatch is amplified by higher dimensionality.

**Verdict:** Cosine schedule is **rejected**. Reverting to linear schedule for all future experiments. Experiment 4 will build on Experiment 2 (our current best).

---
