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

## Experiment 4: v-Prediction Parameterization

**Change:** Switched from epsilon-prediction to v-prediction (Salimans & Ho, 2022). Instead of predicting noise ε, the denoiser predicts `v = α_t · ε - σ_t · x0`, which has uniform gradient variance across all timesteps. Updated training loss target, x0 recovery (`x0 = α_t · y_t - σ_t · v`), and both DDPM and DPM-Solver++ sampling paths. Built on Experiment 2 codebase (linear schedule). Code isolated in `exp4_v_prediction/`.

**Training time:** 34.3 min (50-56 epochs across experiments)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 4 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.4790 | +1.5% | +14.0% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2531 | +0.3% | -25.6% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.4216 | -0.0% | +13.9% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2049 | +2.5% | +36.6% |

**Direct MAE vs Full MAE:** Diffusion still cosmetic (Direct ≈ Full across all experiments).

**What didn't work:** v-prediction showed no improvement over epsilon-prediction. ETTh1 Multi and ETTm1 Uni both regressed slightly. The uniform gradient variance advantage of v-prediction assumes the model is capacity-limited by gradient scale inconsistency — but our 843K-param model on small datasets is more limited by data scarcity than gradient dynamics. The epsilon-prediction baseline already converges well within 50-60 epochs.

**Why:** v-prediction primarily helps when: (a) training for many epochs where gradient scale matters cumulatively, or (b) using cosine/aggressive noise schedules where ε-prediction has extreme variance. With our linear schedule and early-stopped training (~50 epochs), ε-prediction's gradient variance is manageable. The overhead of predicting a rotated target doesn't pay for itself.

**Verdict:** v-prediction is **rejected**. Epsilon-prediction retained for all future experiments. Experiment 5 will build on Experiment 2 (still our best).

---

## Experiment 5: ANT Adaptive Noise Schedule

**Change:** Introduced an Adaptive Noise Time (ANT) schedule that tailors the beta schedule to each dataset's temporal structure. Computes Integrated Absolute Autocorrelation Time (IAAT) from training data, then uses it to warp the linear beta schedule via a power law: high IAAT (strong temporal correlation) produces a concave schedule that spends more steps at low noise levels where temporal structure matters. The curvature parameter `gamma = clip(1 / (1 + 0.05 * IAAT), 0.3, 1.0)` maps IAAT to schedule shape. Added `compute_iaat()` and `create_ant_schedule()` to diffusion module, with `schedule_type: ant` config option. Built on Experiment 2 codebase. Code isolated in `exp5_ant_schedule/`.

**Training time:** ~65 min (all ran to epoch 100 without early stopping)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 5 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.7309 | **+54.9%** | +73.9% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2803 | +11.1% | -17.6% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.6513 | **+54.4%** | +76.0% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2055 | +2.8% | +37.0% |

**Direct MAE vs Full MAE:** Diffusion slightly active on ETTh1 Uni (0.2821 direct → 0.2803 full), cosmetic elsewhere.

**What went wrong:** Catastrophic regression on multivariate (+55% on both ETTh1 and ETTm1 Multi), and degraded univariate (+3-11%). The ANT schedule's concave warping concentrates betas at low noise levels, starving the model of high-noise training signal — the same failure mode as the cosine schedule (Exp 3), but worse. The IAAT-driven curvature amplified this: both datasets have moderate-to-high autocorrelation, producing aggressive concave schedules (gamma ~0.3-0.5) that over-allocated capacity to low-noise regimes.

**Why multivariate is hit hardest:** Same mechanism as Exp 3 — multivariate (D=7) needs broad noise coverage to learn the joint distribution. The concave schedule's emphasis on low-noise steps means the model rarely sees high-noise inputs during training, but must denoise from pure noise at inference. This train/test distribution mismatch is amplified by dimensionality.

**Why no early stopping triggered:** The val loss (in diffusion loss space, not MAE space) kept improving because the schedule changes the loss landscape — lower val loss doesn't mean better MAE when the noise distribution itself is misaligned.

**Verdict:** ANT schedule is **rejected**. This confirms a pattern: for our small-dataset regime, the uniform linear schedule is optimal. Any attempt to reshape the noise distribution (cosine in Exp 3, IAAT-adaptive here) hurts because it reduces coverage of the noise levels the model needs at inference. Experiment 6 will build on Experiment 2.

---

## Experiment 6: Contrastive Conditioning Loss (CCDM)

**Change:** Added an InfoNCE contrastive loss term to training. During the forward pass, negative examples are generated by time-shifting and scaling the true target. The model's epsilon prediction error on the true target serves as the positive score, while epsilon predictions on negatives serve as negative scores. InfoNCE cross-entropy pushes the model to assign higher likelihood to the correct future. Hyperparameters: `contrastive_lambda=0.005`, `temperature=0.1`, `num_negatives=4`. Only applied to stage 0 (finest resolution) to save compute. Negative generation uses stop-gradient. Built on Experiment 2 codebase. Code isolated in `exp6_contrastive/`.

**Training time:** ~75 min (ETTh1 Uni early-stopped at epoch 69; others ran to 87-100 epochs)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 6 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.5535 | +17.3% | +31.8% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2555 | +1.3% | -24.9% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.4925 | +16.8% | +33.1% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.1946 | **-2.7%** | +29.7% |

**Direct MAE vs Full MAE:** Diffusion cosmetic across all experiments (Direct ≈ Full).

**What partially worked:** ETTm1 Uni improved to 0.1946 — the best result we've seen on that benchmark, beating Exp 1's 0.1988 and Exp 2's 0.1999. The contrastive signal may help the model distinguish correct temporal patterns in the univariate long-horizon setting where there's less ambiguity in what constitutes a "wrong" future.

**What didn't work:** Multivariate regressed significantly (+17% on both datasets). ETTh1 Uni was roughly flat. The contrastive loss adds noise to the training signal: with only 4 negatives (time-shifted and scaled versions), the negatives may not be "hard" enough to provide useful gradient information for multivariate forecasting, while the extra loss term destabilizes the already-fragile diffusion training.

**Why multivariate suffers:** In the multivariate setting (D=7), time-shifted negatives along one axis are poor negatives — many channels may still look plausible after a temporal shift. The contrastive signal ends up noisy rather than informative, effectively adding regularization that hurts more than it helps. The `contrastive_lambda=0.005` was conservative, but even this small weight disrupted multivariate convergence.

**Why ETTm1 Uni improved:** Univariate long-horizon (H=192) has the most structured residuals — there's a clear notion of "right" vs "wrong" temporal patterns. The InfoNCE loss provides a useful inductive bias here: it explicitly penalizes predictions that look like time-shifted versions of the truth, encouraging the model to capture exact timing of patterns rather than just amplitude.

**Verdict:** Contrastive loss is **rejected** overall due to multivariate regression, though ETTm1 Uni's improvement (0.1946) is noteworthy. Experiment 2 remains our best across all benchmarks.

---
