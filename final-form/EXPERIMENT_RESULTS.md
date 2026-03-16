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

## Experiment 7: Multi-Granularity Guided Diffusion (MG-TSD)

**Change:** Added a multi-granularity guidance loss to training. Computes 3 progressively smoothed versions of the target using average pooling, then adds a weighted MSE loss that guides the denoiser's x0 predictions: at high noise levels (large k), the loss pushes toward coarse structure; at low noise levels (small k), toward fine details. Loss weight: 0.05. Built on Experiment 2 codebase. Code isolated in `exp7_multi_granularity/`.

**Training time:** ~91 min total (all 4 experiments ran to 98-100 epochs — no early stopping triggered)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 7 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.5653 | +19.8% | +34.6% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2558 | +1.4% | -24.8% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.4819 | +14.3% | +30.2% |
| ETTm1 Uni | 0.2011 | 0.1999 | **0.1913** | **-4.3%** | +27.5% |

**Direct MAE vs Full MAE:** Diffusion cosmetic across all experiments (Direct ≈ Full).

**What worked:** ETTm1 Uni improved to 0.1913 — our **new best** on that benchmark, beating Exp 6's 0.1946 and Exp 2's 0.1999. The multi-granularity guidance provides a curriculum-like training signal that helps the denoiser on univariate long-horizon data where the smoothed targets are genuinely informative intermediate representations.

**What didn't work:** ETTh1 Multi regressed +19.8% and ETTm1 Multi regressed +14.3%. The guidance loss adds noise to multivariate training because the avg-pooling smoothing operates temporally but ignores cross-channel structure. For D=7, the "coarse" version of the target blurs meaningful inter-variable dynamics. The models also ran to 100 epochs without early stopping, suggesting the guidance loss altered the loss landscape enough that val loss kept slowly decreasing while test MAE worsened — a classic case of optimizing a proxy (guided diffusion loss) that diverges from the true metric (forecast MAE).

**Why ETTm1 Uni benefits:** With D=1, the multi-granularity targets are clean temporal smoothings of a single variable. The coarse→fine guidance gives the denoiser a meaningful trajectory from trend to detail, and with H=192 there's enough temporal structure for this to help. Multivariate (D=7) doesn't have this property — smoothing 7 channels independently produces a poor "coarse" target.

**Verdict:** Multi-granularity guidance is **rejected** overall due to multivariate regression, though ETTm1 Uni's 0.1913 is our new best on that benchmark.

---

## Experiment 8: Channel-Aware Denoising Architecture

**Change:** Replaced the shared Conv1d encoder in the denoising network with a `ChannelIndependentEncoder`: each of the D=7 variables gets its own temporal Conv1d processing, followed by cross-channel MultiheadAttention to mix information across variables. For univariate (D=1), falls back to the standard encoder. Built on Experiment 2 codebase. Code isolated in `exp8_channel_aware/`.

**Training time:** ~89.5 min total (ETTh1 ~10 min, ETTm1 ~79 min; epochs ranged 66-100)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 8 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.9313 | **+97.3%** | +121.7% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2552 | +1.1% | -24.9% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.9136 | **+116.6%** | +147.0% |
| ETTm1 Uni | 0.2011 | 0.1999 | **0.1928** | **-3.6%** | +28.5% |

**Direct MAE vs Full MAE:** Diffusion cosmetic across all experiments (Direct ≈ Full). ETTm1 Uni: 0.1926 direct vs 0.1928 full.

**What worked:** ETTm1 Uni improved to 0.1928 — a solid gain over Exp 2's 0.1999 (-3.6%), though not quite matching Exp 6's best of 0.1946. The standard encoder fallback for D=1 means this univariate gain comes from other cumulative effects (Exp 1+2), not the channel-aware architecture itself.

**What catastrophically failed:** Multivariate performance nearly doubled in error (+97% ETTh1, +117% ETTm1). The per-channel encoder with cross-channel attention massively overparameterized the denoising path for our small datasets. With D=7 channels each getting independent Conv1d stacks plus attention overhead, the denoiser has far more capacity than it can usefully fill with ~8K-10K training samples.

**Why:** The channel-independent + attention architecture assumes there's enough data to learn per-variable temporal patterns AND cross-variable correlations. With ETT's small training sets, the per-channel encoders overfit to noise in each variable independently, and the cross-channel attention learns spurious correlations. The shared Conv1d in the baseline acts as implicit regularization — forcing all channels through the same kernels prevents overfitting to per-channel noise. This is a classic case of architecture complexity exceeding data capacity.

**Verdict:** Channel-aware denoising is **rejected**. The architecture is sound in principle but catastrophically wrong for our data regime. Experiment 2 remains our best across all benchmarks.

---

## Experiment 10: Direct x0-Prediction with Trend/Seasonality Decomposition

**Change:** Switched from epsilon-prediction to direct x0-prediction with an explicit decomposition head in the denoiser. The decoder outputs separate trend (learned moving average kernel) and seasonality (top-K Fourier basis, K=5) components, summed to form x0. Training loss is MSE directly on x0 rather than on predicted noise. The FFT auxiliary loss is now coherent — applied to the model's actual output instead of a noisy x0 recovered from epsilon. Built on Experiment 2 codebase. Code isolated in `exp10_x0_decomposition/`.

**Training time:** ~55.1 min total (51-58 epochs, early stopping triggered on all — fastest training of any experiment)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 10 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.4842 | +2.6% | +15.3% |
| ETTh1 Uni | 0.2535 | 0.2523 | **0.2508** | **-0.6%** | **-26.2%** |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.4194 | -0.6% | +13.4% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.1969 | -1.5% | +31.3% |

**Direct MAE vs Full MAE:** On ETTh1 Uni, diffusion **actively contributed** for the first time: Direct 0.2540 → Full 0.2508 (-1.3%). Elsewhere, diffusion was neutral-to-slightly-negative.

**What worked:** This is the most balanced result across all experiments:
- ETTh1 Uni hit **0.2508** — our new best, beating Exp 2's 0.2523 and extending our lead over the paper (-26.2%).
- ETTm1 Multi at 0.4194 essentially matched Exp 2 (-0.6%) — the first experiment since baseline that didn't regress multivariate ETTm1.
- ETTm1 Uni at 0.1969 improved over Exp 2's 0.1999 (-1.5%).
- **Diffusion is no longer purely cosmetic on ETTh1 Uni.** The x0-prediction parameterization with decomposition gives the denoiser a structured output space (trend + seasonality) that it can meaningfully learn, rather than predicting arbitrary noise vectors.
- Training converged faster (51-58 epochs vs 50-100 for other experiments), suggesting the direct x0 loss landscape is smoother.

**What didn't work:** ETTh1 Multi regressed slightly (+2.6%), and on 3 of 4 benchmarks diffusion still hurt slightly (Full > Direct). The decomposition heads add inductive bias that helps univariate seasonal data but may constrain multivariate representations.

**Why x0-prediction + decomposition helps:** Epsilon-prediction asks the denoiser to predict arbitrary noise — there's no structure in the target. Direct x0-prediction with trend/seasonality decomposition gives the denoiser strong inductive bias matching the actual data structure. The FFT loss is now applied to the model's direct output rather than a noisy derived quantity, creating a coherent training signal. For univariate data with clear seasonal patterns, this is exactly the right prior.

**Verdict:** x0-prediction with decomposition shows the **most promising direction** of all experiments. First time diffusion actively contributed (ETTh1 Uni). Most balanced across benchmarks. New best on ETTh1 Uni (0.2508). However, the gains are modest and multivariate still doesn't improve meaningfully.

---

## Experiment 9: Patch + Attention History Encoder (PatchTST-style)

**Change:** Replaced the 3-layer Conv1d history encoder in `ConditioningNetwork` with a PatchTST-style transformer: the lookback window is divided into non-overlapping patches (patch_size=24), each projected to hidden_dim with learnable positional embeddings, then processed by a 2-layer TransformerEncoder (4 heads, GELU activation). This gives the conditioning network global receptive field over the entire lookback window, vs the Conv1d's ~21-timestep effective receptive field. Built on Experiment 2 codebase. Code isolated in `exp9_patch_attention/`.

**Training time:** ~94 min total (66-100 epochs; transformer encoder is slower per epoch)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 9 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.5388 | +14.2% | +28.3% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2541 | +0.7% | -25.3% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.4900 | +16.2% | +32.4% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.1984 | -0.8% | +32.3% |

**Direct MAE vs Full MAE:** Diffusion cosmetic across all experiments (Direct ≈ Full).

**What marginally worked:** ETTm1 Uni improved slightly to 0.1984 (-0.8% vs Exp 2). The global receptive field from self-attention over 60 patches (1440/24) can in theory capture long-range seasonality (daily cycles at 96 timesteps) that the Conv1d misses.

**What didn't work:** ETTh1 Multi (+14.2%) and ETTm1 Multi (+16.2%) regressed significantly. ETTh1 Uni was essentially flat. The patch+attention encoder adds parameters and compute overhead (94 min vs ~35 min baseline) with no multivariate benefit.

**Why:** The PatchTST architecture was designed for standalone forecasting where the encoder IS the model. In our architecture, the history encoder feeds into a conditioning network that then informs the diffusion denoiser — it's an intermediate representation, not a final prediction. The Conv1d encoder produces smooth, local features that the downstream denoiser can work with. The transformer encoder produces sharper, more complex representations that the small denoiser (843K params) can't effectively exploit. Additionally, for ETTh1 (L=336, only 14 patches), self-attention over 14 tokens provides minimal benefit over convolution. The overhead is highest on ETTm1 (60 patches), where the attention computation dominates training time without proportional benefit.

**Why multivariate suffers:** The patch embedding treats all D channels as a flat input per patch. With D=7, each patch token mixes all variables before attention — the transformer can't learn per-variable temporal patterns. This is the inverse of Exp 8's problem: Exp 8 split channels too aggressively; Exp 9 conflates them too early. Neither extreme works.

**Verdict:** Patch+attention conditioning is **rejected**. The Conv1d encoder is the right choice for our conditioning architecture — it's lightweight, provides appropriate local features, and doesn't overwhelm the downstream denoiser.

---

## Summary: Full Campaign Results (Experiments 1-10)

| Exp | Improvement | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni | Verdict |
|---|---|---|---|---|---|---|
| — | **Baseline** | **0.4744** | **0.2535** | **0.4204** | **0.2011** | — |
| — | **Paper** | **0.42** | **0.34** | **0.37** | **0.15** | — |
| 1 | Remove detach | 0.4765 | 0.2543 | 0.4224 | 0.1988 | Rejected |
| 2 | Self-conditioning | 0.4719 | 0.2523 | 0.4218 | 0.1999 | **Best overall** |
| 3 | Cosine schedule | 0.6709 | 0.2813 | 0.6433 | 0.2141 | Rejected |
| 4 | v-prediction | 0.4790 | 0.2531 | 0.4216 | 0.2049 | Rejected |
| 5 | ANT schedule | 0.7309 | 0.2803 | 0.6513 | 0.2055 | Rejected |
| 6 | Contrastive loss | 0.5535 | 0.2555 | 0.4925 | 0.1946 | Rejected |
| 7 | MG-TSD guidance | 0.5653 | 0.2558 | 0.4819 | **0.1913** | Rejected |
| 8 | Channel-aware | 0.9313 | 0.2552 | 0.9136 | 0.1928 | Rejected |
| 9 | Patch+attention | 0.5388 | 0.2541 | 0.4900 | 0.1984 | Rejected |
| 10 | x0+decomposition | 0.4842 | **0.2508** | **0.4194** | 0.1969 | **Most promising** |

**Best per benchmark:**
- ETTh1 Multi: Exp 2 (0.4719) — paper: 0.42, gap: +12.4%
- ETTh1 Uni: **Exp 10 (0.2508)** — paper: 0.34, **beats paper by 26.2%**
- ETTm1 Multi: **Exp 10 (0.4194)** — paper: 0.37, gap: +13.4%
- ETTm1 Uni: **Exp 7 (0.1913)** — paper: 0.15, gap: +27.5%

**Key takeaways from the full campaign:**
1. **Diffusion remains fundamentally cosmetic.** Across 10 experiments and 40 evaluations, diffusion contributed meaningfully exactly once (Exp 10, ETTh1 Uni: -1.3%).
2. **Every improvement that helps univariate hurts multivariate.** This is the campaign's defining pattern — no single change improved both.
3. **The DLinear backbone is the real model.** All meaningful performance comes from the 113K-param direct predictor.
4. **Small datasets resist architectural complexity.** Experiments 3, 5, 7, 8, 9 all added complexity and all regressed on multivariate.
5. **Exp 10 (x0+decomposition) is the most promising direction** — best balance, only experiment where diffusion helped, and didn't catastrophically hurt multivariate.

---

## Experiment 11: Deep Backbone with Standard Residuals (Control)

**Change:** Replaced the flat DLinear backbone (2 linear projections, 113K params) with a deep backbone: 4 Conv1d blocks (kernel=3, hidden_channels=64) with standard additive residual connections, for both trend and residual paths independently. Each path: input_proj → 4x [Conv1d → GroupNorm(16) → LeakyReLU → Dropout(0.3) + additive residual] → length_proj → output_proj. This is the control experiment for Experiments 12-13, isolating whether backbone depth alone helps. Built on Experiment 2 codebase. Code isolated in `exp11_deep_backbone/`.

**Training time:** ~119 min total (50-87 epochs, early stopping on all)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 11 MAE | vs Exp 2 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.6634 | **+40.6%** | +57.9% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2822 | +11.9% | -17.0% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.5440 | +29.0% | +47.0% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2056 | +2.9% | +37.1% |

**Params:** 946K total (215K backbone, up from 113K — roughly 2x the DLinear backbone).

**What didn't work:** Catastrophic multivariate regression (+29-41%). The 4 conv blocks with 64 hidden channels roughly doubled the backbone parameters, and the multivariate data (D=7, ~10K samples) doesn't have enough signal to fill that capacity. The model overfits to training noise.

**What partially worked:** ETTm1 Uni (0.2056) regressed only +2.9% — the closest any deep backbone experiment came to baseline on any benchmark. With 40K training samples and D=1, there's enough data to support modest depth.

**Why:** The flat DLinear backbone is already near-optimal for this data regime. Linear projections provide implicit regularization — they can't overfit to local temporal noise the way conv blocks can. Adding depth adds capacity that manifests as overfitting, not expressiveness, when data is scarce.

**Verdict:** Deep backbone with standard residuals is **rejected**. Confirms the control hypothesis: depth alone doesn't help at this data scale.

---

## Experiment 12: Deep Backbone with Attention Residuals (AttnRes)

**Change:** Same deep backbone architecture as Experiment 11, but replaced standard additive residual connections with Full Attention Residuals (Kimi team, 2026). Each of the 4 layers gets a learned pseudo-query vector `w_l ∈ R^64` (zero-initialized) that computes softmax attention over all prior layer outputs via RMSNorm'd keys. This lets each layer selectively retrieve information from any earlier layer rather than accumulating everything uniformly. Built on Experiment 2 codebase. Code isolated in `exp12_attnres_backbone/`.

**Training time:** ~145 min total (51-76 epochs, early stopping on all)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 12 MAE | vs Exp 2 | vs Exp 11 | vs Paper |
|---|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.6599 | **+39.8%** | -0.5% | +57.1% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2729 | +8.2% | **-3.3%** | -19.7% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.5681 | +34.7% | +4.4% | +53.5% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2051 | +2.6% | **-0.2%** | +36.7% |

**Params:** 947K total (216K backbone — only 640 params more than Exp 11 for the pseudo-queries + RMSNorm).

**AttnRes vs standard residuals (Exp 12 vs Exp 11):**
- ETTh1 Uni: **0.2729 vs 0.2822 = -3.3%** — AttnRes's selective retrieval helped
- ETTm1 Uni: **0.2051 vs 0.2056 = -0.2%** — essentially tied
- ETTh1 Multi: 0.6599 vs 0.6634 = -0.5% — marginal
- ETTm1 Multi: 0.5681 vs 0.5440 = +4.4% — AttnRes slightly worse

**What the AttnRes zero-init safety net delivered:** As predicted, AttnRes never performed catastrophically worse than standard residuals. The zero-initialized pseudo-queries start as uniform attention (equivalent to standard residuals), and the model only deviates when it finds beneficial selective patterns. On ETTh1 Uni, this selective retrieval provided a meaningful 3.3% improvement.

**Why the overall results still regress from baseline:** The fundamental problem is backbone depth, not residual connection type. AttnRes can't fix overfitting caused by having too many conv parameters — it can only improve how information flows through those layers. The right comparison is AttnRes vs standard residuals at the same depth (Exp 12 vs 11), where AttnRes shows a consistent small edge.

**Verdict:** AttnRes provides a **genuine small improvement over standard residuals** (especially ETTh1 Uni -3.3%), but can't overcome the backbone overfitting problem. The mechanism is validated; the architecture scale is wrong.

**Reference:** "Attention Residuals" (Kimi Team, 2026). [GitHub](https://github.com/MoonshotAI/Attention-Residuals)

---

## Experiment 13: AttnRes Backbone + Learned Stage Aggregation

**Change:** Built on Experiment 12's AttnRes backbone. Additionally replaced the fixed equal-weight summation of diffusion stage outputs with a learned `StageAggregator`: each stage prediction is flattened and projected to a 64-dim key space, a learned query vector computes softmax attention over stages, and the output is a weighted sum. Zero-initialized query → uniform (equal sum) at start. Built on Experiment 2 codebase. Code isolated in `exp13_attnres_stage_agg/`.

**Training time:** ~137 min total (51-59 epochs, early stopping on all — fastest convergence of the three)

| Experiment | Baseline MAE | Exp 2 MAE | Exp 13 MAE | vs Exp 12 | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.6387 | **-3.2%** | +51.8% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2846 | +4.3% | -16.3% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.5678 | -0.1% | +35.1% |
| ETTm1 Uni | 0.2011 | 0.1999 | 0.2305 | **+12.4%** | +53.7% |

**Params:** 1,022K total (216K backbone + 75K aggregator).

**What the aggregator did:** On ETTh1 Multi, the learned aggregation helped (-3.2% vs Exp 12), likely by suppressing harmful diffusion stages. But on ETTm1 Uni, it catastrophically hurt (+12.4% vs Exp 12) — the 75K-param aggregator overfitting on the stage-weighting task, learning to amplify diffusion noise rather than suppress it.

**Why ETTm1 Uni regressed so badly:** The aggregator has 75K params (input_dim × forecast_length → 64 projection per stage). For ETTm1 (H=192, D=1), this is 192×1→64 per stage — modest. But the aggregator sees stage predictions that are near-zero noise (diffusion is cosmetic), so it's learning to weight random noise. With enough parameters, it fits training noise perfectly and transfers nothing.

**Verdict:** Learned stage aggregation is **rejected**. The fixed equal-weight sum is actually a form of regularization — it prevents the model from overfitting to stage-level noise patterns. When diffusion stages produce near-zero useful signal, smart aggregation of nothing is worse than dumb aggregation of nothing.

---

## Summary: Experiments 11-13 (AttnRes Campaign)

| Exp | Architecture | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni | Total Time |
|---|---|---|---|---|---|---|
| — | **Baseline (DLinear)** | **0.4744** | **0.2535** | **0.4204** | **0.2011** | ~34m |
| 2 | **Self-conditioning** | **0.4719** | **0.2523** | **0.4218** | **0.1999** | ~33m |
| 11 | Deep backbone (std res) | 0.6634 | 0.2822 | 0.5440 | 0.2056 | ~119m |
| 12 | Deep backbone (AttnRes) | 0.6599 | **0.2729** | 0.5681 | **0.2051** | ~145m |
| 13 | AttnRes + stage agg | **0.6387** | 0.2846 | 0.5678 | 0.2305 | ~137m |

**Campaign conclusion:** The AttnRes mechanism is validated — it consistently outperforms standard residuals at the same depth (Exp 12 beats Exp 11 on 3/4 benchmarks). But deepening the backbone from 2 linear layers to 4 conv blocks is the wrong move at this data scale. The flat DLinear backbone's implicit regularization (linearity) is a feature, not a bug. Future work should either (a) apply AttnRes at a scale where depth is warranted, or (b) find ways to add expressiveness to the backbone without adding depth/capacity.

---

## Experiment 15: Tiny Direct Transformer (PatchTST-Style, No Diffusion)

**Change:** Replaced the entire mr-Diff architecture with a tiny PatchTST-style transformer. No diffusion, no conditioning networks, no denoising — just patch embedding, 2-layer TransformerEncoder (d_model=64, 4 heads, dim_ff=128, pre-norm, GELU), and a linear head. Patches are non-overlapping (patch_size=16 for ETTh1, 16 for ETTm1). Input patches contain all D channels concatenated. Xavier initialization with gain=0.5 for stability. Code isolated in `exp15_tiny_transformer/`.

**Training time:** 4.7 min total across all 4 benchmarks. All hit min_epochs=30 without early stopping triggering — the model converges fast and plateaus.

| Experiment | Baseline MAE | Exp 2 MAE | Exp 15 MAE | vs Baseline | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4719 | 0.5607 | +18.2% | +33.5% |
| ETTh1 Uni | 0.2535 | 0.2523 | 0.2538 | **+0.1%** | -25.4% |
| ETTm1 Multi | 0.4204 | 0.4218 | 0.5514 | +31.2% | +49.1% |
| ETTm1 Uni | 0.2011 | 0.1999 | **0.2002** | **-0.4%** | +33.5% |

**Params:** 1,657K (ETTh1 multi) / 1,180K (ETTm1 uni). The flatten→linear head dominates — n_patches × d_model → T × D is a large projection.

**What's remarkable:**
- **ETTm1 Uni: 0.2002** — matches baseline (0.2011, -0.4%) and Exp 2 (0.1999). A 2-layer transformer trained in **1.8 minutes** matched what our full 843K-param diffusion pipeline achieves in 34 minutes.
- **ETTh1 Uni: 0.2538** — dead even with baseline (0.2535, +0.1%). Again, diffusion adds nothing that a tiny transformer can't match instantly.
- **Training speed:** 150-200 it/s on ETTh1 (vs ~20 it/s for mr-Diff). 4.7 min total vs ~34 min. **7.2x faster.**
- **No diffusion overhead:** No sampling, no multi-step denoising, no DPM-Solver++. Inference is a single forward pass.

**What didn't work:** Multivariate regressed (+18-31%), though less catastrophically than the deep backbone experiments (Exp 11 was +40%). The self-attention over patches captures cross-channel patterns but 10K multivariate samples isn't enough to learn them reliably. The flatten→linear head is also overparameterized for multivariate — n_patches × 64 → 168 × 7 = ~140K params in the head alone.

**Why univariate matches the full pipeline:** On univariate, the forecast is a function of temporal patterns only. Self-attention over 21 patches (ETTh1) or 90 patches (ETTm1) gives global receptive field that captures seasonality directly. The DLinear backbone achieves the same via a single linear projection — both are sufficient for the univariate temporal structure. Diffusion adds nothing on top of either.

**The strategic implication:** This result proves that diffusion is pure overhead for our task. The entire mr-Diff architecture — conditioning networks, multi-stage decomposition, denoising networks, DPM-Solver++ — can be replaced by a 2-layer transformer with zero loss in forecast quality on univariate and significant speedup. **The path forward is optimizing this transformer, not the diffusion model.** With 1.8-minute training cycles, we can test 20 ideas in the time one diffusion experiment takes.

**Verdict:** The tiny transformer **matches baseline on univariate** with 7x speedup and opens a fundamentally faster iteration loop. Multivariate needs work, but the architecture is sound. This is the new foundation.

**Reference:** "A Time Series is Worth 64 Words" (PatchTST, ICLR 2023). [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

---

## Experiment 14: Multi-Scale AttnRes DLinear (Width, Not Depth) — STOPPED EARLY

**Change:** Replaced the single-kernel DLinear backbone with 4 parallel DLinear projections at different temporal scales (kernel_sizes=5, 15, 25, 51), fused with AttnRes-style learned softmax attention. Each scale has independent trend/residual linear projections. A zero-initialized query vector attends over RMSNorm'd scale outputs to learn which temporal granularity matters most. Still wrapped in the full mr-Diff diffusion pipeline. Code isolated in `exp14_multiscale_attnres/`.

**Training stopped after 2 of 4 datasets** — multivariate regressed and the 4x DLinear projections inflated params beyond useful capacity. ETTm1 Multi ballooned to 3.6M params (4x baseline).

| Experiment | Baseline MAE | Exp 14 MAE | vs Baseline | vs Paper |
|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.5685 | +19.8% | +35.4% |
| ETTh1 Uni | 0.2535 | 0.2701 | +6.5% | -20.6% |
| ETTm1 Multi | 0.4204 | *stopped* | — | — |
| ETTm1 Uni | 0.2011 | *stopped* | — | — |

**Params:** 1,184K (ETTh1) / 3,605K (ETTm1) — the 4 parallel linear projections scale with lookback_length², so ETTm1 (L=1440) explodes to 3.6M params.

**What partially worked:** ETTh1 Uni at 0.2701 beat the deep backbone experiments (Exp 11: 0.2822, Exp 12: 0.2729), confirming the multi-scale idea has some merit for univariate. But still worse than baseline's 0.2535.

**Why it was killed:** (1) Multivariate regression (+20%) shows the same overfitting pattern as Exp 11-13. (2) The 4x parallel projections don't share parameters — total backbone is 453K vs DLinear's 113K. (3) Diffusion overhead means each experiment takes ~90 min vs the tiny transformer's 5 min. The juice isn't worth the squeeze when Exp 15's transformer matched baseline in 1.8 min.

**Verdict:** Multi-scale AttnRes DLinear is **rejected**. The width-not-depth idea was sound but the parameter scaling is wrong — 4 independent DLinear projections is 4x the capacity. A better approach would share the projection weights across scales and only differentiate at the trend-extraction kernel level. But with Exp 15 showing transformers can match baseline at 7x speed, the DLinear backbone paradigm is no longer the priority.

---
