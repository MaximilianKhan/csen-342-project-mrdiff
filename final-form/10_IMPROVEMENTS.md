# 10 Improvements Over Our mr-Diff Baseline

> **Baseline established:** March 2026 | **Model:** 843K params (730K diffusion + 113K DLinear)
> **Core finding:** Diffusion contributes ~0% beyond the DLinear backbone. That's the #1 problem.

## Our Baseline Results

| Experiment | Our MAE | Paper MAE | Gap |
|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.42 | +13% |
| **ETTh1 Uni** | **0.2535** | **0.34** | **-25% (beats paper)** |
| ETTm1 Multi | 0.4204 | 0.37 | +14% |
| ETTm1 Uni | 0.2011 | 0.15 | +34% |

**Three weaknesses to attack:**
1. Diffusion is cosmetic — it contributes nothing on top of the backbone
2. Multivariate lags the paper more than univariate (+13-14% vs mixed univariate)
3. 843K params may be underleveraged with the right architectural changes

---

## Improvement 1: Remove the Detach — Joint End-to-End Training

**What:** Change `forecast - direct_pred.detach()` to `forecast - direct_pred` in `mr_diff.py:184`. Let diffusion loss gradients flow back through the DLinear backbone.

**Why this is critical for us:** By detaching, we train the backbone and diffusion as two independent systems. The backbone has zero incentive to produce residuals that diffusion can model — it just minimizes its own MSE greedily. This means the backbone captures everything it can, leaving only noise-like residuals that diffusion *cannot* meaningfully denoise. Joint training creates a cooperative dynamic: the backbone learns to leave structured, predictable residuals for diffusion, while diffusion learns to exploit the patterns the backbone deliberately leaves behind.

**This is arguably our single biggest architectural mistake.**

**Expected impact:** High across all experiments, especially multivariate where there's more room for the backbone to reshape its predictions. Could close 3-8% of the gap. May need to scale diffusion loss by 0.1-0.5 relative to direct loss to prevent gradient dominance.

**Complexity:** Easy — one line change + loss scaling.

**Reference:** Residual Denoising Diffusion Models ([arXiv:2308.13712](https://arxiv.org/abs/2308.13712))

---

## Improvement 2: Self-Conditioning

**What:** During training, with 50% probability, run the denoiser twice per step — first to get a preliminary x0 estimate (stop-gradient), then concatenate that estimate alongside the noisy input for the real forward pass. During sampling, this is free since we already have x0 from the previous DPM-Solver++ step.

**Why it helps us:** Our diffusion never learns to *refine* its own predictions. It sees noisy input + conditioning but has no memory of what it previously estimated. Self-conditioning gives the denoiser access to its own best guess, turning it from a one-shot predictor into an iterative refiner. The denoiser learns to *correct errors* rather than *predict from scratch* — a fundamentally easier task that directly addresses exposure bias.

**Expected impact:** All 4 experiments. Most impactful on ETTm1 Uni (our worst gap at +34%) because the residuals are small and structured — exactly where iterative refinement shines. 5-15% improvement on diffusion contribution.

**Complexity:** Easy — modify `DenoisingNetwork.forward` to accept optional `x0_prev` tensor (concatenated to input, adds `input_dim` channels to input projection). In `training_step`, 50% of time do a no-grad forward pass first. In sampling, pass previous step's x0. ~50 lines.

**Reference:** Chen et al., "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning" (ICLR 2023). [arXiv:2208.04202](https://arxiv.org/abs/2208.04202)

---

## Improvement 3: Cosine Schedule + Reduced Diffusion Steps

**What:** Switch from linear beta schedule (1e-4 to 0.1) to cosine schedule. Reduce total diffusion steps from 100 to 50 during training. Use DPM-Solver++ with 10-15 steps for inference.

**Why it helps us:** We already proved DPM-Solver++ at 20 steps = DDPM at 100 steps exactly. This means the forward process has redundant steps. The cosine schedule (Nichol & Dhariwal 2021) provides smoother noise progression with less wasted capacity at the endpoints where noise levels change slowly. Halving the steps means each training sample sees a more informative distribution of noise levels — effectively doubling training efficiency. With only ~8K-10K training samples, every gradient step counts.

**Expected impact:** Moderate indirect improvement through better training efficiency. 2-4% across all experiments. Also speeds up training.

**Complexity:** Easy — config changes. Cosine schedule is already implemented in `DiffusionSchedule`. ~5 lines.

**Reference:** Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)

---

## Improvement 4: Switch to v-Prediction Parameterization

**What:** Instead of predicting epsilon (noise), predict `v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x0`. This is a weighted combination with more uniform variance across all timesteps.

**Why it helps us:** With epsilon prediction, the learning signal is wildly uneven across noise levels. At low noise (small k), epsilon is nearly the full signal; at high noise (large k), epsilon carries almost no information about x0. This creates oscillating training dynamics. V-prediction normalizes the target variance across all timesteps, giving the model a consistent learning signal. For our small 843K-param model on small datasets, we can't afford wasted gradient steps on poorly-scaled targets.

**Expected impact:** Moderate on all experiments. Most likely to help ETTm1 Uni where inconsistent gradient scales may be preventing diffusion from learning useful features. 2-5% improvement.

**Complexity:** Medium — modify `forward_diffusion` to return v-target, change loss in `training_step`, update epsilon-to-x0 conversion in sampling, update DPM-Solver++ model output. ~80 lines across `diffusion.py` and `mr_diff.py`.

**Reference:** Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models" (ICLR 2022)

---

## Improvement 5: Adaptive Noise Schedule (ANT)

**What:** Replace the fixed schedule with one computed from dataset-specific non-stationarity statistics. ANT uses the Integrated Absolute Autocorrelation Time (IAAT) to characterize each dataset's temporal structure, then designs a schedule that linearly reduces non-stationarity across diffusion steps so every step does equal work.

**Why it helps us:** Our linear schedule was borrowed from image diffusion. Time series have fundamentally different structure. ETTh1 (hourly, 336 lookback) and ETTm1 (15-min, 1440 lookback) have very different non-stationarity profiles, so a one-size-fits-all schedule is suboptimal for both. ANT showed **9.5% average CRPS improvement** on TSDiff — that's massive.

**Expected impact:** Moderate-to-high, especially ETTm1 (longer sequences = more non-stationarity = more schedule mismatch). 3-8% improvement.

**Complexity:** Medium — compute IAAT offline from training data, parameterize a non-linear beta schedule. ~100 lines + preprocessing script. Code: [github.com/seunghan96/ANT](https://github.com/seunghan96/ANT)

**Reference:** "ANT: Adaptive Noise Schedule for Time Series Diffusion Models" (NeurIPS 2024). [arXiv:2410.14488](https://arxiv.org/abs/2410.14488)

---

## Improvement 6: Contrastive Conditioning Loss (from CCDM)

**What:** Add an InfoNCE-style contrastive term to the training loss. For each sample, compute `exp(-||eps - eps_pred||^2 / tau)` as the score, and contrast the true future against temporally-augmented negatives (time-shifted, scaled, or channel-shuffled versions of the target).

**Why it helps us:** Our diffusion learns to denoise residuals but has no explicit incentive to actually *use* the conditioning signal from history. It can just predict mean-zero noise and get reasonable loss. The contrastive loss forces the denoiser to maximize mutual information between conditioning and forecast — it must distinguish the true future from plausible alternatives. CCDM ablations showed **+8.3% MSE degradation** when removing the contrastive term.

**Expected impact:** High on multivariate experiments (ETTh1 Multi, ETTm1 Multi) where cross-channel predictive information is being ignored. 3-7% improvement on multivariate.

**Complexity:** Medium-Hard — implement negative sample generation (temporal augmentations), compute InfoNCE loss using denoiser prediction errors, add as weighted auxiliary loss. ~150 lines. Temperature tau=0.1, lambda in [0.0001, 0.01].

**Reference:** "Channel-aware Contrastive Conditional Diffusion for Multivariate Probabilistic Time Series Forecasting" (2024). [arXiv:2410.02168](https://arxiv.org/abs/2410.02168)

---

## Improvement 7: Multi-Granularity Guided Diffusion (MG-TSD)

**What:** Use coarse-grained versions of the target as intermediate guidance during the diffusion process. At diffusion step k, constrain the latent state to be close to the target at granularity proportional to k — heavily smoothed targets for high noise, fine-grained for low noise.

**Why it helps us:** Our 3-stage hierarchy decomposes the target but each stage runs independent diffusion with independent noise. The stages don't cooperate during the diffusion *trajectory*. MG-TSD structures the diffusion path to go from coarse to fine rather than random noise to signal, giving the process meaningful intermediate targets. MG-TSD reported **4.7-35.8% improvement** — this is the most transformative single change for making diffusion non-cosmetic.

**Expected impact:** High across all experiments. Strongest on longer horizons. 5-15% improvement. Most likely to transform diffusion from cosmetic to genuinely useful.

**Complexity:** Hard — compute multi-granularity targets (downsampled at different temporal resolutions), modify forward diffusion to use these as intermediate constraints, derive the modified loss. ~200 lines. Code: [github.com/Hundredl/MG-TSD](https://github.com/Hundredl/MG-TSD)

**Reference:** "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process" (ICLR 2024). [arXiv:2403.05751](https://arxiv.org/abs/2403.05751)

---

## Improvement 8: Channel-Aware Denoising Architecture

**What:** Replace the shared Conv1d processing (all channels through same kernels) with: (a) channel-independent temporal encoders per variable, then (b) lightweight cross-channel attention to aggregate information across variables.

**Why it helps us:** Our multivariate gap is larger than univariate. The current Conv1d treats all D=7 channels identically — it cannot learn that OT (oil temperature) has different dynamics than HUFL (high useful load). Channel-independent encoding respects each variable's unique patterns; cross-channel attention captures correlations (load predicts temperature) without conflating them.

**Expected impact:** High on multivariate specifically. Could close 5-10% of ETTh1 Multi and ETTm1 Multi gaps. No-op on univariate (D=1).

**Complexity:** Hard — restructure denoising encoder for per-channel processing, add multi-head attention for channel mixing. ~200 lines, must watch parameter count.

**Reference:** CCDM ([arXiv:2410.02168](https://arxiv.org/abs/2410.02168)), CrossFormer ([OpenReview](https://openreview.net/forum?id=vSVLM2j9eie)), iTransformer

---

## Improvement 9: Patch + Attention History Encoder

**What:** Replace the 3-layer Conv1d history encoder in `ConditioningNetwork` with PatchTST-style: segment lookback into non-overlapping patches (16-24 timesteps each), project each to hidden_dim, apply 2-3 layers of self-attention over patches.

**Why it helps us:** The Conv1d encoder with kernel_size=7 has an effective receptive field of ~21 timesteps after 3 layers. For ETTm1 with lookback=1440, most of the history is invisible to any given conditioning token. Self-attention over patches gives global receptive field immediately, capturing long-range dependencies like daily seasonality (96 timesteps in ETTm1). Better conditioning = diffusion gets more useful information = diffusion can actually contribute.

**Expected impact:** Moderate-to-high. Most impactful on ETTm1 (long lookback = most receptive field limitation). 2-5% improvement. Also helps multivariate where cross-time patterns are more complex.

**Complexity:** Medium — replace Conv1d stack with patch embedding + `nn.TransformerEncoder` (2-3 layers, 4 heads). ~120 lines. 2 layers at dim=64 adds ~35K params.

**Reference:** "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (PatchTST, ICLR 2023). [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

---

## Improvement 10: Direct x0-Prediction with Decomposition (Diffusion-TS)

**What:** Switch from epsilon-prediction to direct x0-prediction with an explicit decomposition head: the denoiser outputs separate trend and seasonality components (learnable moving average + top-K Fourier basis), summed to form x0. Train with combined MSE + FFT loss directly on x0.

**Why it helps us:** Our current FFT auxiliary loss (line 251-253 of `mr_diff.py`) is applied to x0 *recovered from epsilon prediction* — a noisy, derived quantity. The FFT loss is fighting against the epsilon objective. Direct x0 prediction with built-in decomposition gives the network a strong inductive bias matching the data's actual structure (trend + seasonality). The FFT loss becomes coherent since it's on the model's actual output.

**Expected impact:** Moderate. Especially helpful for ETTh1 Uni (could push further beyond paper) and ETTm1 Uni (strong seasonality). 2-5% improvement.

**Complexity:** Medium — add trend extraction head (learned moving average) and seasonality head (top-K Fourier) to decoder output, switch training loss, update DPM-Solver++. ~150 lines across denoising.py and mr_diff.py.

**Reference:** "Diffusion-TS: Interpretable Diffusion for General Time Series Generation" (ICLR 2024). [arXiv:2403.01742](https://arxiv.org/abs/2403.01742), [GitHub](https://github.com/Y-debug-sys/Diffusion-TS)

---

## Implementation Priority

Ordered by expected-impact-to-effort ratio:

| # | Improvement | Effort | Target | Why This Order |
|---|---|---|---|---|
| 1 | **Remove Detach** | 1 line | All | Root cause of cosmetic diffusion |
| 2 | **Self-Conditioning** | ~50 LOC | All | Proven technique, easy, high payoff |
| 3 | **Cosine + Fewer Steps** | Config only | All | Free improvement, no code risk |
| 4 | **v-Prediction** | ~80 LOC | All | Fixes gradient dynamics |
| 5 | **ANT Adaptive Schedule** | ~100 LOC | ETTm1 | Dataset-specific, strong evidence |
| 6 | **Contrastive Loss** | ~150 LOC | Multi | Directly targets multivariate gap |
| 7 | **MG-TSD Guidance** | ~200 LOC | All | Most transformative for diffusion |
| 8 | **Channel-Aware Denoising** | ~200 LOC | Multi | Targeted at multivariate gap |
| 9 | **Patch+Attention Conditioning** | ~120 LOC | ETTm1 | Better long-range conditioning |
| 10 | **x0 + Decomposition** | ~150 LOC | Uni | Cleaner architecture |

**Quick wins (1 afternoon):** Improvements 1 + 2 + 3 combined. These three together could realistically close 5-10% of our gap across all experiments with minimal code changes.

**Full campaign:** All 10 improvements, tested incrementally, could bring our multivariate results to within 5% of the paper and push our univariate results further ahead.

---

## References

- Chen et al., "Analog Bits" (ICLR 2023) — [arXiv:2208.04202](https://arxiv.org/abs/2208.04202)
- "ANT: Adaptive Noise Schedule" (NeurIPS 2024) — [arXiv:2410.14488](https://arxiv.org/abs/2410.14488)
- "CCDM: Channel-aware Contrastive Conditional Diffusion" (2024) — [arXiv:2410.02168](https://arxiv.org/abs/2410.02168)
- "MG-TSD: Multi-Granularity Time Series Diffusion" (ICLR 2024) — [arXiv:2403.05751](https://arxiv.org/abs/2403.05751)
- "Diffusion-TS" (ICLR 2024) — [arXiv:2403.01742](https://arxiv.org/abs/2403.01742)
- "PatchTST" (ICLR 2023) — [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Nichol & Dhariwal, "Improved DDPM" (ICML 2021)
- Salimans & Ho, "Progressive Distillation" (ICLR 2022)
- Residual Denoising Diffusion Models — [arXiv:2308.13712](https://arxiv.org/abs/2308.13712)
- "TimeDiff" (ICML 2023) — [arXiv:2306.05043](https://arxiv.org/abs/2306.05043)
- "TSDiff" (NeurIPS 2023) — [arXiv:2307.11494](https://arxiv.org/abs/2307.11494)
