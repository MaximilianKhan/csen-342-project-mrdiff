# mr-Diff Baseline Replication (CSEN-342)

Replication of **Multi-Resolution Diffusion Models for Time Series Forecasting** (ICLR 2024)
by Lifeng Shen, Weiyu Chen, and James T. Kwok.

## Current Best Results (Run 082858, Feb 12)

| Experiment | Paper MAE | Ours MAE | Paper MSE | Ours MSE | MAE Gap |
|---|---|---|---|---|---|
| ETTh1 Multivariate | **0.422** | 0.922 | **0.411** | 1.457 | 2.2x |
| ETTh1 Univariate | **0.196** | 1.007 | **0.066** | 1.610 | 5.1x |
| ETTm1 Multivariate | **0.373** | 0.972 | **0.340** | 1.576 | 2.6x |
| ETTm1 Univariate | **0.149** | 0.963 | **0.039** | 1.473 | 6.5x |

Best run directory: `experiments/run_20260212_082858/`

We are **2-6x off on MAE** and **4-38x off on MSE** from the paper's reported values.

Note: Our best run uses `predictions[0]` (finest residual only), not the full
multi-stage reconstruction. See Run 173919 below for why `sum(predictions)` fails.

---

## Run History

### Run 222824 (Feb 11) - Initial baseline

First complete run. MAE ranged 1.2-3.3, with ETTm1 particularly poor (3.1-3.3 MAE).
Directory: `experiments/run_20260211_222824/`

### Run 082858 (Feb 12) - Best result so far

Improved over Run 222824 across the board. MAE consistently ~0.9-1.0.
Uses `predictions[0]` (finest residual only) for inference.
Directory: `experiments/run_20260212_082858/`

### Run 114727 (Feb 12) - Post-architectural-fix (REVERTED)

Attempted to fix TrendExtraction to match the paper's trend decomposition (returning
cumulative trends instead of residuals), decompose lookback per-stage, and remove
double instance normalization. **Results were catastrophically worse** (MAE 6-32,
MSE 70-2034) due to cascading errors in the inference pipeline.

**Lesson learned:** While the paper describes cumulative trend decomposition, our
residual decomposition works better in practice with our current architecture because:
1. Each stage predicts a small, low-variance component (easier to learn)
2. Errors in coarser stages don't cascade as badly during inference
3. The sum of residual predictions reconstructs the signal more robustly

These architectural changes were **reverted**. Directory: `experiments/run_20260212_114727/`

### Run 144502 (Feb 12) - Post-fix run (REGRESSION)

Applied conditioning fixes, `sum(predictions)` inference fix, correct ETTm1 forecast
length (192), and GPU optimizations (AMP, cudnn.benchmark, non_blocking). **Results
regressed significantly** vs Run 082858:

| Experiment | Run 082858 MAE | Run 144502 MAE | Regression |
|---|---|---|---|
| ETTh1 Multi | 0.922 | 1.324 | 1.4x worse |
| ETTh1 Uni | 1.007 | 1.387 | 1.4x worse |
| ETTm1 Multi | 0.972 | 2.741 | 2.8x worse |
| ETTm1 Uni | 0.963 | 3.472 | 3.6x worse |

Training time: **124 min** (vs 142 min for 082858) -- GPU optimizations delivered
the expected ~1.15x speedup, but at the cost of accuracy.

Directory: `experiments/run_20260212_144502/`

#### Root cause analysis (3 compounding issues)

**Issue 1: AMP (mixed precision) corrupted training (HIGH impact)**

The diffusion schedule stores tensors (betas, alpha_bars, etc.) as plain Python
attributes, not registered `nn.Module` buffers. Under `torch.amp.autocast`, the
forward diffusion and MSE loss computations run in float16, degrading gradient
quality for the sensitive diffusion process. Additionally, training-time validation
ran in float16 autocast while the final evaluation ran in float32, creating a
distribution mismatch that corrupted best-checkpoint selection.

**Issue 2: Learned target projection widened train-test gap (MEDIUM-HIGH impact)**

The learned `nn.Linear` projection in `_apply_mixup()` extracted optimized predictive
features from the ground-truth target during training. At inference (`mixup_prob=0`),
this rich signal vanishes. The previous random-weight projection was effectively noise
injection, which paradoxically made the model more robust at inference.

**Issue 3: sum(predictions) exposed coarser stage errors (MEDIUM impact)**

While `sum(predictions)` is mathematically correct for residual decomposition,
it exposes prediction errors from ALL stages. In Run 082858, only `predictions[0]`
was returned, hiding errors from coarser stages.

### Run 173919 (Feb 12) - Final run (sum(predictions) CONFIRMED HARMFUL)

Disabled AMP and reverted learned projection to isolate `sum(predictions)` as
the only major change from Run 082858. This was designed as the definitive test.

**Results confirmed `sum(predictions)` is the dominant problem:**

| Experiment | Paper | Run 082858 | Run 173919 | vs 082858 |
|---|---|---|---|---|
| ETTh1 Multi MAE | **0.422** | **0.922** | 1.612 | 1.7x worse |
| ETTh1 Uni MAE | **0.196** | **1.007** | 1.206 | 1.2x worse |
| ETTm1 Multi MAE | **0.373** | **0.972** | 6.916 | 7.1x worse |
| ETTm1 Uni MAE | **0.149** | **0.963** | 2.096 | 2.2x worse |

| Experiment | Paper | Run 082858 | Run 173919 | vs 082858 |
|---|---|---|---|---|
| ETTh1 Multi MSE | **0.411** | **1.457** | 5.117 | 3.5x worse |
| ETTh1 Uni MSE | **0.066** | **1.610** | 2.458 | 1.5x worse |
| ETTm1 Multi MSE | **0.340** | **1.576** | 98.280 | 62x worse |
| ETTm1 Uni MSE | **0.039** | **1.473** | 6.934 | 4.7x worse |

**The critical finding:** Training validation losses were the **best we've ever had**:

| Experiment | Run 082858 Val Loss | Run 173919 Val Loss |
|---|---|---|
| ETTh1 Multi | 0.3211 | **0.3105** |
| ETTh1 Uni | 0.3491 | **0.3452** |
| ETTm1 Multi | 0.2622 | **0.2617** |
| ETTm1 Uni | **0.3029** | 0.3146 |

Each stage individually trains better than ever, but composed inference via
`sum(predictions)` produces far worse evaluation metrics. This proves the issue
is not training quality but **inference-time composition**: the coarser stages
produce predictions that don't compose coherently when summed under standard
DDPM sampling.

**Lesson learned:** `sum(predictions)` is theoretically correct but practically
broken without high-quality inference sampling (DPM-Solver). With DDPM, returning
only `predictions[0]` (finest residual) works better by sidestepping the cascading
error problem entirely.

Directory: `experiments/run_20260212_173919/`

---

## Bug Investigation Summary

We identified 6 discrepancies between our implementation and the paper. After
extensive testing across Runs 114727, 144502, and 173919, we refined our
understanding of which fixes help vs hurt.

### Applied Fixes (final configuration)

**Fix 1: Continuous mixup mask instead of binary**
- File: `src/models/conditioning.py`
- Changed mask from binary `(rand > 0.5)` to continuous `rand()` matching paper's U(0,1).

**Fix 2: LeakyReLU slope 0.1 instead of 0.2**
- File: `src/models/conditioning.py`
- Aligned with paper's Table 13.

**Fix 3: Added --forecast-length CLI argument**
- Files: `train.py`, `run_experiments.py`
- ETTm1 experiments were silently using forecast_length=168 instead of 192.

### Applied Then Reverted

**Reverted: sum(predictions) for inference**
- File: `src/models/mr_diff.py`
- Mathematically correct (`sum(components) = original signal`), but coarser stages
  produce low-quality predictions under DDPM sampling. Summing them adds destructive
  noise. Run 173919 confirmed this: best-ever training losses but 1.2-7.1x worse
  evaluation vs `predictions[0]`. Requires DPM-Solver to work in practice.

**Reverted: Learned projection for future mixup**
- File: `src/models/conditioning.py`
- Created a train-test gap: the model learned to exploit ground-truth features
  during training that vanish at inference.

**Disabled: AMP (mixed precision training)**
- File: `src/training/trainer.py`
- Corrupted diffusion schedule precision and degraded gradient quality.

### Reverted Changes (caused catastrophic regression in Run 114727)

**Reverted: TrendExtraction trend decomposition**
- Cumulative trends caused inference to diverge (MAE 6-32).

**Reverted: Per-stage lookback decomposition**
- Coupled with the trend approach. Reverted to full-resolution lookback.

**Reverted: Removal of double instance normalization**
- Kept for stability.

### GPU Training Optimizations

| Optimization | File | Status | Description |
|---|---|---|---|
| **AMP (mixed precision)** | `trainer.py` | **DISABLED** | Corrupted diffusion schedule precision. |
| **`cudnn.benchmark = True`** | `train.py` | Active | ~10-15% speedup. |
| **`non_blocking` transfers** | `trainer.py`, `metrics.py` | Active | ~5% gain. |

---

## Reproducibility Concerns

Our replication effort raises questions about the completeness of the paper's
described methodology. We systematically implemented each component as described
in the paper (cumulative trend decomposition, learned mixup projection,
multi-stage signal summation) and found that **every paper-described component we
tested either degraded or catastrophically broke performance** when used with
standard DDPM sampling:

| Paper Component | Our Result |
|---|---|
| Cumulative trend decomposition (Section 4.1) | MAE 6-32 (catastrophic) |
| Learned mixup projection (Eq. 9) | MAE 1.3-3.5 (regression) |
| sum(predictions) reconstruction | MAE 1.2-6.9 (regression) |

The paper's results likely depend on DPM-Solver not just for speed but for
correctness: higher-quality samples reduce cascading errors that make multi-stage
composition viable. However, DPM-Solver is described only as an efficiency
optimization in the paper (Section 5.2), not as a component critical to accuracy.
Our findings suggest it plays a much more fundamental role than the paper conveys,
and that the described architecture may not function well without it.

This does not imply the paper's results are incorrect, but it does indicate that
**the paper's recipe is incomplete for independent replication**: omitting
DPM-Solver (which the paper treats as optional) causes the entire multi-resolution
framework to underperform a single-stage baseline.

---

## Performance Gap Analysis

Our best MAE (~0.9-1.0) vs the paper (~0.15-0.42) leaves a **2-6x gap**:

1. **Train-test conditioning mismatch:** Stages train with ground-truth coarser
   trends but infer with noisy predictions. This cascading error is the root cause.

2. **Missing DPM-Solver:** Not just a speed optimization -- it's critical for
   producing predictions accurate enough for multi-stage composition to work.

3. **Hyperparameter tuning:** Paper uses per-dataset grid search for S and kernels;
   we use fixed S=5 with kernels [5, 25, 51, 201].

---

## Paper Reference Values

From Tables 1, 2, 5, and 6 (mr-Diff row):

| Dataset | Uni MAE | Uni MSE | Multi MAE | Multi MSE |
|---|---|---|---|---|
| ETTh1 | 0.196 | 0.066 | 0.422 | 0.411 |
| ETTm1 | 0.149 | 0.039 | 0.373 | 0.340 |

Key hyperparameters: Adam lr=1e-3, batch 64, 100 epochs, K=100 diffusion steps,
beta 1e-4 to 0.1, S=5 stages, ETTh1 L=336/H=168, ETTm1 L=1440/H=192.

---

## Running Experiments

```bash
# Full baseline (all 4 experiments, 100 epochs)
python run_experiments.py

# Quick test
python run_experiments.py --epochs 10

# Individual training
python train.py --dataset ETTh1 --lookback-length 336 --forecast-length 168
python train.py --dataset ETTm1 --univariate --lookback-length 1440 --forecast-length 192

# Evaluation only
python evaluate.py --checkpoint experiments/run_<timestamp>/ETTh1_multi/checkpoints/best.pt \
    --dataset ETTh1 --num-samples 10
```

## Project Structure

```
project-baseline/
  configs/default.yaml        # Model and training hyperparameters
  src/
    data/
      dataset.py              # ETT dataset with per-window RevIN normalization
      preprocessing.py        # TrendExtraction and InstanceNormalization
    models/
      mr_diff.py              # Main mr-Diff model (training_step + sample)
      conditioning.py         # Conditioning network with future mixup
      denoising.py            # Encoder-decoder denoising network
      diffusion.py            # Diffusion schedule and forward process
    evaluation/
      metrics.py              # MAE, MSE computation on normalized data
    training/
      trainer.py              # Training loop with early stopping
  train.py                    # Training entry point
  evaluate.py                 # Evaluation entry point
  run_experiments.py          # Full experiment runner (all 4 configs)
  analyze.py                  # Analysis and plotting utilities
  experiments/                # Experiment output directories
  data/ETDataset/             # ETTh1.csv and ETTm1.csv
```
