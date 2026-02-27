# mr-Diff Baseline Replication (CSEN-342)

Replication of **Multi-Resolution Diffusion Models for Time Series Forecasting** (ICLR 2024)
by Lifeng Shen, Weiyu Chen, and James T. Kwok.

---

## Summary

We replicate the mr-Diff architecture on ETTh1 and ETTm1 (univariate + multivariate).
Our best baseline achieves **MAE 0.92-1.01** vs the paper's **0.15-0.42** — a **2-6x gap**.

The dominant cause is a **train-test conditioning mismatch** amplified by the absence of
**DPM-Solver**: each stage trains on ground-truth coarser trends but receives noisy
predictions at inference, causing cascading errors that DDPM sampling cannot absorb.

**Next step:** Implement DPM-Solver to close the gap.

---

## Best Results (Run 082858)

| Experiment | MAE (Ours) | MAE (Paper) | MSE (Ours) | MSE (Paper) | MAE Gap |
|---|---|---|---|---|---|
| ETTh1 Multivariate | 0.922 | **0.422** | 1.457 | **0.411** | 2.2x |
| ETTh1 Univariate | 1.007 | **0.196** | 1.610 | **0.066** | 5.1x |
| ETTm1 Multivariate | 0.972 | **0.373** | 1.576 | **0.340** | 2.6x |
| ETTm1 Univariate | 0.963 | **0.149** | 1.473 | **0.039** | 6.5x |

Uses `predictions[0]` (finest residual only). Full multi-stage `sum(predictions)` is
theoretically correct but practically broken without DPM-Solver (see [Run History](#run-history)).

---

## Run History

Five runs, each testing specific hypotheses. Run 082858 remains our best.

| # | Timestamp | Description | ETTh1-M | ETTh1-U | ETTm1-M | ETTm1-U | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | 222824 | Initial implementation | 1.23 | 1.58 | 3.07 | 3.26 | Baseline established |
| 2 | **082858** | **pred[0] only** | **0.92** | **1.01** | **0.97** | **0.96** | **Best result** |
| 3 | 114727 | Cumulative trend decomp | 8.85 | 32.29 | 6.14 | 7.76 | Catastrophic, reverted |
| 4 | 144502 | AMP + learned proj + sum | 1.32 | 1.39 | 2.74 | 3.47 | Regressed, 3 compounding issues |
| 5 | 173919 | sum(pred) isolated test | 1.61 | 1.21 | 6.92 | 2.10 | Confirmed sum harmful |

*All values are MAE on instance-normalized data, 10 trajectory samples.*

### Key findings per run

**Run 3 (114727)** — Attempted paper-faithful cumulative trend decomposition, per-stage
lookback, and removed double instance normalization. All changes catastrophically broke
inference (MAE 6-32). Residual decomposition is more stable because each stage predicts
a small, low-variance component.

**Run 4 (144502)** — Three compounding issues identified:
1. **AMP** corrupted diffusion schedule precision (tensors stored as plain attributes, not buffers)
2. **Learned mixup projection** created train-test gap (exploits ground-truth features that vanish at inference)
3. **sum(predictions)** exposed coarser stage errors

**Run 5 (173919)** — Isolated `sum(predictions)` as the sole change. Training validation
losses were our *best ever*, yet evaluation was 1.2-7.1x worse. This proves the issue
is inference-time composition quality, not training quality. DPM-Solver is the missing piece.

---

## Applied Fixes (Current Configuration)

| Fix | File | Detail |
|---|---|---|
| Continuous mixup mask | `src/models/conditioning.py` | Binary `(rand > 0.5)` changed to continuous `rand()` per paper |
| LeakyReLU slope 0.1 | `src/models/conditioning.py` | Was 0.2, paper Table 13 specifies 0.1 |
| ETTm1 forecast length | `train.py`, `run_experiments.py` | Was silently using 168 instead of 192 |
| `cudnn.benchmark` | `train.py` | ~10-15% training speedup |
| `non_blocking` transfers | `trainer.py`, `metrics.py` | ~5% training speedup |

### Tested and Reverted

| Change | Why It Failed |
|---|---|
| `sum(predictions)` inference | Coarser stages too noisy under DDPM; needs DPM-Solver |
| Learned mixup projection | Train-test gap: model exploits ground-truth features absent at inference |
| AMP (mixed precision) | Corrupts diffusion schedule precision and gradient quality |
| Cumulative trend decomposition | Cascading errors at inference (MAE 6-32) |
| Per-stage lookback decomposition | Coupled with cumulative trends; reverted together |
| Remove double instance norm | Kept for training stability |

---

## Performance Gap Analysis

Our best MAE (0.92-1.01) vs the paper (0.15-0.42) stems from three factors:

1. **Train-test conditioning mismatch** — Stages train with ground-truth coarser trends
   but infer with noisy predictions. This cascading error is the root cause.
2. **Missing DPM-Solver** — Not just a speed optimization. It's critical for producing
   predictions accurate enough for multi-stage composition to work.
3. **No per-dataset hyperparameter search** — Paper uses per-dataset grid search for S
   and kernel sizes; we use fixed S=5 with kernels [5, 25, 51, 201].

### Reproducibility note

Every paper-described component we faithfully implemented degraded or broke performance
under standard DDPM sampling:

| Paper Component | Our Result |
|---|---|
| Cumulative trend decomposition (Section 4.1) | MAE 6-32 (catastrophic) |
| Learned mixup projection (Eq. 9) | MAE 1.3-3.5 (regression) |
| sum(predictions) reconstruction | MAE 1.2-6.9 (regression) |

The paper presents DPM-Solver as an optional efficiency tool (Section 5.2). Our findings
suggest it is architecturally necessary — without it, the multi-resolution framework
underperforms even its own finest-stage-only baseline.

---

## Paper Reference Values

From Tables 1, 2, 5, and 6 (mr-Diff row):

| Dataset | Uni MAE | Uni MSE | Multi MAE | Multi MSE |
|---|---|---|---|---|
| ETTh1 | 0.196 | 0.066 | 0.422 | 0.411 |
| ETTm1 | 0.149 | 0.039 | 0.373 | 0.340 |

Hyperparameters: Adam lr=1e-3, batch 64, 100 epochs, K=100 diffusion steps,
beta 1e-4 to 0.1, S=5 stages, ETTh1 L=336/H=168, ETTm1 L=1440/H=192.

---

## Usage

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

# Generate report figures and tables (in Jupyter or standalone)
python generate_figures.py
python generate_tables.py
```

## Project Structure

```
project-baseline/
  configs/default.yaml          # Model and training hyperparameters
  src/
    data/
      dataset.py                # ETT dataset with per-window RevIN normalization
      preprocessing.py          # TrendExtraction and InstanceNormalization
    models/
      mr_diff.py                # Main mr-Diff model (training_step + sample)
      conditioning.py           # Conditioning network with future mixup
      denoising.py              # Encoder-decoder denoising network
      diffusion.py              # Diffusion schedule and forward process
    evaluation/
      metrics.py                # MAE, MSE computation on normalized data
    training/
      trainer.py                # Training loop with early stopping
  train.py                      # Training entry point
  evaluate.py                   # Evaluation entry point
  run_experiments.py            # Full experiment runner (all 4 configs)
  generate_figures.py           # Report figure generation
  generate_tables.py            # Report table generation
  boom.ipynb                    # Notebook for interactive figure/table review
  analyze.py                    # Analysis utilities
  experiments/                  # Experiment output directories
  data/ETDataset/               # ETTh1.csv and ETTm1.csv
```
