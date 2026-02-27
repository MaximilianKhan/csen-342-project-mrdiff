# mr-Diff Experiment Results

## Current Best: Run 082858 (Feb 12)

| Dataset | Mode | MAE (Ours) | MAE (Paper) | MSE (Ours) | MSE (Paper) | MAE Gap |
|---|---|---|---|---|---|---|
| ETTh1 | Multivariate | 0.922 | 0.422 | 1.457 | 0.411 | 2.2x |
| ETTh1 | Univariate | 1.007 | 0.196 | 1.610 | 0.066 | 5.1x |
| ETTm1 | Multivariate | 0.972 | 0.373 | 1.576 | 0.340 | 2.6x |
| ETTm1 | Univariate | 0.963 | 0.149 | 1.473 | 0.039 | 6.5x |

Run directory: `experiments/run_20260212_082858/`

---

## Paper Benchmarks

From Tables 1, 2, 5, 6 of the ICLR 2024 paper:

| Dataset | Mode | Horizon | MAE | MSE |
|---|---|---|---|---|
| ETTh1 | Univariate | 168 | 0.196 | 0.066 |
| ETTh1 | Multivariate | 168 | 0.422 | 0.411 |
| ETTm1 | Univariate | 192 | 0.149 | 0.039 |
| ETTm1 | Multivariate | 192 | 0.373 | 0.340 |

---

## Implementation vs Paper

### Matching hyperparameters

| Parameter | Value | Match |
|---|---|---|
| Learning rate | 1e-3 | Yes |
| Batch size | 64 | Yes |
| Max epochs | 100 | Yes |
| Diffusion steps K | 100 | Yes |
| Beta schedule | 1e-4 to 0.1 | Yes |
| Stages S | 5 | Yes |
| Embedding dim | 128 | Yes |
| Hidden dim | 256 | Yes |
| Kernel sizes | (5, 25, 51, 201) | Yes |
| Conv kernel | 3 | Yes |
| Dropout | 0.1 | Yes |
| LeakyReLU slope | 0.1 | Yes (fixed from 0.2) |
| ETTm1 lookback | 1440 | Yes (fixed from 336) |

### Known differences

| Component | Paper | Ours | Impact |
|---|---|---|---|
| Reverse diffusion | DPM-Solver | DDPM (100 steps) | Primary gap source |
| Trend decomposition | Cumulative | Residual | Cumulative broke inference |
| Signal reconstruction | sum(all stages) | predictions[0] only | sum requires DPM-Solver |
| Mixup projection | Learned (Eq. 9) | Random weights | Learned widens train-test gap |
| Kernel/S selection | Per-dataset grid search | Fixed | Minor impact |

---

## Progress History

### Before per-window normalization (global StandardScaler)
- ETTh1 Uni MAE: 23.93, Multi MAE: 9.61

### After per-window normalization (RevIN-style)
- ETTh1 Uni MAE: 1.17 (20x improvement), Multi MAE: 1.84 (5x improvement)

### After bug fixes and optimization (Run 082858)
- All experiments MAE 0.92-1.01 (best achieved with DDPM)

---

## Key Paper Insights (Appendix)

1. **Stages S** (Table 9): S=3-5 optimal; S=1 is worst
2. **Diffusion steps K** (Table 10): K=100 optimal
3. **Beta schedule** (Table 11): beta_K=0.1 optimal
4. **Future mixup** (Table 12): Uniform distribution (gamma=1) works best
5. **Instance normalization**: RevIN-style per-window is used

---

## Environment

- GPU: NVIDIA RTX 5090
- Framework: PyTorch
- Training time (best run): ~63 min training, ~79 min evaluation
