# mr-Diff Experiment Results

## Overview

This document tracks our experimental results for the mr-Diff (Multi-Resolution Diffusion) model implementation, comparing against the ICLR 2024 paper benchmarks.

## Latest Results (2026-02-12)

Run: `experiments/run_20260211_222824`

### ETTh1 Dataset (Hourly, Forecast Horizon = 168)

| Mode | MAE | MAE Std | MSE | MSE Std | RMSE | Epochs | Training Time |
|------|-----|---------|-----|---------|------|--------|---------------|
| Univariate | 1.169 | 0.356 | 2.350 | 1.478 | 1.533 | 53 | 241s |
| Multivariate | 1.839 | 0.389 | 5.775 | 2.194 | 2.403 | 51 | 232s |

### ETTm1 Dataset (15-min, Forecast Horizon = 192)

| Mode | MAE | MAE Std | MSE | MSE Std | RMSE | Epochs | Training Time |
|------|-----|---------|-----|---------|------|--------|---------------|
| Univariate | 3.063 | 0.769 | 14.655 | 7.240 | 3.828 | 63 | 1143s |
| Multivariate | 3.314 | 0.579 | 27.672 | 13.510 | 5.260 | 50 | 913s |

**Total Training Time:** ~95 minutes (5710 seconds)

---

## Paper Benchmarks (Table 1)

From the mr-Diff ICLR 2024 paper:

| Dataset | Mode | Horizon | Paper MAE | Paper MSE |
|---------|------|---------|-----------|-----------|
| ETTh1 | Univariate | 168 | 0.196 | 0.066 |
| ETTh1 | Multivariate | 168 | 0.422 | 0.411 |
| ETTm1 | Univariate | 192 | 0.149 | 0.039 |
| ETTm1 | Multivariate | 192 | 0.373 | 0.340 |

---

## Gap Analysis

| Dataset | Mode | Our MAE | Paper MAE | Ratio |
|---------|------|---------|-----------|-------|
| ETTh1 | Univariate | 1.169 | 0.196 | 6.0x |
| ETTh1 | Multivariate | 1.839 | 0.422 | 4.4x |
| ETTm1 | Univariate | 3.063 | 0.149 | 20.5x |
| ETTm1 | Multivariate | 3.314 | 0.373 | 8.9x |

---

## Paper vs. Our Implementation: Detailed Comparison

### CRITICAL BUG FOUND

| Component | Paper (Algorithm 2) | Our Implementation | Status |
|-----------|---------------------|-------------------|--------|
| **Final output** | `return Ŷ⁰₀` (stage 0 only) | `sum(predictions)` (all stages) | **BUG** |

**Issue:** We sum ALL stage predictions together, but the paper returns only the finest resolution prediction `Ŷ⁰₀`. The coarser stages are used as **conditions**, not summed.

**Location:** `src/models/mr_diff.py` lines 284-286

### Hyperparameter Differences

| Parameter | Paper | Ours | Match? |
|-----------|-------|------|--------|
| Learning rate | 10⁻³ | 10⁻³ | ✓ |
| Batch size | 64 | 64 | ✓ |
| Max epochs | 100 | 100 | ✓ |
| Diffusion steps K | 100 | 100 | ✓ |
| Beta schedule | β₁=10⁻⁴ to βK=10⁻¹ | β₁=10⁻⁴ to βK=10⁻¹ | ✓ |
| Num stages S | 5 | 5 | ✓ |
| Embedding dim | 128 | 128 | ✓ |
| Hidden dim | 256 | 256 | ✓ |
| Kernel sizes | (5, 25, 51, 201) | (5, 25, 51, 201) | ✓ |
| Conv kernel size | 3 | 3 | ✓ |
| Dropout | 0.1 | 0.1 | ✓ |
| **LeakyReLU slope** | **0.1** | **0.2** | **✗** |
| **ETTm1 lookback** | **1440** | **336** | **✗** |
| **DPM-Solver** | **Used** | **Not implemented** | **✗** |

### Architecture Details (Paper Appendix F, Table 13)

| Layer | Paper Config | Our Config | Match? |
|-------|--------------|------------|--------|
| Conv1d | in=256, out=256, kernel=3, stride=1, padding=1 | Same | ✓ |
| BatchNorm1d | features=256 | features=256 | ✓ |
| LeakyReLU | negative_slope=0.1 | negative_slope=0.2 | ✗ |
| Dropout | rate=0.1 | rate=0.1 | ✓ |

### Lookback Length (Paper Table 8)

Paper's optimal lookback lengths:
| Dataset | Optimal L | Our L |
|---------|-----------|-------|
| ETTh1 | 336 | 336 | ✓ |
| ETTm1 | **1440** | **336** | ✗ |

---

## Required Fixes (Priority Order)

### 1. CRITICAL: Fix Output Reconstruction
**File:** `src/models/mr_diff.py`
**Lines:** 284-286
**Current:**
```python
forecast = sum(predictions)
```
**Should be:**
```python
forecast = predictions[0]  # Return only finest resolution
```

### 2. HIGH: Fix ETTm1 Lookback Length
**File:** `run_experiments.py` or experiment configs
**Change:** Use lookback_length=1440 for ETTm1 (currently 336)

### 3. MEDIUM: Fix LeakyReLU Slope
**File:** `src/models/denoising.py`
**Line:** 44
**Current:** `nn.LeakyReLU(0.2)`
**Should be:** `nn.LeakyReLU(0.1)`

### 4. LOW: Implement DPM-Solver (Optional)
The paper uses DPM-Solver for faster sampling (reduces ~100 steps to ~10).
This improves inference speed but shouldn't significantly affect accuracy.

---

## Progress History

### Before Per-Window Normalization (Global StandardScaler)
- ETTh1 Univariate MAE: 23.93
- ETTh1 Multivariate MAE: 9.61

### After Per-Window Normalization (RevIN-style)
- ETTh1 Univariate MAE: 1.17 (20x improvement)
- ETTh1 Multivariate MAE: 1.84 (5x improvement)

### Expected After Fixes
With the critical bug fix (returning predictions[0] instead of sum), we expect:
- ETTh1 results to be much closer to paper benchmarks
- ETTm1 results to improve significantly with longer lookback (1440)

---

## Environment

- GPU: NVIDIA RTX 5090
- Framework: PyTorch
- Batch Size: 64
- Max Epochs: 100
- Learning Rate: 0.001

---

## Key Paper Insights (from Appendix)

1. **Number of stages S** (Table 9): S=3-5 works best; S=1 is worst
2. **Diffusion steps K** (Table 10): K=100 is optimal
3. **Beta schedule** (Table 11): βK=0.1 is optimal
4. **Future mixup** (Table 12): Uniform distribution (γ=1) works best
5. **Instance normalization**: RevIN-style per-window normalization is used

---

## Next Steps

- [ ] Fix critical bug: return predictions[0] instead of sum(predictions)
- [ ] Fix LeakyReLU slope from 0.2 to 0.1
- [ ] Use lookback_length=1440 for ETTm1
- [ ] Re-run experiments after fixes
- [ ] Consider implementing DPM-Solver for faster inference
