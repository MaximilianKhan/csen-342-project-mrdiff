# Project Summary

**Course:** CSEN-342, Winter 2026
**Paper:** Multi-Resolution Diffusion Models for Time Series Forecasting (mr-Diff, ICLR 2024)
**Datasets:** ETTh1 (hourly electricity transformer data) and ETTm1 (15-minute), each univariate and multivariate (7 variables)

---

## What We Did

We set out to replicate mr-Diff — a diffusion-based time series forecasting model — and improve upon it. The paper's code was not available, so we built the entire pipeline from scratch: data loading, RevIN normalization, multi-resolution trend decomposition, conditional diffusion with DPM-Solver++ sampling, and a DLinear backbone. This from-scratch implementation became **our baseline** — the starting point against which all improvements are measured.

Over the course of **27 experiments in a single day**, we systematically explored diffusion improvements, backbone modifications, attention mechanisms, and ultimately discovered that the diffusion component was architecturally inert. This led us to replace the entire diffusion pipeline with a lightweight transformer that trains 20x faster and produces better forecasts.

---

## The Baseline We Built

Since the paper's code was unavailable, we constructed our own faithful implementation of mr-Diff:

- **843K parameters** (730K diffusion + 113K DLinear backbone)
- 3-stage multi-resolution diffusion with self-conditioning
- DPM-Solver++ (20-step) sampling
- RevIN per-window normalization
- ~34 minutes training time for all 4 benchmarks

**Our constructed baseline results (globally-standardized MAE):**

| Benchmark | Our Baseline | Paper's Claimed |
|---|---|---|
| ETTh1 Multivariate | 0.4744 | 0.42 |
| ETTh1 Univariate | 0.2535 | 0.34 |
| ETTm1 Multivariate | 0.4204 | 0.37 |
| ETTm1 Univariate | 0.2011 | 0.15 |

Our baseline already **beats the paper on ETTh1 Univariate** (0.2535 vs 0.34, -25%). The multivariate gaps (+13%) reflect differences in our from-scratch implementation vs the paper's unpublished code. **All improvements below are measured against our constructed baseline — not the paper's numbers.**

---

## The Critical Discovery

During evaluation, we compared the DLinear backbone's predictions (without diffusion) against the full model's predictions (with diffusion). They were identical. **The diffusion component — 730K parameters, 86% of the model — contributed exactly 0% to forecast accuracy.** The entire predictive power came from two simple linear projections totaling 113K parameters.

We spent 14 experiments trying to make diffusion useful (noise schedules, loss functions, architecture changes, attention mechanisms). None succeeded. This led to the paradigm shift.

---

## The Paradigm Shift: From Diffusion to Transformer

We replaced the entire 843K-parameter diffusion pipeline with a **channel-independent patch transformer**:

- Each of the 7 variables is patched and processed **independently** through a **shared** transformer
- Input is decomposed into trend and residual (DLinear's proven inductive bias) before patching
- 2-3 layer TransformerEncoder with self-attention over temporal patch tokens
- Lightweight output head: `Linear(N_patches → T)` + `Linear(d_model → 1)`
- **54-182K total parameters** — smaller than the DLinear backbone alone on most benchmarks
- **2-15 minutes** training time — 20x faster than diffusion
- Single forward pass inference (vs 60 denoiser calls for diffusion)

The channel-independent design was the key breakthrough: it effectively multiplies training data by D (each channel is an independent training sequence) and eliminates cross-channel overfitting that plagued every prior experiment.

For ETTh1 Multivariate (our hardest benchmark), we additionally leverage **Attention Residuals** from the Kimi team's March 2026 paper — replacing standard transformer residual connections with learned selective retrieval over all prior layers — combined with gentle data augmentation. This is deployed as a 3-model heterogeneous ensemble (different architectures and hyperparameters averaged at inference).

---

## Final Results: Our Improvements Over Our Baseline

| Benchmark | Our Baseline | Our Best | **Improvement** | Method |
|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | **0.4829** | +1.8% | 3-model ensemble (CI+Decomp + AttnRes+Aug) |
| ETTh1 Uni | 0.2535 | **0.2505** | **-1.2%** | Single CI+Decomp Transformer |
| ETTm1 Multi | 0.4204 | **0.4094** | **-2.6%** | Single CI+Decomp Transformer |
| ETTm1 Uni | 0.2011 | **0.1881** | **-6.5%** | Single CI+Decomp Transformer |

**We beat our constructed baseline on 3 of 4 benchmarks.** The one remaining gap (ETTh1 Multi, +1.8%) is the smallest it has ever been, achieved through ensembling architecturally diverse models.

### MSE Results

| Benchmark | Our Best MAE | Our Best MSE | Paper MAE | Paper MSE |
|---|---|---|---|---|
| ETTh1 Multi | 0.4829 | 0.4795 | 0.42 | 0.411 |
| ETTh1 Uni | 0.2505 | 0.1060 | 0.34 | 0.066 |
| ETTm1 Multi | 0.4094 | 0.3709 | 0.37 | 0.340 |
| ETTm1 Uni | 0.1881 | 0.0620 | 0.15 | 0.039 |

---

## The 27-Experiment Journey

| Phase | Experiments | What We Tried | What We Learned |
|---|---|---|---|
| **Baseline** | — | Built mr-Diff from scratch | Diffusion contributes 0% to forecasting |
| **Diffusion fixes** | 1-10 | Schedules, losses, architectures, parameterizations | 10 experiments, nothing makes diffusion useful on small data |
| **Backbone depth** | 11-13 | AttnRes, deep conv blocks | Depth overfits; AttnRes mechanism validated but wrong application |
| **Multi-scale** | 14 | Parallel DLinear at different scales | Too many parameters, killed early |
| **Transformer v1** | 15 | Naive PatchTST (no CI) | Matched uni baseline in 1.8 min; head was 95% of params |
| **Transformer v2** | 16-17 | Channel-independent + decomposition | **Breakthrough:** beat baseline on ETTm1 Multi and Uni |
| **Hyperparameter sweep** | 18 | 30 configs × 4 benchmarks | Found optimal per-benchmark configs; set records on 3/4 |
| **Bolt-on improvements** | 19-25 | Training schedule, augmentation, freq branch, channel mixing | None beat sweep — architecture had plateaued |
| **AttnRes + augmentation** | 26 | Kimi's Attention Residuals + gentle data augmentation | Cracked ETTh1 Multi wall (0.4875) |
| **Ensemble** | 27 | Per-dataset heterogeneous ensemble | ETTh1 Multi → 0.4829, ETTh1 Uni → 0.2505 |

---

## Architecture Comparison

| | Paper / Our Baseline | Our Submission |
|---|---|---|
| **Type** | Conditional Diffusion Model | Patch Transformer |
| **Inference** | 60 denoiser forward passes | 1 forward pass |
| **Parameters** | 843K | 54-182K |
| **Training** | ~34 min | ~2-15 min |
| **Channel handling** | Shared across all D | Independent per channel (shared weights) |
| **Decomposition** | 3-5 stage multi-resolution hierarchy | Single trend/residual split |

---

## Key Takeaways

1. **Diffusion is overhead for deterministic time series forecasting on small datasets.** 14 experiments, zero useful contribution.

2. **Channel independence is the single most important design choice.** Processing each variable independently through shared weights provides implicit regularization equivalent to multiplying the training data by D.

3. **Right-sizing beats architecture.** Our 54K-param transformer outperforms an 843K-param diffusion model. On small datasets, fewer parameters = less overfitting = better generalization.

4. **Speed enables discovery.** The shift from 34-minute to 2-minute training cycles allowed us to run a 30-config hyperparameter sweep that found per-benchmark optimal configurations impossible to discover at diffusion speed.

5. **Attention Residuals need diverse data.** The Kimi team's AttnRes mechanism (selective depth-wise retrieval) only helps when combined with data augmentation that provides the signal diversity for learned queries to differentiate on.

---

## Submission Contents

| File | What It Contains |
|---|---|
| `SUMMARY.md` | This file — 5-minute overview |
| `REPORT.md` | Full 10-section technical report of the 27-experiment journey |
| `RESULTS.md` | Per-benchmark best models with exact configs and hyperparameters |
| `MSE_COMPARISON.md` | Complete MAE + MSE analysis across all experiments |
| `ARCHITECTURE_COMPARISON.md` | Detailed architectural breakdown: diffusion vs transformer |
| `train_single.py` | Train the single best model per benchmark |
| `train_ensemble.py` | Train the full ensemble (includes ETTh1 Multi 3-model ensemble) |
| `src/models/ci_decomp_transformer.py` | Channel-Independent Decomposed Transformer (wins 3/4 benchmarks) |
| `src/models/ci_attnres_transformer.py` | AttnRes variant (used in ETTh1 Multi ensemble) |
