# Architecture Comparison: Paper vs Baseline vs Submission

## Architecture Classification

| Model | Architecture Family | Core Mechanism | Parameters |
|---|---|---|---|
| **Paper (mr-Diff)** | Diffusion Model + Linear Backbone | Multi-resolution denoising diffusion (DDPM) with DLinear direct predictor | ~843K |
| **Our Baseline** | Diffusion Model + Linear Backbone | Same — our replication of mr-Diff | 843K |
| **Our Submission** | Patch Transformer | Channel-independent PatchTST-style encoder with linear head | 54-182K |

---

## What Each Model Is

### The Paper / Our Baseline: Conditional Denoising Diffusion Model

A **conditional denoising diffusion probabilistic model** for time series forecasting. Specifically:

- **Generative model** — learns a distribution over possible forecasts via iterative denoising
- **Multi-resolution hierarchy** — decomposes the forecast target into 3 stages (fine, mid, coarse) via average-pooling trend extraction at different kernel sizes
- **Per-stage diffusion** — each resolution stage has its own conditioning network (Conv1d history encoder), denoising network (U-Net style convolutional encoder-decoder with skip connections), and noise schedule
- **DLinear backbone** — a simple linear model (two `nn.Linear` projections for trend and residual) that provides the base forecast; diffusion refines the residual
- **Sampling** — DPM-Solver++ (20 steps) per stage, 3 stages = 60 denoiser forward passes at inference
- **Self-conditioning** — the denoiser receives its own previous x0 estimate as additional input for iterative refinement

**Key components:**
| Component | Type | Params | Role |
|---|---|---|---|
| DLinear backbone | 2× Linear(H→T) | 113K | Direct forecast (trend + residual projection) |
| Conditioning networks | Conv1d encoder (×3 stages) | ~200K | Encode lookback history for each stage |
| Denoising networks | U-Net conv encoder-decoder (×3 stages) | ~500K | Predict noise at each diffusion step |
| Step embedding | Sinusoidal + MLP | ~17K | Encode diffusion timestep |
| Trend extraction | Avg-pool (no params) | 0 | Decompose target into resolution stages |

**Architecture family:** Diffusion + Linear hybrid. The linear backbone does deterministic forecasting; the diffusion component attempts stochastic residual refinement.

---

### Our Submission: Channel-Independent Patch Transformer

A **channel-independent patch transformer** with trend/residual decomposition for time series forecasting. Specifically:

- **Discriminative model** — directly predicts the forecast in a single forward pass
- **Patch tokenization** — the lookback window is divided into non-overlapping fixed-size patches (8-16 timesteps each), each projected to a d-dimensional token
- **Self-attention over time** — a standard TransformerEncoder (2-3 layers, pre-norm, GELU) processes the patch token sequence, capturing temporal dependencies at all ranges
- **Channel-independent (CI) processing** — each variable is treated as an independent sequence through the same shared transformer; no cross-channel mixing during encoding
- **DLinear-style decomposition** — input is split into trend (avg-pool) and residual before patching; both processed by the same shared transformer with separate output heads
- **Linear output head** — `Linear(N_patches→T)` for temporal projection + `Linear(d_model→1)` for channel projection; parameters are independent of the number of variables D
- **Optional Attention Residuals** — on ETTh1 Multi ensemble, one model replaces standard transformer residual connections with AttnRes (Kimi 2026): each layer selectively attends over all prior layer outputs via learned pseudo-queries

**Key components:**
| Component | Type | Params | Role |
|---|---|---|---|
| Patch embedding | Linear(patch_size→d_model) | ~0.5-1K | Convert time patches to tokens |
| Positional embedding | Learnable [N_patches, d_model] | ~1-6K | Encode temporal position |
| TransformerEncoder | 2-3 layers, pre-norm, GELU | ~35-67K | Temporal pattern extraction via self-attention |
| Trend output head | Linear(N→T) + Linear(d→1) | ~4-12K | Project trend forecast |
| Residual output head | Linear(N→T) + Linear(d→1) | ~4-12K | Project residual forecast |
| Trend extraction | Avg-pool (no params) | 0 | Decompose input into trend + residual |
| *AttnRes queries* | *3× learned vectors (optional)* | *~192* | *Selective depth-wise retrieval* |

**Architecture family:** Encoder-only Transformer. No decoder, no cross-attention, no autoregressive generation, no diffusion.

---

## Side-by-Side Comparison

| Aspect | Paper / Baseline | Our Submission |
|---|---|---|
| **Architecture family** | Diffusion (generative) | Transformer (discriminative) |
| **Prediction type** | Stochastic (sample from learned distribution) | Deterministic (single forward pass) |
| **Temporal modeling** | Conv1d in denoiser + linear projection | Self-attention over patch tokens |
| **Channel handling** | Shared Conv1d across all D channels | Channel-independent (each channel separately, shared weights) |
| **Decomposition** | Multi-resolution hierarchy (3-5 stages, kernels 5/25/51/201) | Single trend/residual split (1 kernel, size 15 or 25) |
| **Normalization** | RevIN (per-window, in dataloader) | RevIN (per-window, in dataloader) |
| **Output mechanism** | `direct_pred + Σ diffusion_stages` via 60 denoiser calls | `trend_head(transformer(trend)) + resid_head(transformer(resid))` — 2 forward passes |
| **Inference cost** | 60 denoiser forward passes (DPM-Solver++ × 3 stages) | 1-2 forward passes (trend + residual through shared transformer) |
| **Parameters** | 843K (730K diffusion + 113K backbone) | 54-182K total |
| **Training time (all 4 benchmarks)** | ~34 minutes | ~2-15 minutes |
| **Learnable residual connections** | Standard additive (`x + f(x)`) | Standard or AttnRes (`Σ α_i · v_i`, Kimi 2026) |

---

## The Common Denominator of All Submission Models

Every model in our submission shares this architectural DNA:

1. **Encoder-only Transformer** — no decoder, no cross-attention, no autoregressive generation
2. **Channel-Independent (CI)** — each variable is an independent sequence through the same shared model (the key regularization that prevents multivariate overfitting)
3. **Patch-based tokenization** — lookback window divided into non-overlapping fixed-size patches
4. **DLinear-style trend/residual decomposition** — avg-pool trend extraction as preprocessing before the transformer
5. **Linear CI output head** — `Linear(N_patches→T)` temporal projection + `Linear(d_model→1)` channel projection; params independent of D

### Per-Benchmark Model Variants

| Benchmark | Base Architecture | Modifications | Total Params |
|---|---|---|---|
| ETTh1 Multi | CI+Decomp Transformer | **3-model ensemble**: 1× with AttnRes + gentle augmentation, 2× base with different configs | 3×(55-86K) |
| ETTh1 Uni | CI+Decomp Transformer | None (pure base architecture) | 54K |
| ETTm1 Multi | CI+Decomp Transformer | None (pure base architecture) | 182K |
| ETTm1 Uni | CI+Decomp Transformer | None (pure base architecture) | 77K |

The ETTh1 Multi ensemble is the only benchmark using AttnRes (Attention Residuals from Kimi 2026). This modifies how information flows between transformer layers — replacing fixed additive residuals `x = x + layer(x)` with learned selective retrieval `x = Σ softmax(query · RMSNorm(prior_outputs)) · prior_outputs + layer(x)` — but the model remains fundamentally a transformer.

---

## Why We Moved From Diffusion to Transformer

The shift was driven by empirical evidence across 14 diffusion experiments:

| Finding | Evidence | Implication |
|---|---|---|
| Diffusion contributes 0% to forecast accuracy | Direct MAE = Full MAE across all benchmarks | The 730K diffusion params are pure overhead |
| Small datasets resist complexity | Every experiment >200K params regressed on multivariate | Need smaller, not larger models |
| Channel independence is critical | CI reduced multivariate error by 16 percentage points (Exp 15→16) | Cross-channel mixing overfits on 10K samples |
| Speed enables discovery | 34 min → 2 min per experiment = 17x more iterations | Found optimal configs impossible to discover at diffusion speed |
| Linear decomposition is a good prior | DLinear's trend/residual split helps transformers too (Exp 17) | Keep the inductive bias, replace the architecture |

**In one sentence:** We replaced a 843K-parameter conditional diffusion model with a 54-182K-parameter channel-independent patch transformer — same task, same data, same evaluation, 20x faster, 3 of 4 benchmarks improved.
