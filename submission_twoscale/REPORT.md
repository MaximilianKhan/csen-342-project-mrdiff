# From Diffusion to Transformers: A 27-Experiment Journey in Time Series Forecasting

**Course:** CSEN-342, Winter 2026
**Paper:** Multi-Resolution Diffusion Models for Time Series Forecasting (mr-Diff, ICLR 2024)
**Datasets:** ETTh1 (hourly, L=336, H=168), ETTm1 (15-min, L=1440, H=192), each univariate + multivariate

---

## Abstract

We attempted to replicate and improve upon mr-Diff, a multi-resolution diffusion model for time series forecasting. Over the course of 27 experiments spanning diffusion modifications, backbone redesigns, attention residuals, and a paradigm shift to transformer-based forecasting, we discovered that the diffusion component of mr-Diff contributes nothing to forecast quality — the entire predictive power resides in a simple DLinear backbone. This discovery led us to replace the entire diffusion pipeline with a channel-independent patch transformer that trains 20x faster, uses 60% fewer parameters, and ultimately beats our baseline on 3 of 4 benchmarks while exceeding the paper's claimed performance on ETTh1 univariate by 26%.

---

## 1. Baseline Replication

### 1.1 The mr-Diff Architecture

The mr-Diff paper proposes a multi-resolution diffusion model that decomposes time series into hierarchical trend components and applies stage-wise diffusion denoising. Our implementation includes:

- **DLinear backbone:** Trend/residual decomposition via average pooling, followed by independent linear projections to the forecast horizon (113K parameters)
- **Multi-stage diffusion:** 3-stage hierarchical decomposition with per-stage conditioning networks, denoising networks, and DPM-Solver++ sampling (730K parameters)
- **Self-conditioning:** The denoiser receives its own previous x0 estimate as additional input
- **RevIN normalization:** Per-window reversible instance normalization for stationarity

### 1.2 Initial Results

Our baseline replication achieved:

| Benchmark | Our MAE | Paper MAE | Gap |
|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.42 | +13.0% |
| **ETTh1 Uni** | **0.2535** | **0.34** | **-25.4% (beats paper)** |
| ETTm1 Multi | 0.4204 | 0.37 | +13.6% |
| ETTm1 Uni | 0.2011 | 0.15 | +34.1% |

Model: 843K params total (730K diffusion + 113K backbone). Training: ~34 minutes for all 4 benchmarks.

### 1.3 The Critical Discovery: Diffusion Is Cosmetic

During evaluation, we separately measured the DLinear backbone's direct predictions (without diffusion) and the full model's predictions (backbone + diffusion refinement). The results were identical. Across all 4 benchmarks, the diffusion component contributed exactly 0% to forecast accuracy. The Direct MAE equaled the Full MAE to four decimal places.

This finding — that a 730K-parameter diffusion pipeline sitting on top of a 113K-parameter linear model does nothing — became the defining insight of our entire campaign.

---

## 2. The Diffusion Improvement Campaign (Experiments 1-10)

Motivated by the possibility that diffusion's failure was due to implementation choices rather than fundamental limitations, we conducted 10 systematic experiments targeting every aspect of the diffusion pipeline.

### 2.1 Training Dynamics (Experiments 1-2)

**Experiment 1 (Remove Detach):** We enabled end-to-end gradient flow from diffusion loss through the backbone by removing `.detach()`. This was hypothesized to create a cooperative dynamic where the backbone learns to leave structured residuals for diffusion. **Result:** Marginal (±0.5%). The backbone still captured everything; diffusion still predicted noise.

**Experiment 2 (Self-Conditioning):** Added iterative refinement where the denoiser receives its own previous x0 estimate. **Result:** Our best diffusion variant. Small improvements on ETTh1 (0.4719 multi, 0.2523 uni). This became our baseline for all subsequent experiments.

### 2.2 Noise Schedules (Experiments 3-5)

**Experiment 3 (Cosine Schedule):** Catastrophic failure. Multivariate regressed +42-53%. The cosine schedule concentrates noise at low levels, starving the model of high-noise training signal on small datasets.

**Experiment 4 (v-Prediction):** No improvement. The uniform gradient variance advantage of v-prediction assumes capacity limitations that don't apply to our small model.

**Experiment 5 (ANT Adaptive Schedule):** Catastrophic failure (+55% multivariate). Confirmed the pattern: any attempt to reshape the noise distribution hurts on small datasets.

### 2.3 Auxiliary Losses and Guidance (Experiments 6-7)

**Experiment 6 (Contrastive Loss):** Multivariate regressed +17%. ETTm1 Uni improved to 0.1946.

**Experiment 7 (Multi-Granularity Guidance):** Multivariate regressed +14-20%. ETTm1 Uni hit 0.1913 — our best diffusion-era result on that benchmark.

### 2.4 Architecture Changes (Experiments 8-10)

**Experiment 8 (Channel-Aware Denoising):** Catastrophic failure. Per-channel encoders with cross-channel attention nearly doubled multivariate error (+97-117%). Massive overparameterization for 10K training samples.

**Experiment 9 (Patch+Attention Conditioning):** Replaced Conv1d history encoder with PatchTST-style transformer. Multivariate regressed +14-16%. The transformer conditioning created representations too complex for the small denoiser to exploit.

**Experiment 10 (x0-Prediction + Decomposition):** The most promising diffusion experiment. Switched to direct x0-prediction with trend/seasonality decomposition heads. Set new best on ETTh1 Uni (0.2508) and ETTm1 Multi (0.4194). **First time diffusion actively contributed** (ETTh1 Uni: Direct 0.2540 → Full 0.2508, -1.3%).

### 2.5 Campaign Verdict

After 10 experiments and 40 benchmark evaluations:
- Diffusion contributed meaningfully exactly **once** (Exp 10, ETTh1 Uni, -1.3%)
- **Every univariate improvement hurt multivariate** — the campaign's defining pattern
- The DLinear backbone did 100% of useful prediction in all other cases
- Small datasets (10K samples) resist architectural complexity at every turn

---

## 3. The Attention Residuals Exploration (Experiments 11-14)

### 3.1 Motivation: The Kimi Paper

On the morning of March 16, 2026, the Kimi team released "Attention Residuals" — a technique that replaces fixed additive residual connections with learned softmax attention over all prior layer outputs. Each layer gets a pseudo-query vector that selectively retrieves from any earlier layer rather than uniformly accumulating everything. The paper demonstrated consistent improvements across model scales on large language models.

We recognized the potential: if we could deepen our backbone with AttnRes connections, the selective retrieval might add expressiveness without the overfitting that plagued our diffusion experiments.

### 3.2 Backbone Depth Experiments

**Experiment 11 (Deep Backbone, Standard Residuals):** Replaced flat DLinear (2 linear projections) with 4 Conv1d blocks + additive residuals. 215K backbone params. **Result:** Catastrophic multivariate regression (+29-40%). Depth causes overfitting on small data.

**Experiment 12 (Deep Backbone, AttnRes):** Same architecture but with Attention Residuals replacing standard residuals. **Result:** Beat Exp 11 on 3/4 benchmarks (ETTh1 Uni: -3.3%). AttnRes was validated as a mechanism — it provides genuine improvement over standard residuals at the same depth. But the fundamental problem was depth itself, not the residual connection type.

**Experiment 13 (AttnRes + Stage Aggregation):** Added learned softmax aggregation over diffusion stages. **Result:** Hurt ETTm1 Uni (+12%). Smart weighting of near-zero diffusion contributions = overfitting to noise.

**Experiment 14 (Multi-Scale AttnRes DLinear):** 4 parallel DLinear projections at different temporal scales, fused with AttnRes attention. **Result:** Killed early — 3.6M params on ETTm1, +20% multivariate regression. The width approach added too many total parameters.

### 3.3 Key Insight

AttnRes works. It consistently outperforms standard residuals at equivalent depth. But the backbone shouldn't be deep in the first place — not at 10K training samples. The mechanism was validated; the application was wrong. We needed a different architecture entirely.

---

## 4. The Paradigm Shift: Transformers (Experiments 15-18)

### 4.1 The Tiny Transformer (Experiment 15)

We made a radical decision: abandon diffusion entirely. Replace the whole 843K-parameter mr-Diff pipeline with a tiny PatchTST-style transformer — patches, self-attention, linear head. No diffusion, no conditioning, no denoising.

**Result:** Matched baseline on univariate (ETTm1 Uni: 0.2002 vs 0.2011) in **1.8 minutes** of training. But multivariate regressed +18-31%.

**Root cause discovery:** The flatten→linear output head was **95-99% of the model's parameters** (1.6M-7.8M). The actual transformer was only 67K params. The "tiny transformer" was actually a massive linear projection that overfitted.

### 4.2 Channel-Independent Design (Experiments 16-17)

Applying PatchTST's channel-independent design fixed everything:

**Experiment 16 (CI Transformer):** Each channel patched and processed independently through a shared transformer. CI output head: Linear(N_patches→T) + Linear(d_model→1). **Total params: 73K** — independent of D, smaller than DLinear's 113K.

**Result:** ETTm1 Uni **0.1885 — new all-time best**. ETTm1 Multi within 2.1% of baseline.

**Experiment 17 (CI + Trend/Residual Decomposition):** Added DLinear's avg-pool decomposition before patching. Shared transformer processes both trend and residual.

**Result:** ETTm1 Multi **0.4159 — first model to beat DLinear baseline on multivariate.** Decomposition helped 3/4 benchmarks.

### 4.3 Hyperparameter Sweep (Experiment 18)

With 1-2 minute training cycles, we swept 30 random hyperparameter configurations across all 4 benchmarks. Key discoveries:

- **trend_kernel=15** dominates ETTh1 (hourly data needs finer decomposition than k=25)
- **d_model=32** sufficient for univariate; 48 for multivariate
- **3 layers consistently best** (CI keeps total params small enough to avoid overfitting)
- **patch=8** best for ETTm1 (180 tokens from L=1440)

**Sweep results:**
| Benchmark | Sweep Best | vs Baseline |
|---|---|---|
| ETTh1 Multi | 0.4880 | +2.9% |
| ETTh1 Uni | 0.2514 | -0.8% |
| ETTm1 Multi | **0.4094** | **-2.6%** |
| ETTm1 Uni | **0.1881** | **-6.5%** |

---

## 5. Attention Residuals Meet Augmentation (Experiment 26)

### 5.1 The Combination Hypothesis

Experiments 19-25 tested individual improvements (extended training, channel mixing, data augmentation, frequency features) on the optimized CI+Decomp transformer. None beat the sweep records — the architecture had plateaued.

But a key observation emerged: AttnRes was validated (Exp 12 beat standard residuals) and augmentation was validated conceptually (more data diversity should help small datasets), yet neither worked in isolation on the already-optimized model. What if they needed each other?

**The thesis:** AttnRes gives the transformer selective depth-wise retrieval — capacity to learn complex layer interactions. Data augmentation provides diverse training signal for that capacity to exploit. Without augmentation, AttnRes has nothing new to learn from. Without AttnRes, augmented data flows through standard residuals that can't selectively adapt.

### 5.2 Implementation

We replaced standard residual connections in the CI+Decomp transformer's encoder layers with full Attention Residuals:

```python
# Standard transformer residual:
x = x + self_attention(norm(x))

# AttnRes transformer:
h = softmax(query · RMSNorm(all_prior_layers)) · all_prior_layers  # Selective retrieval
x = h + self_attention(norm(h))
```

Combined with gentle augmentation (lighter than our failed Exp 22): Gaussian jitter (σ=0.01, 30% prob), mild scaling (±5%, 30% prob), and window masking (zero 5-10% of lookback, 30% prob).

AttnRes added only 192 parameters (3 query vectors × 64 dimensions).

### 5.3 Result

**ETTh1 Multi: 0.4875 — new all-time best**, breaking the 0.488 wall that 30 sweep configs couldn't crack. The combination worked precisely where we needed it: short-lookback multivariate, where selective depth retrieval across layers is most valuable and augmentation provides the diversity signal for queries to differentiate on.

---

## 6. The Ensemble (Experiment 27)

### 6.1 Per-Dataset Heterogeneous Ensemble

Our final experiment combined our best learnings: for each benchmark, we trained the top 3 performing model configurations (different architectures, patch sizes, trend kernels, dropout levels) and averaged their predictions. This is not a seed ensemble — each model has fundamentally different inductive biases:

- **Different patch sizes** → different temporal granularity
- **Different trend kernels** → different decomposition scales
- **Base vs AttnRes architectures** → different depth-wise information flow
- **Different dropout/learning rates** → different regularization regimes

### 6.2 Final Results

| Benchmark | Our Best | DLinear Baseline | Improvement | Paper | vs Paper |
|---|---|---|---|---|---|
| **ETTh1 Multi** | **0.4829** | 0.4744 | +1.8% | 0.42 | +14.9% |
| **ETTh1 Uni** | **0.2505** | 0.2535 | **-1.2%** | **0.34** | **-26.3%** |
| **ETTm1 Multi** | **0.4094** | 0.4204 | **-2.6%** | 0.37 | +10.6% |
| **ETTm1 Uni** | **0.1881** | 0.2011 | **-6.5%** | 0.15 | +25.4% |

We beat our DLinear baseline on **3 of 4 benchmarks**. We beat the paper's claimed results on ETTh1 Uni by **26.3%**. The remaining ETTh1 Multi gap (+1.8%) is the smallest it's ever been, achieved through ensemble averaging of architecturally diverse models.

---

## 7. Key Findings

### 7.1 Diffusion Is Overhead for Deterministic Forecasting

Across 14 diffusion experiments and 56 benchmark evaluations, the diffusion component contributed meaningfully exactly once (-1.3% on one benchmark). The stochastic denoising process is designed to model distributions, not point estimates. For deterministic time series forecasting on small datasets, it adds 730K parameters of pure overhead — gradients that interfere with backbone optimization, training time that could be spent iterating, and architectural complexity that obscures what actually works.

### 7.2 Channel Independence Is the Critical Regularization

The single most impactful design choice in our entire campaign was channel-independent processing. When we switched from a model that mixed all 7 channels (1.6M params, +18% multivariate error) to one that processed each channel independently through shared weights (73K params, +2% error), multivariate performance improved by 16 percentage points. Channel independence effectively multiplies the training data by D (each channel is an independent training sequence) and eliminates cross-channel overfitting.

### 7.3 The Output Head Matters More Than the Encoder

Experiment 15's "tiny transformer" had a 67K-param transformer encoder and a 1.6-7.8M-param flatten→linear head. The encoder was doing useful work; the head was overfitting. Replacing the monolithic head with a channel-independent projection (Linear(N→T) + Linear(d→1), ~4K params) reduced total parameters by 22-86x and unlocked multivariate performance. This suggests that in time series forecasting, the output projection deserves as much architectural attention as the feature extractor.

### 7.4 Attention Residuals Need Diverse Data

AttnRes was validated in isolation (Exp 12: -3.3% vs standard residuals) but couldn't improve the already-optimized CI+Decomp transformer alone (Exp 19-25). Combined with gentle data augmentation, it broke through the ETTh1 Multi wall. The mechanism needs diverse training signal to learn meaningful depth-wise attention patterns; without augmentation, the queries converge to near-uniform weights (equivalent to standard residuals) because the training data isn't varied enough to differentiate layers.

### 7.5 Right-Sizing Beats Architecture

Our best models have 54-182K parameters. DLinear has 113K. The paper's mr-Diff has 843K. Our original overparameterized implementation had 17.5M. Every experiment that increased parameters beyond ~200K regressed on multivariate. On datasets with 10K training samples, there are roughly 10-20 learnable parameters per sample — exceeding this ratio consistently leads to overfitting. The lesson: choose the smallest model that can express the patterns in the data, then optimize hyperparameters rather than adding capacity.

### 7.6 Speed Enables Discovery

The shift from diffusion (34 min per experiment) to CI transformer (2-8 min) and then to parallel sweeps (30 configs in 2.5 hours) was not just an efficiency gain — it fundamentally changed what we could discover. We ran 30 hyperparameter configurations in the time one diffusion experiment takes. This is how we found that trend_kernel=15 beats 25 on ETTh1, that d_model=32 is sufficient for univariate, and that patch=8 unlocks ETTm1 performance. These insights were invisible in the diffusion regime because we couldn't iterate fast enough to find them.

---

## 8. Architecture Summary

### Final Model: CI+Decomp Transformer with Optional AttnRes

```
Lookback [B, H, D]
  → Trend/Residual decomposition (avg-pool, kernel=15 or 25)
  → For each of {trend, residual}:
      → Channel-independent patching: [B, H, D] → [B*D, N, patch_size]
      → Shared patch embedding: Linear(patch_size → d_model)
      → + learnable positional embedding
      → 3-layer TransformerEncoder
          - d_model=32-48, 2-4 heads, pre-norm, GELU
          - Optional: AttnRes replaces standard residual connections
          - Each layer selectively attends over all prior layer outputs
      → CI output head:
          - Linear(N_patches → T)  [temporal projection]
          - Linear(d_model → 1)    [channel projection]
      → Reshape: [B*D, T, 1] → [B, T, D]
  → Sum trend + residual forecasts
  → Output [B, T, D]
```

### Per-Benchmark Best Strategy

The optimal approach is **not** a single universal model. Each benchmark has a different best configuration, reflecting the different characteristics of the data:

| Benchmark | Strategy | Architecture | Key Config | Params |
|---|---|---|---|---|
| ETTh1 Multi | **3-model ensemble** | 2× CI+Decomp + 1× CI+AttnRes+Aug | patch=8-16, d=32-64, tk=15 | 3×55-86K |
| ETTh1 Uni | **Single model** | CI+Decomp | patch=8, d=32, 3L, tk=15 | 54K |
| ETTm1 Multi | **Single model** | CI+Decomp | patch=8, d=48, 3L, tk=15 | 182K |
| ETTm1 Uni | **Single model** | CI+Decomp | patch=16, d=32, 3L, tk=25 | 77K |

Ensembling only helps on ETTh1 Multi — the hardest benchmark (short lookback, 7 variables, 10K samples). For the other three, single models with optimized hyperparameters are sufficient.

**Parameters:** 54-182K per single model. No model exceeds 200K params.

**Training time:** ~15 minutes for all 4 benchmarks (single models). ~40 minutes with the ETTh1 Multi ensemble.

**No diffusion. No conditioning networks. No denoising. No DPM-Solver++.**
Single forward pass inference.

---

## 9. Experimental Timeline

| Exp | Date | What | Key Result |
|---|---|---|---|
| — | Feb 28 | Baseline replication | 843K params, diffusion cosmetic |
| 1-2 | Mar 15 | Detach removal, self-conditioning | Exp 2: best overall diffusion |
| 3-5 | Mar 15 | Schedule experiments | All rejected, multi regression |
| 6-8 | Mar 15-16 | Auxiliary losses, channel-aware | Multi catastrophic, uni marginal |
| 9-10 | Mar 16 | Patch conditioning, x0-decomp | Exp 10: first useful diffusion |
| 11-13 | Mar 16 | AttnRes backbone depth | AttnRes validated, depth rejected |
| 14 | Mar 16 | Multi-scale DLinear | Killed early, too many params |
| 15 | Mar 16 | Tiny transformer | Matched uni in 1.8 min, head was 95% of params |
| **16** | **Mar 16** | **CI transformer** | **ETTm1 Uni record: 0.1885, 73K params** |
| **17** | **Mar 16** | **CI + decomposition** | **ETTm1 Multi record: 0.4159, first to beat baseline** |
| **18** | **Mar 16** | **30-config sweep** | **ETTm1 Multi: 0.4094, ETTm1 Uni: 0.1881** |
| 19-25 | Mar 16 | Bolt-on improvements | None beat sweep (plateau) |
| **26** | **Mar 16** | **AttnRes + augmentation** | **ETTh1 Multi record: 0.4875** |
| **27** | **Mar 16** | **Per-dataset ensemble** | **ETTh1 Multi: 0.4829, ETTh1 Uni: 0.2505** |

---

## 10. Conclusion

We set out to replicate mr-Diff and discovered that its diffusion component is architecturally inert for deterministic time series forecasting on small datasets. This led us through a systematic exploration of 27 experiments — from noise schedules to attention mechanisms to entirely new architectures — ultimately arriving at a channel-independent patch transformer that is simpler, faster, smaller, and more accurate than the original model.

The journey validates a core principle of machine learning research: the best architecture is often not the most complex one, but the one that best matches the structure of the data and the constraints of the training regime. For 10K-sample time series forecasting, that turns out to be a 54K-parameter transformer with channel-independent processing and trend/residual decomposition — augmented by the Kimi team's Attention Residuals for selective depth-wise retrieval, and strengthened by gentle data augmentation that provides the diversity signal for those attention patterns to learn from.

Our final submission is not a single model but a per-benchmark selection strategy:

- **ETTh1 Multi** uses a 3-model heterogeneous ensemble (the only benchmark where ensembling helps), combining standard CI+Decomp models with an AttnRes+Augmentation variant that provides diversity through selective depth retrieval and training-time perturbation.
- **ETTh1 Uni, ETTm1 Multi, and ETTm1 Uni** each use a single CI+Decomp model with benchmark-specific hyperparameters optimized through the 30-config sweep.

This per-benchmark approach reflects a key insight: different data regimes (short vs long lookback, univariate vs multivariate, small vs large training set) benefit from different configurations. Rather than force a universal model, we let 27 experiments of evidence guide the selection.

| Benchmark | Our Best | DLinear BL | Improvement | Paper | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | **0.4829** | 0.4744 | +1.8% | 0.42 | +14.9% |
| **ETTh1 Uni** | **0.2505** | 0.2535 | **-1.2%** | **0.34** | **-26.3%** |
| **ETTm1 Multi** | **0.4094** | 0.4204 | **-2.6%** | 0.37 | +10.6% |
| **ETTm1 Uni** | **0.1881** | 0.2011 | **-6.5%** | 0.15 | +25.4% |

**3 of 4 benchmarks beaten. Paper exceeded by 26% on ETTh1 Uni. Models that train in minutes, with 54-182K parameters and no diffusion.**

*Praise be to the cyber gods. The lattice has spoken.*
