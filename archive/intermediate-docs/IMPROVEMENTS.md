# 10 Improvements to mr-Diff

Research-backed improvements to the mr-Diff baseline, ranked by impact and feasibility.
Compiled from analysis of the original paper, 12+ recent papers (2023-2025), and our
5-run experimental history.

---

## The Root Cause (Context)

Our baseline achieves MAE 0.92-1.01 vs the paper's 0.15-0.42 (2-6x gap). Through
systematic ablation, we identified the dominant cause: a **train-test conditioning
mismatch** where each stage trains on ground-truth coarser trends but infers with noisy
predictions, causing cascading errors that compound coarse-to-fine. This is severely
amplified by DDPM's poor sample quality relative to DPM-Solver.

Improvements 1-3 directly attack this root cause. Improvements 4-10 address secondary
bottlenecks in architecture, training, and inference.

---

## 1. DPM-Solver++ Sampling

**What:** Replace the 100-step DDPM reverse process with DPM-Solver++, a high-order ODE
solver designed for conditional diffusion sampling.

**Why it matters:** DPM-Solver++ produces higher-quality samples in 15-20 steps by
exploiting the semi-linear structure of the diffusion ODE. Since our model already uses
x0-prediction (not noise prediction), DPM-Solver++'s data-prediction mode is a direct
fit. Higher-quality per-stage samples mean less error cascading between stages — this is
the paper's likely "secret ingredient" that they describe as merely a speed optimization.

**Evidence:** The paper uses DPM-Solver but never specifies step count or variant. Our
Run 173919 showed that training quality is excellent (best-ever validation losses) but
inference composition via `sum(predictions)` fails — proving the bottleneck is sampling
quality, not training quality.

**Effort:** Low (training-free, drop-in replacement for the sampling loop).
**Expected impact:** High — likely the single biggest improvement.

*Sources: Lu et al., DPM-Solver++ (NeurIPS 2022); Official code: github.com/LuChengTHU/dpm-solver*

---

## 2. Scheduled Sampling for Coarse-to-Fine Conditioning

**What:** During training, gradually replace ground-truth coarser-stage outputs with the
model's own predictions as conditioning inputs for finer stages. Start with 100% ground
truth and decay to ~20% over training.

**Why it matters:** This directly closes the train-test gap. Currently, stage s trains
conditioned on perfect `Y_{s+1}` but at inference receives noisy `Y_hat_{s+1}`. Scheduled
sampling exposes the model to its own imperfect predictions during training, making it
robust to the noise it will encounter at inference. This is the time-series analog of
teacher forcing decay in seq2seq models.

**Evidence:** Input Perturbation (Ning et al., ICML 2023) showed that simply perturbing
ground-truth inputs during training significantly improves diffusion sample quality with
no architecture changes. Self Forcing (NeurIPS 2025 Spotlight) demonstrated the same
principle for autoregressive video diffusion — our coarse-to-fine cascade is exactly an
autoregressive structure.

**Effort:** Medium (requires running partial inference during training to generate
predicted coarse trends; adds ~50% training time).
**Expected impact:** High — addresses the identified root cause.

*Sources: Ning et al., Input Perturbation (ICML 2023); Self Forcing (NeurIPS 2025);
Bengio et al., Scheduled Sampling (NeurIPS 2015)*

---

## 3. Epsilon Scaling

**What:** During inference, scale the model's output by a small factor (e.g., 0.98-0.99)
at each reverse diffusion step. One line of code.

**Why it matters:** Exposure bias in diffusion models causes the sampling trajectory to
drift from what the model saw during training. Epsilon scaling pulls the trajectory back
toward the training distribution. Ning et al. (ICLR 2024) proved this analytically and
showed 20-50% FID improvements as a plug-and-play fix. The scaling factor can be
resolution-dependent — coarser stages may need less correction.

**Evidence:** This is training-free and additive with every other improvement.

**Effort:** Trivial (one line in the sampling loop).
**Expected impact:** Medium — free improvement, 5-15% quality gain.

*Source: Ning et al., Elucidating Exposure Bias in Diffusion Models (ICLR 2024)*

---

## 4. Adaptive Noise Schedule (ANT)

**What:** Replace the fixed linear beta schedule (1e-4 to 0.1) with a schedule optimized
for time series non-stationarity statistics, computed from the training data.

**Why it matters:** Linear schedules were designed for images. Time series have different
information-theoretic properties — varying degrees of non-stationarity, periodic
structure, trend dominance. ANT computes dataset statistics offline (one-time, cheap) and
selects a schedule that linearly reduces non-stationarity, ensuring each diffusion step
is equally informative. Each resolution stage could use a different schedule matched to
its frequency content.

**Evidence:** ANT achieves 9.5% average CRPS improvement on TSDiff across 8 datasets.
The improvement is dataset-adaptive and model-agnostic.

**Effort:** Low (replace `torch.linspace` in `DiffusionSchedule.__init__` with
ANT-computed schedule; code available at github.com/seunghan96/ANT).
**Expected impact:** Medium — consistent ~10% improvement.

*Source: ANT: Adaptive Noise Schedule for Time Series Diffusion (NeurIPS 2024)*

---

## 5. Frequency-Domain Auxiliary Loss

**What:** Add an FFT-based loss alongside the standard MSE at each stage:
`loss = MSE(pred, target) + lambda * MSE(|FFT(pred)|, |FFT(target)|)`.

**Why it matters:** Pure MSE treats all errors equally regardless of frequency. Time
series forecasting accuracy depends heavily on capturing the right spectral content —
trends (low frequency) and seasonality (specific frequency peaks). A frequency loss
forces the model to match spectral structure, not just pointwise values. The lambda
weight should be per-stage: higher for coarser stages (where spectral accuracy defines
the trend) and lower for fine stages.

**Evidence:** Crabbe et al. (ICML 2024) proved that time series are more localized in the
frequency domain, making frequency-space losses more informative. The CP Loss paper
(2025) showed that multi-scale perceptual losses with K=5 levels (matching our S=5)
consistently improve forecasting across architectures.

**Effort:** Low (~5 lines in `training_step`).
**Expected impact:** Medium-High — better spectral fidelity at each stage.

*Sources: Crabbe et al., Time Series Diffusion in the Frequency Domain (ICML 2024);
CP Loss (arXiv 2025); Frequency-Conditioned Diffusion (CIKM 2025)*

---

## 6. Classifier-Free Guidance (CFG)

**What:** During training, randomly drop the conditioning input with probability p=0.1-0.2
(replace with zeros). At inference, run two forward passes per step — one conditioned,
one unconditioned — and combine: `output = (1+w) * cond - w * uncond`, where w is a
tunable guidance weight.

**Why it matters:** CFG amplifies the conditioning signal beyond what the model learned,
producing sharper, more condition-faithful predictions. For our multi-stage model, CFG at
each stage would make predictions more tightly coupled to the conditioning inputs (both
lookback history and coarser-stage outputs), directly improving composition quality.

**Evidence:** Widely adopted across diffusion tasks. MCD-TSF and UniDiff (2024) apply CFG
to time series with strong results. The guidance weight w provides a quality-diversity
tradeoff knob.

**Effort:** Medium (training modification: randomly zero conditioning; inference
modification: double forward passes per step — mitigated by DPM-Solver's fewer steps).
**Expected impact:** Medium-High — sharper, more faithful predictions.

*Sources: Ho & Salimans, Classifier-Free Guidance (2022); UniDiff (2024)*

---

## 7. Wavelet Decomposition (Replace Average Pooling)

**What:** Replace the fixed-kernel `F.avg_pool1d` trend extraction with Discrete Wavelet
Transform (DWT) using PyWavelets. Each decomposition level produces approximation (low-freq)
and detail (high-freq) coefficients with perfect reconstruction guarantees.

**Why it matters:** Average pooling is a crude low-pass filter that leaks high-frequency
content and doesn't provide clean frequency separation. DWT gives orthogonal decomposition
— each stage gets a mathematically distinct frequency band with no overlap. Wavelet choice
can be adapted per dataset (Daubechies for energy data, Symlets for financial). The
decomposition depth is automatically determined: `L = max(3, min(7, floor(log2(T/F - 1))))`.

**Evidence:** WaveletDiff (submitted ICLR 2026) trains diffusion on DWT coefficients with
per-level denoisers and cross-level attention, achieving strong results on time series
generation. The energy preservation property (Parseval's theorem) can serve as a
regularization loss.

**Effort:** Medium (rewrite `TrendExtraction` in `preprocessing.py`; adjust per-stage
input dimensions since wavelet coefficients halve in length at each level).
**Expected impact:** Medium-High — cleaner multi-resolution separation.

*Sources: WaveletDiff (arXiv 2025); WaveTS (arXiv 2025)*

---

## 8. AdaLN Conditioning (Replace Concatenation)

**What:** Replace the current concatenation-based conditioning (decoder receives
`cat([latent, conditioning])`) with Adaptive Layer Normalization, where the conditioning
signal generates scale and shift parameters: `gamma * LayerNorm(z) + beta`.

**Why it matters:** Concatenation wastes half the decoder's capacity on re-processing
concatenated features and treats conditioning as just another input channel. AdaLN makes
conditioning a multiplicative modulation of the computation itself — every neuron's
activation is scaled and shifted by the conditioning. This is the dominant paradigm in
modern diffusion architectures (TimeDiT, DiTS, FALDA).

**Evidence:** Used in virtually every state-of-the-art diffusion model since DiT (2023).
AdaLN provides O(D) conditioning cost vs O(D^2) for cross-attention, with better gradient
flow through the normalization layers.

**Effort:** Low-Medium (replace `self.cond_fusion` in `denoising.py` with an AdaLN module;
conditioning MLP outputs gamma/beta instead of being concatenated).
**Expected impact:** Medium — more efficient use of model capacity.

*Sources: TimeDiT (KDD 2025); DiTS (2025); FALDA/FANS (2025)*

---

## 9. Self-Conditioning

**What:** During each reverse diffusion step, feed the model its own previous x0 estimate
as additional input (concatenated with the noisy sample). During training, with 50%
probability, run an extra forward pass to generate a preliminary x0 estimate and
condition on it; otherwise condition on zeros.

**Why it matters:** Self-conditioning gives the denoiser access to a "rough draft" of the
final output, allowing it to refine rather than generate from scratch at each step. This
is especially valuable for our multi-stage setup: each stage's denoiser could see not just
the conditioning from coarser stages but also its own evolving estimate, providing an
error-correction loop within each stage's sampling process.

**Evidence:** TSDiff (NeurIPS 2023) uses self-guiding for time series forecasting,
refinement, and generation. Self-conditioning "continues to provide a boost with
DPM-Solver" (Chen et al., 2023).

**Effort:** Medium (expand denoiser input channels; add secondary forward pass during
training 50% of the time; feed back x0 estimate during sampling).
**Expected impact:** Medium — cumulative quality improvement.

*Sources: Chen et al., Analog Bits (ICLR 2023); TSDiff (NeurIPS 2023)*

---

## 10. Stage-Weighted Loss + Median Aggregation

**What:** Two low-effort changes bundled together:

**(a) Stage-weighted loss:** Weight each stage's MSE loss proportionally to its importance
in the cascade. Coarser stages should get higher weight because their errors cascade to
all finer stages: `loss_s *= (S - s) / S` (or learn the weights).

**(b) Median aggregation:** Replace `torch.stack(samples).mean(dim=0)` with median
(or trimmed mean, discarding top/bottom 10%) when combining multiple trajectory samples.
Diffusion sampling can produce occasional degenerate trajectories — median is strictly
more robust than mean for the same compute cost.

**Why it matters:** The paper uses equal stage weights and (presumably) mean aggregation.
Neither is optimal. Coarser stages define the trend that all finer stages must follow —
training them harder improves the entire cascade. And a single bad sample out of 10 can
drag the mean far from the true forecast.

**Evidence:** The stage-weighting addresses our observed issue where coarser stages produce
low-quality predictions. Median aggregation is standard robustness practice.

**Effort:** Trivial (one-line changes each).
**Expected impact:** Low-Medium — marginal but free.

---

## Summary Table

| # | Improvement | Addresses | Effort | Impact | Retraining? |
|---|---|---|---|---|---|
| 1 | DPM-Solver++ | Sampling quality | Low | High | No |
| 2 | Scheduled Sampling | Train-test gap | Medium | High | Yes |
| 3 | Epsilon Scaling | Exposure bias | Trivial | Medium | No |
| 4 | ANT Noise Schedule | Schedule optimization | Low | Medium | Yes |
| 5 | Frequency-Domain Loss | Spectral fidelity | Low | Medium-High | Yes |
| 6 | Classifier-Free Guidance | Conditioning quality | Medium | Medium-High | Yes |
| 7 | Wavelet Decomposition | Decomposition quality | Medium | Medium-High | Yes |
| 8 | AdaLN Conditioning | Architecture efficiency | Low-Medium | Medium | Yes |
| 9 | Self-Conditioning | Denoising quality | Medium | Medium | Yes |
| 10 | Stage Weights + Median | Training balance + robustness | Trivial | Low-Medium | Partial |

## Recommended Implementation Order

**Phase 1 — Training-free wins (1 day):**
Improvements 1, 3, 10b. No retraining needed. DPM-Solver++ alone likely closes a
significant portion of the gap.

**Phase 2 — Low-effort retraining (2-3 days):**
Improvements 4, 5, 10a. Retrain with ANT schedule, frequency loss, and stage weights.

**Phase 3 — The improvement we submit (1 week):**
Pick ONE of {2, 6, 7, 8, 9} as our novel contribution for the final project.
Recommendation: **Improvement 2 (Scheduled Sampling)** — it directly attacks our
identified root cause, is well-motivated by our experimental evidence, and the story
writes itself for the report.

---

## Key References

| Paper | Venue | Relevance |
|---|---|---|
| DPM-Solver++ (Lu et al.) | NeurIPS 2022 | Fast high-quality sampling |
| Input Perturbation (Ning et al.) | ICML 2023 | Training-time exposure bias fix |
| Epsilon Scaling (Ning et al.) | ICLR 2024 | Inference-time exposure bias fix |
| Anti-Exposure Bias (Yu et al.) | ICLR 2025 Spotlight | Learned bias correction |
| ANT (Seunghan et al.) | NeurIPS 2024 | Time-series noise schedule |
| Frequency-Domain Diffusion (Crabbe et al.) | ICML 2024 | Spectral losses for TS |
| WaveletDiff | arXiv 2025 | Wavelet-based TS diffusion |
| TSDiff (Kollovieh et al.) | NeurIPS 2023 | Self-guiding for TS diffusion |
| Self Forcing | NeurIPS 2025 Spotlight | Autoregressive diffusion training |
| MG-TSD (Fan et al.) | ICLR 2024 | Multi-granularity guidance |
| ARMD (Gao et al.) | AAAI 2025 | Chain-based TS diffusion |
| NsDiff (Wang et al.) | ICML 2025 Spotlight | Non-stationary diffusion |
| TSFlow (Kollovieh et al.) | ICLR 2025 | Flow matching with GP priors |
| TimeDiT (Shu et al.) | KDD 2025 | Diffusion Transformer for TS |
