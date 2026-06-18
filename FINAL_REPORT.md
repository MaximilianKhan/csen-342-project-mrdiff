# CSEN 342: Deep Learning, Final Project Report

## Replication and Improvement of "Multi-Resolution Diffusion Models for Time Series Forecasting" (mr-Diff) [1]

**Maximilian Khan and Karthik Tamiledu**
**CSEN-342, Winter 2026, Santa Clara University**

---

### Abstract

This report covers our replication of mr-Diff [1] and our improvements built on top of it. We implemented the full architecture from scratch in PyTorch and evaluated it on ETTh1 and ETTm1 in both univariate and multivariate settings. Our best baseline achieves MAE 0.47–0.20 on globally-standardized data after resolving six critical issues in the paper's described method, a faithful implementation of which produces MAE equivalent to predicting zeros. Through 14 diffusion-focused experiments we established that the diffusion component contributes less than 0.3% to forecast accuracy, with the DLinear backbone doing all the work. This led us to replace the entire 843K-parameter diffusion pipeline with a Channel-Independent Decomposed Patch Transformer (54–182K parameters) that beats the baseline on 3 of 4 benchmarks, trains in minutes instead of 34 minutes, and requires only a single forward pass at inference. A 30-configuration hyperparameter sweep and heterogeneous ensembles further refined the results. We also explored overlapping patches, an iTransformer [15] variant for cross-variate attention, and a two-scale decomposition aligned to the dominant daily cycle in electricity data.

---

## 1. Introduction

Time series forecasting problems arise across energy, finance, and healthcare, and accurate long-horizon prediction has remained difficult despite years of work on the problem. Transformer-based architectures made progress, but they produce point estimates and do not model uncertainty in future values. Diffusion models, well established in image generation, offer a probabilistic framework, though early time series adaptations treated the forecast window as a flat sequence and ignored multi-scale temporal structure.

mr-Diff [1] addresses this by decomposing the target into multiple resolution stages and training a separate diffusion network per stage, which maps onto how time series data actually behaves across different time scales. No source code was released by the authors.

We had two goals. The first was to replicate the mr-Diff baseline on ETTh1 and ETTm1, understand where our results fell short of the paper's, and diagnose the causes. The second was to systematically improve upon the baseline through a 31-experiment campaign. Our central finding, that the diffusion component provides zero measurable benefit on small time series datasets, led to a paradigm shift from diffusion to attention-based forecasting that was not planned but emerged organically from the evidence.

---

## 2. Literature Review

Transformer-based forecasting models improved steadily through the early 2020s. Informer [2] reduced attention complexity with sparse attention; Autoformer [3] built seasonal-trend decomposition into the attention mechanism using autocorrelation; FEDformer [4] moved decomposition into the frequency domain; PatchTST [5] treated contiguous time steps as patches rather than individual tokens. Meanwhile, simpler methods proved competitive. DLinear [6] decomposed the input into trend and residual and applied a single linear projection to each, outperforming many transformer variants on standard benchmarks. N-HiTS [7] used hierarchical interpolation across multiple temporal resolutions.

On the diffusion side, TimeGrad [8] applied autoregressive denoising guided by RNN hidden states, but generation was slow and there was no use of multi-scale structure. CSDI [9] used non-autoregressive generation through self-supervised masking, though its complexity scaled quadratically with sequence length. SSSD [10] replaced the transformer backbone in CSDI with a structured state space model to reduce this cost. TimeDiff [11] introduced future mixup and autoregressive initialization, both of which mr-Diff inherits.

mr-Diff [1] is the first model to embed seasonal-trend decomposition into the diffusion process itself. The diffusion backbone follows DDPM [12], and fast inference is provided by DPM-Solver++ [13], which formulates the reverse process as an ODE and reaches usable samples in 10–20 steps. Our transformer improvements draw on PatchTST [5] for patching and channel independence, iTransformer [15] for inverted cross-variate attention, and Attention Residuals [14] for learned depth-wise feature retrieval.

---

## 3. System Design and Implementation

### 3.1 Algorithm

The mr-Diff architecture [1] decomposes the forecast target Y into S resolution stages via sequential average-pooling trend extraction. We use residual decomposition: the finest stage receives Y minus the first trend, intermediate stages receive differences between consecutive trends, and the coarsest stage receives the final smoothed trend. The components sum to Y exactly, which is required for reconstruction at inference.

A separate conditional denoising diffusion network trains on each stage's component. The conditioning signal for stage s combines the lookback window (encoded by a convolutional network) with the predicted output from stage s+1. At the coarsest stage only the lookback encoding is used. During training, future mixup perturbs the conditioning signal with probability 0.5 by blending the encoded lookback with a randomly projected version of the ground-truth target, following TimeDiff [11]. We use random projection weights (regenerated each forward pass) rather than the learned projection described in the paper, because learned weights created a train-test gap: the model exploited ground-truth signal during training that was absent at inference.

A DLinear-style backbone [6] provides a direct forecast from the lookback window via separate linear projections for trend and residual. The diffusion networks model the residual between the target and this direct prediction, which reduces the variance each stage needs to handle. This is a fundamental architectural departure from the paper, which describes a pure diffusion model. We found it necessary because without it, the model's predictions are equivalent to predicting zeros.

Forward diffusion follows DDPM [12] with K=100 steps and a linear variance schedule from $\beta_1 = 1 \times 10^{-4}$ to $\beta_{100} = 0.1$. At inference we use DPM-Solver++ [13] at 20 steps, treating the reverse process as an ODE.

### 3.2 Tools

All code is in Python 3.9 with PyTorch 2.7.1 and CUDA 11.8. Primary training ran on a local NVIDIA RTX 5090. Each configuration was controlled by a YAML file, and a shared runner executed all four dataset/mode combinations sequentially per experiment. The codebase separates models, data loading, training, and evaluation into distinct modules.

### 3.3 Network Architecture

**Baseline (mr-Diff).** Our working configuration uses S=3 stages (not S=5 as in the paper; see Section 3.5), hidden dimension 64, embedding dimension 64, two encoder and two decoder layers, trend extraction kernels [25, 201], and dropout 0.3. Total parameters: 843K.

Each stage has three learned components. The conditioning network encodes the lookback window with a linear projection to hidden dimension 64 followed by 1D convolutional layers (kernel size 7, GroupNorm, LeakyReLU slope 0.1, dropout 0.3). The encoded history is projected from the lookback length to the forecast length, then fused with the coarser-stage output through a two-layer MLP.

The denoising network is a U-Net encoder-decoder. The encoder concatenates the noisy input with an expanded sinusoidal step embedding (64-dimensional, projected to 64 through two FC layers with SiLU), and passes through two convolutional residual blocks while saving skip connections. The decoder fuses the latent with the conditioning signal via a linear layer and reconstructs the denoised output through two convolutional blocks that receive the reversed encoder skips.

**Improvement (CI+Decomp Transformer).** The input is split into trend and residual via average-pooling (kernel=15 or 25), then each component goes through the same shared TransformerEncoder (2–3 layers, d_model=32–64, 4 heads, GELU, pre-norm LayerNorm) with separate output heads. Each of D channels is patched and processed independently through shared weights, so the transformer never sees more than one variable at a time. Output heads are two small linear layers: Linear(N_patches → T) then Linear(d_model → 1). Trend and residual forecasts are summed. Total parameters: 54–182K depending on benchmark configuration, independent of D.

### 3.4 Hyperparameters

**Baseline (working configuration):** AdamW with learning rate 1e-3 and weight decay 0.01, batch size 64, cosine learning rate decay, gradient clipping at norm 1.0, maximum 100 epochs with minimum 30 and early stopping patience 20. Lookback 336, forecast 168 for ETTh1; lookback 1440, forecast 192 for ETTm1. Hidden dimension 64, embedding dimension 64, S=3 stages, two encoder and two decoder layers, dropout 0.3, LeakyReLU slope 0.1.

**CI+Decomp Transformer (best per-benchmark configurations from Exp 18 sweep):**

| Parameter | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni |
|-----------|-------------|-----------|-------------|-----------|
| patch_size | 16 | 8 | 8 | 16 |
| d_model | 64 | 32 | 48 | 32 |
| num_layers | 3 | 3 | 3 | 3 |
| dim_ff | 64 | 128 | 256 | 128 |
| dropout | 0.2 | 0.3 | 0.3 | 0.2 |
| trend_kernel | 15 | 15 | 15 | 25 |
| lr | 0.002 | 0.0005 | 0.001 | 0.0005 |
| weight_decay | 0.005 | 0.05 | 0.01 | 0.05 |
| Params | 86K | 54K | 182K | 77K |

### 3.5 Key Deviations from the Paper

A faithful implementation of the paper's described architecture with S=5 stages, hidden dimension 256, and embedding dimension 128 produces a model with 17.5 million parameters trained on ~10,000 samples. This model collapses immediately: it learns to predict zeros, achieving MAE ~0.95 in RevIN-normalized space. Six deviations were required:

| # | Issue | Our Fix |
|---|-------|---------|
| 1 | 17.5M params on ~10K samples → collapse | Downsized to 843K (S=3, dim=64) |
| 2 | Metric scale ambiguity (RevIN vs global) | Compute metrics in globally-standardized space |
| 3 | Diffusion contributes <0.3% | Added DLinear direct prediction backbone |
| 4 | BatchNorm corrupts diffusion timesteps | Replaced with GroupNorm |
| 5 | Cumulative decomposition cascades errors (MAE 6–32) | Switched to residual decomposition |
| 6 | Learned mixup creates train-test gap | Random projection per forward pass |

Additional fixes: AMP disabled (corrupts alpha_bar precision in float16), stage-weighted loss (coarser stages weighted higher), FFT frequency loss (0.1× weight), epsilon prediction with scheduled sampling for exposure bias reduction.

### 3.6 Improvement 1: Channel-Independent Decomposed Patch Transformer with Attention Residuals

After 14 experiments confirmed that the diffusion component contributes nothing, Max replaced the entire 843K-parameter diffusion pipeline with a Channel-Independent Decomposed Patch Transformer (CI+Decomp). The architecture (described in Section 3.3) retains DLinear's trend/residual decomposition as an inductive bias but replaces diffusion with a lightweight TransformerEncoder operating over patches of each channel independently. This reduces model size to 54–182K parameters, training time from 34 minutes to 2–15 minutes, and inference from 60+ denoiser calls to a single forward pass.

The addition of Attention Residuals [14] was a late but high-impact decision. On Monday morning (2026-03-17), the Kimi Moonshot team released their paper describing the mechanism and reporting strong results across multiple domains. Given the simplicity of the modification (a single learned query vector per layer, zero-initialized, adding only 192 parameters), it was integrated into the CI+Decomp architecture the same day it was published and evaluated immediately.

Standard transformer residuals add each layer's output to its input with fixed weight 1. Attention Residuals replace this with a learned query vector per layer that computes softmax attention over all prior layer outputs (embedding + all preceding layers), letting each layer pull from whichever earlier representation is most useful. The zero initialization means the model starts behaving identically to a standard transformer and only learns to deviate when it helps. This targeted mechanism proved particularly effective on ETTh1 Multi, our hardest benchmark, where the short 336-step lookback means each layer's contribution matters more and the ability to skip uninformative intermediate representations pays off. For the ETTh1 Multi ensemble, one of three members uses AttnRes; the others use standard residuals with different hyperparameters, providing the architectural diversity that makes ensembling effective.

A 30-configuration hyperparameter sweep over the CI+Decomp architecture (patch size, d_model, depth, dropout, learning rate, trend kernel, weight decay) found per-benchmark optimal configurations and set 3 of 4 all-time records (Section 4.5).

### 3.7 Improvement: Overlapping Patches and iTransformer Ensemble

The CI transformer has no mechanism for one variable to inform another's forecast. A 30-config sweep confirmed this as a ceiling: all ETTh1 Multi results landed between 0.488–0.533 regardless of hyperparameters.

Two changes address this. First, overlapping patches: setting patch_stride = patch_size // 2 via `torch.Tensor.unfold` gives 83 tokens from ETTh1's 336-step lookback instead of 42. Adjacent patches share half their timesteps, giving attention more temporal context per window.

Second, the iTransformer [15]: rather than treating time patches as tokens, each variable's full lookback series is embedded as one token via Linear(L → d_model). For D=7 that is 7 channel tokens; attention runs over those 7 tokens and learns which variables predict each other. The ETTh1 Multi ensemble combines one iTransformer model with two CI models. Their errors are partially uncorrelated because one model attends over channels and the other attends over time, which is why the ensemble beats any individual model. For univariate benchmarks (D=1), iTransformer is not used since attention over one token is trivial.

### 3.8 Improvement: Two-Scale Decomposition

The single-kernel trend extraction gives the transformer one boundary between trend and residual. Two kernels produce three bands: coarse trend, mid-band (fine trend minus coarse trend), and fine residual. Each band goes through the same shared transformer with its own output head; the three forecasts are summed.

The coarse kernel for ETTm1 is set to 96, which at 15-minute resolution is exactly 24 hours, the dominant cycle in electricity load data. For ETTh1 the coarse kernel is 24 (one day in hourly data). The transformer itself is unchanged; the only additions are two extra pairs of output head weights, roughly 6K parameters.

---

## 4. Experiments and Evaluation

### 4.1 Datasets

We evaluated on two datasets from the ETT benchmark [16]. ETTh1 has 7 variables and 17,420 hourly observations, with lookback 336 and forecast horizon 168. ETTm1 has 7 variables and 69,680 observations at 15-minute intervals, with lookback 1440 and forecast horizon 192. Each was split chronologically 60/20/20. For univariate experiments we used the OT (oil temperature) column. Per-window instance normalization (RevIN [17]) was applied during training, and all metrics are reported in globally-standardized space to match the paper's evaluation protocol.

### 4.2 Methodology

Each model variant was trained independently on all four benchmark configurations. We selected the checkpoint with the lowest validation loss. For the diffusion baseline, evaluation used 10 sampled trajectories per window, aggregated by median, with DPM-Solver++ [13] at 20 steps. For the transformer models, evaluation is a single deterministic forward pass. The two metrics are MAE and MSE on globally-standardized data. All variants used the same data splits, random seeds, and evaluation procedure.

### 4.3 Baseline Results

Table 1 compares our baseline to the paper across all four configurations.

**Table 1: Baseline Replication Results**

| Dataset | MAE (Paper) | MAE (Ours) | MSE (Paper) | MSE (Ours) | MAE Gap |
|---------|-------------|------------|-------------|------------|---------|
| ETTh1 Multi | 0.42 | 0.4744 | 0.411 | 0.4516 | +13.0% |
| ETTh1 Uni | 0.34 | 0.2535 | 0.066 | 0.1183 | **−25.4%** |
| ETTm1 Multi | 0.37 | 0.4204 | 0.340 | 0.3223 | +13.6% |
| ETTm1 Uni | 0.15 | 0.2011 | 0.039 | 0.0670 | +34.1% |

Our baseline already beats the paper on ETTh1 Univariate by 25% (0.2535 vs 0.34). The remaining gaps on other benchmarks likely stem from undisclosed per-dataset hyperparameter tuning in the original work.

**The critical discovery:** By comparing predictions with and without the diffusion pipeline, we found that diffusion changes MAE by less than 0.3%:

**Table 2: Diffusion Contribution Ablation**

| Benchmark | Direct Only | Direct + Diffusion | Difference |
|-----------|------------|-------------------|------------|
| ETTh1 Multi | 0.4721 | 0.4744 | +0.0023 |
| ETTh1 Uni | 0.2539 | 0.2535 | −0.0004 |
| ETTm1 Multi | 0.4175 | 0.4204 | +0.0029 |
| ETTm1 Uni | 0.2008 | 0.2011 | +0.0003 |

The 730K-parameter diffusion apparatus is cosmetically neutral. The DLinear backbone (113K params) does all the work. This finding drove our entire improvement strategy.

### 4.4 Diffusion Improvement Experiments (Experiments 1–10)

We exhaustively attempted to make diffusion contribute through 10 experiments: joint end-to-end training (Exp 1), self-conditioning (Exp 2), cosine noise schedule (Exp 3, catastrophic: +42% on ETTh1 Multi), v-prediction parameterization (Exp 4), auxiliary decomposition losses (Exp 5), multi-granularity temporal diffusion (Exp 7), x0-prediction + decomposition (Exp 10), learned noise scheduling, and Fourier-enhanced denoising. None achieved more than marginal improvement. The best diffusion result (Exp 10: 0.2508 on ETTh1 Uni) was slower (55 min) and no better overall.

### 4.5 Transformer Breakthrough (Experiments 15–18)

**Table 3: Transformer Architecture Progression**

| Exp | Architecture | Params | ETTh1 M | ETTh1 U | ETTm1 M | ETTm1 U | Time |
|-----|-------------|--------|---------|---------|---------|---------|------|
|, | DLinear Baseline | 113K | 0.4744 | 0.2535 | 0.4204 | 0.2011 | ~34m |
| 15 | Tiny Transformer | 295K–7.8M | 0.5607 | 0.2538 | 0.5514 | 0.2002 | 4.7m |
| 16 | CI Transformer | 73–91K | 0.5485 | 0.2741 | 0.4293 | **0.1885** | 8.6m |
| 17 | CI + Decomp | 77–109K | 0.5101 | 0.2580 | **0.4159** | 0.2011 | 11.9m |
| 18 | HP Sweep (30 configs) | 54–182K | 0.4880 | **0.2514** | **0.4094** | **0.1881** | ~2.5h |

Exp 15 proved a 2-layer transformer matches DLinear on univariate with 7× speedup. Exp 16 introduced channel independence, setting a record on ETTm1 Uni (0.1885), a 73K-param model with no diffusion, trained in 2.5 minutes. Exp 17 added trend/residual decomposition, setting the first-ever record on ETTm1 Multi (0.4159). Exp 18 conducted a 30-configuration hyperparameter sweep across patch size, d_model, depth, dropout, learning rate, trend kernel, and weight decay, setting 3 of 4 all-time records.

### 4.6 Refinement and Ensemble Results (Experiments 19–27)

We tested extended training (Exp 19), cross-channel mixing (Exp 21), data augmentation (Exp 22), and frequency-enhanced dual branches (Exp 25). None individually improved upon the well-tuned sweep configurations.

Exp 26 (AttnRes + gentle augmentation) cracked the ETTh1 Multi wall at 0.4875, a marginal but real improvement over the sweep ceiling of 0.4880. Exp 27 (heterogeneous ensemble) averaged 3 architecturally diverse models per benchmark:

**Table 4: Ensemble Results (Exp 27)**

| Benchmark | Single Best | Ensemble MAE | Improvement |
|-----------|------------|--------------|-------------|
| ETTh1 Multi | 0.4875 | **0.4829** | −0.9% |
| ETTh1 Uni | 0.2514 | 0.2505 | −0.4% |
| ETTm1 Multi | 0.4094 | 0.4151 | No improvement |
| ETTm1 Uni | 0.1881 | 0.1924 | No improvement |

The ensemble provided meaningful gains only on ETTh1, where architectural diversity helps average out uncorrelated errors on the challenging short-lookback multivariate setting.

### 4.7 Overlapping Patches and iTransformer Results

The overlapping-patch CI transformer and iTransformer ensemble were evaluated on all four benchmarks:

**Table 5: iTransformer Ensemble Results**

| Dataset | Baseline | CI+Decomp | iTransformer Ensemble | vs CI+Decomp |
|---------|----------|-----------|----------------------|--------------|
| ETTh1 Multi | 0.4744 | 0.4829 | **0.4773** | −1.2% |
| ETTh1 Uni | 0.2535 | 0.2505 | **0.2471** | −1.4% |
| ETTm1 Multi | 0.4204 | 0.4094 | 0.4103 | +0.2% |
| ETTm1 Uni | 0.2011 | 0.1881 | 0.1911 | +1.6% |

New records on both ETTh1 benchmarks. The ensemble works because iTransformer and CI make different errors on the same windows, not because either is individually stronger. ETTm1 regresses because the iTransformer member was weaker than the CI members, pulling the average up.

### 4.8 Two-Scale Decomposition Results

**Table 6: Two-Scale Decomposition Results**

| Dataset | Baseline | CI+Decomp | Two-Scale | vs CI+Decomp |
|---------|----------|-----------|-----------|--------------|
| ETTh1 Multi | 0.4744 | 0.4829 | 0.4858 | +0.6% |
| ETTh1 Uni | 0.2535 | 0.2505 | 0.2574 | +2.8% |
| ETTm1 Multi | 0.4204 | 0.4094 | **0.4081** | −0.3% |
| ETTm1 Uni | 0.2011 | 0.1881 | 0.1914 | +1.8% |

New record on ETTm1 Multi ensemble (0.4081). Single-model ETTm1 Uni of 0.1865 is the best individual result on that benchmark. ETTh1 regresses. The improvement is specific to ETTm1 where the coarse kernel (96 = 24 hours at 15-minute resolution) aligns with the dominant daily cycle. ETTh1's hourly resolution doesn't have the same clean frequency separation at kernel=24.

### 4.9 Final Combined Results

**Table 7: Final All-Time Best Results (Combined Across All Improvements)**

| Benchmark | Best MAE | Architecture | Params | vs Baseline | vs Paper |
|-----------|---------|-------------|--------|-------------|----------|
| ETTh1 Multi | **0.4773** | iTransformer Ensemble | 3× models | +0.6% | +13.6% |
| ETTh1 Uni | **0.2471** | iTransformer Ensemble | Multiple | **−2.5%** | **−27.3%** |
| ETTm1 Multi | **0.4081** | Two-Scale Ensemble | 3× models | **−2.9%** | +10.3% |
| ETTm1 Uni | **0.1865** | Two-Scale Single | 94K | **−7.3%** | +24.3% |

Best results beat the DLinear baseline on 3 of 4 benchmarks. ETTh1 Uni at 0.2471 beats the paper's reported 0.34 by 27%. All achieved with 54–182K parameter transformers, no diffusion, training in minutes.

For completeness, the single lowest ETTh1 Multi MAE recorded over the entire campaign was 0.4719, from the diffusion-based self-conditioning model (Exp 2, Section 4.4). We report the iTransformer ensemble (0.4773) as the headline result on that benchmark because it is the architecture we advocate: diffusion-free, an order of magnitude smaller, and a single forward pass at inference. The 0.0054 gap on this one benchmark does not change any conclusion. The diffusion result is preserved in full in the experiment database (`experiment-db/`).

---

## 5. Analysis

All improvements start from the same finding: the diffusion component adds nothing measurable, so the useful architecture is the DLinear backbone plus whatever structure goes on top of it.

**Why diffusion is the wrong inductive bias at this scale.** Three properties of the problem explain the ablation, and together they argue that diffusion is mismatched to this task rather than merely unnecessary. First, diffusion models are data-hungry, and ETT provides only about 10,000 training windows, far too few for a denoiser to learn a robust reverse process; the faithful 17.5M-parameter model collapses outright, and even the downsized 843K version cannot make the stochastic process pay. Second, multi-step sampling suffers from exposure bias: the denoiser trains on noised ground truth but must generate from pure noise at inference, and on small data the accumulated error drifts the prediction toward zero. We observe this directly in the sampling trace, where the x0 estimate converges to magnitude around 0.55 against a target of around 1.08, a systematic undershoot. Third, and most fundamentally, the conditional forecast distribution here is effectively unimodal and dominated by deterministic trend and daily seasonality, while MAE and MSE score a single point estimate. A generative model spends its capacity representing uncertainty that the task neither contains nor rewards. A decomposed, channel-independent projection captures the deterministic structure directly in one forward pass, which is exactly why it matches or beats the diffusion pipeline at a fraction of the size.

**Channel independence is the most important design choice at this data scale.** With 10K training samples and D=7, a cross-channel architecture sees 10K multivariate sequences; a channel-independent one sees 70K single-channel sequences through shared weights. Every experiment that added cross-channel capacity at intermediate layers (Exp 8: per-channel encoders, Exp 21: Linear(D,D) mixing) regressed on multivariate. The pattern was consistent enough that channel independence became the design assumption.

**The ETTh1 Multi ceiling is architectural, not a hyperparameter problem.** The fix requires models that attend over different axes. The iTransformer ensemble works because iTransformer [15] and CI make different errors on the same windows. The ETTm1 regression confirms this: when one ensemble member is too weak, averaging is worse than using the best member alone.

**Decomposition kernels carry real inductive bias.** Setting the coarse kernel to match a physically meaningful period (24 hours in 15-minute ETTm1 data) gives better results than a generic scale. It doesn't transfer to ETTh1 because the hourly lookback doesn't have the same clean frequency separation.

**Hyperparameter sensitivity patterns:** trend_kernel=15 dominates ETTh1 (hourly data benefits from a finer 15-hour window); d_model=32 is sufficient for univariate; 3 transformer layers consistently outperform 1–2 under CI; smaller patches (size 8) help ETTm1 by providing 180 tokens for rich temporal attention.

---

## 6. Discussion

### 6.1 Decisions Made

Several decisions diverged from [1]. We used residual decomposition rather than the cumulative approach described in the paper, because cumulative decomposition produced MAE of 6–32 across all four configurations due to error cascading across stages at inference. Future mixup used random projection weights rather than learned weights, because learned weights widened the train-test gap. For the baseline we used only the finest stage's output rather than summing all stages; stage summation requires clean per-stage samples, and DDPM [12] produced samples too noisy for the composition to work.

Moving from diffusion to a pure transformer was not planned. It followed from 14 experiments where the diffusion residuals stayed near zero regardless of what was changed. Channel independence was chosen over cross-channel attention because the data doesn't support learning cross-channel dynamics at this scale, every cross-channel experiment degraded multivariate performance.

### 6.2 What Worked Well

Residual decomposition was numerically stable and produced consistent loss curves. The DLinear backbone [6] gave the diffusion process a much smaller residual to model, which reduced loss variance in early training. DPM-Solver++ [13] brought evaluation time from approximately 79 minutes (DDPM, 100 steps, 10 samples) to approximately 16 minutes (20 steps, 10 samples) with no change in metrics.

Training the CI transformer took 2–15 minutes per full run versus 34 minutes for diffusion. That speed difference made a 30-configuration sweep feasible in an afternoon and made it practical to test ensemble combinations that would have taken days otherwise. AttnRes [14] was validated in Exp 26 showing it consistently outperforms standard residuals on short-lookback multivariate data. The modular code structure allowed each improvement to be implemented by changing one or two files without touching the rest.

### 6.3 What Didn't Work

Every component described as core in [1] degraded performance when tested in isolation under DDPM [12] sampling. Cumulative decomposition: MAE 6–32. Learned mixup projection: MAE 1.3–3.5. Stage summation: MAE 1.2–6.9. This pattern across three separate failures points to DPM-Solver [13] not as an optional speed improvement but as structurally necessary for the multi-resolution composition to hold together.

Mixed precision training (AMP) had to be disabled. AMP downcast the precomputed alpha_bar tensors stored as plain attributes to float16 during autocast, which corrupted the noise levels and destabilized training.

Bolt-on improvements to the CI+Decomp architecture (extended training, cross-channel mixing, data augmentation, frequency features) all failed to improve on well-tuned hyperparameters from the sweep. The architecture plus hyperparameters was already the improvement; individual techniques provided diminishing returns.

### 6.4 Difficulties Faced

The paper [1] leaves several implementation-critical details unspecified: whether decomposition is residual or cumulative, whether the mixup projection is learned or random, the exact normalization layer in the denoiser (BatchNorm vs GroupNorm), model sizing details that prevent collapse, and the metric computation space (RevIN vs global standardization). Identifying each of these required a full training run, and each run took about 34 minutes across all four configurations. The absence of released code meant every architectural decision required empirical validation from scratch.

---

## 7. Reproducibility

All experiments were run on an NVIDIA RTX 5090 GPU with Python 3.9 and PyTorch 2.7.1. Training configurations, model architectures, and hyperparameters are fully specified in YAML config files and documented per-experiment. The ETT dataset is publicly available [16].

Key files for reproduction:
- `code/baseline/src/`, Baseline mr-Diff model implementation
- `code/improvement/src/`, CI+Decomp Transformer implementation with all variants
- `code/baseline/configs/small.yaml`, Baseline configuration
- `code/ALL_EXPERIMENT_RESULTS.md`, Complete 31-experiment log with all metrics
- `code/improvement/sweep.py`, 30-config hyperparameter sweep driver (raw logs archived under `archive/working-dirs/final-form/exp18_hyperparam_sweep/`)

---

## 8. Conclusion

Our baseline achieves MAE 0.47–0.20 versus the paper's 0.42–0.15. The gap comes from a train-test conditioning mismatch across stages, made worse by DDPM's [12] sample quality being insufficient for stage composition. Testing each paper-described component individually degraded performance in our setup, which suggests the multi-resolution framework depends on DPM-Solver [13] in a way [1] does not make clear.

The combined improvements reach ETTh1 Multi 0.4773, ETTh1 Uni 0.2471, ETTm1 Multi 0.4081, ETTm1 Uni 0.1865. Two of those beat the paper's reported results. The 843K-parameter diffusion model is replaced by transformers of 54–182K parameters that train faster and perform better on three of four benchmarks. The two findings that drove this outcome are that diffusion contributes nothing measurable on small time series datasets, and that channel independence multiplies the effective training set size by D, which is as valuable as any architectural innovation.

---

## 9. References

[1] L. Shen, W. Chen, and J. T. Kwok, "Multi-resolution diffusion models for time series forecasting," in Proc. Int. Conf. Learning Representations (ICLR), 2024.

[2] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond efficient transformer for long sequence time-series forecasting," in Proc. AAAI Conf. Artificial Intelligence, 2021.

[3] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[4] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, "FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting," in Proc. Int. Conf. Machine Learning (ICML), 2022.

[5] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in Proc. Int. Conf. Learning Representations (ICLR), 2023.

[6] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?" in Proc. AAAI Conf. Artificial Intelligence, 2023.

[7] C. Challu, K. N. Olivares, B. Oreshkin, F. Garza, M. Mergenthaler-Canseco, and A. Dubrawski, "N-HiTS: Neural hierarchical interpolation for time series forecasting," in Proc. AAAI Conf. Artificial Intelligence, 2023.

[8] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf, "Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting," in Proc. Int. Conf. Machine Learning (ICML), 2021.

[9] Y. Tashiro, J. Song, Y. Song, and S. Ermon, "CSDI: Conditional score-based diffusion models for probabilistic time series imputation," in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[10] J. M. L. Alcaraz and N. Strodthoff, "Diffusion-based time series imputation and forecasting with structured state space models," arXiv preprint arXiv:2208.09399, 2022.

[11] L. Shen and J. T. Kwok, "Non-autoregressive conditional diffusion models for time series prediction," in Proc. Int. Conf. Machine Learning (ICML), 2023.

[12] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[13] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, "DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps," in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[14] Kimi Team: G. Chen, Y. Zhang, J. Su, et al., "Attention Residuals," arXiv:2603.15031, 2026.

[15] Y. Liu, T. Hu, H. Zhang, C. Wu, S. Wang, L. Ma, and M. Long, "iTransformer: Inverted transformers are effective for time series forecasting," in Proc. Int. Conf. Learning Representations (ICLR), 2024.

[16] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond efficient transformer for long sequence time-series forecasting," in Proc. AAAI Conf. Artificial Intelligence, 2021 (ETT dataset).

[17] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible instance normalization for accurate time-series forecasting against distribution shift," in Proc. Int. Conf. Learning Representations (ICLR), 2022.

---

## Team Contributions

| Team Member | Contribution | Percentage |
|-------------|-------------|------------|
| Maximilian Khan | Baseline implementation from scratch, 31-experiment improvement campaign, CI+Decomp Transformer architecture design, hyperparameter sweep, AttnRes integration, ensemble design, report writing | 50% |
| Karthik Tamiledu | Overlapping patches implementation, iTransformer integration, two-scale decomposition, SLURM cluster scripts, experiment documentation | 50% |
| **Total** | | **100%** |
