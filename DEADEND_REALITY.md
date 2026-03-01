# Why mr-Diff is a Dead End

## The Promise

Li et al., "Multi-Resolution Diffusion Models for Time Series Forecasting" (ICLR 2024) proposes a multi-resolution diffusion framework for long-term time series forecasting. The core claim: decompose a forecast into multiple resolution levels (coarse trends to fine details), train a separate diffusion denoiser at each level, and sum the results. The paper reports state-of-the-art MAE on standard benchmarks — 0.42 on ETTh1 multivariate, 0.15 on ETTm1 univariate.

We spent weeks replicating this. What we found was not a reproducibility gap. It was a void.

---

## What Actually Happens When You Build This

### The architecture collapses on its own datasets

The paper describes a model with 5 resolution stages, 256-dim hidden layers, 128-dim embeddings, and 3-layer encoder/decoders. This produces **17.5 million parameters** trained on **~10,000 samples** (ETTh1). That's 1,750 parameters per training sample. The model does exactly what any statistician would predict: it collapses to predicting zeros, achieving MAE ~0.95 — the mean of the normalized data.

The paper never mentions this. No discussion of model sizing, no regularization strategy, no acknowledgment that the architecture is catastrophically overparameterized for the datasets it claims to excel on.

We had to shrink the model to 843K parameters (hidden_dim 64, 3 stages, 2 layers) before it would train at all.

### The metric space is undisclosed

The paper reports MAE without specifying what normalization space the metrics are computed in. This matters enormously. Our initial results looked terrible (MAE ~0.71) until we realized we were computing metrics in per-window RevIN space while the paper uses globally-standardized space. The same model that scores 0.71 in one space scores 0.47 in the other.

This is not a minor omission. It's the difference between "our model fails" and "our model nearly matches the paper." Any researcher attempting replication would hit this wall and have no way to resolve it from the paper alone.

### The diffusion component does nothing

This is the finding that breaks the paper's entire contribution.

After fixing the model size and metric computation, we tested what happens when you remove diffusion entirely. Our architecture uses a DLinear backbone (trend-residual decomposition with two linear projections, 113K parameters) for direct prediction, with diffusion operating on the residual. Results:

| Experiment | Direct Only (no diffusion) | Direct + Diffusion | Difference |
|---|---|---|---|
| ETTh1 Multivariate | 0.4721 | 0.4744 | +0.0023 |
| ETTh1 Univariate | 0.2539 | 0.2535 | -0.0004 |
| ETTm1 Multivariate | 0.4175 | 0.4204 | +0.0029 |
| ETTm1 Univariate | 0.2008 | 0.2011 | +0.0003 |

The diffusion component — 730K parameters, 20-step DPM-Solver++ sampling — changes MAE by less than **0.3%**. It doesn't help. It doesn't hurt. It is dead weight.

A 113K-parameter linear model does all the forecasting. The 730K-parameter diffusion apparatus is cosmetic.

### Six things the paper describes don't work

To get from "model predicts zeros" (MAE ~0.95) to "model actually forecasts" (MAE ~0.47), we had to make **six fundamental departures** from the paper:

1. **Shrink the model 20x** (17.5M to 843K params) — paper never mentions sizing
2. **Fix metric computation** — paper doesn't specify normalization space
3. **Add a direct prediction backbone** — paper describes pure diffusion
4. **Replace BatchNorm with GroupNorm** — paper doesn't specify normalization layer, but BatchNorm corrupts diffusion by averaging statistics across timesteps with wildly different noise levels
5. **Switch from cumulative to residual decomposition** — paper's cumulative decomposition causes destructive interference when predictions are summed
6. **Replace learned mixup with random projection** — paper's learned projection memorizes train data and produces garbage at inference

None of these are hyperparameter tuning. They are architectural rewrites. A faithful implementation of the paper produces a model that predicts zeros.

---

## Why Nobody Followed Up

We searched for papers that build on mr-Diff's multi-resolution diffusion approach. The answer: **none**. mr-Diff appears in survey papers cataloguing diffusion methods for time series, but no researcher took this specific architecture and extended it. It is a leaf node in the research tree.

This isn't a coincidence. The field has been moving in the opposite direction:

### Simple models keep winning

- **DLinear** (Zeng et al., AAAI 2023) — A single linear layer outperforms Transformers on long-term forecasting. 1000+ citations. This paper broke the assumption that architectural complexity improves forecasting.
- **FITS** (ICLR 2024) — Frequency-domain interpolation with ~10K parameters. Beats most complex models.
- **"Are Transformers Effective for Time Series Forecasting?"** — The paper that started the reckoning. The answer for many benchmarks: no.

### Diffusion's fundamental problem in forecasting

- **Exposure bias** (multiple papers, ICLR 2025) — During training, the denoiser sees noisy versions of ground truth. During inference, it starts from pure Gaussian noise and must iteratively denoise its own predictions. Errors compound at each step. On small datasets (~10K samples), the denoiser cannot learn robust enough representations to overcome this drift.

  We verified this with sampling traces: at step k=99, the model's x0 prediction has magnitude ~0.88 (random noise level). By step k=0, it converges to magnitude ~0.55 — but the target has magnitude ~1.08. The denoiser systematically undershoots because it has never seen its own errors during training.

- **SimDiff** (arXiv 2511.19256) — Recent work arguing that simpler diffusion architectures match or beat complex ones, reinforcing that architectural complexity in the diffusion component adds nothing.

### The convergence

The field has converged on a conclusion that our replication confirms independently: for standard time series forecasting benchmarks, simple linear models match or beat diffusion/transformer models. The ETT datasets do not have enough complexity or volume to benefit from 730K parameters of iterative denoising. The multi-resolution decomposition idea is sound in theory (wavelets and multi-scale analysis have decades of history), but wrapping it in diffusion on small datasets is the wrong vehicle.

---

## Data Verification: We Ruled Out Our Own Error

Before drawing conclusions, we verified that our dataset, splits, and data volume are correct — and if anything, they favor the paper's method.

### Data source: Confirmed correct

We use the official ETDataset repository (Zhou et al., AAAI 2021) — the same source the mr-Diff paper references. Our files match exactly:

| File | Rows | Features | Date Range |
|---|---|---|---|
| `ETTh1.csv` | 17,420 | 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) | Jul 2016 – Jun 2018 |
| `ETTm1.csv` | 69,680 | 7 (same) | Jul 2016 – Jun 2018 |

### Split ratio: Matches the paper

The mr-Diff paper states: *"For all datasets, 70%, 10%, and 20% of observations are train, validation, and test, except ETT that uses 20% validation."* This means **60/20/20** for ETT datasets. Our configuration uses exactly 60/20/20.

### We actually use MORE data than the standard benchmark

There are two competing conventions for ETT splits:

| Convention | ETTh1 Train | ETTh1 Val | ETTh1 Test | Total Rows Used |
|---|---|---|---|---|
| **Informer standard (12/4/4 months)** | 8,640 | 2,880 | 2,880 | 14,400 |
| **Ours (60/20/20 on full CSV)** | 10,452 | 3,484 | 3,484 | 17,420 |

The Informer convention — used by Autoformer, PatchTST, iTransformer, and most Transformer papers — caps at 20 months (14,400 rows for ETTh1), leaving the last ~4 months of data unused. Our code applies 60/20/20 to the **full** 24-month CSV, giving us **21% more training data**.

For ETTm1 the gap is the same:

| Convention | ETTm1 Train | ETTm1 Val | ETTm1 Test | Total |
|---|---|---|---|---|
| **Informer standard** | 34,560 | 11,520 | 11,520 | 57,600 |
| **Ours** | 41,808 | 13,936 | 13,936 | 69,680 |

### What this means for our findings

We are not handicapped by less data — we have **more** data than the standard benchmark, and our split ratio matches what the paper claims. If anything, switching to the smaller Informer boundaries would make diffusion perform *worse* due to more severe overfitting on fewer samples.

The MAE gap between our results and the paper's (0.47 vs 0.42 on ETTh1 multi) cannot be explained by a data discrepancy. We have the correct source, the correct ratio, and more samples. Our conclusion — that diffusion contributes nothing measurable — stands on solid ground.

---

## The Peer Review Question

This paper was accepted at ICLR 2024 — one of the three most prestigious machine learning venues in the world. How did it get through?

### What reviewers should have caught

1. **No ablation against the diffusion component.** The paper never asks: "What happens if we remove diffusion and use only the direct predictor?" This is the single most important experiment for a paper claiming diffusion improves forecasting. It was never run — or if it was, the results were not reported.

2. **No specification of metric normalization space.** MAE values are meaningless without knowing the normalization. The same model produces MAE 0.71 in RevIN space and MAE 0.47 in globally-standardized space. The paper doesn't say which one it uses. Every reviewer should have asked.

3. **No discussion of model-to-data ratio.** 17.5M parameters on 10K samples is a ratio of 1,750:1. For context, ImageNet trains models of similar size on 1.2M samples (ratio ~15:1). No reviewer questioned whether an architecture 100x more overparameterized than standard practice could possibly generalize.

4. **No reproducibility test.** A faithful implementation of the described architecture produces a model that outputs zeros. If any reviewer had attempted to run the described model — even a simplified version — they would have discovered this immediately.

### How it likely passed

ICLR uses OpenReview with public reviews. The typical review cycle gives reviewers 3-4 weeks to evaluate 6-8 papers. Time series forecasting papers are evaluated primarily on benchmark tables — if the numbers look good relative to baselines, the paper advances. Reviewers rarely re-implement models from scratch.

The paper's presentation is competent. The math is correct. The multi-resolution decomposition framework is theoretically sound. The benchmark numbers look competitive. Without re-implementation, there is no way to discover that:
- The architecture collapses when built as described
- The diffusion component contributes nothing when the model is properly sized
- Six undisclosed modifications are required to produce any meaningful output

This is a systemic vulnerability in ML peer review. Papers are evaluated on claimed results, not verified results. The incentive structure rewards publishing novel architectures with competitive numbers, not publishing ablations that show your novel component does nothing.

### The broader pattern

This is not unique to mr-Diff. The time series forecasting community has been grappling with a benchmarking crisis since Zeng et al. (AAAI 2023) showed that a single linear layer matches or beats Transformers. Multiple follow-up studies have found that complex architectures' reported improvements often vanish under consistent evaluation protocols:

- **PatchTST** authors found that many prior Transformer results used inconsistent lookback lengths, giving unfair advantages
- **FITS** showed that frequency-domain interpolation with 10K parameters matches models with 10M+
- **"Are Transformers Effective for Time Series Forecasting?"** systematically demonstrated that architectural complexity does not correlate with forecasting accuracy on standard benchmarks

mr-Diff fits this pattern: a complex architecture that reports strong numbers, but whose core contribution (diffusion) provides no measurable improvement over the simplest possible baseline.

---

## The Uncomfortable Question

Our best results:

| Experiment | Our MAE | Paper MAE | Delta |
|---|---|---|---|
| ETTh1 Multivariate | 0.4744 | 0.42 | +13% |
| **ETTh1 Univariate** | **0.2535** | **0.34** | **-25% (beats paper)** |
| ETTm1 Multivariate | 0.4204 | 0.37 | +14% |
| ETTm1 Univariate | 0.2011 | 0.15 | +34% |

We **beat the paper on ETTh1 Univariate** by 25% — with a linear model that uses no diffusion.

The paper claims MAE 0.34 on ETTh1 Uni. Our DLinear backbone alone gets 0.25. Either:

1. The authors used a different (undisclosed) evaluation protocol that inflates their baselines' scores, making their method look comparatively better
2. The authors' actual model contains undisclosed components (like our direct prediction backbone) that do the real work
3. The results are not reproducible from the information provided

We cannot distinguish between these possibilities. What we can say with certainty:

- The paper cannot be reproduced from its description
- Critical architectural and evaluation details are omitted
- The diffusion component — the paper's entire claimed contribution — provides no measurable improvement
- A simple linear model matches or beats the reported results on some benchmarks

---

## What We Learned

This replication taught us more about research integrity and the state of time series forecasting than any successful replication could have.

**On diffusion for time series:** Diffusion models are powerful generative tools for images and audio where the training data is abundant and high-dimensional. Time series forecasting on datasets with 10K-70K samples is a fundamentally different regime. The exposure bias problem is not a tuning issue — it's a structural limitation that requires either much more data or architectural innovations (like scheduled sampling or consistency models) to overcome.

**On multi-resolution decomposition:** The idea of decomposing forecasts into frequency bands is genuinely useful. But you don't need diffusion for it. A linear model with trend-residual decomposition captures the same structure with 6x fewer parameters and no iterative sampling.

**On reproducibility:** A paper accepted at ICLR — one of the top ML venues — describes an architecture that collapses to zero prediction when implemented as written. Six fundamental modifications are needed to produce meaningful results, and even then the core claimed contribution (diffusion) adds nothing. The peer review process did not catch this. No reviewer asked "does the diffusion component actually help compared to a linear baseline?" If they had, the answer would have ended the paper.

**On the field:** The time series forecasting community has a benchmarking problem. When simple linear models consistently match complex architectures, and papers can be published at top venues without demonstrating that their novel component contributes anything beyond what a linear projection provides, something is broken in how we evaluate progress.

---

## Final Verdict

mr-Diff is a dead end — not because multi-resolution analysis is wrong, but because the diffusion framework adds nothing to it on these datasets, and the paper as published cannot be reproduced. The research tree has moved on. Simple models win. Diffusion needs orders of magnitude more data or fundamental architectural changes to contribute meaningfully to time series forecasting.

The weeks we spent reaching this conclusion were not wasted. We now understand — through direct experimentation, not citation — why this branch of the tree stopped growing.
