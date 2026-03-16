# Next Experiments — What 13 Failures Taught Us

> **Date:** March 16, 2026
> **Status:** 13 experiments complete. Zero have improved multivariate. Time to change the paradigm.

## The Pattern We Can't Ignore

Every experiment followed the same arc: add complexity → multivariate regresses → univariate sometimes improves slightly. Thirteen times. The signal is deafening:

**The model is not undertrained. The model is not underparameterized. The model is overparameterized for its data.**

The DLinear backbone (113K params, 2 linear projections) achieves 0.47 MAE on ETTh1 Multi with ~10K training samples. That's ~11 params per sample. The diffusion side (730K params) contributes nothing. Every experiment that added parameters made things worse on multivariate.

The path forward is not "add more stuff to the model." It's either:
1. **Make the existing model work better** (regularization, training tricks, ensemble)
2. **Remove the parts that don't work** (strip diffusion, lean into what's actually good)
3. **Change the game entirely** (data augmentation, knowledge distillation, fundamentally different architecture)

---

## Tier 1: High-Confidence Experiments (Should Run First)

### Experiment 14: Backbone-Only Mode (Kill Diffusion)

**The thesis:** Diffusion is not cosmetic — it's *actively harmful* through gradient interference. Even with `diffusion_loss_scale=0.3`, the diffusion loss contributes gradients that flow back through the backbone (Exp 1 removed the detach). What if diffusion training is subtly corrupting the backbone's optimization landscape?

**The test:** Train the model with diffusion completely disabled — just the DLinear backbone + direct MSE loss. No diffusion steps, no conditioning networks, no denoising. Pure DLinear with the same training schedule (AdamW, cosine LR, early stopping).

**What this tells us:**
- If backbone-only matches or beats baseline → diffusion training is actively hurting the backbone
- If backbone-only is worse → the diffusion gradients provide useful regularization (unlikely but possible)
- The gap between backbone-only and Full MAE tells us diffusion's *true* contribution when it's not interfering with backbone optimization

**Expected:** Backbone-only will match or slightly beat the current baseline, especially on multivariate. This would confirm that our "best" model is being held back by its own diffusion component.

**Complexity:** Trivial — remove diffusion from training loop, train just direct_loss.

---

### Experiment 15: DLinear Ensemble (3-5 Models, Different Seeds)

**The thesis:** If DLinear is the real model, make it better the simplest way possible — ensemble multiple independently trained DLinear models. This is the lowest-risk, highest-expected-value experiment we haven't tried.

**The test:** Train 5 DLinear-only models (from Exp 14) with different random seeds. At inference, average their predictions. This is O(5) cost but captures prediction uncertainty and reduces variance.

**What this tells us:**
- The variance reduction from ensembling tells us how much of our error is optimization noise vs fundamental
- If ensembling helps multivariate more than univariate, it confirms multi is limited by optimization rather than architecture

**Expected:** 2-5% improvement across all benchmarks. Ensembling is the most reliable technique in ML for exactly this data regime (small data, high variance).

**Complexity:** Easy — train 5 models, average predictions.

---

### Experiment 16: Aggressive Regularization (Backbone Dropout + Weight Decay Sweep)

**The thesis:** The baseline uses dropout=0.3 and weight_decay=0.01. What if we're still underdoing it? On ~10K multivariate samples, even DLinear might benefit from stronger regularization.

**The test:** Sweep:
- Dropout: [0.3, 0.5, 0.7]
- Weight decay: [0.01, 0.05, 0.1]
- Add explicit L1 penalty on backbone weights (encourages sparsity in the linear projections)

**Why this could close the multivariate gap:** Our multivariate error is +13% above the paper. If the backbone is memorizing training-specific correlations between the 7 ETT channels, stronger regularization forces it to learn more generalizable cross-channel patterns.

**Complexity:** Easy — config changes + L1 loss term.

---

## Tier 2: Creative Experiments (Higher Risk, Higher Reward)

### Experiment 17: Data Augmentation — Temporal Jitter + Channel Dropout

**The thesis:** We can't get more data, but we can make the existing data harder to memorize.

**Augmentations:**
1. **Temporal jitter:** Randomly shift the lookback window by ±1-5 timesteps (different per channel for multivariate). Forces the model to be robust to slight misalignment.
2. **Channel dropout:** For multivariate, randomly zero out 1-2 of the 7 channels during training. Forces the model to not rely on any single channel. At inference, all channels are present.
3. **Cutmix for time series:** Splice segments from different training samples together. Creates synthetic samples with realistic local statistics but novel global patterns.

**Why this specifically helps multivariate:** The multivariate gap (+13%) is larger than univariate. Augmentation prevents memorization of the specific cross-channel correlations in the training set. Channel dropout is particularly powerful — it's like dropout but in the data space rather than hidden space.

**Expected:** 3-8% multivariate improvement. This is the highest-impact experiment for closing the multi gap.

**Complexity:** Medium — augmentation transforms in the dataloader.

---

### Experiment 18: Residual Boosting (Iterative DLinear)

**The thesis:** Instead of using diffusion to predict the residual after DLinear, use *another DLinear* to predict the residual. Then another one for the residual-of-the-residual. This is gradient boosting with DLinear as the weak learner.

**Architecture:**
```
pred_1 = DLinear_1(lookback)                    # First model
resid_1 = target - pred_1
pred_2 = DLinear_2(lookback, resid_1_detached)   # Second model, conditioned on residual
resid_2 = target - pred_1 - pred_2
pred_3 = DLinear_3(lookback, resid_2_detached)   # Third model
final = pred_1 + shrinkage * pred_2 + shrinkage^2 * pred_3
```

**Key: shrinkage factor (0.1-0.3).** This prevents later models from overfitting — they contribute less and less, exactly like learning rate in gradient boosting.

**Why this replaces diffusion better than diffusion:** Diffusion tries to model residuals as a stochastic process (add noise, learn to denoise). But the residuals after DLinear are small and structured — they're not noise-like. A second DLinear that directly predicts the residual pattern is a much better inductive bias than a denoiser.

**Expected:** 2-5% improvement across all benchmarks. The shrinkage prevents overfitting that killed Exp 11-13.

**Complexity:** Medium — ~60 lines. No diffusion infrastructure needed.

---

### Experiment 19: AttnRes on DLinear (Width, Not Depth)

**The thesis:** Experiments 11-13 showed that AttnRes works (beats standard residuals) but backbone depth is the problem. What if we apply AttnRes *laterally* instead of *vertically*? Instead of 4 deep layers, use multiple parallel DLinear projections that attend over each other.

**Architecture:**
```
# N parallel DLinear projections (N=4-8), each with different kernel sizes
proj_1 = DLinear(lookback, kernel=5)    # Fine patterns
proj_2 = DLinear(lookback, kernel=15)   # Medium patterns
proj_3 = DLinear(lookback, kernel=25)   # Coarse patterns (current baseline)
proj_4 = DLinear(lookback, kernel=51)   # Very coarse patterns

# AttnRes-style attention to combine them
query = learned_vector  # [hidden_dim]
keys = RMSNorm([proj_1, proj_2, proj_3, proj_4])
weights = softmax(query^T · keys)
output = weighted_sum(projections, weights)
```

**Why width instead of depth avoids overfitting:** Each parallel projection has the same parameter count as the current DLinear. We're not adding depth (which caused overfitting) but adding *perspectives* — multiple views of the same lookback at different temporal scales. The AttnRes attention learns which scale is most informative for each dataset, and zero-initialization ensures it starts as an equal average.

**This is basically multi-scale DLinear with learned attention fusion.** It's the multi-resolution idea from the paper, but applied to the backbone instead of diffusion.

**Expected:** 2-6% improvement. Particularly promising for multivariate where different channels may benefit from different temporal scales.

**Complexity:** Easy — ~50 lines. No diffusion changes needed.

---

### Experiment 20: Knowledge Distillation from Exp 10 + Exp 7

**The thesis:** Exp 10 is our best on ETTh1 Uni (0.2508) and ETTm1 Multi (0.4194). Exp 7 is our best on ETTm1 Uni (0.1913). Neither is best everywhere. What if we distill both into a single student model?

**The test:** Train a new model where the loss includes:
- Standard MSE on ground truth (direct_loss)
- KL divergence from Exp 10's predictions (soft targets)
- KL divergence from Exp 7's predictions (soft targets, weighted lower)
- Temperature scaling to soften the teacher predictions

**Why distillation helps small data:** The teacher models encode learned patterns that go beyond the raw training data. Distillation transfers these patterns to the student as a form of data augmentation — the student sees not just "what the answer is" but "what two good models think the answer distribution looks like."

**Expected:** Could combine the strengths of both teachers. 2-4% improvement on the benchmarks where each teacher is weak.

**Complexity:** Medium — train two teachers, add distillation loss to student training.

---

## Tier 3: Paradigm Shifts

### Experiment 21: Replace Everything with Tiny Transformer (PatchTST-style, Direct)

**The thesis:** Stop trying to make diffusion work. Replace the entire mr-Diff architecture with a tiny PatchTST-style transformer. Our Exp 9 failed because the transformer was used as a *conditioning* encoder feeding into diffusion. What if the transformer IS the model?

**Architecture:**
```
lookback [B, H, D] → patch_embed (patch_size=16, stride=16) → [B, N_patches, D_model]
  → 2-layer Transformer (D_model=64, 4 heads, dim_ff=128)
  → flatten → Linear → [B, T, D]
```

**Target param count:** ~100-150K — deliberately matched to DLinear's regime.

**Why this is different from Exp 9:** Exp 9 put a transformer *inside* the conditioning network of a diffusion model. The transformer's output had to be consumed by a denoiser, creating an impedance mismatch. Here, the transformer directly produces the forecast — no diffusion, no conditioning bottleneck.

**Expected:** Competitive with DLinear on multivariate (transformers handle multi-channel better than linear projections). Possibly 5-10% improvement on multi if the architecture is right-sized.

**Complexity:** Medium — new model class, ~100 lines.

---

### Experiment 22: Diffusion on the Right Thing (Probabilistic Residuals)

**The thesis:** Diffusion fails because we ask it to predict deterministic residuals. But diffusion is fundamentally a *generative* model — it's designed to model distributions, not point estimates. What if we use it correctly?

**The shift:** Instead of `output = backbone + diffusion_residual`, use diffusion to model the *distribution* of forecast errors:
1. Train backbone normally (DLinear, no diffusion interference)
2. Collect backbone residuals on validation set → these form the empirical error distribution
3. Train a small diffusion model to generate samples from this distribution
4. At inference: `forecast = backbone_pred + α * diffusion_sample` where α is a learned/tuned blending weight

**Why this could actually work:** The diffusion model never interferes with backbone training. It models a genuine distribution (forecast errors), not a deterministic quantity. The blending weight α can be set to zero if diffusion doesn't help — safe by design.

**Expected:** This won't improve point MAE much (backbone is already good). But it gives calibrated uncertainty estimates, which could be valuable for the report.

**Complexity:** Medium-Hard — two-stage training pipeline.

---

## Recommended Execution Order

| Priority | Experiment | Expected Impact | Risk | Time |
|---|---|---|---|---|
| 1 | **14: Backbone-only** | Establishes true baseline | None | ~30m |
| 2 | **15: Ensemble** | 2-5% all benchmarks | None | ~2.5h |
| 3 | **17: Data augmentation** | 3-8% multivariate | Low | ~1h |
| 4 | **19: Multi-scale AttnRes DLinear** | 2-6% all | Low | ~1h |
| 5 | **18: Residual boosting** | 2-5% all | Low | ~1h |
| 6 | **16: Regularization sweep** | 1-3% multivariate | None | ~3h |
| 7 | **20: Distillation** | 2-4% weak benchmarks | Medium | ~2h |
| 8 | **21: Tiny transformer** | 5-10% multivariate | Medium | ~1h |
| 9 | **22: Probabilistic residuals** | Uncertainty, not MAE | Medium | ~2h |

**The big picture:** Experiments 14-15 establish the floor (what pure DLinear can do). Experiments 17-19 try to beat that floor with principled, low-risk innovations. Experiments 20-22 are swings for the fences.

If we could only run three: **14 (backbone-only), 17 (data augmentation), 19 (multi-scale AttnRes DLinear)**. These three test orthogonal hypotheses with minimal overlap and maximum information gain.
