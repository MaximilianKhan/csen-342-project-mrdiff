# Road to 30%: Closing the Gap

> **Date:** March 16, 2026
> **Target:** 30% improvement from personal baseline across benchmarks
> **Current state:** 18 experiments, CI+Decomp Transformer winning 3/4 benchmarks

---

## The Gap

| Benchmark | Baseline | Current Best | Improvement | **30% Target** | **Gap to 30%** |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.4880 | -2.9% (worse) | 0.3321 | **need -32% more** |
| ETTh1 Uni | 0.2535 | 0.2514 | -0.8% | 0.1774 | **need -29% more** |
| ETTm1 Multi | 0.4204 | 0.4094 | -2.6% | 0.2943 | **need -28% more** |
| ETTm1 Uni | 0.2011 | 0.1881 | -6.5% | 0.1408 | **need -25% more** |

We're at 1-6% improvement. We need 30%. That's a 5-30x multiplier on our current gains. This requires a fundamental step change, not incremental tuning.

For context, the **paper claims**: ETTh1 Multi 0.42 (11.5% from our baseline), ETTm1 Multi 0.37 (12%), ETTm1 Uni 0.15 (25%). The 30% target is **beyond the paper** on every benchmark except ETTm1 Uni. This is championship-level territory.

---

## What's Working vs What We Need

**What we've proven works:**
- CI transformer architecture (channel-independence, shared weights)
- Trend/residual decomposition before patching
- Small models (54-182K params)
- Fast iteration (1-8 min per benchmark)
- patch=8 for ETTm1, trend_kernel=15 for ETTh1, 3 layers, d_model=32-64

**What hasn't been tried yet (high potential):**
1. Cross-channel mixing (queued — targets ETTh1 Multi)
2. Data augmentation (never tried on the transformer)
3. Loss function engineering (only MSE so far)
4. Longer training / learning rate warmup (all models hit min_epochs=30)
5. Multi-scale patching (multiple patch sizes simultaneously)
6. Frequency-domain features (FFT branch)
7. Test-time augmentation (flip/jitter at inference, average)
8. RevIN as learnable layer (adaptive normalization)

---

## The Plan: 5 Phases

### Phase 1: Low-Hanging Fruit — Training Improvements (Exp 19-20)

Every model in the sweep hit `min_epochs=30` or early-stopped. We may be undertrained.

**Exp 19: Extended Training + Warmup**
- Increase max_epochs to 200 (from 100)
- Add linear warmup for first 10 epochs before cosine decay
- Reduce min_epochs to 10 (let early stopping decide)
- Increase patience to 30 (from 20)
- Use the best configs from sweep: cfg01 for ETTh1, cfg10 for ETTm1 Multi, cfg06 for ETTm1 Uni
- **Expected: 2-5% improvement from better convergence**
- **Time: ~15-20 min**

**Exp 20: Loss Function Engineering**
- Replace MSE with Huber loss (δ=1.0) — reduces sensitivity to outliers
- Add frequency-domain loss: `0.1 * MSE(FFT(pred), FFT(target))` — matches spectral structure
- Try combined: `Huber + 0.1 * FFT_MSE`
- Run best configs from Exp 19
- **Expected: 2-5% from better loss landscape**
- **Time: ~15-20 min**

### Phase 2: Cross-Channel Mixing (Exp 21)

The queued idea. ETTh1 Multi is our worst benchmark specifically because CI can't capture cross-variable dynamics.

**Exp 21: CI Transformer + Lightweight Channel Mixing**
```python
# After CI transformer produces [B, T, D]:
forecast = ci_decomp_transformer(lookback)     # [B, T, D]
# Lightweight cross-channel residual correction
forecast = forecast + self.channel_mix(forecast)  # Linear(D→D) residual
```

Three variants:
- **A:** Simple `Linear(D, D)` — 49 params (D=7)
- **B:** Single-head attention over D channels per timestep — ~200 params
- **C:** `Linear(D, D)` with per-timestep gating: `gate = sigmoid(Linear(D, 1))` — 56 params

All use residual connection (+ not replace) so CI path is preserved as fallback.
- **Expected: 3-8% on ETTh1 Multi specifically, neutral elsewhere**
- **Time: ~5 min per variant**

### Phase 3: Data Augmentation (Exp 22-23)

We've never augmented the training data. With 10K samples (ETTh1) this is potentially the biggest unlock.

**Exp 22: Temporal Augmentation**
- **Jitter:** Add small Gaussian noise to lookback (σ=0.01-0.05 of input std)
- **Scaling:** Random scale each sample by U(0.9, 1.1)
- **Temporal shift:** Shift lookback by ±1-3 timesteps (circular)
- **Window crop:** Random sub-window of lookback (90-100% length), pad to full
- Apply 50% of the time during training
- **Expected: 5-10% from implicit regularization and data diversity**
- **Time: ~10 min (just augmentation in dataloader)**

**Exp 23: Channel Augmentation (Multivariate Only)**
- **Channel dropout:** Zero out 1-2 of 7 channels randomly during training (keep all at test)
- **Channel shuffle:** Randomly permute channel order 20% of the time
- Forces model to be robust to missing/reordered channels
- CI architecture makes this natural — channels are already independent
- **Expected: 3-8% on multivariate specifically**
- **Time: ~5 min**

### Phase 4: Architecture Refinements (Exp 24-26)

**Exp 24: Multi-Scale Patching (Dual Patch)**
- Use two patch sizes simultaneously (e.g., 8 and 24) → fine and coarse tokens
- Concatenate or alternate in the sequence
- Self-attention naturally mixes fine and coarse temporal information
- This is the multi-resolution idea from the original paper, but without diffusion
```python
patches_fine = patch(lookback, size=8)    # [B*D, N_fine, d]
patches_coarse = patch(lookback, size=24)  # [B*D, N_coarse, d]
tokens = cat([patches_fine, patches_coarse], dim=1)  # [B*D, N_fine+N_coarse, d]
# Shared transformer attends over all
output = transformer(tokens + pos_embed)
```
- **Expected: 3-7% from capturing multi-scale temporal patterns**
- **Time: ~10 min**

**Exp 25: Frequency-Enhanced Dual Branch**
- Parallel FFT branch alongside patch transformer
- FFT captures exact periodicities (daily, weekly cycles) that attention approximates
```python
# Time branch (existing)
time_forecast = ci_decomp_transformer(lookback)
# Freq branch (new, lightweight)
freq = rfft(lookback, dim=1)  # [B, F, D] complex
freq_real = cat([freq.real, freq.imag], dim=-1)  # [B, F, 2D]
freq_forecast = linear(freq_real)  # [B, T, D]
# Combine
output = time_forecast + alpha * freq_forecast  # alpha learned or fixed at 0.1
```
- **Expected: 3-8% from complementary spectral information**
- **Time: ~5 min**

**Exp 26: RevIN as Learnable Layer**
- Move RevIN normalization into the model as a differentiable layer
- Learnable affine parameters per channel: `x_norm = (x - μ) / σ * γ + β`
- Denormalize on output with the same statistics
- The model can learn optimal normalization, not just mean/std
- **Expected: 1-3% from adaptive normalization**
- **Time: ~5 min**

### Phase 5: Combination + Polish (Exp 27+)

Take the winning techniques from Phases 1-4 and combine:
- Best training schedule (Exp 19)
- Best loss function (Exp 20)
- Cross-channel mixing (Exp 21)
- Best augmentation (Exp 22-23)
- Best architecture refinement (Exp 24-26)

The multiplicative effect of combining 3-5% improvements:
- 5 improvements × 4% each (multiplicative) = (0.96)^5 = 0.815 → **18.5% total improvement**
- 5 improvements × 6% each = (0.94)^5 = 0.734 → **26.6% total improvement**
- 5 improvements × 7% each = (0.93)^5 = 0.696 → **30.4% total improvement**

**This is how we get to 30%.** No single change does it. The compounding of 5-6 independent improvements, each providing 4-7%, gets us there.

---

## Execution Order (Prioritized by Expected Impact / Time)

| Priority | Exp | What | Expected Impact | Time | Cumulative |
|---|---|---|---|---|---|
| 1 | **22** | Temporal augmentation | 5-10% | 10 min | 5-10% |
| 2 | **19** | Extended training + warmup | 2-5% | 15 min | 7-15% |
| 3 | **21** | Cross-channel mixing | 3-8% (multi) | 5 min | 10-20% |
| 4 | **25** | Frequency-enhanced branch | 3-8% | 5 min | 13-25% |
| 5 | **20** | Loss function (Huber + FFT) | 2-5% | 15 min | 15-28% |
| 6 | **24** | Multi-scale patching | 3-7% | 10 min | 18-32% |
| 7 | **23** | Channel augmentation | 3-8% (multi) | 5 min | 20-35% |
| 8 | **26** | Learnable RevIN | 1-3% | 5 min | 21-36% |
| 9 | **27+** | Best combo | multiplicative | 15 min | **25-40%** |

**Total estimated time: ~85 min of experiments** (many parallelizable)

---

## The Critical Insight

Looking at the naive baselines:
- Naive (repeat last value): MAE 0.58-1.29
- Our best models: MAE 0.19-0.49
- 30% targets: MAE 0.14-0.33

Our models are already capturing 60-70% of the predictable signal. The remaining 30% improvement requires extracting the **hard** patterns — subtle cross-channel correlations, precise periodicity matching, and long-range dependencies that neither linear models nor basic attention catch.

The frequency branch (Exp 25) is the highest-conviction play for this. Time series have strong spectral structure — daily cycles at 24h (ETTh1) and 96 timesteps (ETTm1). Attention approximates periodicity through position; FFT captures it exactly. This is the kind of complementary information that could unlock a step change.

Data augmentation (Exp 22) is the other high-conviction play. We have 10K samples for ETTh1. Every prior experiment that added parameters overfitted. Augmentation effectively multiplies the dataset without adding parameters — it's the one lever we haven't pulled.

**If I had to pick 3 experiments to run right now:** Exp 22 (augmentation), Exp 25 (frequency branch), and Exp 21 (channel mixing). These attack three different bottlenecks (data scarcity, spectral structure, cross-channel dynamics) and their effects should multiply.

---

## What 30% Looks Like

| Benchmark | Current | 30% Target | What Gets Us There |
|---|---|---|---|
| ETTh1 Multi | 0.4880 | 0.3321 | Channel mixing + augmentation + frequency branch |
| ETTh1 Uni | 0.2514 | 0.1774 | Frequency branch + extended training + augmentation |
| ETTm1 Multi | 0.4094 | 0.2943 | Augmentation + channel mixing + multi-scale patches |
| ETTm1 Uni | 0.1881 | 0.1408 | Frequency branch + augmentation + loss engineering |

The hardest target is ETTh1 Uni (0.2514 → 0.1774). We already beat the paper (0.34) by 26%. Getting to 0.1774 would mean beating the paper by 48%. That's extremely aggressive but the frequency branch could unlock it — ETTh1 Uni has the strongest daily periodicity.

The most achievable target is ETTm1 Uni (0.1881 → 0.1408). The paper claims 0.15, so 0.1408 is within 6% of that. With 40K training samples and strong temporal structure, augmentation + frequency features have the best shot here.

Let's hunt.
