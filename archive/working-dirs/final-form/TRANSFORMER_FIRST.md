# TRANSFORMER_FIRST: Optimizing the Tiny Transformer

> **Date:** March 16, 2026
> **Foundation:** Experiment 15 — 2-layer PatchTST, matches baseline in 1.8 min
> **Goal:** Beat the DLinear baseline on ALL benchmarks. Close the multivariate gap. Win.

---

## Where We Stand

Exp 15 proved the thesis: diffusion is overhead. A tiny transformer matches the full mr-Diff pipeline on univariate in 1/20th the training time. But multivariate is +18-31% worse. That's the target.

| Benchmark | DLinear Baseline | Exp 15 Transformer | Gap |
|---|---|---|---|
| ETTh1 Multi | 0.4744 | 0.5607 | **+18.2% — must fix** |
| ETTh1 Uni | 0.2535 | 0.2538 | +0.1% — matched |
| ETTm1 Multi | 0.4204 | 0.5514 | **+31.2% — must fix** |
| ETTm1 Uni | 0.2011 | 0.2002 | -0.4% — matched |

---

## The Diagnosis: Why Multivariate Fails

I ran the param breakdown on Exp 15. The results are damning:

```
ETTh1_multi: total=1,657,368 | head=1,581,720 (95%) | transformer=66,944 (4%)
ETTh1_uni:   total=  295,464 | head=  225,960 (76%) | transformer=66,944 (23%)
ETTm1_multi: total=7,822,848 | head=7,742,784 (99%) | transformer=66,944 (1%)
ETTm1_uni:   total=1,180,032 | head=1,106,112 (94%) | transformer=66,944 (6%)
```

**The transformer is 67K params. The flatten→linear head is 1.6M-7.8M params.**

The "tiny transformer" is actually a massive linear projection with a tiny transformer on top. The head does `n_patches × d_model → T × D`:
- ETTh1 Multi: 21×64 = 1,344 → 168×7 = 1,176 → **1.58M params**
- ETTm1 Multi: 90×64 = 5,760 → 192×7 = 1,344 → **7.74M params**

This is worse than our original 17.5M model that collapsed to zero prediction. We got lucky on univariate (where T×D is small), but multivariate is being murdered by this head.

**The transformer itself (67K) is doing useful work. The head (1.6-7.8M) is the entire problem.**

---

## The 15 Lessons That Guide Us

From 15 experiments, these are the iron laws of our data regime:

1. **Linearity is regularization.** DLinear works because it can't overfit. (Baseline)
2. **113K params is the sweet spot.** Every experiment that exceeded this regressed on multi. (Exp 3-14)
3. **Channel independence helps.** DLinear processes channels through shared weights. (Baseline vs Exp 8)
4. **Depth hurts more than width.** 4 conv layers = +40% regression. (Exp 11-13)
5. **Diffusion is pure overhead.** Zero contribution across 15 experiments. (All)
6. **Univariate is easy.** Almost anything works — the signal is strong and D=1 means small params. (All)
7. **Multivariate is the real challenge.** D=7 with 10K samples = every extra param is a trap. (All)
8. **AttnRes works when depth is warranted.** Small but real gains over standard residuals. (Exp 12 vs 11)
9. **Training speed enables iteration.** Exp 15 trains 20x faster = 20x more experiments per hour. (Exp 15)
10. **The head is the bottleneck.** 95-99% of Exp 15's params are in the output projection. (Diagnosis above)
11. **RevIN normalization is essential.** Per-window normalization provides 5-20x improvement. (Early work)
12. **Early stopping at patience=20 is reliable.** Models converge in 30-80 epochs. (All)
13. **Trend/residual decomposition is a good prior.** DLinear's trend extraction helps. (Baseline, Exp 10)
14. **x0-prediction with decomposition is the best diffusion variant.** If we ever revisit diffusion. (Exp 10)
15. **Transformers are fast enough to iterate.** 150+ it/s vs 20 it/s. (Exp 15)

---

## The Optimization Plan

### Phase 1: Fix the Head (The 95% Problem)

The flatten→linear head is 95-99% of parameters and the sole cause of multivariate regression. Three approaches, ordered by conservatism:

#### A. Channel-Independent Head (CI-Head)

The PatchTST paper actually recommends this. Process each channel independently through a shared head:

```python
# Current (Exp 15): flatten ALL patches × ALL channels → one giant linear
x = x.reshape(B, -1)                    # [B, N * d_model]
x = self.head(x)                        # [B, T * D] — 1.6M params for ETTh1 Multi

# Proposed: shared per-channel linear projection
x = x.reshape(B, N, d_model)            # [B, N, d_model]
x = self.head(x.transpose(1,2))         # head: Linear(N → T), applied per d_model dim
                                          # Then project: Linear(d_model → D)
```

**Param count:** Linear(N→T) = 21×168 = 3,528 + Linear(64→7) = 448 → **~4K params** vs 1.58M. That's a **400x reduction**.

This forces channel independence in the output — each channel's forecast depends only on the transformer's learned temporal patterns, not cross-channel output correlations. This is exactly DLinear's implicit regularization, but in the transformer.

#### B. Patch-to-Forecast Linear (Per-Patch Projection)

Instead of flattening all patches, project each patch token independently to a forecast segment, then reassemble:

```python
# Each patch token produces a segment of the forecast
segment_length = T // N  # or use overlap + average
x = self.patch_head(x)   # Linear(d_model → segment_length * D) per patch
x = x.reshape(B, T, D)   # Reassemble segments
```

**Param count:** Linear(64 → 8×7) = 3,584 → **~4K params**. Similar reduction.

#### C. Learned Temporal Upsampling

Use a small MLP to upsample from N patch tokens to T forecast steps:

```python
x = self.upsample(x.transpose(1,2))  # Linear(N → T) on the temporal dim
x = self.proj(x.transpose(1,2))       # Linear(d_model → D)
```

**Param count:** Linear(21→168) + Linear(64→7) = ~4K. Same ballpark.

**Recommendation: Start with A (CI-Head).** It's the most principled — it inherits DLinear's channel-independence insight while keeping the transformer's temporal attention. The PatchTST paper validates it empirically.

---

### Phase 2: Channel-Independent Patching (CI-Patch)

Current Exp 15 concatenates all D channels into each patch: `patch = [t1_ch1, t1_ch2, ..., t1_ch7, t2_ch1, ...]`. This means:
- Patch embedding is `Linear(patch_size × D → d_model)` = `Linear(112 → 64)` for D=7
- The model mixes channels BEFORE attention — cross-channel patterns are baked into patch tokens
- For D=1, this is fine. For D=7, it conflates variables that have different dynamics

The PatchTST paper's key insight: **patch each channel independently, share transformer weights across channels.**

```python
# Current: [B, H, D] → [B, N, P*D] → embed → [B, N, d_model]
# Proposed: [B, H, D] → [B*D, H, 1] → [B*D, N, P] → embed → [B*D, N, d_model]
#           → transformer (shared) → head → [B*D, T, 1] → [B, T, D]
```

**Why this helps multivariate:**
- Patch embedding is `Linear(P → d_model)` = `Linear(16 → 64)` regardless of D. **Same params for uni and multi.**
- Each channel goes through the same transformer independently — the model learns *temporal* patterns, not spurious cross-channel correlations
- Shared weights across channels = implicit regularization (exactly like DLinear)
- Total params become independent of D — no more 7.8M head for ETTm1 Multi

**Param count (ETTh1, any D):**
- Patch embed: Linear(16 → 64) = 1,088
- Transformer: 66,944 (same)
- Head: Linear(21 → 168) + Linear(64 → 1) = 3,592 + 65 = 3,657
- **Total: ~72K params** — for ANY value of D.

This is **less than DLinear's 113K**. And it has self-attention with global receptive field.

---

### Phase 3: Trend/Residual Decomposition (The DLinear Prior)

DLinear decomposes input into trend (avg-pool) and residual before projection. This is a good inductive bias for time series. Add it to the transformer:

```python
# Before patching:
trend = avg_pool1d(lookback, kernel=25)   # [B, H, D]
residual = lookback - trend                # [B, H, D]

# Option A: Two separate transformers (trend + residual), sum outputs
forecast = transformer_trend(trend) + transformer_resid(residual)

# Option B: Concatenate as extra channels (trend as context)
x = torch.cat([residual, trend], dim=-1)  # [B, H, 2D]
# Then CI-patch treats this as 2D channels through shared transformer

# Option C: Decompose output, not input
forecast = model(lookback)
# Post-hoc: no decomposition needed if the model learns it
```

**Recommendation: Option A with shared weights.** Use the SAME transformer for both trend and residual (like DLinear uses the same architecture for both), just with different linear heads. This doubles the effective training data for the transformer (it sees both trend and residual sequences) without doubling params.

---

### Phase 4: Hyperparameter Optimization (The Speed Advantage)

With 1-2 minute training cycles, we can sweep hyperparameters that would take days in the diffusion regime:

| Parameter | Exp 15 Value | Sweep Range | Why |
|---|---|---|---|
| patch_size | 16 | [8, 12, 16, 24, 32] | Controls receptive field per token |
| d_model | 64 | [32, 48, 64, 96] | Model capacity |
| nhead | 4 | [2, 4, 8] | Attention diversity |
| num_layers | 2 | [1, 2, 3] | Depth (careful — Exp 11-13 lesson) |
| dim_feedforward | 128 | [64, 128, 256] | FFN capacity |
| dropout | 0.3 | [0.2, 0.3, 0.4, 0.5] | Regularization |
| learning_rate | 0.0005 | [0.0001, 0.0005, 0.001, 0.002] | Convergence speed |
| weight_decay | 0.01 | [0.001, 0.01, 0.05, 0.1] | L2 regularization |

A full grid sweep of the top 4 params (patch_size × d_model × num_layers × dropout) at 5×4×3×4 = 240 configs would take ~240 × 2 min = **8 hours**. A smart random search of 30-50 configs = **1-2 hours**. In the diffusion regime, 50 experiments would take 50 × 90 min = **75 hours**.

---

### Phase 5: Advanced Techniques (After Phases 1-4 Are Locked)

#### A. Frequency-Enhanced Attention

Add a parallel FFT branch that operates on frequency-domain representations. Time series have strong spectral structure — attention over Fourier coefficients captures periodicity directly.

```python
x_time = self.transformer(patches)           # Temporal attention
x_freq = self.freq_linear(fft(lookback))     # Frequency-domain projection
forecast = time_head(x_time) + freq_head(x_freq)  # Combine
```

This is the FEDformer idea but minimal — just a linear projection on FFT coefficients, no complex architecture. The FFT branch adds ~2K params and provides complementary spectral information the attention might miss.

#### B. Reversible Instance Normalization (RevIN) Integration

We already use RevIN for per-window normalization. But we could integrate it more tightly:

```python
# Current: RevIN applied in dataloader, model sees normalized data
# Proposed: RevIN as first/last layer of model, with learnable affine
class RevINLayer(nn.Module):
    def __init__(self, D):
        self.affine_weight = nn.Parameter(ones(D))
        self.affine_bias = nn.Parameter(zeros(D))

    def forward(self, x):  # normalize
        mean, std = x.mean(1, keepdim=True), x.std(1, keepdim=True)
        x_norm = (x - mean) / (std + eps)
        return x_norm * self.affine_weight + self.affine_bias, mean, std

    def inverse(self, x, mean, std):  # denormalize
        return (x - self.affine_bias) / self.affine_weight * std + mean
```

The learnable affine parameters let the model adjust normalization per-channel — subtle but helpful for multivariate where channels have different statistical properties.

#### C. Mixture of Linear Experts (MoLE)

Replace the single linear head with a lightweight mixture-of-experts:

```python
# K=4 small linear heads, gated by a learned router
gate = softmax(router(x_pooled))  # [B, K]
forecasts = [head_k(x) for k in range(K)]  # K × [B, T, D]
output = sum(gate[:, k] * forecasts[k] for k in range(K))
```

Each expert is a tiny Linear(d_model → T) — total params is K × d_model × T + router. With K=4, this adds ~16K params but lets the model specialize different experts for different temporal patterns (e.g., one expert for trending sequences, one for seasonal, one for flat).

#### D. AttnRes Across Transformer Layers

We validated AttnRes in Exp 12. With the CI-patch transformer, we'd have a ~72K-param model with 2 layers. AttnRes adds ~128 params (2 query vectors × 64 dims) and lets the output layer selectively attend back to the input embedding or first layer's output. At this model size, the overhead is negligible and the mechanism is validated.

---

## Concrete Experiment Sequence

Based on expected impact and our speed advantage:

### Round 1: The Fix (Critical — Should Run First)

| Exp | Change | Target | Time Est |
|---|---|---|---|
| **16** | CI-Patch + CI-Head (Phases 1+2 combined) | Fix multivariate, target <100K params | ~5 min |
| **17** | Exp 16 + Trend/Residual decomposition (Phase 3) | Add DLinear's inductive bias | ~5 min |

**These two experiments will tell us if the transformer can beat DLinear on multivariate.** If CI-Patch + CI-Head brings the multivariate param count from 1.6M to ~72K — below DLinear's 113K — we should see the overfitting problem disappear.

### Round 2: Tuning (After Round 1 establishes new baseline)

| Exp | Change | Target | Time Est |
|---|---|---|---|
| **18** | Hyperparameter sweep on best Round 1 model | Optimize patch_size, d_model, dropout | ~60 min (30 configs) |
| **19** | Frequency-enhanced dual branch | Capture spectral structure | ~5 min |

### Round 3: Polish (After Round 2 locks hyperparams)

| Exp | Change | Target | Time Est |
|---|---|---|---|
| **20** | AttnRes on transformer layers | Selective depth-wise retrieval | ~5 min |
| **21** | MoLE output head | Expert specialization | ~5 min |
| **22** | RevIN as learnable model layer | Adaptive per-channel normalization | ~5 min |

---

## The Winning Configuration (Prediction)

If our analysis is right, the winning model looks like:

```
Lookback [B, H, D]
  → RevIN normalize (learnable affine)
  → Trend/Residual decomposition (avg-pool k=25)
  → For each of {trend, residual}:
      → Channel-independent patching [B*D, N, P]
      → Shared patch embedding Linear(P → d_model) [B*D, N, d_model]
      → + positional embedding
      → 2-layer TransformerEncoder (d_model=48-64, 4 heads, pre-norm)
      → Temporal projection Linear(N → T) [B*D, d_model, T]
      → Channel projection Linear(d_model → 1) [B*D, T, 1]
      → Reshape [B, T, D]
  → Sum trend + residual forecasts
  → RevIN denormalize
  → Output [B, T, D]
```

**Estimated params:** ~80-120K for any dataset/dimension. Right in DLinear's sweet spot.

**Estimated training time:** ~2-5 min for all 4 benchmarks.

**Why this should win:**
- Channel-independent patching = DLinear's regularization + transformer's temporal attention
- Trend/residual decomposition = proven inductive bias
- Shared transformer = doubles effective data (sees trend and residual)
- ~100K params = right-sized for 10K samples
- Global receptive field via self-attention = captures long-range seasonality DLinear can't
- 20x iteration speed = we can tune every detail

The DLinear baseline has had 15 experiments of advantages. Time to take the crown.

---

## What Success Looks Like

| Benchmark | Current Best | Target | Stretch |
|---|---|---|---|
| ETTh1 Multi | 0.4719 (Exp 2) | **< 0.46** | < 0.44 |
| ETTh1 Uni | 0.2508 (Exp 10) | **< 0.25** | < 0.24 |
| ETTm1 Multi | 0.4194 (Exp 10) | **< 0.41** | < 0.39 |
| ETTm1 Uni | 0.1913 (Exp 7) | **< 0.19** | < 0.18 |

If we hit the targets, we'll have beaten our own baseline on every benchmark AND extended our lead over the paper on ETTh1 Uni. If we hit the stretch goals, we start closing the multivariate gap with the paper itself.

Let's hunt.
