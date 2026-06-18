# Experiment 28: Overlapping Patches

**Change:** Added a `patch_stride` parameter to both model classes. Setting `patch_stride < patch_size` produces overlapping patches via `torch.Tensor.unfold`, doubling the token count for ETTh1 benchmarks (42 → 83 tokens with stride=4 on patch_size=8). ETTm1 configs are left at their current non-overlapping stride since they have rich token counts already and are beating baseline.

**Applied to:** ETTh1 Multi and ETTh1 Uni only. `patch_stride=4` with `patch_size=8` → 50% overlap.

**Not applied to:** ETTm1 Multi (patch=8, stride=8, 180 tokens — already adequate) and ETTm1 Uni (patch=16, stride=16, 90 tokens — already beating baseline by 6.5%).

---

## Motivation

ETTh1 Multi is the only remaining benchmark above the DLinear baseline (+1.8%). The cause is well-documented in Exp 18: all 30 sweep configs cluster between 0.488–0.533 — there is a structural ceiling. The dominant constraint is token count.

With `patch_size=8` and non-overlapping stride, ETTh1's 336-timestep lookback produces only **42 tokens**. Self-attention over 42 tokens means each token's receptive field spans at most 42 positions — but seasonal patterns in hourly data often repeat at 24-hour and 168-hour intervals. With 42 tokens, patterns separated by 24+ time steps (3+ tokens at 8 steps/token) are at the edge of what the positional embedding can reliably distinguish.

Overlapping patches with stride=4 produce **83 tokens** from the same 336-step lookback. Adjacent patches share 4 timesteps, giving richer local context per token and a smoother positional signal. The `Linear(N→T)` temporal projection head still maps 83 → 168, which is a less aggressive compression than 42 → 168.

This is why ETTm1 improved earlier and faster than ETTh1: ETTm1 with patch=8 already had 180 tokens from its 1440-step lookback, providing ample coverage. ETTh1 was starved.

---

## Implementation

The change is three lines in each model class.

**`n_patches` calculation** (was `lookback_length // patch_size`):
```python
self.patch_stride = patch_stride if patch_stride is not None else patch_size
self.n_patches = (lookback_length - patch_size) // self.patch_stride + 1
```

**`_encode_and_project`** (was `reshape`):
```python
# old:
x = x_1d[:, :self.n_patches * self.patch_size]
x = x.reshape(-1, self.n_patches, self.patch_size)

# new:
x = x_1d.unfold(-1, self.patch_size, self.patch_stride)
```

`unfold(dimension, size, step)` slides a window of `patch_size` timesteps with `patch_stride` step, returning `[B*D, n_patches, patch_size]` directly — same shape as before, no downstream changes needed. `pos_embed`, `trend_temporal`, `resid_temporal`, and `head_norm` all depend on `n_patches`, which is recalculated correctly.

`patch_stride=None` (the default) falls back to `patch_size`, preserving the original non-overlapping behaviour identically. ETTm1 configs remain unchanged.

---

## Files changed

| File | Change |
|---|---|
| `src/models/ci_decomp_transformer.py` | Added `patch_stride` param; updated `n_patches`; `unfold` in `_encode_and_project` |
| `src/models/ci_attnres_transformer.py` | Same three changes |
| `train_single.py` | Added `"patch_stride": 4` to ETTh1 Multi and ETTh1 Uni configs; `create_model` reads and forwards it |
| `train_ensemble.py` | Added `"patch_stride": 4` (or `8` for patch=16) to all 6 ETTh1 ensemble models; `create_model` updated |

---

## Expected impact

ETTh1 Multi is the primary target. The architecture ceiling in Exp 18 was attributed specifically to the short token count — this directly addresses that. ETTh1 Uni is also getting the change since it uses the same patch=8 config and is currently only -1.2% vs baseline (room to improve further or at least not regress).

The token count increase from 42→83 roughly doubles the attention compute per batch on ETTh1. Training time will increase moderately (~15-25% on ETTh1). ETTm1 training time is unchanged.

Parameter count increases slightly: `trend_temporal` goes from `Linear(42→168)` to `Linear(83→168)`, adding ~(83-42)×168 = 6,888 params. Negligible.

---

## Relationship to prior experiments

Exp 9 tried a PatchTST-style conditioning encoder with global self-attention and regressed on multivariate (+14%). That was a different architecture in a different role (history encoder feeding into diffusion conditioning) with a different failure mode (conflating channels before attention). This change is a targeted token count increase on the already-validated CI+Decomp Transformer architecture, leaving everything else identical.

The `unfold` approach is also cleaner than the prior `reshape` — `reshape` silently truncated the lookback to `n_patches * patch_size` (line: `x = x_1d[:, :self.n_patches * self.patch_size]`), discarding up to `patch_size - 1 = 7` timesteps at the end. `unfold` uses the full sequence.
