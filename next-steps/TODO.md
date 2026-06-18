# Next Steps

Planned follow-up experiments and integration work for the mr-Diff project.

## Conventions

- The submission code under `code/` is frozen. Do all experiment work in a copy
  inside this directory, for example `next-steps/exp01-backbone-alpha-sweep/`.
  Never edit `code/` in place.
- Each TODO states its hypothesis, the exact changes, how to run it, how to log
  results, the decision criteria, and a definition of done. No step should
  require guessing.
- Status markers: `[ ]` not started, `[~]` in progress, `[x]` done.

---

## [ ] TODO 1: Backbone-strength (alpha) sweep, is diffusion inert or just starved?

### Why
Our headline finding is that the diffusion stack changes MAE by less than 0.3%.
That was measured with the DLinear backbone at full strength, so the residual
handed to diffusion is small and noise-like. We tested only the two endpoints:
full backbone (diffusion idle) and no backbone (the faithful-paper config, which
collapsed to predicting zeros). We never swept the middle. This experiment puts
a single scalar knob on how much of the forecast the backbone is responsible
for, so we can tell whether diffusion is fundamentally mismatched to the task or
simply starved of residual signal.

### Hypothesis and what each outcome means
Define the backbone trust scalar `alpha` in `[0, 1]`. Reconstruction at
inference is `forecast_hat = alpha * direct_pred + diffusion_output`, and the
diffusion target during training is the residual `forecast - alpha * direct_pred`.

- `alpha = 1.0` reproduces the current model (small residual).
- `alpha = 0.0` is pure diffusion (residual is the full target).

Outcomes:
- **Diffusion is inert (expected):** the diffusion contribution stays near zero
  and the captured residual is uncorrelated with the true residual at every
  alpha. This upgrades our claim from "diffusion adds nothing at full backbone"
  to "diffusion adds nothing at any backbone strength," which is far stronger.
- **Diffusion is merely starved (possible):** at some intermediate alpha the
  diffusion contribution becomes clearly positive and the captured residual
  correlates with the truth. That identifies a sweet spot where diffusion does
  real work, and we report it.

### The single knob (do not change anything else)
Keep S=3 stages, K=100 training steps, DPM-Solver++ at 20 steps for evaluation,
10 sampled trajectories aggregated by median, dropout 0.3, the same optimizer,
epoch budget, early stopping, and random seed as the frozen baseline. `alpha` is
the only variable.

### Step 1: Make a working copy (respect the freeze)
```bash
cd <repo-root>
cp -r code/baseline next-steps/exp01-backbone-alpha-sweep
cd next-steps/exp01-backbone-alpha-sweep
```

### Step 2: Add the alpha parameter to the model
File: `src/models/mr_diff.py`.

1. Constructor `MRDiff.__init__` (around line 22). Add `backbone_alpha=1.0` to
   the signature and store it. For example, change the last signature line:
   ```python
                num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
   ```
   to:
   ```python
                num_encoder_layers=3, num_decoder_layers=3, dropout=0.1,
                backbone_alpha=1.0):
   ```
   and add near the other `self.*` assignments:
   ```python
   self.backbone_alpha = backbone_alpha
   ```

2. `create_model(config)` in the same file: pass the value through from config:
   ```python
   backbone_alpha=config.get("backbone_alpha", 1.0),
   ```
   (add it to the existing `MRDiff(...)` call).

3. `training_step` (line 94). Change:
   ```python
   residual = forecast - direct_pred.detach()
   ```
   to:
   ```python
   residual = forecast - self.backbone_alpha * direct_pred.detach()
   ```
   Leave `direct_loss = mse_loss(direct_pred, forecast)` unchanged: the backbone
   still learns to predict the full forecast; we only change how much of it is
   subtracted to form the diffusion target.

4. `sample` (lines 172 to 176). Scale the backbone term in both branches, and
   also expose the raw diffusion output so we can measure residual fidelity.
   Change:
   ```python
   direct_pred = self.direct_predict(lookback)
   if num_samples == 1:
       samples = direct_pred + samples
   else:
       samples = direct_pred.unsqueeze(1) + samples
   return samples
   ```
   to:
   ```python
   direct_pred = self.direct_predict(lookback)
   diffusion_only = samples  # raw diffusion output, before adding the backbone
   if num_samples == 1:
       samples = self.backbone_alpha * direct_pred + samples
   else:
       samples = self.backbone_alpha * direct_pred.unsqueeze(1) + samples
   if return_components:
       return samples, direct_pred, diffusion_only
   return samples
   ```
   Add `return_components=False` to the `sample(...)` signature.

### Step 3: Add the config key
File: `configs/small.yaml` (and `small_k20.yaml` if used for evaluation). Add:
```yaml
backbone_alpha: 1.0   # overridden per run by the sweep driver
```

### Step 4: Metrics to capture (all in globally-standardized space)
For each `alpha` and each benchmark, compute three numbers on the test set,
using the existing global-standardization protocol in
`src/evaluation/metrics.py`:

1. `mae_direct`: MAE of `alpha * direct_pred` against the ground-truth forecast
   (diffusion off). Obtain `direct_pred` from `sample(..., return_components=True)`.
2. `mae_full`: MAE of `alpha * direct_pred + diffusion_output` against the
   ground truth (the normal reconstruction).
3. `diffusion_contribution = mae_direct - mae_full`. Positive means diffusion
   helps; near zero means it does nothing; negative means it hurts.

Plus the residual-fidelity check, which is the direct test of the question:
- True residual per window: `r = forecast - alpha * direct_pred`.
- Predicted residual per window: `r_hat = diffusion_only` (median over the 10 samples).
- `residual_r2 = 1 - sum((r - r_hat)^2) / sum((r - mean(r))^2)`, averaged over windows.
- `residual_cosine = mean over windows of cos(r, r_hat)`.
Both computed in standardized space.

### Step 5: Run the sweep
Write `sweep_alpha.py` in the experiment directory that loops over
`alpha in [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]`, for each alpha writes the config,
trains via `train.py`, evaluates with `return_components=True` to get
`mae_direct`, `mae_full`, `residual_r2`, `residual_cosine`, and appends one row
per (alpha, benchmark) to `results_alpha_sweep.csv` with columns:
```
alpha, benchmark, mae_direct, mae_full, diffusion_contribution, residual_r2, residual_cosine
```
Run all four benchmarks. If compute is tight, prioritize ETTh1 Multi (our
hardest, where any diffusion help would matter most) and ETTm1 Uni (our best).

### Step 6: Integrate results into the experiment database
File: `experiment-db/build_db.py`. Add a table and load the CSV:
```sql
CREATE TABLE backbone_alpha_sweep (
    alpha REAL, benchmark TEXT,
    mae_direct REAL, mae_full REAL,
    diffusion_contribution REAL, residual_r2 REAL, residual_cosine REAL,
    PRIMARY KEY (alpha, benchmark)
);
```
Read `next-steps/exp01-backbone-alpha-sweep/results_alpha_sweep.csv` into it,
include it in the JSON dump, then regenerate: `python3 experiment-db/build_db.py`.

### Decision criteria (state the verdict explicitly)
- **Inert:** if `diffusion_contribution <= 0.002` (the noise floor of the
  original ablation) for every alpha, and `residual_r2 <= 0.05` throughout, then
  diffusion is mismatched, not starved.
- **Useful in a regime:** if for some alpha `diffusion_contribution >= 0.005`
  (about 1% relative) with `residual_r2 >= 0.10`, diffusion captures genuine
  residual structure once given enough of it. Report that alpha and its numbers.

### Definition of done
- [ ] `results_alpha_sweep.csv` filled for all swept (alpha, benchmark) pairs.
- [ ] `next-steps/exp01-backbone-alpha-sweep/RESULTS.md` with the results table,
      the diffusion-contribution and residual-fidelity curves versus alpha, and a
      one-paragraph verdict using the criteria above.
- [ ] `experiment-db/` rebuilt with the `backbone_alpha_sweep` table.
- [ ] One paragraph added to `FINAL_REPORT.md` Section 5 (Analysis) and the
      paper's Analysis section reporting the stronger conclusion. If diffusion
      turns out useful at some alpha, also revisit the abstract and conclusion.
- [ ] No em dashes in any added prose. Keep numbers consistent with the DB.

### Estimated compute
A full baseline run over all four benchmarks is about 34 minutes. Six alpha
values is roughly 3.5 hours of training plus evaluation on the RTX 5090. Well
within an afternoon.
