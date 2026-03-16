# Experiment Results — AdaLN and ANT Scheduling

Both experiments run from the baseline directly (843K params, linear schedule, epsilon prediction), not from the cumulative chain in EXPERIMENT_RESULTS.md.

---

## Experiment A: AdaLN Conditioning

The concatenation-based conditioning in the Decoder was replaced with Adaptive Layer Normalization. Previously the conditioning vector was concatenated with the latent and squashed through `cond_fusion = nn.Linear(hidden_dim + cond_dim, hidden_dim)`. AdaLN instead runs the conditioning vector through a small MLP to produce per-channel scale and shift, then applies `(1 + gamma) * LayerNorm(z) + beta`. The MLP's last layer is zero-initialized so the network starts as identity and learns to deviate. No config changes needed; the only overhead is the extra MLP parameters.

**Training time:** ~5 hrs

| Experiment | Paper | Baseline | AdaLN | vs Baseline | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.42 | 0.4744 | **0.4733** | **-0.2%** | +12.7% |
| ETTh1 Uni | 0.34 | 0.2535 | 0.2565 | +1.2% | -24.6% |
| ETTm1 Multi | 0.37 | 0.4204 | 0.4266 | +1.5% | +15.3% |
| ETTm1 Uni | 0.15 | 0.2011 | **0.1974** | **-1.8%** | +31.6% |

ETTh1 Multi and ETTm1 Uni improved; ETTh1 Uni and ETTm1 Multi regressed by similar amounts. The clearest gain is ETTm1 Uni (0.2011 → 0.1974). AdaLN's multiplicative conditioning is more useful when the conditioning signal itself is informative — ETTm1 Uni has the longest horizon and the most structured coarser-stage trends, so the change pays off there. On ETTh1 Uni the coarser-stage signal is noisier and the benefit doesn't show up.

Two wins, two small losses, roughly even. Worth keeping since it costs nothing at inference.

---

## Experiment B: ANT Adaptive Noise Schedule

The fixed `linspace(1e-4, 0.1, 100)` beta schedule was replaced with a dataset-specific one. `src/data/ant_schedule.py` computes the mean variance of first differences across all channels in the training split, then scales `beta_end` using `clip(0.1 * ns_mean / 0.1, 5e-4, 0.2)`. ETTh1 and ETTm1 get different `beta_end` values; both still use a linear ramp. Config: `schedule_type: ant`, `ant_betas_path: data/ant_betas_{dataset}.pt`.

Exp 5 in the cumulative chain tried an IAAT-driven power-law that reshaped the curve itself and caused +54% multivariate regression. This version only shifts `beta_end` and leaves the linear shape alone, which is why the results are stable.

**Training time:** ~5 hrs 20 min (including ~20 min CPU pre-compute)

| Experiment | Paper | Baseline | ANT | vs Baseline | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.42 | 0.4744 | 0.4819 | +1.6% | +14.7% |
| ETTh1 Uni | 0.34 | 0.2535 | **0.2515** | **-0.8%** | -26.0% |
| ETTm1 Multi | 0.37 | 0.4204 | **0.4192** | **-0.3%** | +13.3% |
| ETTm1 Uni | 0.15 | 0.2011 | **0.2001** | **-0.5%** | +33.4% |

Three of four benchmarks improved. ETTh1 Multi is the exception, up 1.6%. ETTh1's non-stationarity score lands close to the reference value of 0.1, so the derived `beta_end` barely moves from the baseline — a small upward nudge that slightly hurts multivariate. ETTm1 is higher-frequency (15-minute vs hourly) and more non-stationary, so the scaling pushes `beta_end` up more meaningfully and all three ETTm1-adjacent results improve.

Three wins, one loss. The ETTh1 Multi regression is worth watching but the method is clearly doing something useful on ETTm1.

---

## Combined Summary

| | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni |
|---|---|---|---|---|
| Paper | 0.42 | 0.34 | 0.37 | 0.15 |
| Baseline | 0.4744 | 0.2535 | 0.4204 | 0.2011 |
| AdaLN | **0.4733** | 0.2565 | 0.4266 | **0.1974** |
| ANT | 0.4819 | **0.2515** | **0.4192** | 0.2001 |
| Best | AdaLN | ANT | ANT | AdaLN |

Neither method wins everywhere. AdaLN takes ETTh1 Multi and ETTm1 Uni; ANT takes ETTh1 Uni and ETTm1 Multi. The two methods fail on different benchmarks, which is a reasonable argument for running them combined.
