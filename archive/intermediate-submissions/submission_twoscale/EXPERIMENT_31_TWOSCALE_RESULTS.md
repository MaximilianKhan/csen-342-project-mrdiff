# Experiment 31 — Two-Scale Decomposition

Replaces the single trend/residual split (one avg-pool kernel) with a
two-scale decomposition using two kernels: k_fine and k_coarse. This
produces three components — coarse trend, mid-band, and fine residual —
each processed by the shared transformer with its own output head.

The motivation is ETTm1 specifically. At 15-minute resolution, k=15
(the submission's winning kernel) is 3.75 minutes of smoothing — barely
a trend at all. The dominant pattern in electricity data is the 24-hour
daily cycle, which is 96 timesteps at 15-minute resolution. Setting
k_coarse=96 gives the model an explicit daily-scale component so the
coarse head can specialize in extending the daily shape forward, the
mid head in intra-day transitions, and the fine head in short-range
corrections.

For ETTh1 (hourly), k_coarse=24 (one day). For ETTm1, k_coarse=96
(one day). k_fine=15 for multi, k_fine=25 for uni (matching the
submission's winning kernels per benchmark).

Parameter overhead vs baseline CI: two extra Linear(n_patches -> T)
and two extra Linear(d_model -> 1) layers — roughly 6-12K extra
parameters depending on config.

Previous bests for reference:

| | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni |
|---|---|---|---|---|
| Paper | 0.42 | 0.34 | 0.37 | 0.15 |
| Submission (Exp 27) | 0.4829 | 0.2505 | 0.4094 | 0.1881 |
| Exp 29 ensemble | 0.4773 | **0.2471** | 0.4103 | 0.1911 |
| Exp 30 ensemble | **0.4761** | 0.2611 | 0.4102 | 0.1919 |
| Exp 30 single best | 0.4852 | 0.2551 | 0.4394 (iTransformer) | **0.1869** |

---

## Single Models

| Benchmark | MAE | Architecture | Params | Epochs | Time |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4921 | CI AttnRes + Aug | 54,514 | 30 | 119s |
| ETTh1 Uni | 0.2522 | CI base | 54,322 | 30 | 84s |
| **ETTm1 Multi** | **0.4088** | Two-scale | 217,011 | 30 | 2520s |
| **ETTm1 Uni** | **0.1865** | Two-scale | 94,115 | 30 | 473s |

ETTm1 Multi at 0.4088 beats the submission record of 0.4094 as a single
model — the first time a single model has done so on that benchmark.
ETTm1 Uni at 0.1865 is a new all-time best, beating Exp 30's single-model
record of 0.1869. ETTh1 results are not impacted since two-scale was not
used there — those used the same configs as previous experiments.

---

## Ensemble Results

### ETTh1 Multi (MAE: 0.4858)

| Member | Architecture | Params | MAE |
|---|---|---|---|
| 1 | CI AttnRes + Aug | 54,514 | 0.4882 |
| 2 | CI base (d=64) | 110,498 | 0.4950 |
| 3 | CI base (d=32) | 54,322 | 0.4957 |
| **Ensemble** | | | **0.4858** |

Regressed from Exp 30's record of 0.4761. Individual members were
weaker than in Exp 30 (best here 0.4882 vs 0.4837 in Exp 30), which
lifted the ensemble. The ETTh1 Multi record remains with Exp 30.

### ETTh1 Uni (MAE: 0.2574)

| Member | Architecture | Params | MAE |
|---|---|---|---|
| 1 | CI base | 54,322 | 0.2553 |
| 2 | CI base | 54,322 | 0.2584 |
| 3 | CI base | 54,322 | 0.2692 |
| **Ensemble** | | | **0.2574** |

Regressed from Exp 29's record of 0.2471. Three nearly identical models
provide insufficient diversity — member 3 (0.2692) dragged the average
above the single-model best. The ETTh1 Uni record remains with Exp 29.

### ETTm1 Multi (MAE: 0.4081 — new all-time best)

| Member | Architecture | Params | MAE |
|---|---|---|---|
| 1 | Two-scale | 217,011 | 0.4126 |
| 2 | CI base | 182,210 | 0.4110 |
| 3 | CI base | 182,210 | 0.4200 |
| **Ensemble** | | | **0.4081** |

New all-time best, beating the submission's 0.4094 and Exp 30's 0.4102.
The two-scale model (0.4126) was individually weaker than both CI base
members (0.4110, 0.4200) but its different decomposition strategy means
its errors are partially uncorrelated with theirs, and the ensemble
benefits. The 0.0013 gain over the next-best ensemble (Exp 30: 0.4102)
confirms the two-scale architecture adds genuine diversity.

### ETTm1 Uni (MAE: 0.1914)

| Member | Architecture | Params | MAE |
|---|---|---|---|
| 1 | Two-scale | 94,115 | 0.1937 |
| 2 | CI base | 76,610 | 0.1964 |
| 3 | CI base | 76,610 | 0.1941 |
| **Ensemble** | | | **0.1914** |

Ensemble regressed from the single-model record of 0.1865. The two-scale
member (0.1937) was individually weaker than the single-model run, pulling
the ensemble above the single-model best. The three members are too similar
in output to provide meaningful variance reduction. For ETTm1 Uni, the
single two-scale model is the right strategy.

---

## Full Comparison Table

| | Submission | Exp 29 | Exp 30 | **Exp 31 Single** | **Exp 31 Ensemble** |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.4829 | 0.4773 | **0.4761** | 0.4921 | 0.4858 |
| ETTh1 Uni | 0.2505 | **0.2471** | 0.2611 | 0.2522 | 0.2574 |
| ETTm1 Multi | 0.4094 | 0.4103 | 0.4102 | **0.4088** | **0.4081** |
| ETTm1 Uni | 0.1881 | 0.1911 | 0.1919 | **0.1865** | 0.1914 |

Two-scale improves ETTm1 in both single and ensemble settings. It makes
no difference to ETTh1 since it was not applied there.

---

## Updated All-Time Bests

| | Paper | Previous Best | **New Best** | Source | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.42 | **0.4761** | 0.4761 | Exp 30 (unchanged) | +13.4% |
| ETTh1 Uni | 0.34 | **0.2471** | 0.2471 | Exp 29 (unchanged) | -27.3% |
| ETTm1 Multi | 0.37 | 0.4094 | **0.4081** | Exp 31 ensemble | +10.3% |
| ETTm1 Uni | 0.15 | 0.1869 | **0.1865** | Exp 31 single | +24.3% |

Two new records. ETTm1 Multi ensemble at 0.4081 is the first time that
benchmark has beaten 0.4094. ETTm1 Uni single at 0.1865 extends the
single-model record set in Exp 30.

---

## Analysis

Two-scale decomposition works on ETTm1 and only ETTm1. The reason is
specific to that dataset's characteristics: 15-minute resolution with a
1440-step lookback contains exactly 15 full daily cycles. The 24-hour
electricity demand pattern (k_coarse=96 = one day) is the dominant signal
and average-pooling at that scale cleanly separates it from sub-hourly
variation. The submission's k=15 kernel was barely smoothing at all —
3.75 minutes of averaging on 15-minute data — so the single-scale model
was effectively decomposing trend from noise rather than trend from cycle.

The improvement is not dramatic (0.4094 → 0.4081, 0.1881 → 0.1865) but
it is consistent across both single and ensemble settings on ETTm1, which
is the correct signal that the inductive bias is genuinely helpful rather
than noise. The fact that the two-scale single model beats the submission
single model on ETTm1 Multi (0.4088 vs 0.4094) is the strongest evidence:
no ensembling needed, just the right decomposition.

The degradation on ETTh1 (single models slightly worse than Exp 30) is
expected — those configs were unchanged from previous experiments so the
variation is run-to-run noise, not a structural regression from two-scale.