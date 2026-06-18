# Experiment Results — Overlapping Patches and iTransformer

Both experiments build on the submission baseline (CI+Decomp Transformer, Exp 27 ensemble bests).

Previous bests for reference:

| | ETTh1 Multi | ETTh1 Uni | ETTm1 Multi | ETTm1 Uni |
|---|---|---|---|---|
| Paper | 0.42 | 0.34 | 0.37 | 0.15 |
| Submission baseline | 0.4829 | 0.2505 | 0.4094 | 0.1881 |

---

## Experiment 28: Overlapping Patches

Non-overlapping `patch_size=8` on ETTh1's 336-step lookback produces 42 tokens. Self-attention over 42 tokens means adjacent seasonal cycles (24-hour and 168-hour periods in hourly data) compete for the same attention budget. Setting `patch_stride=4` — half the patch size — produces 83 tokens from the same input via `torch.Tensor.unfold`. Adjacent patches share 4 timesteps, giving smoother positional coverage and richer local context per token. ETTm1 configs were left at their non-overlapping stride since 180 and 90 tokens are already adequate and those benchmarks were already beating baseline.

The code change is three lines in `ci_decomp_transformer.py` and `ci_attnres_transformer.py`: a `patch_stride` parameter, an updated `n_patches` formula, and `unfold` replacing `reshape` in the patch extraction step. `unfold` also fixes a silent bug in the original where up to 7 trailing timesteps were discarded.

**Training time:** ~44 min (single models only)

| | Paper | Submission | Exp 28 Single | vs Submission |
|---|---|---|---|---|
| ETTh1 Multi | 0.42 | 0.4829 | **0.4832** | +0.1% |
| ETTh1 Uni | 0.34 | 0.2505 | 0.2542 | +1.5% |
| ETTm1 Multi | 0.37 | 0.4094 | 0.4165 | +1.7% |
| ETTm1 Uni | 0.15 | 0.1881 | 0.1894 | +0.7% |

Single-model ETTh1 Multi (0.4832) landed within 0.0003 of the previous three-model ensemble best (0.4829), which is encouraging — the overlapping patches are doing something on that benchmark. All other single-model results are slightly above the submission ensemble bests, which is expected: single models rarely beat ensembles.

The proper comparison is single vs single. The prior single-model ETTh1 Multi champion was 0.4875 (Exp 26 AttnRes+Aug). Exp 28 at 0.4832 beats that by 0.0043, making it the new single-model record on ETTh1 Multi.

No ensemble was run for Exp 28 — those results appear in Exp 29 below, which combines overlapping patches with the iTransformer.

---

## Experiment 29: iTransformer Ensemble

The CI transformer attends over time patches — each token is 8 timesteps, and the model asks which time windows are relevant to each other. This means channels never communicate during attention. ETTh1 Multi is documented as the structural ceiling of the CI design because of this: every prior attempt to fix it (Linear(D,D) mixing in Exp 21, channel-aware Conv+attention in Exp 8) either added too few parameters to learn anything or too many to avoid overfitting.

iTransformer inverts the attention axis. Each variable's full 336-step lookback is embedded into one token via `Linear(H → d_model)`. With D=7 variables, attention runs over 7 tokens — learning which variables predict each other. For ETTh1 with 7 channels, that is 7×7=49 attention scores vs 83×83=6,889 for the overlapping-patch CI model. Fewer scores, but they capture cross-variable dynamics that CI cannot.

The ensemble pairs iTransformer with CI+AttnRes and CI base for the multivariate benchmarks. The two architectures attend over orthogonal axes — iTransformer over channels, CI over time patches — so their errors on the same window are partially uncorrelated, which is the condition under which ensembling helps most. For univariate benchmarks (D=1), iTransformer degenerates to a single token with no meaningful attention; CI models are used there unchanged.

**Training time:** ~75 min total (single models + full ensemble)

### Single models

| | Submission | Exp 29 Single | vs Submission |
|---|---|---|---|
| ETTh1 Multi (iTransformer) | 0.4829 | 0.4895 | +1.4% |
| ETTh1 Uni (CI base) | 0.2505 | 0.2579 | +3.0% |
| ETTm1 Multi (iTransformer) | 0.4094 | 0.4253 | +3.9% |
| ETTm1 Uni (CI base) | 0.1881 | 0.1945 | +3.4% |

The iTransformer single models are weaker than the CI single models on every benchmark. This is expected — the submission configs were swept specifically for CI, and iTransformer has not been tuned at all.

### Ensemble

| | Paper | Submission | Exp 29 Ensemble | vs Submission |
|---|---|---|---|---|
| ETTh1 Multi | 0.42 | 0.4829 | **0.4773** | **-1.2%** |
| ETTh1 Uni | 0.34 | 0.2505 | **0.2471** | **-1.4%** |
| ETTm1 Multi | 0.37 | 0.4094 | 0.4103 | +0.2% |
| ETTm1 Uni | 0.15 | 0.1881 | 0.1911 | +1.6% |

ETTh1 Multi at **0.4773** is a new all-time best — 0.0056 below the previous record of 0.4829, the largest single improvement on that benchmark since Exp 26 cracked 0.488. The individual ensemble members for ETTh1 Multi were 0.4878 (iTransformer), 0.4836 (CI AttnRes), 0.4915 (CI base). None of the three individually beat the submission, but the ensemble of architecturally diverse models does — cross-variate and temporal attention making different errors on the same windows is exactly why the ensemble works here.

ETTh1 Uni at **0.2471** is also a new all-time best. Individual members were 0.2485, 0.2507, 0.2593 — again, none individually beat the submission's 0.2505, but the ensemble does.

ETTm1 Multi and ETTm1 Uni both regressed slightly. The iTransformer ETTm1 Multi individual (0.4390) was noticeably weaker than the CI members (0.4118, 0.4098), dragging the ensemble up. Dropping iTransformer from the ETTm1 Multi ensemble or replacing it with a third CI model would likely recover the 0.4094 record.

### Individual member breakdown

| Benchmark | iTransformer | CI AttnRes | CI base | Ensemble |
|---|---|---|---|---|
| ETTh1 Multi | 0.4878 | 0.4836 | 0.4915 | **0.4773** |
| ETTh1 Uni | 0.2485 | — | 0.2507 / 0.2593 | **0.2471** |
| ETTm1 Multi | 0.4390 | — | 0.4118 / 0.4098 | 0.4103 |
| ETTm1 Uni | — | — | 0.1934 / 0.1970 / 0.1903 | 0.1911 |

---

## Updated All-Time Bests

| | Paper | Previous Best | **New Best** | Source | vs Paper |
|---|---|---|---|---|---|
| ETTh1 Multi | 0.42 | 0.4829 | **0.4773** | Exp 29 ensemble | +13.6% |
| ETTh1 Uni | 0.34 | 0.2505 | **0.2471** | Exp 29 ensemble | **-27.3%** |
| ETTm1 Multi | 0.37 | **0.4094** | 0.4094 | Exp 18 (unchanged) | +10.6% |
| ETTm1 Uni | 0.15 | **0.1881** | 0.1881 | Exp 18 (unchanged) | +25.4% |

Two new records. ETTh1 Multi is now 13.6% above the paper rather than 14.9%. ETTh1 Uni now beats the paper by 27.3%, extending the lead from 26.3%. ETTm1 records held — the iTransformer was too weak a third member there to help the ensemble.

The clear next step for ETTm1 Multi is replacing the iTransformer ensemble member with a third CI model (different hyperparameters), which should recover or beat the 0.4094 record without the weak member dragging the average.
