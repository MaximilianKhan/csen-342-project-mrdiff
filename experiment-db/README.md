# Experiment database

A structured record of the full mr-Diff campaign for fast lookup, ablation
comparison, and state-space exploration.

## Files

| File | Purpose |
|------|---------|
| `build_db.py` | Source of truth. Holds every documented result inline and generates the two artifacts below. |
| `experiments.db` | SQLite database. Query it directly. |
| `experiments.json` | Human-readable, diffable dump of the same data. |

Regenerate after editing the data: `python3 experiment-db/build_db.py`.

All MAE/MSE values are on globally-standardized data (the paper's metric space),
transcribed from `code/ALL_EXPERIMENT_RESULTS.md`.

## Schema

```
benchmarks(benchmark, dataset, mode, lookback, horizon, n_vars, paper_mae, baseline_mae)
experiments(exp_id, exp_num, name, category, era, builds_on,
            params_min_k, params_max_k, train_time_min, verdict, change_summary)
results(exp_id, benchmark, variant, mae, mse, is_final_best, was_record)
ensemble_members(exp_id, benchmark, member_idx, architecture, params_k, mae)
sweep_best_configs(benchmark, config_name, patch_size, d_model, num_layers,
                   dim_ff, dropout, trend_kernel, lr, weight_decay, params_k, mae)
```

- `variant` is `single`, `ensemble`, or `reported` (paper). Ensemble experiments
  (27, 29, 31) carry both a `single` and an `ensemble` row per benchmark.
- `is_final_best` flags the four canonical headline results (the numbers in the
  report's Table 7). `was_record` flags any result that was an all-time best when
  it was produced.
- `category`: `reference`, `baseline`, `diffusion`, `transformer`, `ensemble`,
  `baseline_variant`. `era`: `diffusion` or `transformer`.

### Views

- `v_results` joins each result to its paper and baseline MAE and adds
  `vs_paper_pct` and `vs_baseline_pct`.
- `v_final_bests` is the four canonical headline results.
- `v_best_per_benchmark` is the true minimum MAE per benchmark across every run.

## Example queries

```sql
-- The four headline results (report Table 7)
SELECT benchmark, exp_id, variant, mae FROM v_final_bests ORDER BY benchmark;

-- True best MAE ever recorded per benchmark (not just the presented architecture)
SELECT * FROM v_best_per_benchmark;

-- Diffusion era vs transformer era: best result on each benchmark
SELECT era, benchmark, MIN(mae) AS best
FROM results r JOIN experiments e ON e.exp_id = r.exp_id
WHERE era IN ('diffusion','transformer')
GROUP BY era, benchmark ORDER BY benchmark, era;

-- Every result on the hardest benchmark, ranked, with deltas
SELECT exp_id, name, variant, mae, vs_paper_pct, vs_baseline_pct
FROM v_results WHERE benchmark = 'ETTh1_Multi' ORDER BY mae;

-- Ablation: what did each rejected diffusion change cost on ETTh1 Multi?
SELECT e.exp_num, e.name, r.mae,
       ROUND(r.mae - 0.4719, 4) AS vs_exp2
FROM results r JOIN experiments e ON e.exp_id = r.exp_id
WHERE r.benchmark = 'ETTh1_Multi' AND e.era = 'diffusion' AND e.exp_num >= 1
ORDER BY r.mae;

-- Ensemble member breakdown for the iTransformer ensemble (Exp 29)
SELECT benchmark, architecture, params_k, mae
FROM ensemble_members WHERE exp_id = '29' ORDER BY benchmark, mae;

-- Parameter efficiency: best MAE per benchmark under 100K params
SELECT r.benchmark, MIN(r.mae) AS best_mae, e.params_max_k
FROM results r JOIN experiments e ON e.exp_id = r.exp_id
WHERE e.params_max_k <= 100 GROUP BY r.benchmark;

-- The winning sweep config for each benchmark
SELECT benchmark, config_name, patch_size, d_model, num_layers, trend_kernel, lr, mae
FROM sweep_best_configs ORDER BY benchmark;
```

## A note the data makes visible

On ETTh1 Multi, the diffusion experiment Exp 2 (self-conditioning) reached
MAE 0.4719, which is lower than the presented headline of 0.4773 (Exp 29
transformer ensemble). The headline reflects the final transformer architecture
we present as the contribution; Exp 2's lower number on this one benchmark is
preserved here faithfully and is easy to surface with `v_best_per_benchmark`.
