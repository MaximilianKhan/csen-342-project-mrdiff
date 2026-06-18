# mr-Diff: Replication and Improvement (CSEN-342)

**Maximilian Khan & Karthik Tamiledu — Santa Clara University, Winter 2026**

Replication and improvement of **"Multi-Resolution Diffusion Models for Time Series
Forecasting"** (mr-Diff, ICLR 2024) by Lifeng Shen, Weiyu Chen, and James T. Kwok,
evaluated on ETTh1 and ETTm1 in both univariate and multivariate settings.

---

## TL;DR

We re-implemented mr-Diff from scratch (no code was released by the authors), reproduced
its baseline, and then ran a 31-experiment improvement campaign. The campaign's central
finding is that **the diffusion component contributes less than 0.3%** to forecast accuracy
on these datasets — a DLinear backbone does all the work. That led us to replace the
843K-parameter diffusion pipeline with a **Channel-Independent Decomposed Patch Transformer**
(54–182K parameters) that trains in minutes, runs in a single forward pass, and beats the
baseline on 3 of 4 benchmarks. Full numbers, tables, and analysis live in
**[`FINAL_REPORT.md`](FINAL_REPORT.md)** and
**[`final-final-form/ALL_EXPERIMENT_RESULTS.md`](final-final-form/ALL_EXPERIMENT_RESULTS.md)**.

## Headline results

MAE on globally-standardized data (the paper's metric space). Lower is better.

| Benchmark | Paper | Our baseline (mr-Diff) | **Our best** | Best architecture | vs Paper |
|-----------|------:|-----------------------:|-------------:|-------------------|---------:|
| ETTh1 Multi | 0.42 | 0.4744 | **0.4773** | Exp 29 iTransformer ensemble | +13.6% |
| ETTh1 Uni   | 0.34 | 0.2535 | **0.2471** | Exp 29 iTransformer ensemble | **−27.3%** |
| ETTm1 Multi | 0.37 | 0.4204 | **0.4081** | Exp 31 two-scale ensemble | +10.3% |
| ETTm1 Uni   | 0.15 | 0.2011 | **0.1865** | Exp 31 two-scale single | +24.3% |

Best results beat the DLinear baseline on 3 of 4 benchmarks and beat the paper's reported
result on ETTh1 Univariate by 27%. All with 54–182K-parameter transformers, no diffusion,
trained in minutes. See **[`FINAL_REPORT.md`](FINAL_REPORT.md) Table 7** for the authoritative
breakdown.

---

## Repository layout

| Path | What it is |
|------|------------|
| **[`FINAL_REPORT.md`](FINAL_REPORT.md)** | The final project report — abstract, design, all experiments, result tables, references. **Start here.** |
| **`final-final-form/`** | Canonical, runnable code submission (see below). |
| **`final-final-form/ALL_EXPERIMENT_RESULTS.md`** | The complete experiment log: baseline technical changes, Experiments 1–31, and the AdaLN/ANT variants. The source of truth for every number. |
| `graphs-and-charts/` | Report figures and architecture diagrams, plus `generate_all.py` to regenerate them. |
| `karthic-interim-report.pdf` | The earlier interim report. |
| `ICLR-2024-...Paper-Conference.pdf` | The original mr-Diff paper we replicated. |
| `description.txt` | The assignment specification. |
| `dataset-link.txt` | Link to the ETT dataset. |
| `archive/` | Project history — superseded code, intermediate submission snapshots, and working notes. Kept for provenance; not needed to run or read anything current. |

### Inside `final-final-form/`

```
final-final-form/
├── requirements.txt             # Python dependencies
├── download_data.py             # ETT dataset downloader
├── ALL_EXPERIMENT_RESULTS.md    # Full experiment log (Exps 1–31 + AdaLN/ANT)
├── baseline/                    # mr-Diff replication (843K params)
│   ├── train.py / evaluate.py / run_experiments.py
│   ├── dpm_solver_pp.py         # DPM-Solver++ fast sampler
│   ├── configs/  src/           # YAML configs + model source
│   └── exp_adaln/  exp_ant/     # AdaLN and ANT baseline variants
├── improvement/                 # All transformer architectures
│   ├── train_single*.py         # CI+Decomp, overlapping-patch, iTransformer, split, two-scale
│   ├── train_ensemble*.py       # Heterogeneous 3-model ensembles
│   ├── sweep.py / sweep_shard.py # 30-config hyperparameter sweep
│   └── src/models/              # ci_decomp_*, ci_attnres_*, ci_twoscale_*, itransformer
└── slurm/                       # SLURM job scripts (SCU HPC)
```

## Quick start

```bash
cd final-final-form
pip install -r requirements.txt
python download_data.py

# Baseline mr-Diff replication
cd baseline && python train.py --config configs/default.yaml

# Best improvement (CI+Decomp+AttnRes with overlapping patches)
cd ../improvement && python train_single_overlapping_patches.py
```

## Experiment progression at a glance

1. **Baseline technical changes** — six deviations from the paper required to get a working
   model (model downsizing, global-std metrics, DLinear backbone, GroupNorm, residual
   decomposition, random-projection mixup).
2. **Exps 1–13** — diffusion-focused attempts; all confirm diffusion is cosmetic (< 0.3%).
3. **Exps 14–17** — CI+Decomp Transformer replaces diffusion; first records fall.
4. **Exp 18** — 30-config hyperparameter sweep; 3 of 4 records set.
5. **Exps 19–27** — refinements, AttnRes integration, and heterogeneous ensembles.
6. **Exps 28–29** — overlapping patches and an iTransformer ensemble (cross-variate attention).
7. **Exp 31** — two-scale decomposition aligned to the daily cycle in ETTm1.
8. **AdaLN / ANT** — two baseline-diffusion variants explored in parallel.

## `archive/` contents

- `intermediate-submissions/` — `submission/`, `submission_itransformer/`, `submission_twoscale/` (superseded snapshots; their `REPORT.md` files were byte-identical).
- `working-dirs/` — `final-form/` (per-experiment working dir) and `analysis/`.
- `root-baseline-code/` — original root-level baseline code now consolidated into `final-final-form/baseline/`.
- `scripts/` — one-off diagnostics, figure/table generators, and `boom.ipynb`.
- `intermediate-docs/` — working notes (`TECHNICAL_CHANGES.md`, `IMPROVEMENTS.md`, `DEADEND_REALITY.md`) and the old baseline-only README.
