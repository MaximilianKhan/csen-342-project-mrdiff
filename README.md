# mr-Diff: Replication and Improvement (CSEN-342)

**Maximilian Khan & Karthik Tamiledu — Santa Clara University, Winter 2026**

Replication and improvement of **"Multi-Resolution Diffusion Models for Time Series
Forecasting"** (mr-Diff, ICLR 2024) by Shen, Chen, and Kwok, evaluated on ETTh1 and ETTm1
(univariate + multivariate).

## TL;DR

We replicated mr-Diff from scratch, found through a 27-experiment campaign that the
diffusion component contributes **< 0.3%** to forecast accuracy (the DLinear backbone does
the work), and replaced the 843K-parameter diffusion pipeline with a **Channel-Independent
Decomposed Patch Transformer** (54–182K params) that trains in minutes and beats the
baseline on 3 of 4 benchmarks. Full numbers, tables, and analysis are in
**[`FINAL_REPORT.md`](FINAL_REPORT.md)** — the authoritative source for all results.

## Repository layout

| Path | What it is |
|------|------------|
| **[`FINAL_REPORT.md`](FINAL_REPORT.md)** | The final project report — abstract, design, all experiments, result tables, references. **Start here.** |
| **`final-final-form/`** | Canonical, consolidated code submission. `baseline/` (mr-Diff replication), `improvement/` (all transformer variants + ensembles + sweep), `slurm/`, and `ALL_EXPERIMENT_RESULTS.md` (full Exp 1–31 log). |
| `graphs-and-charts/` | Report figures and diagrams + `generate_all.py` to regenerate them. |
| `karthic-interim-report.pdf` | Earlier interim report. |
| `ICLR-2024-...Paper-Conference.pdf` | The original mr-Diff paper we replicated. |
| `description.txt` | The assignment specification. |
| `dataset-link.txt` | Link to the ETT dataset. |
| `archive/` | Project history — superseded code, intermediate submission snapshots, and working-notes docs. Kept for provenance; not needed to run anything. |

## Quick start

All runnable code lives in `final-final-form/`:

```bash
cd final-final-form
pip install -r requirements.txt
python download_data.py

# Baseline mr-Diff replication
cd baseline && python train.py --config configs/default.yaml

# Best improvement (CI+Decomp+AttnRes, overlapping patches)
cd ../improvement && python train_single_overlapping_patches.py
```

## Results summary

Best results beat the DLinear baseline on 3 of 4 benchmarks and beat the paper on ETTh1
Univariate (0.2471 vs 0.34). See **[`FINAL_REPORT.md`](FINAL_REPORT.md) Table 7** for the
complete final numbers and per-architecture breakdowns.

## `archive/` contents

- `intermediate-submissions/` — `submission/`, `submission_itransformer/`, `submission_twoscale/` (superseded snapshots; `REPORT.md` files were identical across them).
- `working-dirs/` — `final-form/` (per-experiment working dir) and `analysis/`.
- `root-baseline-code/` — original root-level baseline code now consolidated into `final-final-form/baseline/` (`src/`, `configs/`, `slurm/`, `train.py`, `evaluate.py`, etc.).
- `scripts/` — one-off diagnostics, figure/table generators, and `boom.ipynb`.
- `intermediate-docs/` — working notes (`TECHNICAL_CHANGES.md`, `IMPROVEMENTS.md`, `DEADEND_REALITY.md`) and the old baseline-only README.
