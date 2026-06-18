# CSEN-342 Final Project: Multi-Resolution Diffusion Models for Time Series Forecasting

**Santa Clara University, Winter 2026**

Baseline replication and improvement of "Multi-Resolution Diffusion Models for Time Series Forecasting" (mr-Diff, ICLR 2024).

## Directory Structure

```
code/
├── requirements.txt             # Python dependencies
├── download_data.py             # ETT dataset downloader
│
├── baseline/                    # mr-Diff replication (843K params)
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation with DDPM/DPM-Solver sampling
│   ├── run_experiments.py       # Batch experiment runner
│   ├── dpm_solver_pp.py         # DPM-Solver++ implementation
│   ├── configs/                 # YAML configs (default, small, small_k20)
│   ├── src/                     # Model source code
│   │   ├── data/                # Dataset loading and preprocessing (RevIN)
│   │   └── models/              # mr-Diff: diffusion, denoising, conditioning
│   ├── exp_adaln/               # AdaLN experiment (Karthik)
│   │   ├── configs/             # AdaLN-specific configs
│   │   └── src/models/          # Modified models with adaptive layer norm
│   └── exp_ant/                 # ANT experiment (Karthik)
│       ├── configs/             # ANT-specific configs
│       └── src/models/          # Modified models with adaptive noise schedule
│
├── improvement/                 # All improvement architectures
│   ├── train_single.py          # CI+Decomp Transformer trainer
│   ├── train_single_baseline.py # CI+Decomp baseline (no overlap)
│   ├── train_single_overlapping_patches.py  # Overlapping patches variant
│   ├── train_single_i.py        # iTransformer trainer
│   ├── train_single_split.py    # Split-head variant
│   ├── train_single_2cale_decomp.py  # Two-scale decomposition
│   ├── train_ensemble.py        # Heterogeneous ensemble (3 models)
│   ├── train_ensemble_baseline.py
│   ├── train_ensemble_i.py
│   ├── train_ensemble_overlapping_patches.py
│   ├── train_ensemble_twoscale.py
│   ├── sweep.py                 # 30-config hyperparameter sweep
│   ├── sweep_shard.py           # Distributed sweep shard runner
│   ├── configs/                 # Improvement configs
│   └── src/
│       ├── data/                # Shared dataset module
│       ├── evaluation/          # Metrics (MAE, MSE, RMSE, MAPE)
│       ├── training/            # Trainer class, schedulers, early stopping
│       ├── utils/               # Logging and visualization utilities
│       └── models/              # All model architectures
│           ├── ci_decomp_transformer.py           # CI+Decomp (Exp 14-17)
│           ├── ci_decomp_transformer_baseline.py  # Without overlapping patches
│           ├── ci_decomp_transformer_overlapping_patches.py  # 50% overlap
│           ├── ci_decomp_transformer_split.py     # Split trend/resid heads
│           ├── ci_attnres_transformer.py          # CI+Decomp+AttnRes (Exp 26)
│           ├── ci_attnres_transformer_baseline.py # AttnRes without overlap
│           ├── ci_attnres_transformer_overlapping_patches.py  # AttnRes+overlap
│           ├── ci_twoscale_transformer.py         # Two-scale decomposition
│           └── itransformer.py                    # Inverted transformer
│
├── ALL_EXPERIMENT_RESULTS.md    # Full experiment log: technical changes, Exps 1-31, AdaLN/ANT
│
└── slurm/                       # SLURM job scripts (SCU HPC)
    ├── job_2scale.sh
    ├── job_overlapping_patches+itransformer.sh
    └── job_overlapping_patches+itransformer+separate.sh
```

## Key Results

MAE on globally-standardized data (the paper's metric space). Lower is better.

| Benchmark | Paper MAE | Baseline MAE (mr-Diff) | Best MAE | Best architecture |
|-----------|----------:|-----------------------:|---------:|-------------------|
| ETTh1 Multi | 0.42 | 0.4744 | **0.4773** | Exp 29 iTransformer ensemble |
| ETTh1 Uni   | 0.34 | 0.2535 | **0.2471** | Exp 29 iTransformer ensemble |
| ETTm1 Multi | 0.37 | 0.4204 | **0.4081** | Exp 31 two-scale ensemble |
| ETTm1 Uni   | 0.15 | 0.2011 | **0.1865** | Exp 31 two-scale single |

The baseline is our mr-Diff replication (843K params). Best results use 54–182K-parameter
transformers with no diffusion, and beat the baseline on 3 of 4 benchmarks (and the paper on
ETTh1 Uni by 27%). See [`ALL_EXPERIMENT_RESULTS.md`](ALL_EXPERIMENT_RESULTS.md) for the full
log behind every number.

## Quick Start

```bash
pip install -r requirements.txt
python download_data.py

# Run baseline
cd baseline && python train.py --config configs/default.yaml

# Run best improvement (CI+Decomp+AttnRes with overlapping patches)
cd improvement && python train_single_overlapping_patches.py
```

## Experiment Progression

1. **Baseline technical changes**: six deviations from the paper required to get a model that
   trains at all (downsizing 17.5M→843K params, global-std metrics, DLinear backbone,
   GroupNorm, residual decomposition, random-projection mixup). Working baseline MAE 0.47–0.20.
2. **Exps 1–13**: diffusion-focused attempts (self-conditioning, v-prediction, cosine/ANT
   schedules, contrastive loss, MG-TSD, channel-aware denoising, deep AttnRes backbones).
   All confirm diffusion is cosmetic: it changes MAE by < 0.3%.
3. **Exps 14–17**: CI+Decomp Transformer replaces diffusion (MAE 0.41–0.55). First all-time
   records on ETTm1 Uni (0.1885) and ETTm1 Multi (0.4159).
4. **Exp 18**: 30-config hyperparameter sweep, sets 3 of 4 benchmark records.
5. **Exps 19–27**: refinements (extended training, channel mixing, augmentation, frequency
   branch), Attention Residuals integration (Exp 26), and heterogeneous 3-model ensembles (Exp 27).
6. **Exps 28–29**: overlapping patches and an iTransformer ensemble (cross-variate attention).
   New records on both ETTh1 benchmarks (0.4773 Multi, 0.2471 Uni).
7. **Exp 31**: two-scale decomposition aligned to the daily cycle in ETTm1. New records on
   ETTm1 Multi (0.4081) and ETTm1 Uni (0.1865).
8. **AdaLN / ANT**: two baseline-diffusion variants explored in parallel (`baseline/exp_adaln`,
   `baseline/exp_ant`).
