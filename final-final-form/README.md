# CSEN-342 Final Project: Multi-Resolution Diffusion Models for Time Series Forecasting

**Santa Clara University, Winter 2026

Baseline replication and improvement of "Multi-Resolution Diffusion Models for Time Series Forecasting" (mr-Diff, ICLR 2024).

## Directory Structure

```
final-final-form/
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

| Benchmark | Paper MAE | Baseline MAE | Best MAE | Model |
|-----------|-----------|-------------|----------|-------|
| ETTh1 Multi | 0.422 | 0.922 | 0.379 | CI+Decomp+AttnRes+Overlap |
| ETTh1 Uni | 0.340 | 1.007 | 0.488 | Ensemble (3 models) |
| ETTm1 Multi | 0.373 | 0.972 | 0.299 | CI+Decomp+AttnRes+Overlap |
| ETTm1 Uni | 0.149 | 0.963 | 0.307 | CI+Decomp+AttnRes+Overlap |

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

1. **Exps 1-5**: Baseline mr-Diff replication (MAE 0.92-1.01, 2-6x gap)
2. **Exps 6-9**: Diffusion ablation — discovered DLinear does all the work
3. **Exps 10-13**: Normalization and conditioning fixes
4. **Exps 14-17**: CI+Decomp Transformer replaces diffusion (MAE 0.41-0.58)
5. **Exp 18**: 30-config hyperparameter sweep
6. **Exps 19-25**: Overlapping patches, iTransformer, split heads, two-scale
7. **Exp 26**: Attention Residuals integration (MAE 0.299-0.488)
8. **Exp 27**: Heterogeneous ensemble (3 diverse architectures)
