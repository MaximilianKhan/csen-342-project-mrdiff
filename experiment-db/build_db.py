#!/usr/bin/env python3
"""
Build the mr-Diff experiment database.

This script is the single source of truth for the experiment campaign data. It
holds every documented result inline (transcribed from
code/ALL_EXPERIMENT_RESULTS.md) and emits two artifacts:

  experiment-db/experiments.json   human-readable, diffable normalized dump
  experiment-db/experiments.db     SQLite database for ad-hoc queries

Schema (SQLite):
  experiments(exp_id, exp_num, name, category, era, builds_on,
              params_min_k, params_max_k, train_time_min, verdict, change_summary)
  results(exp_id, benchmark, variant, mae, mse, is_final_best, was_record)
  ensemble_members(exp_id, benchmark, member_idx, architecture, params_k, mae)
  sweep_best_configs(benchmark, config_name, patch_size, d_model, num_layers,
                     dim_ff, dropout, trend_kernel, lr, weight_decay, params_k, mae)
  benchmarks(benchmark, dataset, mode, lookback, horizon, n_vars,
             paper_mae, baseline_mae)

Convenience views:
  v_results       results joined with paper/baseline MAE + % deltas
  v_final_bests   the four canonical headline results
  v_best_per_benchmark  the true minimum MAE per benchmark across all runs

All MAE/MSE values are on globally-standardized data (the paper's metric space).
Run:  python3 experiment-db/build_db.py
"""

import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "experiments.db")
JSON_PATH = os.path.join(HERE, "experiments.json")

BENCHMARKS = ["ETTh1_Multi", "ETTh1_Uni", "ETTm1_Multi", "ETTm1_Uni"]

# benchmark metadata: dataset, mode, lookback L, horizon H, n_vars, paper MAE, our baseline MAE
BENCH_META = {
    "ETTh1_Multi": ("ETTh1", "multivariate", 336, 168, 7, 0.42, 0.4744),
    "ETTh1_Uni":   ("ETTh1", "univariate",   336, 168, 1, 0.34, 0.2535),
    "ETTm1_Multi": ("ETTm1", "multivariate", 1440, 192, 7, 0.37, 0.4204),
    "ETTm1_Uni":   ("ETTm1", "univariate",   1440, 192, 1, 0.15, 0.2011),
}

def m(h1m, h1u, m1m, m1u):
    """Helper: pack the four benchmark MAEs into a dict (None allowed)."""
    return {"ETTh1_Multi": h1m, "ETTh1_Uni": h1u, "ETTm1_Multi": m1m, "ETTm1_Uni": m1u}

# ---------------------------------------------------------------------------
# Experiment metadata + results.
# `results` maps variant -> {benchmark: mae}. Most experiments report a single
# model ("single"); ensemble experiments also report an "ensemble" variant.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    dict(id="paper", num=None, name="mr-Diff (paper reported)", category="reference",
         era="reference", builds_on=None, params=(None, None), time=None, verdict="reference",
         change="Reported results from Shen, Chen and Kwok (ICLR 2024). No source code released.",
         results={"reported": m(0.42, 0.34, 0.37, 0.15)},
         mse={"reported": m(0.411, 0.066, 0.340, 0.039)}),

    dict(id="baseline", num=0, name="Our mr-Diff baseline (DLinear + diffusion)", category="baseline",
         era="diffusion", builds_on=None, params=(113, 843), time=34.0, verdict="baseline",
         change="Working replication after six deviations (downsizing, global-std metrics, DLinear "
                "backbone, GroupNorm, residual decomposition, random-projection mixup).",
         results={"single": m(0.4744, 0.2535, 0.4204, 0.2011)},
         mse={"single": m(0.4516, 0.1183, 0.3223, 0.0670)}),

    # --- Diffusion campaign (Exps 1-14) -----------------------------------
    dict(id="1", num=1, name="Remove detach (joint end-to-end)", category="diffusion", era="diffusion",
         builds_on="baseline", params=(843, 843), time=31.1, verdict="rejected",
         change="Removed .detach() so diffusion gradients flow into the DLinear backbone; diffusion_loss_scale=0.3.",
         results={"single": m(0.4765, 0.2543, 0.4224, 0.1988)}),
    dict(id="2", num=2, name="Self-conditioning", category="diffusion", era="diffusion",
         builds_on="1", params=(843, 843), time=32.7, verdict="best (diffusion era)",
         change="Self-conditioning in the denoiser: preliminary x0 estimate concatenated to the noisy input 50% of the time.",
         results={"single": m(0.4719, 0.2523, 0.4218, 0.1999)}),
    dict(id="3", num=3, name="Cosine noise schedule", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=25.0, verdict="rejected",
         change="Cosine beta schedule (Nichol and Dhariwal). Catastrophic on multivariate.",
         results={"single": m(0.6709, 0.2813, 0.6433, 0.2141)}),
    dict(id="4", num=4, name="v-prediction", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=34.3, verdict="rejected",
         change="v-prediction parameterization (Salimans and Ho) instead of epsilon-prediction.",
         results={"single": m(0.4790, 0.2531, 0.4216, 0.2049)}),
    dict(id="5", num=5, name="ANT adaptive noise schedule (IAAT power-law)", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=65.0, verdict="rejected",
         change="IAAT-driven concave beta warping. Catastrophic on multivariate (+54%).",
         results={"single": m(0.7309, 0.2803, 0.6513, 0.2055)}),
    dict(id="6", num=6, name="Contrastive conditioning loss (CCDM)", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=75.0, verdict="rejected",
         change="InfoNCE contrastive loss on stage-0 epsilon predictions with time-shifted negatives.",
         results={"single": m(0.5535, 0.2555, 0.4925, 0.1946)}),
    dict(id="7", num=7, name="Multi-granularity guided diffusion (MG-TSD)", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=91.0, verdict="rejected",
         change="Coarse-to-fine guidance loss on x0 predictions across noise levels.",
         results={"single": m(0.5653, 0.2558, 0.4819, 0.1913)}),
    dict(id="8", num=8, name="Channel-aware denoising", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=89.5, verdict="rejected",
         change="Per-channel Conv1d encoders + cross-channel attention. Catastrophic multivariate overfitting.",
         results={"single": m(0.9313, 0.2552, 0.9136, 0.1928)}),
    dict(id="9", num=9, name="Patch + attention history encoder (PatchTST-style)", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=94.0, verdict="rejected",
         change="Replaced Conv1d conditioning encoder with a 2-layer transformer over lookback patches.",
         results={"single": m(0.5388, 0.2541, 0.4900, 0.1984)}),
    dict(id="10", num=10, name="x0-prediction + trend/seasonality decomposition", category="diffusion", era="diffusion",
         builds_on="2", params=(843, 843), time=55.1, verdict="promising",
         change="Direct x0-prediction with decomposition head (trend + top-K Fourier). Only run where diffusion actively helped (ETTh1 Uni).",
         results={"single": m(0.4842, 0.2508, 0.4194, 0.1969)}),
    dict(id="11", num=11, name="Deep backbone, standard residuals (control)", category="diffusion", era="diffusion",
         builds_on="2", params=(946, 946), time=119.0, verdict="rejected",
         change="4 Conv1d residual blocks replacing the flat DLinear backbone. Overfit at this data scale.",
         results={"single": m(0.6634, 0.2822, 0.5440, 0.2056)}),
    dict(id="12", num=12, name="Deep backbone, Attention Residuals", category="diffusion", era="diffusion",
         builds_on="11", params=(947, 947), time=145.0, verdict="rejected",
         change="Same deep backbone with AttnRes (Kimi 2026). Beats Exp 11 on 3/4 but backbone depth is wrong.",
         results={"single": m(0.6599, 0.2729, 0.5681, 0.2051)}),
    dict(id="13", num=13, name="AttnRes backbone + learned stage aggregation", category="diffusion", era="diffusion",
         builds_on="12", params=(1022, 1022), time=137.0, verdict="rejected",
         change="Added a 75K-param learned StageAggregator over diffusion stages. Overfit the weighting task.",
         results={"single": m(0.6387, 0.2846, 0.5678, 0.2305)}),
    dict(id="14", num=14, name="Multi-scale AttnRes DLinear (stopped early)", category="diffusion", era="diffusion",
         builds_on="baseline", params=(1184, 3605), time=None, verdict="rejected",
         change="4 parallel DLinear projections fused with AttnRes, still diffusion-wrapped. Stopped after 2/4 (param blow-up).",
         results={"single": m(0.5685, 0.2701, None, None)}),

    # --- Transformer breakthrough (Exps 15-18) ----------------------------
    dict(id="15", num=15, name="Tiny direct transformer (PatchTST-style, no diffusion)", category="transformer", era="transformer",
         builds_on=None, params=(295, 7823), time=4.7, verdict="foundation",
         change="Dropped diffusion entirely. 2-layer transformer over patches with a flatten->linear head. Matched baseline on univariate at 7x speed.",
         results={"single": m(0.5607, 0.2538, 0.5514, 0.2002)}),
    dict(id="16", num=16, name="Channel-Independent patch transformer", category="transformer", era="transformer",
         builds_on="15", params=(73, 91), time=8.6, verdict="record",
         change="CI design: each channel patched and run through a shared transformer; params independent of D. Record ETTm1 Uni 0.1885.",
         results={"single": m(0.5485, 0.2741, 0.4293, 0.1885)}),
    dict(id="17", num=17, name="CI + trend/residual decomposition", category="transformer", era="transformer",
         builds_on="16", params=(77, 109), time=11.9, verdict="record",
         change="Added DLinear-style decomposition before CI patching, shared transformer, separate heads. Record ETTm1 Multi 0.4159.",
         results={"single": m(0.5101, 0.2580, 0.4159, 0.2011)}),
    dict(id="18", num=18, name="Hyperparameter sweep (30 configs)", category="transformer", era="transformer",
         builds_on="17", params=(54, 182), time=150.0, verdict="record",
         change="30-config random sweep on the CI+Decomp architecture. Set 3 of 4 all-time records.",
         results={"single": m(0.4880, 0.2514, 0.4094, 0.1881)}),

    # --- Refinements (Exps 19-26) -----------------------------------------
    dict(id="19", num=19, name="Extended training + LR warmup", category="transformer", era="transformer",
         builds_on="18", params=(54, 182), time=None, verdict="rejected",
         change="max_epochs 200, 10-epoch warmup, patience 30. Models still converged in ~35 epochs.",
         results={"single": m(0.4912, 0.2593, 0.4120, 0.1962)}),
    dict(id="21", num=21, name="Cross-channel mixing", category="transformer", era="transformer",
         builds_on="18", params=(54, 182), time=None, verdict="rejected",
         change="Zero-init Linear(D,D) residual after the CI transformer. Never learned useful cross-channel structure.",
         results={"single": m(0.4937, 0.2786, 0.4145, 0.2001)}),
    dict(id="22", num=22, name="Temporal data augmentation", category="transformer", era="transformer",
         builds_on="18", params=(54, 182), time=None, verdict="rejected",
         change="Jitter, scaling, temporal shift at 50% prob. Added noise to already-clean RevIN signals.",
         results={"single": m(0.4902, 0.2634, 0.4197, 0.1998)}),
    dict(id="25", num=25, name="Frequency-enhanced dual branch", category="transformer", era="transformer",
         builds_on="18", params=(140, 496), time=None, verdict="rejected",
         change="Parallel rFFT branch blended with learned alpha. Most consistent of the bolt-ons but adds params for no gain.",
         results={"single": m(0.4884, 0.2575, 0.4166, 0.1971)}),
    dict(id="26", num=26, name="CI+Decomp+AttnRes + gentle augmentation", category="transformer", era="transformer",
         builds_on="18", params=(54, 182), time=17.8, verdict="record",
         change="Attention Residuals (+192 params) + gentle augmentation. Cracked the ETTh1 Multi wall: single-model record 0.4875.",
         results={"single": m(0.4875, 0.2645, 0.4197, 0.1904)}),

    # --- Ensembles and later architectures (Exps 27-31) -------------------
    dict(id="27", num=27, name="Per-dataset heterogeneous ensemble", category="ensemble", era="transformer",
         builds_on="26", params=(54, 182), time=40.0, verdict="record",
         change="Average 3 architecturally diverse models per benchmark. Ensemble record ETTh1 Multi 0.4829; individual record ETTh1 Uni 0.2505.",
         results={"ensemble": m(0.4829, 0.2526, 0.4151, 0.1924),
                  "single":   m(0.4875, 0.2505, 0.4130, 0.1940)}),
    dict(id="28", num=28, name="Overlapping patches (single models)", category="transformer", era="transformer",
         builds_on="27", params=(54, 182), time=44.0, verdict="record",
         change="patch_stride = patch_size//2 via unfold (83 tokens vs 42 on ETTh1). Single-model record ETTh1 Multi 0.4832.",
         results={"single": m(0.4832, 0.2542, 0.4165, 0.1894)}),
    dict(id="29", num=29, name="iTransformer ensemble", category="ensemble", era="transformer",
         builds_on="27", params=(54, 182), time=75.0, verdict="record",
         change="iTransformer (cross-variate attention over D tokens) ensembled with CI models. Records ETTh1 Multi 0.4773 and ETTh1 Uni 0.2471.",
         results={"ensemble": m(0.4773, 0.2471, 0.4103, 0.1911),
                  "single":   m(0.4895, 0.2579, 0.4253, 0.1945)}),
    dict(id="31", num=31, name="Two-scale decomposition", category="transformer", era="transformer",
         builds_on="27", params=(94, 217), time=None, verdict="record",
         change="Two trend kernels (coarse=daily cycle) producing coarse/mid/fine bands. Records ETTm1 Multi 0.4081 (ensemble) and ETTm1 Uni 0.1865 (single).",
         results={"single":   m(0.4921, 0.2522, 0.4088, 0.1865),
                  "ensemble": m(0.4858, 0.2574, 0.4081, 0.1914)}),

    # --- Baseline-diffusion variants (run from baseline, not the chain) ----
    dict(id="adaln", num=None, name="AdaLN conditioning (baseline variant)", category="baseline_variant", era="diffusion",
         builds_on="baseline", params=(843, 843), time=300.0, verdict="kept",
         change="Replaced concatenation-based decoder conditioning with Adaptive LayerNorm (zero-init). Two wins, two small losses.",
         results={"single": m(0.4733, 0.2565, 0.4266, 0.1974)}),
    dict(id="ant", num=None, name="ANT beta_end shift (baseline variant)", category="baseline_variant", era="diffusion",
         builds_on="baseline", params=(843, 843), time=320.0, verdict="kept",
         change="Dataset-specific beta_end from first-difference variance, linear shape preserved. Three wins, one loss.",
         results={"single": m(0.4819, 0.2515, 0.4192, 0.2001)}),
]

# The four canonical headline results (after Max's decision to use documented Exp 29).
# Keys are (exp_id, benchmark, variant) to match the results lookup.
FINAL_BESTS = {
    ("29", "ETTh1_Multi", "ensemble"),
    ("29", "ETTh1_Uni",   "ensemble"),
    ("31", "ETTm1_Multi", "ensemble"),
    ("31", "ETTm1_Uni",   "single"),
}

# All-time records as tracked by the campaign log (the value was a record when set).
WAS_RECORD = {
    ("7", "ETTm1_Uni", "single"),
    ("10", "ETTh1_Uni", "single"),
    ("16", "ETTm1_Uni", "single"),
    ("17", "ETTm1_Multi", "single"),
    ("18", "ETTh1_Uni", "single"), ("18", "ETTm1_Multi", "single"), ("18", "ETTm1_Uni", "single"),
    ("26", "ETTh1_Multi", "single"),
    ("27", "ETTh1_Multi", "ensemble"), ("27", "ETTh1_Uni", "single"),
    ("28", "ETTh1_Multi", "single"),
    ("29", "ETTh1_Multi", "ensemble"), ("29", "ETTh1_Uni", "ensemble"),
    ("31", "ETTm1_Multi", "ensemble"), ("31", "ETTm1_Multi", "single"), ("31", "ETTm1_Uni", "single"),
}

# Ensemble member breakdowns (exp_id, benchmark, members[(architecture, params_k, mae)]).
ENSEMBLE_MEMBERS = [
    ("27", "ETTh1_Multi", [("CI base (cfg02)", 86, 0.4929), ("CI base (cfg07, d=64)", 86, 0.4927), ("CI AttnRes+Aug", 55, 0.4875)]),
    ("27", "ETTh1_Uni",   [("CI base (cfg01)", 54, 0.2505), ("CI base (cfg03, high dropout)", 54, 0.2530), ("CI AttnRes+Aug", 54, 0.2627)]),
    ("27", "ETTm1_Multi", [("CI base (cfg10)", 182, 0.4214), ("CI base (cfg02)", 182, 0.4210), ("CI AttnRes", 182, 0.4130)]),
    ("27", "ETTm1_Uni",   [("CI base (cfg06)", 77, 0.1961), ("CI base (cfg16)", 52, 0.1940), ("CI base (cfg19)", 77, 0.1951)]),
    ("29", "ETTh1_Multi", [("iTransformer", 60, 0.4878), ("CI AttnRes", 55, 0.4836), ("CI base", 86, 0.4915)]),
    ("29", "ETTh1_Uni",   [("iTransformer", 60, 0.2485), ("CI base", 54, 0.2507), ("CI base", 54, 0.2593)]),
    ("29", "ETTm1_Multi", [("iTransformer", 60, 0.4390), ("CI base", 182, 0.4118), ("CI base", 182, 0.4098)]),
    ("29", "ETTm1_Uni",   [("CI base", 77, 0.1934), ("CI base", 77, 0.1970), ("CI base", 77, 0.1903)]),
    ("31", "ETTh1_Multi", [("CI AttnRes+Aug", 55, 0.4882), ("CI base (d=64)", 110, 0.4950), ("CI base (d=32)", 54, 0.4957)]),
    ("31", "ETTh1_Uni",   [("CI base", 54, 0.2553), ("CI base", 54, 0.2584), ("CI base", 54, 0.2692)]),
    ("31", "ETTm1_Multi", [("Two-scale", 217, 0.4126), ("CI base", 182, 0.4110), ("CI base", 182, 0.4200)]),
    ("31", "ETTm1_Uni",   [("Two-scale", 94, 0.1937), ("CI base", 77, 0.1964), ("CI base", 77, 0.1941)]),
]

# Exp 18 winning per-benchmark CI+Decomp configs.
# (benchmark, config, patch, d_model, layers, dim_ff, dropout, trend_kernel, lr, wd, params_k, mae)
SWEEP_BEST = [
    ("ETTh1_Multi", "cfg07", 16, 64, 3, 64,  0.2, 15, 0.002,  0.005, 86,  0.4880),
    ("ETTh1_Uni",   "cfg01", 8,  32, 3, 128, 0.3, 15, 0.0005, 0.05,  54,  0.2514),
    ("ETTm1_Multi", "cfg10", 8,  48, 3, 256, 0.3, 15, 0.001,  0.01,  182, 0.4094),
    ("ETTm1_Uni",   "cfg06", 16, 32, 3, 128, 0.2, 25, 0.0005, 0.05,  77,  0.1881),
]


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE benchmarks (
        benchmark TEXT PRIMARY KEY, dataset TEXT, mode TEXT,
        lookback INTEGER, horizon INTEGER, n_vars INTEGER,
        paper_mae REAL, baseline_mae REAL
    );
    CREATE TABLE experiments (
        exp_id TEXT PRIMARY KEY, exp_num INTEGER, name TEXT, category TEXT,
        era TEXT, builds_on TEXT, params_min_k INTEGER, params_max_k INTEGER,
        train_time_min REAL, verdict TEXT, change_summary TEXT
    );
    CREATE TABLE results (
        exp_id TEXT, benchmark TEXT, variant TEXT, mae REAL, mse REAL,
        is_final_best INTEGER, was_record INTEGER,
        PRIMARY KEY (exp_id, benchmark, variant),
        FOREIGN KEY (exp_id) REFERENCES experiments(exp_id),
        FOREIGN KEY (benchmark) REFERENCES benchmarks(benchmark)
    );
    CREATE TABLE ensemble_members (
        exp_id TEXT, benchmark TEXT, member_idx INTEGER,
        architecture TEXT, params_k INTEGER, mae REAL,
        PRIMARY KEY (exp_id, benchmark, member_idx)
    );
    CREATE TABLE sweep_best_configs (
        benchmark TEXT PRIMARY KEY, config_name TEXT, patch_size INTEGER,
        d_model INTEGER, num_layers INTEGER, dim_ff INTEGER, dropout REAL,
        trend_kernel INTEGER, lr REAL, weight_decay REAL, params_k INTEGER, mae REAL
    );
    """)

    for b, (ds, mode, L, H, nv, pm, bm) in BENCH_META.items():
        cur.execute("INSERT INTO benchmarks VALUES (?,?,?,?,?,?,?,?)",
                    (b, ds, mode, L, H, nv, pm, bm))

    for e in EXPERIMENTS:
        pmin, pmax = e["params"]
        cur.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (e["id"], e["num"], e["name"], e["category"], e["era"], e["builds_on"],
                     pmin, pmax, e["time"], e["verdict"], e["change"]))
        mse_sets = e.get("mse", {})
        for variant, vals in e["results"].items():
            mse_vals = mse_sets.get(variant, {})
            for b, mae in vals.items():
                key = (e["id"], b, variant)
                cur.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                            (e["id"], b, variant, mae, mse_vals.get(b),
                             1 if key in FINAL_BESTS else 0,
                             1 if key in WAS_RECORD else 0))

    for exp_id, b, members in ENSEMBLE_MEMBERS:
        for i, (arch, pk, mae) in enumerate(members):
            cur.execute("INSERT INTO ensemble_members VALUES (?,?,?,?,?,?)",
                        (exp_id, b, i, arch, pk, mae))

    for row in SWEEP_BEST:
        cur.execute("INSERT INTO sweep_best_configs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)

    cur.executescript("""
    CREATE VIEW v_results AS
    SELECT r.exp_id, e.exp_num, e.name, e.category, e.era, r.benchmark, r.variant,
           r.mae, b.paper_mae, b.baseline_mae,
           ROUND(100.0 * (r.mae - b.paper_mae) / b.paper_mae, 1)    AS vs_paper_pct,
           ROUND(100.0 * (r.mae - b.baseline_mae) / b.baseline_mae, 1) AS vs_baseline_pct,
           r.is_final_best, r.was_record
    FROM results r
    JOIN experiments e ON e.exp_id = r.exp_id
    JOIN benchmarks  b ON b.benchmark = r.benchmark
    WHERE r.mae IS NOT NULL;

    CREATE VIEW v_final_bests AS
    SELECT benchmark, exp_id, variant, mae FROM results WHERE is_final_best = 1;

    CREATE VIEW v_best_per_benchmark AS
    SELECT benchmark, MIN(mae) AS best_mae FROM results
    WHERE mae IS NOT NULL AND exp_id NOT IN ('paper')
    GROUP BY benchmark;
    """)

    con.commit()

    # --- emit JSON dump from the live tables -------------------------------
    def dump(table):
        cur.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    payload = {
        "note": "Source of truth: code/ALL_EXPERIMENT_RESULTS.md. "
                "MAE/MSE on globally-standardized data. Generated by experiment-db/build_db.py.",
        "benchmarks": dump("benchmarks"),
        "experiments": dump("experiments"),
        "results": dump("results"),
        "ensemble_members": dump("ensemble_members"),
        "sweep_best_configs": dump("sweep_best_configs"),
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    n_exp = cur.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    n_res = cur.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    con.close()
    print(f"Wrote {DB_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"  {n_exp} experiments, {n_res} result rows, "
          f"{len(ENSEMBLE_MEMBERS)*3} ensemble members, {len(SWEEP_BEST)} sweep configs")


if __name__ == "__main__":
    build()
