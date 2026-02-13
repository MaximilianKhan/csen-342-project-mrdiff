"""Generate report tables from experiment results.

Usage in Jupyter notebook:
    from generate_tables import table1_results, table2_training, table3_runs, table4_differences
    table1_results()
    table2_training()
    table3_runs()
    table4_differences()
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

try:
    EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
except NameError:
    EXPERIMENTS_DIR = Path.cwd() / "experiments"

# The last 5 complete runs, in chronological order
RUNS = [
    ("run_20260211_222824", "Initial impl."),
    ("run_20260212_082858", "Best (pred[0])"),
    ("run_20260212_114727", "Cumul. trends"),
    ("run_20260212_144502", "Cond+AMP+sum"),
    ("run_20260212_173919", "sum isolated"),
]

EXPERIMENTS = ["ETTh1_multi", "ETTh1_uni", "ETTm1_multi", "ETTm1_uni"]

EXPERIMENT_LABELS = {
    "ETTh1_multi": "ETTh1 Multi",
    "ETTh1_uni": "ETTh1 Uni",
    "ETTm1_multi": "ETTm1 Multi",
    "ETTm1_uni": "ETTm1 Uni",
}

PAPER_VALUES = {
    "ETTh1_multi": {"mae": 0.422, "mse": 0.411},
    "ETTh1_uni":   {"mae": 0.196, "mse": 0.066},
    "ETTm1_multi": {"mae": 0.373, "mse": 0.340},
    "ETTm1_uni":   {"mae": 0.149, "mse": 0.039},
}


def _load_run(run_dir: str) -> dict:
    """Load results.json for a run."""
    path = EXPERIMENTS_DIR / run_dir / "results.json"
    with open(path) as f:
        return json.load(f)


def _render_table(df: pd.DataFrame, title: str, col_widths: list = None,
                  highlight_col: str = None):
    """Render a DataFrame as a matplotlib table figure."""
    n_rows, n_cols = df.shape

    fig_width = max(10, n_cols * 1.6)
    fig_height = max(2.0, 0.45 * (n_rows + 1) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index if df.index.name or not df.index.equals(pd.RangeIndex(len(df))) else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.12)

    # Style row labels if present
    has_row_labels = df.index.name or not df.index.equals(pd.RangeIndex(len(df)))
    if has_row_labels:
        for pos, cell in table.get_celld().items():
            row, col = pos
            if col == -1:
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    cell.set_facecolor("#ecf0f1")
                    cell.set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(n_rows):
        color = "#ffffff" if i % 2 == 0 else "#f7f9fa"
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(color)
            cell.set_height(0.10)

    # Highlight column if specified
    if highlight_col and highlight_col in df.columns:
        col_idx = list(df.columns).index(highlight_col)
        for i in range(n_rows):
            cell = table[i + 1, col_idx]
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d5f5e3" if i % 2 == 0 else "#c8f0da")

    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(n_rows + 1):
                table[i, j].set_width(w)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table 1: Baseline Results vs Paper
# ---------------------------------------------------------------------------

def table1_results(render: bool = True):
    """Table 1: Our best baseline results vs the paper's reported values.

    Args:
        render: If True, display a matplotlib figure. If False, return DataFrames.

    Returns:
        Tuple of (uni_df, multi_df) if render=False, else displays figure.
    """
    best = _load_run("run_20260212_082858")

    rows = []
    for exp in EXPERIMENTS:
        label = EXPERIMENT_LABELS[exp]
        ours = best["experiments"][exp]["evaluation"]
        paper = PAPER_VALUES[exp]
        setting = "Univariate" if "uni" in exp else "Multivariate"
        dataset = "ETTh1" if "ETTh1" in exp else "ETTm1"
        gap_mae = f"{ours['mae'] / paper['mae']:.1f}x"
        gap_mse = f"{ours['mse'] / paper['mse']:.1f}x"
        rows.append({
            "Dataset": dataset,
            "Setting": setting,
            "MAE (Ours)": f"{ours['mae']:.3f}",
            "MAE (Paper)": f"{paper['mae']:.3f}",
            "MSE (Ours)": f"{ours['mse']:.3f}",
            "MSE (Paper)": f"{paper['mse']:.3f}",
            "MAE Gap": gap_mae,
        })

    df = pd.DataFrame(rows)

    if not render:
        return df

    _render_table(df, "Table 1: Baseline Results vs. Paper (Best of Last 5 Runs)",
                  highlight_col="MAE (Paper)")
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Table 2: Training Summary
# ---------------------------------------------------------------------------

def table2_training(render: bool = True):
    """Table 2: Training dynamics for the best run.

    Args:
        render: If True, display a matplotlib figure. If False, return DataFrame.

    Returns:
        DataFrame with training summary.
    """
    best = _load_run("run_20260212_082858")

    rows = []
    for exp in EXPERIMENTS:
        label = EXPERIMENT_LABELS[exp]
        train = best["experiments"][exp]["training"]

        # Load best checkpoint to get epoch number
        best_ckpt_path = EXPERIMENTS_DIR / "run_20260212_082858" / exp / "checkpoints" / "best.json"
        best_epoch = "?"
        if best_ckpt_path.exists():
            with open(best_ckpt_path) as f:
                ckpt = json.load(f)
                best_epoch = ckpt.get("epoch", "?")

        rows.append({
            "Experiment": label,
            "Epochs": train["epochs"],
            "Best Val Loss": f"{train['best_val_loss']:.4f}",
            "Best Epoch": best_epoch,
            "Final Train Loss": f"{train['final_train_loss']:.4f}",
            "Time (min)": f"{train['time'] / 60:.1f}",
        })

    df = pd.DataFrame(rows)

    if not render:
        return df

    _render_table(df, "Table 2: Training Summary (Best of Last 5 Runs)")
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Table 3: MAE Progression Across Runs
# ---------------------------------------------------------------------------

def table3_runs(render: bool = True):
    """Table 3: MAE progression across the last 5 runs.

    Args:
        render: If True, display a matplotlib figure. If False, return DataFrame.

    Returns:
        DataFrame with MAE values per run.
    """
    n_runs = len(RUNS)
    data = {}

    for i, (run_dir, desc) in enumerate(RUNS):
        run_label = f"Last-{n_runs - i}\n({desc})"
        run_data = _load_run(run_dir)
        col = {}
        for exp in EXPERIMENTS:
            label = EXPERIMENT_LABELS[exp]
            eval_data = run_data["experiments"][exp].get("evaluation", {})
            mae = eval_data.get("mae")
            col[label] = f"{mae:.2f}" if mae is not None else "N/A"
        data[run_label] = col

    # Add paper column
    paper_col = {}
    for exp in EXPERIMENTS:
        label = EXPERIMENT_LABELS[exp]
        paper_col[label] = f"{PAPER_VALUES[exp]['mae']:.3f}"
    data["Paper"] = paper_col

    df = pd.DataFrame(data)

    if not render:
        return df

    _render_table(df, f"Table 3: MAE Across Last {n_runs} Runs",
                  highlight_col="Paper")
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Table 4: Known Differences from Paper
# ---------------------------------------------------------------------------

def table4_differences(render: bool = True):
    """Table 4: Known differences between our implementation and the paper.

    Args:
        render: If True, display a matplotlib figure. If False, return DataFrame.

    Returns:
        DataFrame with differences.
    """
    rows = [
        {"Component": "Trend decomposition", "Paper": "Cumulative", "Ours": "Residual"},
        {"Component": "Reverse diffusion", "Paper": "DPM-Solver", "Ours": "DDPM (100 steps)"},
        {"Component": "Mixup projection", "Paper": "Learned (Eq. 9)", "Ours": "Random weights"},
        {"Component": "Signal reconstruction", "Paper": "sum(all stages)", "Ours": "predictions[0]"},
        {"Component": "Kernel sizes / S", "Paper": "Per-dataset search", "Ours": "Fixed S=5, [5-201]"},
    ]
    df = pd.DataFrame(rows)

    if not render:
        return df

    _render_table(df, "Table 4: Known Differences from Paper")
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Generate all tables
# ---------------------------------------------------------------------------

def all_tables():
    """Generate and display all four report tables."""
    table1_results()
    table2_training()
    table3_runs()
    table4_differences()


if __name__ == "__main__":
    matplotlib.use("Agg")
    all_tables()
    print("Tables generated.")
