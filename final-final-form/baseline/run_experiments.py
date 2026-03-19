#!/usr/bin/env python3
"""
mr-Diff Full Experiment Runner
==============================

Runs all experiments required for baseline replication:
- ETTh1 Multivariate & Univariate
- ETTm1 Multivariate & Univariate

This script is designed for the CSEN-342 project to replicate the mr-Diff
paper experiments (ICLR 2024: Multi-Resolution Diffusion Models for Time
Series Forecasting).

Usage:
    python run_experiments.py [--epochs N] [--dry-run]

The script will:
1. Run all 4 experiment configurations
2. Display real-time progress with ETAs
3. Log all results to experiments/ directory
4. Generate a summary suitable for the project report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 70
    print()
    print(Colors.BOLD + Colors.CYAN + "=" * width + Colors.END)
    print(Colors.BOLD + Colors.CYAN + f"  {text}".center(width) + Colors.END)
    print(Colors.BOLD + Colors.CYAN + "=" * width + Colors.END)
    print()


def print_experiment_box(exp_num: int, total: int, name: str, status: str = "RUNNING") -> None:
    """Print experiment status box."""
    width = 70
    color = Colors.YELLOW if status == "RUNNING" else Colors.GREEN if status == "DONE" else Colors.RED

    print()
    print(color + "┌" + "─" * (width - 2) + "┐" + Colors.END)
    print(color + f"│  Experiment {exp_num}/{total}: {name}".ljust(width - 1) + "│" + Colors.END)
    print(color + f"│  Status: {status}".ljust(width - 1) + "│" + Colors.END)
    print(color + "└" + "─" * (width - 2) + "┘" + Colors.END)
    print()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_eta(seconds: float) -> str:
    """Format ETA with expected completion time."""
    if seconds <= 0:
        return "Complete"

    eta_time = datetime.now() + timedelta(seconds=seconds)
    duration = format_time(seconds)
    return f"{duration} (ETA: {eta_time.strftime('%H:%M:%S')})"


class ExperimentRunner:
    """Manages running all mr-Diff experiments with progress tracking."""

    # Experiment configurations
    # Lookback lengths from paper Table 8: ETTh1=336, ETTm1=1440
    # Experiment configurations
    # Lookback/forecast lengths from paper: ETTh1: L=336, H=168; ETTm1: L=1440, H=192
    EXPERIMENTS = [
        {
            "name": "ETTh1 Multivariate",
            "dataset": "ETTh1",
            "univariate": False,
            "forecast_length": 168,
            "lookback_length": 336,
            "description": "Hourly electricity data, all 7 features"
        },
        {
            "name": "ETTh1 Univariate",
            "dataset": "ETTh1",
            "univariate": True,
            "forecast_length": 168,
            "lookback_length": 336,
            "description": "Hourly electricity data, OT (Oil Temperature) only"
        },
        {
            "name": "ETTm1 Multivariate",
            "dataset": "ETTm1",
            "univariate": False,
            "forecast_length": 192,
            "lookback_length": 1440,
            "description": "15-minute electricity data, all 7 features"
        },
        {
            "name": "ETTm1 Univariate",
            "dataset": "ETTm1",
            "univariate": True,
            "forecast_length": 192,
            "lookback_length": 1440,
            "description": "15-minute electricity data, OT only"
        },
    ]

    def __init__(
        self,
        max_epochs: int = 100,
        batch_size: int = 64,
        output_dir: str = "experiments",
        dry_run: bool = False,
        eval_samples: int = 10,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.eval_samples = eval_samples

        # Timing estimates (will be updated as experiments run)
        self.time_per_epoch_estimate = 30.0  # Initial estimate in seconds
        self.completed_times: List[float] = []

        # Results storage
        self.results: Dict[str, dict] = {}

        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def estimate_experiment_time(self, exp_config: dict) -> float:
        """Estimate time for a single experiment."""
        # ETTm1 has 4x more data than ETTh1
        data_multiplier = 4.0 if exp_config["dataset"] == "ETTm1" else 1.0

        # Univariate is slightly faster
        uni_multiplier = 0.9 if exp_config["univariate"] else 1.0

        # Use average of completed times if available
        if self.completed_times:
            base_time = sum(self.completed_times) / len(self.completed_times)
        else:
            base_time = self.time_per_epoch_estimate * self.max_epochs

        return base_time * data_multiplier * uni_multiplier

    def print_overall_progress(self, current_exp: int, exp_elapsed: float = 0) -> None:
        """Print overall progress summary."""
        total = len(self.EXPERIMENTS)
        completed = current_exp - 1

        # Calculate times
        completed_time = sum(self.completed_times)
        remaining_experiments = self.EXPERIMENTS[current_exp - 1:]
        remaining_time = sum(self.estimate_experiment_time(e) for e in remaining_experiments)
        remaining_time -= exp_elapsed  # Subtract elapsed time on current

        total_estimated = completed_time + remaining_time + exp_elapsed

        # Progress bar
        bar_width = 40
        filled = int(bar_width * completed / total)
        bar = "█" * filled + "▒" * (bar_width - filled)

        print(Colors.BOLD + "\n┌─ OVERALL PROGRESS " + "─" * 49 + "┐" + Colors.END)
        print(f"│  [{bar}] {completed}/{total} experiments complete")
        print(f"│  ")
        print(f"│  ⏱  Elapsed:    {format_time(completed_time + exp_elapsed)}")
        print(f"│  ⏳ Remaining:  {format_eta(remaining_time)}")
        print(f"│  📊 Total Est:  {format_time(total_estimated)}")
        print(Colors.BOLD + "└" + "─" * 68 + "┘\n" + Colors.END)

    def run_single_experiment(self, exp_num: int, config: dict) -> Tuple[bool, dict]:
        """Run a single experiment and return results."""
        exp_name = f"{config['dataset']}_{'uni' if config['univariate'] else 'multi'}"

        print_experiment_box(exp_num, len(self.EXPERIMENTS), config["name"], "RUNNING")
        print(f"  📝 {config['description']}")
        print(f"  📁 Dataset: {config['dataset']}")
        print(f"  🔢 Mode: {'Univariate' if config['univariate'] else 'Multivariate'}")
        print(f"  📏 Forecast Length: {config['forecast_length']}")
        print(f"  👀 Lookback Length: {config.get('lookback_length', 336)}")
        print(f"  🔄 Max Epochs: {self.max_epochs}")
        print()

        if self.dry_run:
            print(Colors.YELLOW + "  [DRY RUN] Would execute training..." + Colors.END)
            time.sleep(2)
            return True, {"mae": 0.0, "mse": 0.0, "epochs": 0, "time": 0}

        # Create unique checkpoint directory for this experiment
        exp_checkpoint_dir = self.run_dir / exp_name / "checkpoints"
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable, "train.py",
            "--dataset", config["dataset"],
            "--epochs", str(self.max_epochs),
            "--batch-size", str(self.batch_size),
            "--experiment-name", exp_name,
            "--checkpoint-dir", str(exp_checkpoint_dir),
            "--lookback-length", str(config.get("lookback_length", 336)),
            "--forecast-length", str(config.get("forecast_length", 168)),
        ]

        if config["univariate"]:
            cmd.append("--univariate")

        # Run training
        start_time = time.time()

        try:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            output_lines = []
            for line in process.stdout:
                print(line, end="")
                output_lines.append(line)

            process.wait()
            elapsed = time.time() - start_time

            if process.returncode != 0:
                print(Colors.RED + f"\n  ❌ Experiment failed with code {process.returncode}" + Colors.END)
                return False, {"error": f"Exit code {process.returncode}", "time": elapsed}

            # Parse results from output
            results = self._parse_results(output_lines, elapsed)
            self.completed_times.append(elapsed)

            print(Colors.GREEN + f"\n  ✓ Completed in {format_time(elapsed)}" + Colors.END)
            return True, results

        except Exception as e:
            elapsed = time.time() - start_time
            print(Colors.RED + f"\n  ❌ Error: {e}" + Colors.END)
            return False, {"error": str(e), "time": elapsed}

    def _parse_results(self, output_lines: List[str], elapsed: float) -> dict:
        """Parse training output for results."""
        results = {
            "time": elapsed,
            "epochs": 0,
            "best_val_loss": None,
            "final_train_loss": None,
        }

        for line in output_lines:
            if "Best validation loss:" in line:
                try:
                    results["best_val_loss"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Final epoch:" in line:
                try:
                    results["epochs"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "Epoch" in line and "Train Loss:" in line:
                try:
                    parts = line.split("|")
                    for part in parts:
                        if "Train Loss:" in part:
                            results["final_train_loss"] = float(part.split(":")[-1].strip())
                except:
                    pass

        return results

    def run_evaluation(self, exp_name: str, config: dict,
                       checkpoint_path: Path = None,
                       solver: str = "ddpm", solver_steps: int = 20,
                       aggregation: str = "sum", epsilon_scale: float = 1.0,
                       output_dir: Path = None) -> dict:
        """Run evaluation on a trained model."""
        if checkpoint_path is None:
            checkpoint_path = self.run_dir / exp_name / "checkpoints" / "best.pt"

        if not checkpoint_path.exists():
            return {"error": f"Checkpoint not found at {checkpoint_path}"}

        if output_dir is None:
            output_dir = self.run_dir / exp_name / "evaluation"

        cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", str(checkpoint_path),
            "--dataset", config["dataset"],
            "--num-samples", str(self.eval_samples),
            "--output-dir", str(output_dir),
            "--lookback-length", str(config.get("lookback_length", 336)),
            "--solver", solver,
            "--solver-steps", str(solver_steps),
            "--aggregation", aggregation,
            "--epsilon-scale", str(epsilon_scale),
        ]

        if config["univariate"]:
            cmd.append("--univariate")

        try:
            # 8 hour timeout for evaluation (28800 seconds)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)

            if result.returncode != 0:
                print(f"    Evaluation stderr: {result.stderr[:200] if result.stderr else 'None'}")
                return {"error": f"Evaluation failed: {result.returncode}"}

            # Try to load metrics from output
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    return json.load(f)
            else:
                return {"error": f"Metrics file not found at {metrics_path}"}

        except Exception as e:
            return {"error": str(e)}

    def run_all(self) -> None:
        """Run all experiments."""
        print_header("mr-Diff Baseline Replication")

        print(Colors.BOLD + "Project: CSEN-342 Baseline Implementation" + Colors.END)
        print(f"Paper: Multi-Resolution Diffusion Models for Time Series Forecasting (ICLR 2024)")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {self.run_dir}")
        print()

        print(Colors.UNDERLINE + "Experiments to run:" + Colors.END)
        for i, exp in enumerate(self.EXPERIMENTS, 1):
            uni_str = "Univariate" if exp["univariate"] else "Multivariate"
            print(f"  {i}. {exp['dataset']} {uni_str}")
        print()

        # Estimate total time
        total_estimate = sum(self.estimate_experiment_time(e) for e in self.EXPERIMENTS)
        print(f"⏱  Estimated total time: {format_time(total_estimate)}")
        print(f"   (based on {self.max_epochs} epochs per experiment)")
        print()

        if not self.dry_run:
            print(Colors.YELLOW + "Starting in 3 seconds... (Ctrl+C to cancel)" + Colors.END)
            time.sleep(3)

        # Run each experiment
        all_success = True
        start_time = time.time()

        for i, config in enumerate(self.EXPERIMENTS, 1):
            exp_name = f"{config['dataset']}_{'uni' if config['univariate'] else 'multi'}"

            self.print_overall_progress(i)

            success, results = self.run_single_experiment(i, config)
            self.results[exp_name] = {
                "config": config,
                "training": results,
                "success": success,
            }

            if not success:
                all_success = False
                print(Colors.RED + f"\n  ⚠ Experiment {i} failed, continuing..." + Colors.END)
            else:
                # Run evaluation with DPM-Solver++ (5x faster, identical quality)
                print(f"\n  📊 Running evaluation (DPM-Solver++)...")
                eval_results = self.run_evaluation(exp_name, config,
                                                   solver="dpm_solver_pp")
                self.results[exp_name]["evaluation"] = eval_results

            print_experiment_box(i, len(self.EXPERIMENTS), config["name"],
                               "DONE" if success else "FAILED")

        total_time = time.time() - start_time

        # Save results
        self._save_results(total_time)

        # Print summary
        self._print_summary(total_time, all_success)

    def eval_only(self, checkpoint_base: Path, solver: str, solver_steps: int,
                   aggregation: str, epsilon_scale: float = 1.0) -> None:
        """Re-evaluate existing checkpoints with a different solver."""
        print_header(f"mr-Diff Eval-Only: {solver}")

        solver_desc = solver
        if solver == "dpm_solver_pp":
            solver_desc = f"DPM-Solver++ ({solver_steps} steps, agg={aggregation})"
        print(f"Solver: {solver_desc}")
        print(f"Checkpoint dir: {checkpoint_base}")
        print()

        start_time = time.time()

        for i, config in enumerate(self.EXPERIMENTS, 1):
            exp_name = f"{config['dataset']}_{'uni' if config['univariate'] else 'multi'}"

            # Search for best.pt in common locations
            candidates = [
                checkpoint_base / exp_name / "checkpoints" / "best.pt",
                checkpoint_base / exp_name / "best.pt",
                checkpoint_base / "checkpoints" / exp_name / "best.pt",
            ]
            # Also search for any run_* subdirectories
            for run_dir in sorted(checkpoint_base.glob("run_*")):
                candidates.append(run_dir / exp_name / "checkpoints" / "best.pt")

            checkpoint_path = None
            for c in candidates:
                if c.exists():
                    checkpoint_path = c
                    break

            if checkpoint_path is None:
                print(Colors.RED + f"  [{i}/4] {config['name']}: No checkpoint found" + Colors.END)
                self.results[exp_name] = {"config": config, "success": False,
                                          "evaluation": {"error": "No checkpoint"}}
                continue

            print(f"  [{i}/4] {config['name']}: {checkpoint_path}")

            eval_output = self.run_dir / exp_name / "evaluation"
            eval_results = self.run_evaluation(
                exp_name, config,
                checkpoint_path=checkpoint_path,
                solver=solver, solver_steps=solver_steps,
                aggregation=aggregation, epsilon_scale=epsilon_scale,
                output_dir=eval_output,
            )

            self.results[exp_name] = {
                "config": config,
                "success": "error" not in eval_results,
                "evaluation": eval_results,
            }

            if "mae" in eval_results:
                print(Colors.GREEN +
                      f"       MAE={eval_results['mae']:.4f}  MSE={eval_results['mse']:.4f}" +
                      Colors.END)
            else:
                print(Colors.RED + f"       Error: {eval_results.get('error', 'unknown')}" + Colors.END)

        total_time = time.time() - start_time
        self._save_results(total_time)
        self._print_summary(total_time, all(d.get("success") for d in self.results.values()))

    def _save_results(self, total_time: float) -> None:
        """Save all results to JSON."""
        summary = {
            "timestamp": self.timestamp,
            "total_time_seconds": total_time,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "experiments": self.results,
        }

        results_path = self.run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_path}")

    def _print_summary(self, total_time: float, all_success: bool) -> None:
        """Print final summary suitable for project report."""
        print_header("EXPERIMENT SUMMARY")

        status = "✓ ALL COMPLETE" if all_success else "⚠ SOME FAILED"
        color = Colors.GREEN if all_success else Colors.YELLOW

        print(color + f"Status: {status}" + Colors.END)
        print(f"Total Time: {format_time(total_time)}")
        print(f"Results Directory: {self.run_dir}")
        print()

        # Results table
        print(Colors.BOLD + "Results Table (for project report):" + Colors.END)
        print()
        print("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Experiment          │ Val Loss │ Epochs   │ Time     │ Status   │")
        print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

        for name, data in self.results.items():
            training = data.get("training", {})
            val_loss = training.get("best_val_loss", "N/A")
            epochs = training.get("epochs", "N/A")
            exp_time = format_time(training.get("time", 0))
            status = "✓" if data.get("success") else "✗"

            val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)

            print(f"│ {name:<19} │ {val_str:<8} │ {str(epochs):<8} │ {exp_time:<8} │ {status:<8} │")

        print("└─────────────────────┴──────────┴──────────┴──────────┴──────────┘")
        print()

        # Evaluation metrics if available
        has_eval = any("evaluation" in d and "mae" in d.get("evaluation", {})
                       for d in self.results.values())

        if has_eval:
            print(Colors.BOLD + "Evaluation Metrics:" + Colors.END)
            print()
            print("┌─────────────────────┬──────────────┬──────────────┐")
            print("│ Experiment          │ MAE          │ MSE          │")
            print("├─────────────────────┼──────────────┼──────────────┤")

            for name, data in self.results.items():
                eval_data = data.get("evaluation", {})
                mae = eval_data.get("mae", "N/A")
                mse = eval_data.get("mse", "N/A")

                mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
                mse_str = f"{mse:.4f}" if isinstance(mse, float) else str(mse)

                print(f"│ {name:<19} │ {mae_str:<12} │ {mse_str:<12} │")

            print("└─────────────────────┴──────────────┴──────────────┘")

        print()
        print(Colors.BOLD + "Next Steps for Project Report:" + Colors.END)
        print("  1. Copy the results table above to your report")
        print("  2. Generate plots using: python analyze.py plot-loss --log-dir logs/<experiment>")
        print("  3. Compare with paper's Table 1 (MAE) and Appendix D (MSE)")
        print("  4. Document any differences and potential causes")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run all mr-Diff experiments for CSEN-342 baseline replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                    # Full run with 100 epochs
  python run_experiments.py --epochs 10        # Quick test with 10 epochs
  python run_experiments.py --dry-run          # Test without running

For the project report, run with default settings to replicate paper results.
        """
    )

    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum epochs per experiment (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments",
        help="Output directory for results (default: experiments)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=10,
        help="Number of samples for evaluation (default: 10, use 1 for fast testing)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Re-evaluate existing checkpoints without retraining"
    )
    parser.add_argument(
        "--solver", type=str, default="ddpm",
        choices=["ddpm", "dpm_solver_pp"],
        help="Sampling solver for evaluation (default: ddpm)"
    )
    parser.add_argument(
        "--solver-steps", type=int, default=20,
        help="Number of DPM-Solver++ steps (default: 20)"
    )
    parser.add_argument(
        "--aggregation", type=str, default="sum",
        choices=["first", "sum"],
        help="Multi-resolution aggregation: first or sum (default: sum)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory containing experiment checkpoints (for --eval-only)"
    )
    parser.add_argument(
        "--epsilon-scale", type=float, default=1.0,
        help="Epsilon scaling for x0 predictions (default: 1.0, try 0.98)"
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        eval_samples=args.eval_samples,
    )

    try:
        if args.eval_only:
            checkpoint_base = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("experiments")
            runner.eval_only(
                checkpoint_base=checkpoint_base,
                solver=args.solver,
                solver_steps=args.solver_steps,
                aggregation=args.aggregation,
                epsilon_scale=args.epsilon_scale,
            )
        else:
            runner.run_all()
    except KeyboardInterrupt:
        print(Colors.YELLOW + "\n\n⚠ Interrupted by user" + Colors.END)
        print("Partial results may be saved in the experiments directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
