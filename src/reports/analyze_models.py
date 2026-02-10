"""Script to collect and analyze metrics from all trained models.

This module provides functionality to scan experiment directories, extract
training and evaluation metrics, and generate comprehensive comparison reports.

Example:
    Run as module:
        python -m src.reports.analyze_models

    Or import and use programmatically:
        from src.reports.analyze_models import analyze_all_experiments
        results = analyze_all_experiments()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, cast

import pandas as pd

# Constants
CONVERGENCE_THRESHOLD: Final[float] = 200.0
EXPERIMENTS_DIR: Final[Path] = Path("results/experiments")
OUTPUT_DIR: Final[Path] = Path("results/reports")
OUTPUT_CSV: Final[Path] = OUTPUT_DIR / "model_comparison.csv"
OUTPUT_JSON: Final[Path] = OUTPUT_DIR / "model_comparison.json"

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load experiment configuration from JSON file.

    Args:
        config_path: Path to config.json file.

    Returns:
        Dictionary containing config data, or None if loading fails.
    """
    try:
        with open(config_path, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return None


def _extract_config_data(config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    """Extract relevant fields from experiment configuration.

    Args:
        config: Configuration dictionary.
        experiment_id: Experiment identifier.

    Returns:
        Dictionary with extracted config fields.
    """
    return {
        "experiment_id": experiment_id,
        "algorithm": config.get("algorithm"),
        "seed": config.get("seed"),
        "timesteps": config.get("timesteps"),
        "gamma": config.get("gamma"),
        "ent_coef": config.get("ent_coef"),
        "learning_rate": config.get("learning_rate"),
    }


def _extract_training_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    """Extract training metrics from metrics.csv file.

    Args:
        metrics_path: Path to metrics.csv file.

    Returns:
        Dictionary with training metrics, or None if extraction fails.
    """
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            logger.warning(f"Empty metrics file: {metrics_path}")
            return None

        # Get final metrics (last row)
        last_row = df.iloc[-1]

        return {
            "final_train_reward": last_row.get("reward_mean", float("nan")),
            "final_train_std": last_row.get("reward_std", float("nan")),
            "total_training_time": df["walltime"].sum(),
            "convergence_status": (
                last_row.get("reward_mean", float("nan")) > CONVERGENCE_THRESHOLD
            ),
        }
    except (pd.errors.EmptyDataError, KeyError) as e:
        logger.warning(f"Failed to extract training metrics from {metrics_path}: {e}")
        return None


def _extract_eval_metrics(eval_path: Path) -> Optional[Dict[str, float]]:
    """Extract evaluation metrics from eval_log.csv file.

    Args:
        eval_path: Path to eval_log.csv file.

    Returns:
        Dictionary with evaluation metrics, or None if extraction fails.
    """
    try:
        df = pd.read_csv(eval_path)
        if df.empty:
            logger.warning(f"Empty eval log file: {eval_path}")
            return None

        # Best evaluation reward
        best_idx = df["mean_reward"].idxmax()
        best_row = df.loc[best_idx]

        # Final evaluation reward
        final_row = df.iloc[-1]

        return {
            "best_eval_reward": best_row["mean_reward"],
            "best_eval_std": best_row["std_reward"],
            "final_eval_reward": final_row["mean_reward"],
            "final_eval_std": final_row["std_reward"],
        }
    except (pd.errors.EmptyDataError, KeyError) as e:
        logger.warning(f"Failed to extract eval metrics from {eval_path}: {e}")
        return None


def _analyze_experiment(
    experiment_dir: Path, experiment_id: str
) -> Optional[Dict[str, Any]]:
    """Analyze a single experiment directory.

    Args:
        experiment_dir: Path to experiment directory.
        experiment_id: Experiment identifier (directory name).

    Returns:
        Dictionary with all extracted metrics, or None if analysis fails.
    """
    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.csv"
    eval_path = experiment_dir / "eval_log.csv"

    # Check required files
    if not config_path.exists():
        logger.debug(f"No config.json in {experiment_dir}, skipping")
        return None

    if not metrics_path.exists():
        logger.debug(f"No metrics.csv in {experiment_dir}, skipping")
        return None

    # Load and extract data
    config = _load_config(config_path)
    if config is None:
        return None

    config_data = _extract_config_data(config, experiment_id)
    training_metrics = _extract_training_metrics(metrics_path)

    if training_metrics is None:
        return None

    eval_metrics = None
    if eval_path.exists():
        eval_metrics = _extract_eval_metrics(eval_path)

    # Merge all data
    result = {**config_data, **training_metrics}
    if eval_metrics is not None:
        result.update(eval_metrics)

    return result


def _find_experiment_dirs(
    experiments_dir: Path,
) -> List[tuple[Path, str]]:
    """Find all experiment directories containing config.json.

    Args:
        experiments_dir: Root directory containing experiments.

    Returns:
        List of tuples (directory_path, experiment_id) for valid experiments.
    """
    experiment_dirs = []

    for item in experiments_dir.iterdir():
        if not item.is_dir():
            continue

        # Check if this directory has a config.json
        if (item / "config.json").exists():
            experiment_dirs.append((item, item.name))
        else:
            # Recursively check subdirectories (for demo structure)
            for sub_item in item.iterdir():
                if sub_item.is_dir() and (sub_item / "config.json").exists():
                    experiment_dirs.append((sub_item, sub_item.name))

    return experiment_dirs


def analyze_all_experiments(
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """Analyze all experiments and generate comparison DataFrame.

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        DataFrame containing comparison metrics for all experiments.
    """
    logger.info(f"Scanning experiments directory: {experiments_dir}")

    experiment_dirs = _find_experiment_dirs(experiments_dir)
    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    results: List[Dict[str, Any]] = []
    skipped = 0

    for exp_dir, exp_id in experiment_dirs:
        logger.debug(f"Analyzing experiment: {exp_id}")
        result = _analyze_experiment(exp_dir, exp_id)
        if result is not None:
            results.append(result)
        else:
            skipped += 1

    if not results:
        logger.error("No valid experiments found!")
        return pd.DataFrame()

    logger.info(f"Successfully analyzed {len(results)} experiments (skipped {skipped})")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Ensure required columns exist
    required_columns = [
        "experiment_id",
        "algorithm",
        "seed",
        "timesteps",
        "gamma",
        "ent_coef",
        "learning_rate",
        "final_train_reward",
        "final_train_std",
        "total_training_time",
        "convergence_status",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Add evaluation columns if missing
    for col in [
        "best_eval_reward",
        "best_eval_std",
        "final_eval_reward",
        "final_eval_std",
    ]:
        if col not in df.columns:
            df[col] = None

    # Sort by best_eval_reward descending (use final_eval_reward as fallback)
    sort_column = (
        "best_eval_reward" if "best_eval_reward" in df.columns else "final_eval_reward"
    )
    df = df.sort_values(by=sort_column, ascending=False, na_position="last")

    return df


def save_results(
    df: pd.DataFrame,
    output_csv: Path = OUTPUT_CSV,
    output_json: Path = OUTPUT_JSON,
) -> None:
    """Save analysis results to CSV and JSON files.

    Args:
        df: DataFrame containing comparison metrics.
        output_csv: Path for CSV output.
        output_json: Path for JSON output.
    """
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV to {output_csv}")

    # Save to JSON
    df.to_json(output_json, orient="records", indent=2)
    logger.info(f"Saved JSON to {output_json}")


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to console.

    Args:
        df: DataFrame containing comparison metrics.
    """
    if df.empty:
        print("\n❌ No experiments to analyze!")
        return

    total_experiments = len(df)
    converged = df["convergence_status"].sum()
    converged_pct = (
        (converged / total_experiments * 100) if total_experiments > 0 else 0
    )

    print("\n" + "=" * 80)
    print("📊 EXPERIMENT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments analyzed: {total_experiments}")
    print(
        f"Models converged (reward > {CONVERGENCE_THRESHOLD}): {converged} ({converged_pct:.1f}%)"
    )

    # Top 3 models
    print("\n🏆 TOP 3 MODELS (by best evaluation reward)")
    print("-" * 80)

    # Use best_eval_reward for ranking
    if "best_eval_reward" in df.columns and bool(df["best_eval_reward"].notna().any()):
        top_models = df.nlargest(3, "best_eval_reward")
        for idx, (_, row) in enumerate(top_models.iterrows(), 1):
            print(
                f"\n{idx}. {row['experiment_id']} ({row['algorithm']}, seed={row['seed']})"
            )
            print(
                f"   Best eval reward: {row['best_eval_reward']:.2f} ± {row['best_eval_std']:.2f}"
            )
            if bool(pd.notna(row["final_eval_reward"])):
                print(
                    f"   Final eval reward: {row['final_eval_reward']:.2f} ± {row['final_eval_std']:.2f}"
                )
            print(
                f"   Final train reward: {row['final_train_reward']:.2f} ± {row['final_train_std']:.2f}"
            )
            print(f"   Training time: {row['total_training_time']:.1f}s")
            print(
                f"   Config: timesteps={row['timesteps']}, gamma={row['gamma']}, "
                f"lr={row['learning_rate']}"
            )
    else:
        # Fallback to train reward
        top_models = df.nlargest(3, "final_train_reward")
        for idx, (_, row) in enumerate(top_models.iterrows(), 1):
            print(
                f"\n{idx}. {row['experiment_id']} ({row['algorithm']}, seed={row['seed']})"
            )
            print(
                f"   Final train reward: {row['final_train_reward']:.2f} ± {row['final_train_std']:.2f}"
            )
            print(f"   Training time: {row['total_training_time']:.1f}s")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point for the script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting model analysis")

    # Analyze all experiments
    df = analyze_all_experiments()

    if df.empty:
        logger.warning("No experiments to analyze")
        return

    # Save results
    save_results(df)

    # Print summary
    print_summary(df)

    logger.info("Model analysis complete")


if __name__ == "__main__":
    main()
