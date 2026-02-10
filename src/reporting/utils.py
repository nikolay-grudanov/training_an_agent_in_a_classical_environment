"""File utility functions for reporting module."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def read_config(config_path: Path) -> dict[str, Any]:
    """Read experiment configuration from JSON file.

    Args:
        config_path: Path to config.json

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    logger.debug(f"Loaded config from {config_path}: {config}")
    return config


def read_metrics_csv(metrics_path: Path) -> pd.DataFrame:
    """Read training metrics from CSV file.

    Args:
        metrics_path: Path to metrics.csv

    Returns:
        DataFrame with training metrics

    Raises:
        FileNotFoundError: If metrics file doesn't exist
        ValueError: If CSV is empty or invalid
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    if df.empty:
        raise ValueError(f"Metrics file is empty: {metrics_path}")

    logger.debug(f"Loaded {len(df)} rows from {metrics_path}")
    return df


def read_eval_log(eval_log_path: Path) -> pd.DataFrame:
    """Read evaluation log from CSV file.

    Args:
        eval_log_path: Path to eval_log.csv

    Returns:
        DataFrame with evaluation metrics

    Raises:
        FileNotFoundError: If eval log file doesn't exist
        ValueError: If CSV is empty or invalid
    """
    if not eval_log_path.exists():
        raise FileNotFoundError(f"Eval log file not found: {eval_log_path}")

    df = pd.read_csv(eval_log_path)

    if df.empty:
        raise ValueError(f"Eval log file is empty: {eval_log_path}")

    logger.debug(f"Loaded {len(df)} evaluation episodes from {eval_log_path}")
    return df


def extract_final_metrics(df: pd.DataFrame, reward_col: str | None = None, std_col: str | None = None) -> tuple[float, float]:
    """Extract final metrics from DataFrame (last row).

    Args:
        df: DataFrame with metrics
        reward_col: Column name for reward (auto-detected if None)
        std_col: Column name for standard deviation (auto-detected if None)

    Returns:
        Tuple of (mean_reward, std_reward) from last row

    Raises:
        ValueError: If DataFrame is empty or columns don't exist
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Auto-detect reward column
    if reward_col is None:
        for col in ["reward_mean", "mean_reward", "reward"]:
            if col in df.columns:
                reward_col = col
                break

    if reward_col is None:
        raise ValueError(f"Reward column not found in DataFrame. Available columns: {list(df.columns)}")

    last_row = df.iloc[-1]
    mean_reward = float(last_row[reward_col])

    # Auto-detect std column
    if std_col is None:
        for col in ["reward_std", "std_reward", "std"]:
            if col in df.columns:
                std_col = col
                break

    if std_col and std_col in df.columns:
        std_reward = float(last_row[std_col])
    else:
        std_reward = 0.0

    return mean_reward, std_reward


def extract_best_metrics(
    df: pd.DataFrame, reward_col: str | None = None, std_col: str | None = None
) -> tuple[float, float, int]:
    """Extract best metrics from DataFrame (max reward row).

    Args:
        df: DataFrame with metrics
        reward_col: Column name for reward (auto-detected if None)
        std_col: Column name for standard deviation (auto-detected if None)

    Returns:
        Tuple of (best_reward, best_std, best_episode_index)

    Raises:
        ValueError: If DataFrame is empty or columns don't exist
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Auto-detect reward column
    if reward_col is None:
        for col in ["reward_mean", "mean_reward", "reward"]:
            if col in df.columns:
                reward_col = col
                break

    if reward_col is None:
        raise ValueError(f"Reward column not found in DataFrame. Available columns: {list(df.columns)}")

    # Find index of max reward
    best_idx = df[reward_col].idxmax()
    best_row = df.loc[best_idx]

    best_reward = float(best_row[reward_col])

    # Auto-detect std column
    if std_col is None:
        for col in ["reward_std", "std_reward", "std"]:
            if col in df.columns:
                std_col = col
                break

    if std_col and std_col in df.columns:
        best_std = float(best_row[std_col])
    else:
        best_std = 0.0

    # Try to get episode number from 'episode_count' or 'episode' column
    if "episode_count" in df.columns:
        best_episode = int(best_row["episode_count"])
    elif "episode" in df.columns:
        best_episode = int(best_row["episode"])
    else:
        # Use DataFrame index position
        best_episode = int(best_idx) if isinstance(best_idx, (int, float)) else 0

    return best_reward, best_std, best_episode


def calculate_total_training_time(df: pd.DataFrame, time_col: str = "walltime") -> float:
    """Calculate total training time from DataFrame.

    Args:
        df: DataFrame with training metrics
        time_col: Column name for wall time (default: "walltime")

    Returns:
        Total training time in seconds

    Raises:
        ValueError: If time column doesn't exist or DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if time_col not in df.columns:
        return 0.0  # No time data available

    # Calculate sum of walltime (or use last value if it's cumulative)
    time_values = df[time_col].dropna()
    if time_values.empty:
        return 0.0

    total_time = float(time_values.sum())
    return total_time


def discover_experiments(experiments_dir: Path) -> list[Path]:
    """Discover all valid experiments in directory.

    Args:
        experiments_dir: Path to experiments directory

    Returns:
        List of experiment directories containing config.json
    """
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    valid_experiments = []

    for item in experiments_dir.iterdir():
        if item.is_dir():
            config_path = item / "config.json"
            if config_path.exists():
                valid_experiments.append(item)
                logger.debug(f"Found valid experiment: {item.name}")

    logger.info(f"Found {len(valid_experiments)} valid experiments in {experiments_dir}")
    return valid_experiments


def get_model_path(experiment_dir: Path) -> Path:
    """Find the best model path in experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Path to model .zip file

    Raises:
        FileNotFoundError: If no model file found
    """
    # Check for best_model.zip first
    best_model = experiment_dir / "best_model.zip"
    if best_model.exists():
        return best_model

    # Check for model.zip
    model = experiment_dir / "model.zip"
    if model.exists():
        return model

    # Check for *model.zip
    model_files = list(experiment_dir.glob("*model.zip"))
    if model_files:
        return model_files[0]

    # Check checkpoints for best model
    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists():
        # Find all checkpoint files and sort them
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_*.zip"), reverse=True)
        if checkpoint_files:
            return checkpoint_files[0]

    raise FileNotFoundError(f"No model file found in {experiment_dir}")
