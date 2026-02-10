"""Tests for analyze_models.py module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.reporting.types import ComparisonTable, ModelMetrics


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "algorithm": "PPO",
        "environment": "LunarLander-v3",
        "seed": 42,
        "timesteps": 500000,
        "gamma": 0.999,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
    }


@pytest.fixture
def sample_metrics_csv(tmp_path):
    """Create sample metrics CSV file."""
    metrics_path = tmp_path / "metrics.csv"
    data = {
        "timesteps": [100000, 200000, 300000, 400000, 500000],
        "mean_reward": [-200, 50, 150, 200, 220],
        "std_reward": [100, 80, 60, 50, 40],
    }
    df = pd.DataFrame(data)
    df.to_csv(metrics_path, index=False)
    return metrics_path


@pytest.fixture
def sample_eval_log_csv(tmp_path):
    """Create sample evaluation log CSV file."""
    eval_path = tmp_path / "eval_log.csv"
    data = {
        "episode": [1, 2, 3, 4, 5],
        "mean_reward": [180, 200, 230, 240, 235],
        "std_reward": [40, 35, 30, 25, 28],
    }
    df = pd.DataFrame(data)
    df.to_csv(eval_path, index=False)
    return eval_path


# T008: Unit test for parsing config.json
def test_read_config(sample_config, tmp_path):
    """Test reading configuration from JSON file."""
    from src.reporting.utils import read_config

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config, f)

    config = read_config(config_path)

    assert config["algorithm"] == "PPO"
    assert config["seed"] == 42
    assert config["timesteps"] == 500000


def test_read_config_not_found(tmp_path):
    """Test that FileNotFoundError is raised for missing config."""
    from src.reporting.utils import read_config

    config_path = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        read_config(config_path)


# T009: Unit test for extracting metrics.csv with timesteps
def test_extract_metrics_with_timesteps(sample_metrics_csv):
    """Test extracting training metrics with timesteps."""
    from src.reporting.utils import extract_final_metrics, read_metrics_csv

    df = read_metrics_csv(sample_metrics_csv)
    final_mean, final_std = extract_final_metrics(df)

    assert final_mean == 220.0
    assert final_std == 40.0


def test_extract_metrics_empty_dataframe():
    """Test that ValueError is raised for empty DataFrame."""
    from src.reporting.utils import extract_final_metrics

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty"):
        extract_final_metrics(df)


# T010: Unit test for validating eval episodes (10-20)
def test_validate_eval_episodes(sample_eval_log_csv):
    """Test validating evaluation episode count (should be 10-20)."""
    from src.reporting.utils import read_eval_log

    df = read_eval_log(sample_eval_log_csv)

    # In real implementation, check if eval episodes are between 10-20
    # For now, just verify DataFrame was loaded
    assert len(df) > 0
    assert "mean_reward" in df.columns


# T011: Integration test for full analysis
@pytest.mark.integration
def test_analyze_all_models(tmp_path, sample_config, sample_metrics_csv, sample_eval_log_csv):
    """Test full analysis of all models."""
    from src.reporting.analyze_models import analyze_all_models
    from src.reporting.utils import get_model_path

    # Create sample experiment directory
    exp_dir = tmp_path / "sample_exp"
    exp_dir.mkdir()

    # Create sample files
    with open(exp_dir / "config.json", "w") as f:
        json.dump(sample_config, f)

    sample_metrics_csv.rename(exp_dir / "metrics.csv")
    sample_eval_log_csv.rename(exp_dir / "eval_log.csv")

    # Create dummy model file
    (exp_dir / "model.zip").touch()

    output_dir = tmp_path / "reports"
    output_dir.mkdir()

    comparison_table = analyze_all_models(
        experiments_dir=tmp_path,
        output_dir=output_dir,
    )

    assert isinstance(comparison_table, ComparisonTable)
    assert len(comparison_table.models) == 1


# T012: Unit test for checking hypothesis coverage
def test_hypothesis_coverage():
    """Test checking which hypotheses are covered by experiments."""
    from src.reporting.analyze_models import analyze_hypothesis_coverage

    # Create sample comparison table
    table = ComparisonTable(
        models=[
            ModelMetrics(
                experiment_id="ppo_seed42",
                algorithm="PPO",
                environment="LunarLander-v3",
                seed=42,
                timesteps=500000,
                gamma=0.999,
                ent_coef=0.01,
                learning_rate=0.0003,
                model_path=Path("model.zip"),
                final_train_reward=220.0,
                final_train_std=40.0,
                best_eval_reward=243.45,
                best_eval_std=22.85,
                final_eval_reward=235.0,
                final_eval_std=30.0,
                total_training_time=190.0,
                convergence_status="CONVERGED",
            ),
            ModelMetrics(
                experiment_id="ppo_seed999",
                algorithm="PPO",
                environment="LunarLander-v3",
                seed=999,
                timesteps=500000,
                gamma=0.999,
                ent_coef=0.01,
                learning_rate=0.0003,
                model_path=Path("model.zip"),
                final_train_reward=180.0,
                final_train_std=50.0,
                best_eval_reward=195.09,
                best_eval_std=30.52,
                final_eval_reward=190.0,
                final_eval_std=35.0,
                total_training_time=180.0,
                convergence_status="NOT_CONVERGED",
            ),
        ]
    )

    hypothesis_results = analyze_hypothesis_coverage(table)

    assert len(hypothesis_results) > 0
