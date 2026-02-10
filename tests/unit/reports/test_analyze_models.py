"""Unit tests for analyze_models module."""

from pathlib import Path
from typing import Any

import pytest

from src.reports.analyze_models import (
    _extract_config_data,
    _extract_eval_metrics,
    _extract_training_metrics,
    _find_experiment_dirs,
    _load_config,
    analyze_all_experiments,
    CONVERGENCE_THRESHOLD,
)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample experiment configuration.

    Returns:
        Dictionary containing sample experiment configuration.
    """
    return {
        "experiment_id": "test_exp",
        "algorithm": "PPO",
        "seed": 42,
        "timesteps": 100000,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
    }


@pytest.fixture
def sample_metrics_file(tmp_path: Path) -> Path:
    """Create a sample metrics.csv file.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        Path to the created metrics.csv file.
    """
    metrics_file = tmp_path / "metrics.csv"
    metrics_data = """timesteps,walltime,reward_mean,reward_std,episode_count,fps
10000,10.5,150.0,20.0,10,1000
20000,21.0,180.0,15.0,20,950
30000,31.5,210.0,10.0,30,950
"""
    metrics_file.write_text(metrics_data)
    return metrics_file


@pytest.fixture
def sample_eval_file(tmp_path: Path) -> Path:
    """Create a sample eval_log.csv file.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        Path to the created eval_log.csv file.
    """
    eval_file = tmp_path / "eval_log.csv"
    eval_data = """timesteps,mean_reward,std_reward
5000,100.0,50.0
10000,150.0,40.0
15000,200.0,30.0
20000,180.0,35.0
"""
    eval_file.write_text(eval_data)
    return eval_file


def test_load_config(tmp_path: Path) -> None:
    """Test loading configuration from JSON file.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    config_data = {"experiment_id": "test", "algorithm": "PPO"}
    config_file = tmp_path / "config.json"
    config_file.write_text(config_data.__str__().replace("'", '"'))

    result = _load_config(config_file)
    assert result is not None
    assert result["experiment_id"] == "test"


def test_load_config_invalid(tmp_path: Path) -> None:
    """Test loading invalid configuration file.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    config_file = tmp_path / "config.json"
    config_file.write_text("invalid json")

    result = _load_config(config_file)
    assert result is None


def test_extract_config_data(sample_config: dict[str, Any]) -> None:
    """Test extracting data from configuration.

    Args:
        sample_config: Sample configuration fixture.
    """
    result = _extract_config_data(sample_config, "test_exp")
    assert result["experiment_id"] == "test_exp"
    assert result["algorithm"] == "PPO"
    assert result["seed"] == 42
    assert result["timesteps"] == 100000
    assert result["gamma"] == 0.99
    assert result["ent_coef"] == 0.01
    assert result["learning_rate"] == 0.0003


def test_extract_config_data_missing_fields() -> None:
    """Test extracting config with missing fields."""
    config = {"experiment_id": "test", "algorithm": "PPO"}
    result = _extract_config_data(config, "test")
    assert result["seed"] is None
    assert result["gamma"] is None
    assert result["learning_rate"] is None


def test_extract_training_metrics(sample_metrics_file: Path) -> None:
    """Test extracting training metrics from CSV.

    Args:
        sample_metrics_file: Sample metrics file fixture.
    """
    result = _extract_training_metrics(sample_metrics_file)
    assert result is not None
    assert abs(result["final_train_reward"] - 210.0) < 0.01
    assert abs(result["final_train_std"] - 10.0) < 0.01
    assert result["total_training_time"] == pytest.approx(63.0)
    assert result["convergence_status"] == (210.0 > CONVERGENCE_THRESHOLD)


def test_extract_training_metrics_empty_file(tmp_path: Path) -> None:
    """Test extracting metrics from empty file.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("timesteps,walltime,reward_mean,reward_std")
    result = _extract_training_metrics(empty_file)
    assert result is None


def test_extract_eval_metrics(sample_eval_file: Path) -> None:
    """Test extracting evaluation metrics from CSV.

    Args:
        sample_eval_file: Sample evaluation file fixture.
    """
    result = _extract_eval_metrics(sample_eval_file)
    assert result is not None
    assert abs(result["best_eval_reward"] - 200.0) < 0.01
    assert abs(result["best_eval_std"] - 30.0) < 0.01
    assert abs(result["final_eval_reward"] - 180.0) < 0.01
    assert abs(result["final_eval_std"] - 35.0) < 0.01


def test_extract_eval_metrics_empty_file(tmp_path: Path) -> None:
    """Test extracting eval metrics from empty file.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("timesteps,mean_reward,std_reward")
    result = _extract_eval_metrics(empty_file)
    assert result is None


def test_find_experiment_dirs(tmp_path: Path) -> None:
    """Test finding experiment directories.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    # Create directories with config.json
    exp1 = tmp_path / "exp1"
    exp1.mkdir()
    (exp1 / "config.json").write_text('{"experiment_id": "exp1"}')

    exp2 = tmp_path / "exp2"
    exp2.mkdir()
    (exp2 / "config.json").write_text('{"experiment_id": "exp2"}')

    # Create directory without config
    no_config = tmp_path / "no_config"
    no_config.mkdir()
    (no_config / "other.txt").write_text("test")

    # Create regular file
    (tmp_path / "file.txt").write_text("test")

    result = _find_experiment_dirs(tmp_path)
    assert len(result) == 2
    exp_ids = [exp_id for _, exp_id in result]
    assert "exp1" in exp_ids
    assert "exp2" in exp_ids


def test_analyze_all_experiments(tmp_path: Path) -> None:
    """Test analyzing all experiments.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    # Create experiment directory with all required files
    exp_dir = tmp_path / "test_exp"
    exp_dir.mkdir()

    # Config
    config_data = {
        "experiment_id": "test_exp",
        "algorithm": "PPO",
        "seed": 42,
        "timesteps": 100000,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
    }
    (exp_dir / "config.json").write_text(__import__("json").dumps(config_data))

    # Metrics
    metrics_data = """timesteps,walltime,reward_mean,reward_std,episode_count,fps
10000,10.5,150.0,20.0,10,1000
20000,21.0,250.0,10.0,20,950
"""
    (exp_dir / "metrics.csv").write_text(metrics_data)

    # Eval log
    eval_data = """timesteps,mean_reward,std_reward
10000,150.0,50.0
20000,250.0,30.0
"""
    (exp_dir / "eval_log.csv").write_text(eval_data)

    df = analyze_all_experiments(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == "test_exp"
    assert df.iloc[0]["algorithm"] == "PPO"
    assert abs(df.iloc[0]["best_eval_reward"] - 250.0) < 0.01
    assert bool(df.iloc[0]["convergence_status"]) is True


def test_analyze_all_experiments_no_experiments(tmp_path: Path) -> None:
    """Test analyzing with no valid experiments.

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    df = analyze_all_experiments(tmp_path)
    assert len(df) == 0


def test_analyze_all_experiments_incomplete(tmp_path: Path) -> None:
    """Test analyzing with incomplete experiments (missing metrics).

    Args:
        tmp_path: Temporary directory path provided by pytest.
    """
    exp_dir = tmp_path / "incomplete_exp"
    exp_dir.mkdir()

    # Only config, no metrics
    config_data = {"experiment_id": "incomplete", "algorithm": "PPO"}
    (exp_dir / "config.json").write_text(__import__("json").dumps(config_data))

    df = analyze_all_experiments(tmp_path)
    assert len(df) == 0
