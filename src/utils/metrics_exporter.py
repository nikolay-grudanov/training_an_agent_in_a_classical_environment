"""Metrics exporter for RL training experiments.

Per FR-008 and data-model.md requirements:
- Export training metrics to JSON format
- Implement TrainingMetrics entity serialization
- Implement ExperimentResults entity serialization
- Add timestamp and metadata fields
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Training metrics time-series data."""

    def __init__(
        self,
        experiment_id: str,
        algorithm: str,
        environment: str,
        seed: int,
        recording_interval: int = 100,
    ):
        self.metadata = {
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "environment": environment,
            "seed": seed,
            "recording_interval": recording_interval,
            "created_at": datetime.now().isoformat(),
        }
        self.time_series: list[dict[str, Any]] = []
        self.timestep_set: set[int] = set()

    def add_record(
        self,
        timestep: int,
        episode: int,
        reward: float,
        episode_length: int,
        loss: float | None = None,
    ) -> None:
        """Add a metrics record.

        Args:
            timestep: Current training timestep
            episode: Current episode number
            reward: Reward value
            episode_length: Length of current episode
            loss: Optional training loss value
        """
        if timestep in self.timestep_set:
            logger.warning(f"Duplicate timestep {timestep}, skipping")
            return

        record = {
            "timestep": timestep,
            "episode": episode,
            "reward": reward,
            "episode_length": episode_length,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        self.time_series.append(record)
        self.timestep_set.add(timestep)

    def calculate_aggregated(self) -> dict[str, Any]:
        """Calculate aggregated statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.time_series:
            return {
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "reward_min": 0.0,
                "reward_max": 0.0,
                "episode_length_mean": 0.0,
                "total_timesteps": 0,
            }

        import numpy as np

        rewards = [r["reward"] for r in self.time_series]
        episode_lengths = [r["episode_length"] for r in self.time_series]

        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "episode_length_mean": float(np.mean(episode_lengths)),
            "total_timesteps": max(self.timestep_set) if self.timestep_set else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "training_metrics": {
                "metadata": self.metadata,
                "time_series": self.time_series,
                "aggregated": self.calculate_aggregated(),
            }
        }

    def save(self, filepath: Path) -> None:
        """Save metrics to JSON file.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Training metrics saved to {filepath}")


class ExperimentResults:
    """Complete experiment results with metadata, metrics, and model info."""

    def __init__(
        self,
        experiment_id: str,
        algorithm: str,
        environment: str,
        seed: int,
        total_timesteps: int,
        conda_environment: str = "rocm",
    ):
        self.data = {
            "experiment_results": {
                "metadata": {
                    "experiment_id": experiment_id,
                    "algorithm": algorithm,
                    "environment": environment,
                    "seed": seed,
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "total_timesteps": total_timesteps,
                    "conda_environment": conda_environment,
                },
                "model": {
                    "algorithm": algorithm,
                    "policy": "MlpPolicy",
                    "model_file": f"{experiment_id}_model.zip",
                    "model_path": f"results/experiments/{experiment_id}/",
                    "checkpoint_interval": 1000,
                    "checkpoints": [],
                },
                "metrics": {
                    "final_reward_mean": 0.0,
                    "final_reward_std": 0.0,
                    "episode_length_mean": 0.0,
                    "total_episodes": 0,
                    "training_time_seconds": 0.0,
                    "converged": False,
                },
                "hyperparameters": {},
                "environment": {
                    "name": environment,
                    "observation_space": "Box(8,)",
                    "action_space": "Discrete(4)",
                    "reward_threshold": 200.0,
                },
            }
        }

    def update_model_info(
        self,
        model_file: str,
        policy: str = "MlpPolicy",
        checkpoint_interval: int = 1000,
        checkpoints: list[str] | None = None,
    ) -> None:
        """Update model information.

        Args:
            model_file: Model filename
            policy: Policy architecture
            checkpoint_interval: Steps between checkpoints
            checkpoints: List of checkpoint filenames
        """
        self.data["experiment_results"]["model"].update({
            "model_file": model_file,
            "policy": policy,
            "checkpoint_interval": checkpoint_interval,
            "checkpoints": checkpoints or [],
        })

    def update_metrics(
        self,
        final_reward_mean: float,
        final_reward_std: float,
        episode_length_mean: float,
        total_episodes: int,
        training_time_seconds: float,
        converged: bool,
    ) -> None:
        """Update training metrics.

        Args:
            final_reward_mean: Mean reward over evaluation episodes
            final_reward_std: Standard deviation of rewards
            episode_length_mean: Mean episode length
            total_episodes: Total episodes completed
            training_time_seconds: Training duration
            converged: Whether agent converged
        """
        self.data["experiment_results"]["metrics"].update({
            "final_reward_mean": final_reward_mean,
            "final_reward_std": final_reward_std,
            "episode_length_mean": episode_length_mean,
            "total_episodes": total_episodes,
            "training_time_seconds": training_time_seconds,
            "converged": converged,
        })

    def update_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameters.

        Args:
            **kwargs: Hyperparameter name-value pairs
        """
        self.data["experiment_results"]["hyperparameters"].update(kwargs)

    def update_environment_info(
        self,
        name: str,
        observation_space: str,
        action_space: str,
        reward_threshold: float = 200.0,
    ) -> None:
        """Update environment information.

        Args:
            name: Environment name
            observation_space: Observation space specification
            action_space: Action space specification
            reward_threshold: Success threshold
        """
        self.data["experiment_results"]["environment"].update({
            "name": name,
            "observation_space": observation_space,
            "action_space": action_space,
            "reward_threshold": reward_threshold,
        })

    def finalize(self) -> None:
        """Mark experiment as complete."""
        self.data["experiment_results"]["metadata"]["end_time"] = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return self.data

    def save(self, filepath: Path) -> None:
        """Save results to JSON file.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Experiment results saved to {filepath}")


def save_agent_metadata(
    experiment_id: str,
    algorithm: str,
    environment: str,
    seed: int,
    training_timesteps: int,
    filepath: Path,
    conda_environment: str = "rocm",
) -> None:
    """Save agent metadata JSON.

    Args:
        experiment_id: Experiment identifier
        algorithm: Algorithm name
        environment: Environment name
        seed: Random seed
        training_timesteps: Total training timesteps
        filepath: Output file path
        conda_environment: Conda environment name
    """
    from datetime import datetime

    metadata = {
        "agent_metadata": {
            "model_file": f"{experiment_id}_model.zip",
            "algorithm": algorithm,
            "policy": "MlpPolicy",
            "environment": environment,
            "seed": seed,
            "training_timesteps": training_timesteps,
            "training_date": datetime.now().isoformat(),
            "conda_environment": conda_environment,
            "file_size_bytes": 0,
            "checksum": "sha256:pending",
        }
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Agent metadata saved to {filepath}")


if __name__ == "__main__":
    # Test metrics exporter
    print("Testing metrics exporter...")

    # Create training metrics
    metrics = TrainingMetrics(
        experiment_id="ppo_seed42",
        algorithm="PPO",
        environment="LunarLander-v3",
        seed=42,
        recording_interval=100,
    )

    # Add some test records
    for i in range(10):
        metrics.add_record(
            timestep=i * 100,
            episode=i * 10,
            reward=-100 + i * 30,  # Improving over time
            episode_length=100 + i * 5,
            loss=0.5 - i * 0.04,
        )

    # Save training metrics
    metrics_path = Path("results/experiments/ppo_seed42/test_metrics.json")
    metrics.save(metrics_path)
    print(f"✅ Training metrics saved: {metrics_path}")

    # Create experiment results
    results = ExperimentResults(
        experiment_id="ppo_seed42_test",
        algorithm="PPO",
        environment="LunarLander-v3",
        seed=42,
        total_timesteps=1000,
    )

    results.update_metrics(
        final_reward_mean=150.5,
        final_reward_std=12.3,
        episode_length_mean=180.0,
        total_episodes=100,
        training_time_seconds=120.0,
        converged=True,
    )

    results.update_hyperparameters(
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )

    results.finalize()

    results_path = Path("results/experiments/ppo_seed42/test_results.json")
    results.save(results_path)
    print(f"✅ Experiment results saved: {results_path}")

    print("\n✅ Metrics exporter tests passed!")
