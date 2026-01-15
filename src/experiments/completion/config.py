"""Configuration dataclasses for RL experiments.

Provides structured configuration for training experiments including
algorithm, environment, hyperparameters, and experiment metadata.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class Algorithm(str, Enum):
    """Supported RL algorithms."""

    A2C = "A2C"
    PPO = "PPO"


class ExperimentType(str, Enum):
    """Type of experiment."""

    BASELINE = "baseline"
    HYPERPARAMETER = "hyperparameter"


@dataclass
class ExperimentConfig:
    """Configuration for an RL training experiment.

    Attributes:
        experiment_id: Unique identifier
        experiment_type: Type of experiment
        algorithm: RL algorithm to use
        environment: Gymnasium environment ID
        timesteps: Training duration
        seed: Random seed for reproducibility
        gamma: Discount factor
        learning_rate: Learning rate (optional, uses default if not set)
        checkpoint_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of episodes for evaluation
        hypothesis: Expected outcome (optional)
    """

    experiment_id: str
    experiment_type: ExperimentType
    algorithm: Algorithm
    environment: str
    timesteps: int
    seed: int
    gamma: float
    learning_rate: Optional[float] = None
    checkpoint_freq: int = 50_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10
    hypothesis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type.value,
            "algorithm": self.algorithm.value,
            "environment": self.environment,
            "timesteps": self.timesteps,
            "seed": self.seed,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "checkpoint_freq": self.checkpoint_freq,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "hypothesis": self.hypothesis,
        }

    def save_to_json(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ExperimentConfig instance
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            experiment_id=data["experiment_id"],
            experiment_type=ExperimentType(data["experiment_type"]),
            algorithm=Algorithm(data["algorithm"]),
            environment=data["environment"],
            timesteps=data["timesteps"],
            seed=data["seed"],
            gamma=data["gamma"],
            learning_rate=data.get("learning_rate"),
            checkpoint_freq=data.get("checkpoint_freq", 50_000),
            eval_freq=data.get("eval_freq", 5_000),
            n_eval_episodes=data.get("n_eval_episodes", 10),
            hypothesis=data.get("hypothesis"),
        )
