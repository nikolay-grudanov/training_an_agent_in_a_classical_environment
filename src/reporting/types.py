"""Common data types for reporting module."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelMetrics:
    """Metrics collected from a trained RL model.

    Attributes:
        experiment_id: Unique identifier for the experiment
        algorithm: RL algorithm used (PPO, A2C, etc.)
        environment: Gymnasium environment name
        seed: Random seed used for training
        timesteps: Total training timesteps
        gamma: Discount factor
        ent_coef: Entropy coefficient
        learning_rate: Learning rate
        model_path: Path to the trained model (.zip file)
        final_train_reward: Final training reward
        final_train_std: Standard deviation of final training reward
        best_eval_reward: Best evaluation reward
        best_eval_std: Standard deviation of best evaluation reward
        final_eval_reward: Final evaluation reward
        final_eval_std: Standard deviation of final evaluation reward
        total_training_time: Total training time in seconds
        convergence_status: "CONVERGED", "NOT_CONVERGED", or "UNKNOWN"
    """

    experiment_id: str
    algorithm: str
    environment: str
    seed: int | None
    timesteps: int | None
    gamma: float | None
    ent_coef: float | None
    learning_rate: float | None
    model_path: Path
    final_train_reward: float
    final_train_std: float
    best_eval_reward: float
    best_eval_std: float
    final_eval_reward: float
    final_eval_std: float
    total_training_time: float
    convergence_status: str

    def is_converged(self) -> bool:
        """Check if model achieved convergence (reward >= 200)."""
        return self.best_eval_reward >= 200.0


@dataclass
class ComparisonTable:
    """Comparison table for multiple trained models.

    Attributes:
        models: List of model metrics
        top_n: Number of top models to highlight
        generated_at: Timestamp when table was generated
    """

    models: list[ModelMetrics] = field(default_factory=list)
    top_n: int = 3
    generated_at: datetime = field(default_factory=datetime.now)

    def get_top_models(self) -> list[ModelMetrics]:
        """Get top-N models by best evaluation reward."""
        return sorted(self.models, key=lambda m: m.best_eval_reward, reverse=True)[: self.top_n]

    def count_converged(self) -> int:
        """Count models that achieved convergence (reward >= 200)."""
        return sum(1 for m in self.models if m.is_converged())

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """Convert to dictionary format for pandas DataFrame.

        Returns:
            Dictionary with column names as keys and lists as values
        """
        return {
            "experiment_id": [m.experiment_id for m in self.models],
            "algorithm": [m.algorithm for m in self.models],
            "environment": [m.environment for m in self.models],
            "seed": [m.seed for m in self.models],
            "timesteps": [m.timesteps for m in self.models],
            "gamma": [m.gamma for m in self.models],
            "ent_coef": [m.ent_coef for m in self.models],
            "learning_rate": [m.learning_rate for m in self.models],
            "model_path": [str(m.model_path) for m in self.models],
            "final_train_reward": [m.final_train_reward for m in self.models],
            "final_train_std": [m.final_train_std for m in self.models],
            "best_eval_reward": [m.best_eval_reward for m in self.models],
            "best_eval_std": [m.best_eval_std for m in self.models],
            "final_eval_reward": [m.final_eval_reward for m in self.models],
            "final_eval_std": [m.final_eval_std for m in self.models],
            "total_training_time": [m.total_training_time for m in self.models],
            "convergence_status": [m.convergence_status for m in self.models],
        }


@dataclass
class HypothesisResult:
    """Result of testing a hypothesis.

    Attributes:
        hypothesis_id: Unique identifier for hypothesis
        description: Description of the hypothesis
        tested: Whether hypothesis was tested
        supported: Whether hypothesis is supported by evidence
        evidence: Summary of evidence
        recommendation: Recommendation for further experiments
    """

    hypothesis_id: str
    description: str
    tested: bool
    supported: bool | None
    evidence: str
    recommendation: str | None
