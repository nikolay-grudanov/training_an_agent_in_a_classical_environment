"""Results dataclasses for training experiments.

Provides structured classes for capturing and storing training results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingResult:
    """Complete results from a training experiment.

    Attributes:
        experiment_id: Unique identifier
        model_path: Path to saved model
        metrics_path: Path to metrics CSV
        evaluation_result: Final evaluation metrics
        training_duration_seconds: Wall-clock training time
        convergence_achieved: Whether convergence threshold was met
        completed_at: Timestamp of completion
    """

    experiment_id: str
    model_path: Path
    metrics_path: Path
    evaluation_result: Dict[str, float]
    training_duration_seconds: float
    convergence_achieved: bool
    completed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the training result
        """
        return {
            "experiment_id": self.experiment_id,
            "model_path": str(self.model_path),
            "metrics_path": str(self.metrics_path),
            "evaluation_result": self.evaluation_result,
            "training_duration_seconds": self.training_duration_seconds,
            "convergence_achieved": self.convergence_achieved,
            "completed_at": self.completed_at.isoformat(),
        }
