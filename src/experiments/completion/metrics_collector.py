"""Metrics collection utilities for RL training.

Provides classes for collecting, storing, and exporting training metrics
including rewards, timesteps, and episode statistics.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List


class MetricsCollector:
    """Collect and store training metrics during RL experiments.

    Attributes:
        experiment_id: Unique identifier for the experiment
        metrics_path: Path to save metrics CSV file
        data: List of metric dictionaries
    """

    def __init__(self, experiment_id: str, metrics_path: Path) -> None:
        """Initialize metrics collector.

        Args:
            experiment_id: Unique identifier for the experiment
            metrics_path: Path to save metrics CSV file
        """
        self.experiment_id = experiment_id
        self.metrics_path = Path(metrics_path)
        self.data: List[Dict[str, Any]] = []
        self._current_episode_reward = 0.0
        self._episode_count = 0
        self._write_header()

    def _write_header(self) -> None:
        """Write CSV header to metrics file."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timesteps",
                    "walltime",
                    "reward_mean",
                    "reward_std",
                    "episode_count",
                    "fps",
                ],
            )
            writer.writeheader()

    def log(
        self,
        timesteps: int,
        reward_mean: float,
        reward_std: float,
        walltime: float,
        fps: float,
    ) -> None:
        """Log a metrics data point.

        Args:
            timesteps: Cumulative timesteps completed
            reward_mean: Mean reward over evaluation window
            reward_std: Standard deviation of rewards
            walltime: Wall-clock time in seconds
            fps: Training speed (frames per second)
        """
        row = {
            "timesteps": timesteps,
            "walltime": walltime,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "episode_count": self._episode_count,
            "fps": fps,
        }
        self.data.append(row)
        self._append_to_csv(row)

    def _append_to_csv(self, row: Dict[str, Any]) -> None:
        """Append a row to the CSV file."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.data:
            return {}

        rewards = [d["reward_mean"] for d in self.data]
        return {
            "final_reward": rewards[-1] if rewards else None,
            "max_reward": max(rewards) if rewards else None,
            "min_reward": min(rewards) if rewards else None,
            "data_points": len(self.data),
        }
