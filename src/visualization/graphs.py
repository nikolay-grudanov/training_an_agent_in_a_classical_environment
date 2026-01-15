"""Graph generation utilities for RL experiment visualization.

Provides classes for generating:
- Learning curves (reward vs timesteps)
- Comparison plots (A2C vs PPO)
- Gamma comparison plots (hyperparameter study)
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Union


class LearningCurveGenerator:
    """Generator for training learning curve graphs.

    Attributes:
        width: Graph width in inches
        height: Graph height in inches
        dpi: Resolution
    """

    def __init__(self, width: int = 10, height: int = 6, dpi: int = 150) -> None:
        """Initialize graph generator.

        Args:
            width: Graph width
            height: Graph height
            dpi: Resolution
        """
        self.width = width
        self.height = height
        self.dpi = dpi

    def generate_from_metrics(
        self,
        metrics_csv: Union[str, Path],
        output_path: Union[str, Path],
        title: str = "Learning Curve",
        show_ci: bool = True,
    ) -> Figure:
        """Generate learning curve from metrics CSV.

        Args:
            metrics_csv: Path to metrics CSV file
            output_path: Path to save PNG output
            title: Graph title
            show_ci: Show confidence interval

        Returns:
            Matplotlib Figure object
        """
        df = pd.read_csv(metrics_csv)

        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi)

        # Main line
        ax.plot(
            df["timesteps"],
            df["reward_mean"],
            color="#2E86AB",
            linewidth=2,
            label="Mean Reward",
        )

        # Confidence interval
        if show_ci and "reward_std" in df.columns:
            ax.fill_between(
                df["timesteps"],
                df["reward_mean"] - df["reward_std"],
                df["reward_mean"] + df["reward_std"],
                color="#2E86AB",
                alpha=0.2,
                label="±1 Std Dev",
            )

        # Convergence threshold
        ax.axhline(
            y=200,
            color="#E94F37",
            linestyle="--",
            linewidth=1.5,
            label="Convergence Threshold (200)",
        )

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        return fig


class ComparisonPlotGenerator:
    """Generator for comparing multiple experiments."""

    def __init__(self, width: int = 12, height: int = 7, dpi: int = 150) -> None:
        """Initialize comparison plot generator.

        Args:
            width: Graph width
            height: Graph height
            dpi: Resolution
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    def generate(
        self,
        experiment_paths: List[Union[str, Path]],
        labels: List[str],
        output_path: Union[str, Path],
        title: str = "Algorithm Comparison",
    ) -> Figure:
        """Generate comparison plot for multiple experiments.

        Args:
            experiment_paths: List of paths to metrics CSV files
            labels: List of labels for each experiment
            output_path: Path to save PNG output
            title: Graph title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi)

        for i, (path, label) in enumerate(zip(experiment_paths, labels)):
            df = pd.read_csv(path)
            ax.plot(
                df["timesteps"],
                df["reward_mean"],
                color=self.colors[i % len(self.colors)],
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                df["timesteps"],
                df["reward_mean"] - df.get("reward_std", 0),
                df["reward_mean"] + df.get("reward_std", 0),
                color=self.colors[i % len(self.colors)],
                alpha=0.15,
            )

        ax.axhline(
            y=200,
            color="#E94F37",
            linestyle="--",
            linewidth=1.5,
            label="Threshold (200)",
        )

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        return fig


class GammaComparisonPlotGenerator(ComparisonPlotGenerator):
    """Generator specifically for gamma hyperparameter comparison."""

    def generate_gamma_comparison(
        self,
        gamma_results: Dict[float, str],
        output_path: str,
        title: str = "Gamma Hyperparameter Comparison",
    ) -> Figure:
        """Generate comparison plot for gamma experiments.

        Args:
            gamma_results: Dict mapping gamma values to metrics CSV paths
            output_path: Path to save PNG output
            title: Graph title

        Returns:
            Matplotlib Figure object
        """
        labels = [f"γ = {g}" for g in gamma_results.keys()]
        return super().generate(
            experiment_paths=list(gamma_results.values()),
            labels=labels,
            output_path=output_path,
            title=title,
        )
