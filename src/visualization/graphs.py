"""Graph generation utilities for RL experiment visualization.

Provides classes for generating:
- Learning curves (reward vs timesteps)
- Comparison plots (A2C vs PPO)
- Gamma comparison plots (hyperparameter study)
"""
import matplotlib
matplotlib.use('Agg')  # Перед plt!
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
        
        # Convert columns to numeric types
        for col in ['timesteps', 'reward_mean', 'reward_std']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with 0 for reward_std
        if 'reward_std' in df.columns:
            df['reward_std'] = df['reward_std'].fillna(0)
        
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
            
            # Convert columns to numeric types
            for col in ['timesteps', 'reward_mean', 'reward_std']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with 0 for reward_std
            if 'reward_std' in df.columns:
                df['reward_std'] = df['reward_std'].fillna(0)
            
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


def main() -> None:
    """CLI entry point for graph generation."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate performance graphs for RL experiments"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment ID (e.g., 'ppo_seed42') or comma-separated list for comparison",
    )
    parser.add_argument(
        "--type",
        choices=["learning_curve", "comparison", "gamma_comparison"],
        default="learning_curve",
        help="Graph type (default: learning_curve)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--title",
        default="Performance Graph",
        help="Graph title",
    )
    args = parser.parse_args()

    # Determine graph type and generate
    if args.type == "learning_curve":
        # Single experiment
        experiment_id = args.experiment
        metrics_path = Path(f"results/experiments/{experiment_id}/metrics.csv")

        if not metrics_path.exists():
            # Try JSON format
            json_path = Path(f"results/experiments/{experiment_id}/{experiment_id}_metrics.json")
            if json_path.exists():
                # Convert JSON to CSV
                import json
                with open(json_path, 'r') as f:
                    data = json.load(f)
                with open(metrics_path, 'w', newline='') as f:
                    f.write('timesteps,walltime,reward_mean,reward_std,episode_count,fps\n')
                    for m in data.get('metrics', []):
                        ts = m.get('timestep', 0)
                        reward = m.get('reward', 0)
                        # Use numeric values, handle missing fields
                        f.write(f'{ts},0,{reward},0,0,0\n')

        generator = LearningCurveGenerator()
        generator.generate_from_metrics(
            metrics_csv=str(metrics_path),
            output_path=args.output,
            title=args.title,
        )
        print(f"Graph saved: {args.output}")

    elif args.type in ("comparison", "gamma_comparison"):
        # Multiple experiments
        experiments = args.experiment.split(",")
        metrics_paths = [
            Path(f"results/experiments/{exp}/metrics.csv")
            for exp in experiments
        ]

        # Check and convert if needed
        for i, (exp, mp) in enumerate(zip(experiments, metrics_paths)):
            if not mp.exists():
                json_path = Path(f"results/experiments/{exp}/{exp}_metrics.json")
                if json_path.exists():
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    with open(mp, 'w', newline='') as f:
                        f.write('timesteps,walltime,reward_mean,reward_std,episode_count,fps\n')
                        for m in data.get('metrics', []):
                            ts = m.get('timestep', 0)
                            reward = m.get('reward', 0)
                            # Use numeric values, handle missing fields
                            f.write(f'{ts},0,{reward},0,0,0\n')

        if args.type == "gamma_comparison":
            # Extract gamma values from experiment names like "gamma_090"
            gamma_map = {}
            for exp in experiments:
                if exp.startswith("gamma_"):
                    gamma_value = float("0." + exp.split("_")[1])
                    gamma_map[gamma_value] = f"results/experiments/{exp}/metrics.csv"

            generator = GammaComparisonPlotGenerator()
            generator.generate_gamma_comparison(
                gamma_results=gamma_map,
                output_path=args.output,
                title=args.title,
            )
        else:
            generator = ComparisonPlotGenerator()
            generator.generate(
                experiment_paths=[str(mp) for mp in metrics_paths],
                labels=experiments,
                output_path=args.output,
                title=args.title,
            )
        print(f"Comparison graph saved: {args.output}")


if __name__ == "__main__":
    main()
