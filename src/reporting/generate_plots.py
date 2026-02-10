"""Plot generation functions for RL experiment visualization."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting.constants import (
    ALGO_A2C,
    ALGO_PPO,
    COL_EPISODE,
    COL_MEAN_REWARD,
    COL_STD_REWARD,
    COL_TIMESTEPS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE_COMPARISON,
    DEFAULT_FIGSIZE_LEARNING_CURVE,
    EVAL_COL_MEAN_REWARD,
    EVAL_COL_STD_REWARD,
    EVAL_COL_TIMESTEPS,
)
from src.reporting.logging import get_logger
from src.reporting.utils import discover_experiments, read_metrics_csv

logger = get_logger(__name__)

# Plot style settings
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except (OSError, ValueError):
    # Fallback if style not available
    plt.style.use("seaborn-darkgrid")


def _get_top_experiments(
    comparison_df: pd.DataFrame, top_n: int, algorithm: str | None = None
) -> list[dict[str, Any]]:
    """Get top-N experiments from comparison DataFrame.

    Args:
        comparison_df: DataFrame with model comparison data
        top_n: Number of top models to select
        algorithm: Optional algorithm filter (e.g., "PPO", "A2C")

    Returns:
        List of dictionaries with experiment metadata
    """
    # Sort by best_eval_reward descending
    df = comparison_df.sort_values("best_eval_reward", ascending=False)

    # Filter by algorithm if specified
    if algorithm:
        df = df[df["algorithm"] == algorithm]

    # Select top-N
    top_df = df.head(top_n)

    # Convert to list of dictionaries
    top_experiments = []
    for _, row in top_df.iterrows():
        experiment_id = row["experiment_id"]
        model_path = Path(row["model_path"])
        # Get experiment directory from model path
        experiment_dir = model_path.parent if model_path.parent.name != "model" else model_path.parent.parent

        top_experiments.append(
            {
                "experiment_id": experiment_id,
                "algorithm": row["algorithm"],
                "best_eval_reward": row["best_eval_reward"],
                "best_eval_std": row["best_eval_std"],
                "experiment_dir": experiment_dir,
            }
        )

    logger.info(f"Selected top {len(top_experiments)} experiments for plotting")
    return top_experiments


def load_evaluation_data(experiment_dir: Path) -> dict[str, np.ndarray] | None:
    """Load evaluation metrics from eval_log.csv file.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with keys:
        - 'timesteps': array of evaluation timesteps
        - 'mean_rewards': mean reward per evaluation
        - 'std_rewards': std deviation per evaluation
        - 'results': raw results matrix (episodes x evaluations)
        
        Returns None if file not found or loading fails.
    """
    from src.reporting.logging import get_logger
    logger = get_logger(__name__)
    
    eval_path = experiment_dir / "eval_log.csv"
    
    if not eval_path.exists():
        # Try alternative location
        eval_path = experiment_dir / "model" / "eval_log.csv"
    
    if not eval_path.exists():
        logger.warning(f"Evaluation log file not found for {experiment_dir.name}")
        return None
    
    try:
        df = read_metrics_csv(eval_path)
        
        # Validate required columns
        required_cols = ['timesteps', 'mean_reward']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in {eval_path}. Found: {list(df.columns)}")
            return None
        
        timesteps = np.array(df['timesteps'].values)
        mean_rewards = np.array(df['mean_reward'].values)
        
        # Get std deviation if available
        if 'std_reward' in df.columns:
            std_rewards = np.array(df['std_reward'].values)
        else:
            std_rewards = np.zeros_like(mean_rewards)
        
        # Create results matrix (mean rewards as single column)
        results = mean_rewards.reshape(-1, 1)
        
        return {
            'timesteps': timesteps,
            'mean_rewards': mean_rewards,
            'std_rewards': std_rewards,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error loading evaluation data from {eval_path}: {e}")
        return None


# ============================================================================
# T029: Generate reward vs timestep learning curve
# ============================================================================


def generate_reward_vs_timestep(
    experiments_dir: Path,
    output_path: Path,
    top_n: int = 3,
    comparison_csv: Path | None = None,
    use_eval_data: bool = True,
) -> None:
    """Generate learning curve plot (reward vs timestep) with error bands.

    Creates a line plot showing how reward changes over evaluation timesteps,
    with shaded regions showing standard deviation. Uses EVALUATION data from
    evaluations.npz files.

    Args:
        experiments_dir: Path to experiments directory
        output_path: Path to save the plot (PNG)
        top_n: Number of top models to plot
        comparison_csv: Optional path to comparison CSV (for selecting top models)
        use_eval_data: If True, use evaluations.npz (evaluation data); if False, use metrics.csv (training data)
    """
    logger.info(f"Generating reward vs timestep plot for top {top_n} models (use_eval_data={use_eval_data})")

    # Read comparison data if provided
    if comparison_csv and comparison_csv.exists():
        comparison_df = pd.read_csv(comparison_csv)
        top_experiments = _get_top_experiments(comparison_df, top_n)
    else:
        # Discover all experiments and use first N
        all_experiments = discover_experiments(experiments_dir)
        top_experiments = [
            {
                "experiment_id": exp.name,
                "algorithm": "Unknown",
                "experiment_dir": exp,
            }
            for exp in all_experiments[:top_n]
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_LEARNING_CURVE)

    # Plot each model's learning curve
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_experiments)))
    for i, exp_info in enumerate(top_experiments):
        exp_dir = exp_info["experiment_dir"]
        
        try:
            # Load evaluation metrics instead of training metrics
            eval_data = load_evaluation_data(exp_dir)
            
            if eval_data is None:
                logger.warning(f"Could not load evaluation data for {exp_info['experiment_id']}, skipping")
                continue
            
            timesteps = eval_data['timesteps']
            mean_rewards = eval_data['mean_rewards']
            std_rewards = eval_data['std_rewards']
            
            # Plot mean reward line
            ax.plot(
                timesteps,
                mean_rewards,
                label=exp_info["experiment_id"],
                color=colors[i],
                linewidth=2,
            )
            
            # Add std deviation band
            ax.fill_between(
                timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color=colors[i],
                alpha=0.2,
            )
            
        except Exception as e:
            logger.error(f"Error plotting {exp_info['experiment_id']}: {e}")
            continue

    # Set labels (Russian language)
    ax.set_xlabel("Временной шаг", fontsize=12, fontweight="bold")
    ax.set_ylabel("Награда", fontsize=12, fontweight="bold")
    ax.set_title("Кривая обучения: Награда vs Временной шаг", fontsize=14, fontweight="bold")

    # Add reference line for convergence threshold
    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Порог сходимости (200)")

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved reward vs timestep plot to {output_path}")

    plt.close()


# ============================================================================
# T030: Generate reward vs episode learning curve
# ============================================================================


def generate_reward_vs_episode(
    experiments_dir: Path,
    output_path: Path,
    top_n: int = 3,
    comparison_csv: Path | None = None,
    use_eval_data: bool = True,
) -> None:
    """Generate learning curve plot (reward vs evaluation number) with error bands.

    Creates a line plot showing how reward changes over evaluation runs,
    with shaded regions showing standard deviation. Uses EVALUATION data from
    evaluations.npz files.

    Args:
        experiments_dir: Path to experiments directory
        output_path: Path to save the plot (PNG)
        top_n: Number of top models to plot
        comparison_csv: Optional path to comparison CSV (for selecting top models)
        use_eval_data: If True, use evaluations.npz (evaluation data); if False, use metrics.csv (training data)
    """
    logger.info(f"Generating reward vs episode plot for top {top_n} models (use_eval_data={use_eval_data})")

    # Read comparison data if provided
    if comparison_csv and comparison_csv.exists():
        comparison_df = pd.read_csv(comparison_csv)
        top_experiments = _get_top_experiments(comparison_df, top_n)
    else:
        # Discover all experiments and use first N
        all_experiments = discover_experiments(experiments_dir)
        top_experiments = [
            {
                "experiment_id": exp.name,
                "algorithm": "Unknown",
                "experiment_dir": exp,
            }
            for exp in all_experiments[:top_n]
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_LEARNING_CURVE)

    # Plot each model's learning curve
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_experiments)))
    for i, exp_info in enumerate(top_experiments):
        exp_dir = exp_info["experiment_dir"]
        
        try:
            # Load evaluation data
            eval_data = load_evaluation_data(exp_dir)
            
            if eval_data is None:
                logger.warning(f"Could not load evaluation data for {exp_info['experiment_id']}, skipping")
                continue
            
            # For episode plot, use evaluation index as episode number
            eval_indices = np.arange(len(eval_data['mean_rewards']))
            mean_rewards = eval_data['mean_rewards']
            std_rewards = eval_data['std_rewards']
            
            # Plot mean reward line
            ax.plot(
                eval_indices,
                mean_rewards,
                label=exp_info["experiment_id"],
                color=colors[i],
                linewidth=2,
            )
            
            # Add std deviation band
            ax.fill_between(
                eval_indices,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color=colors[i],
                alpha=0.2,
            )
            
        except Exception as e:
            logger.error(f"Error plotting {exp_info['experiment_id']}: {e}")
            continue

    # Set labels (Russian language)
    ax.set_xlabel("Номер оценки (Evaluation #)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Награда", fontsize=12, fontweight="bold")
    ax.set_title("Кривая обучения: Награда vs Номер оценки", fontsize=14, fontweight="bold")

    # Add reference line for convergence threshold
    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Порог сходимости (200)")

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved reward vs episode plot to {output_path}")

    plt.close()


# ============================================================================
# T031: Generate comparison chart with quantitative evaluation
# ============================================================================


def generate_comparison_chart(
    comparison_csv: Path,
    output_path: Path,
    top_n: int = 3,
) -> None:
    """Generate bar chart comparison of top-N models with error bars.

    Creates a bar chart showing mean evaluation reward for top models,
    with error bars showing standard deviation. Each bar displays the
    quantitative value (mean ± std).

    Args:
        comparison_csv: Path to comparison CSV file
        output_path: Path to save the plot (PNG)
        top_n: Number of top models to display
    """
    logger.info(f"Generating comparison chart for top {top_n} models")

    # Read comparison data
    df = pd.read_csv(comparison_csv)

    # Sort by best_eval_reward and select top-N
    df_sorted = df.sort_values("best_eval_reward", ascending=False).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_COMPARISON)

    # Get data for plotting
    x_pos = np.arange(len(df_sorted))
    rewards = df_sorted["best_eval_reward"].values
    stds = df_sorted["best_eval_std"].values
    labels = df_sorted["experiment_id"].values
    algorithms = df_sorted["algorithm"].values

    # Create bar chart with error bars
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(
        x_pos,
        rewards,
        yerr=stds,
        align="center",
        alpha=0.8,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add quantitative labels (mean ± std) on top of bars
    for i, (bar, reward, std) in enumerate(zip(bars, rewards, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 5,
            f"{reward:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Set labels (Russian language)
    ax.set_xlabel("Модель", fontsize=12, fontweight="bold")
    ax.set_ylabel("Награда оценки", fontsize=12, fontweight="bold")
    ax.set_title(f"Сравнение топ-{top_n} моделей", fontsize=14, fontweight="bold")

    # Add convergence threshold line
    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Порог сходимости (200)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved comparison chart to {output_path}")

    plt.close()


# ============================================================================
# T032: Generate multi-algorithm comparison
# ============================================================================


def generate_multi_algorithm_comparison(
    experiments_dir: Path,
    comparison_csv: Path,
    output_path: Path,
    top_n_per_algorithm: int = 2,
    use_eval_data: bool = True,
) -> None:
    """Generate comparison plot for multiple algorithms on same figure.

    Creates a line plot showing learning curves for multiple algorithms
    (e.g., PPO vs A2C), with different colors/linestyles for each.
    Uses EVALUATION data from evaluations.npz files.

    Args:
        experiments_dir: Path to experiments directory
        comparison_csv: Path to comparison CSV file
        output_path: Path to save the plot (PNG)
        top_n_per_algorithm: Number of top models per algorithm to plot
        use_eval_data: If True, use evaluations.npz (evaluation data); if False, use metrics.csv (training data)
    """
    logger.info("Generating multi-algorithm comparison plot")

    # Read comparison data
    comparison_df = pd.read_csv(comparison_csv)

    # Get unique algorithms
    algorithms = comparison_df["algorithm"].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_LEARNING_CURVE)

    # Plot each algorithm's top models
    algorithm_styles = {
        ALGO_PPO: {"color": "blue", "linestyle": "-", "marker": "o"},
        ALGO_A2C: {"color": "green", "linestyle": "--", "marker": "s"},
    }

    for algo in algorithms:
        # Get top models for this algorithm
        top_experiments = _get_top_experiments(comparison_df, top_n_per_algorithm, algo)
        style = algorithm_styles.get(algo, {"color": "black", "linestyle": "-", "marker": "o"})

        for i, exp_info in enumerate(top_experiments):
            exp_dir = exp_info["experiment_dir"]
            
            try:
                # Load evaluation data instead of training metrics
                eval_data = load_evaluation_data(exp_dir)
                
                if eval_data is None:
                    logger.warning(f"Could not load evaluation data for {exp_info['experiment_id']}, skipping")
                    continue
                
                timesteps = eval_data['timesteps']
                mean_rewards = eval_data['mean_rewards']
                
                # Plot with algorithm-specific style
                ax.plot(
                    timesteps,
                    mean_rewards,
                    label=f"{algo}: {exp_info['experiment_id']}",
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3,
                    alpha=0.8,
                )
                
            except Exception as e:
                logger.error(f"Error plotting {exp_info['experiment_id']}: {e}")
                continue

    # Set labels (Russian language)
    ax.set_xlabel("Временной шаг", fontsize=12, fontweight="bold")
    ax.set_ylabel("Награда", fontsize=12, fontweight="bold")
    ax.set_title("Сравнение алгоритмов обучения", fontsize=14, fontweight="bold")

    # Add convergence threshold
    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="Порог сходимости (200)")

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved multi-algorithm comparison to {output_path}")

    plt.close()


# ============================================================================
# T033: Generate summary dashboard
# ============================================================================


def generate_summary_dashboard(
    experiments_dir: Path,
    comparison_csv: Path,
    output_path: Path,
    top_n: int = 3,
    use_eval_data: bool = True,
) -> None:
    """Generate summary dashboard with 2x2 combined visualization.

    Creates a 2x2 grid of plots:
    1. Reward vs timestep (EVALUATION data)
    2. Reward vs evaluation number (EVALUATION data)
    3. Comparison bar chart
    4. Convergence analysis

    Args:
        experiments_dir: Path to experiments directory
        comparison_csv: Path to comparison CSV file
        output_path: Path to save the dashboard (PNG)
        top_n: Number of top models to display
        use_eval_data: If True, use evaluations.npz (evaluation data); if False, use metrics.csv (training data)
    """
    logger.info("Generating summary dashboard")

    # Read comparison data
    comparison_df = pd.read_csv(comparison_csv)
    top_experiments = _get_top_experiments(comparison_df, top_n)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Сводная панель: Анализ результатов обучения", fontsize=16, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_experiments)))

    # Plot 1: Reward vs timesteps (EVALUATION DATA)
    ax1 = axes[0, 0]
    for i, exp_info in enumerate(top_experiments):
        eval_data = load_evaluation_data(exp_info["experiment_dir"])
        if eval_data is not None:
            ax1.plot(
                eval_data['timesteps'],
                eval_data['mean_rewards'],
                label=exp_info["experiment_id"],
                color=colors[i],
            )
            ax1.fill_between(
                eval_data['timesteps'],
                eval_data['mean_rewards'] - eval_data['std_rewards'],
                eval_data['mean_rewards'] + eval_data['std_rewards'],
                color=colors[i],
                alpha=0.15,
            )

    ax1.set_xlabel("Временной шаг", fontweight="bold")
    ax1.set_ylabel("Награда", fontweight="bold")
    ax1.set_title("Кривая обучения", fontweight="bold")
    ax1.axhline(y=200, color="red", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward vs evaluation number
    ax2 = axes[0, 1]
    for i, exp_info in enumerate(top_experiments):
        eval_data = load_evaluation_data(exp_info["experiment_dir"])
        if eval_data is not None:
            eval_indices = np.arange(len(eval_data['mean_rewards']))
            ax2.plot(
                eval_indices,
                eval_data['mean_rewards'],
                label=exp_info["experiment_id"],
                color=colors[i],
            )
            ax2.fill_between(
                eval_indices,
                eval_data['mean_rewards'] - eval_data['std_rewards'],
                eval_data['mean_rewards'] + eval_data['std_rewards'],
                color=colors[i],
                alpha=0.15,
            )

    ax2.set_xlabel("Номер оценки (Evaluation #)", fontweight="bold")
    ax2.set_ylabel("Награда", fontweight="bold")
    ax2.set_title("Награда по оценкам", fontweight="bold")
    ax2.axhline(y=200, color="red", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison bar chart (bottom-left)
    ax3 = axes[1, 0]
    df_sorted = comparison_df.sort_values("best_eval_reward", ascending=False).head(top_n)
    x_pos = np.arange(len(df_sorted))
    ax3.bar(
        x_pos,
        df_sorted["best_eval_reward"].values,
        yerr=df_sorted["best_eval_std"].values,
        align="center",
        alpha=0.8,
        capsize=5,
        color=colors[: len(df_sorted)],
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df_sorted["experiment_id"].values, rotation=15, ha="right", fontsize=8)
    ax3.set_xlabel("Модель", fontweight="bold")
    ax3.set_ylabel("Награда", fontweight="bold")
    ax3.set_title(f"Топ-{top_n} моделей", fontweight="bold")
    ax3.axhline(y=200, color="red", linestyle="--", alpha=0.5)
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Convergence analysis (bottom-right)
    ax4 = axes[1, 1]
    converged_count = comparison_df[comparison_df["best_eval_reward"] >= 200].shape[0]
    not_converged_count = len(comparison_df) - converged_count

    ax4.pie(
        [converged_count, not_converged_count],
        labels=["Сошлись", "Не сошлись"],
        autopct="%1.1f%%",
        colors=["green", "red"],
        startangle=90,
    )
    ax4.set_title(f"Анализ сходимости ({len(comparison_df)} моделей)", fontweight="bold")

    plt.tight_layout()

    # Save dashboard
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved summary dashboard to {output_path}")

    plt.close()


# ============================================================================
# T034: CLI entry point
# ============================================================================


def main() -> None:
    """Main CLI entry point for plot generation.

    Supports subcommands:
    - reward-vs-timestep: Generate learning curve (reward vs timestep)
    - reward-vs-episode: Generate learning curve (reward vs episode)
    - comparison: Generate comparison bar chart
    - multi-algorithm: Generate multi-algorithm comparison
    - dashboard: Generate summary dashboard (2x2)
    """
    parser = argparse.ArgumentParser(description="Generate visualization plots for RL experiments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--experiments-dir",
            type=Path,
            default=Path("results/experiments"),
            help="Path to experiments directory",
        )
        subparser.add_argument(
            "--comparison-csv",
            type=Path,
            default=Path("results/reports/model_comparison.csv"),
            help="Path to comparison CSV file",
        )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("results/reports"),
            help="Path to output directory",
        )
        subparser.add_argument(
            "--top-n",
            type=int,
            default=3,
            help="Number of top models to display",
        )
        subparser.add_argument(
            "--use-eval-data",
            action="store_true",
            default=True,
            help="Use eval_log.csv (evaluation data) instead of metrics.csv (training data). Default: True",
        )
        subparser.add_argument(
            "--use-training-data",
            action="store_true",
            default=False,
            help="Use metrics.csv (training data) instead of eval_log.csv (evaluation data). Default: False",
        )

    # reward-vs-timestep command
    timestep_parser = subparsers.add_parser("reward-vs-timestep", help="Generate reward vs timestep plot")
    add_common_args(timestep_parser)

    # reward-vs-episode command
    episode_parser = subparsers.add_parser("reward-vs-episode", help="Generate reward vs episode plot")
    add_common_args(episode_parser)

    # comparison command
    comparison_parser = subparsers.add_parser("comparison", help="Generate comparison bar chart")
    comparison_parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("results/reports/model_comparison.csv"),
        help="Path to comparison CSV file",
    )
    comparison_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports"),
        help="Path to output directory",
    )
    comparison_parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top models to display",
    )

    # multi-algorithm command
    multi_parser = subparsers.add_parser("multi-algorithm", help="Generate multi-algorithm comparison")
    add_common_args(multi_parser)
    multi_parser.add_argument(
        "--top-n-per-algo",
        type=int,
        default=2,
        help="Number of top models per algorithm",
    )

    # dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate summary dashboard")
    add_common_args(dashboard_parser)

    args = parser.parse_args()

    # Determine use_eval_data flag
    use_eval_data = True
    if hasattr(args, "use_training_data") and args.use_training_data:
        use_eval_data = False
    elif hasattr(args, "use_eval_data") and not args.use_eval_data:
        use_eval_data = False

    # Execute command
    if args.command == "reward-vs-timestep":
        output_path = args.output_dir / "reward_vs_timestep.png"
        generate_reward_vs_timestep(
            experiments_dir=args.experiments_dir,
            output_path=output_path,
            top_n=args.top_n,
            comparison_csv=args.comparison_csv,
            use_eval_data=use_eval_data,
        )
    elif args.command == "reward-vs-episode":
        output_path = args.output_dir / "reward_vs_episode.png"
        generate_reward_vs_episode(
            experiments_dir=args.experiments_dir,
            output_path=output_path,
            top_n=args.top_n,
            comparison_csv=args.comparison_csv,
            use_eval_data=use_eval_data,
        )
    elif args.command == "comparison":
        output_path = args.output_dir / "agent_comparison.png"
        generate_comparison_chart(
            comparison_csv=args.comparison_csv,
            output_path=output_path,
            top_n=args.top_n,
        )
    elif args.command == "multi-algorithm":
        output_path = args.output_dir / "multi_algorithm_comparison.png"
        generate_multi_algorithm_comparison(
            experiments_dir=args.experiments_dir,
            comparison_csv=args.comparison_csv,
            output_path=output_path,
            top_n_per_algorithm=getattr(args, "top_n_per_algo", 2),
            use_eval_data=use_eval_data,
        )
    elif args.command == "dashboard":
        output_path = args.output_dir / "summary_dashboard.png"
        generate_summary_dashboard(
            experiments_dir=args.experiments_dir,
            comparison_csv=args.comparison_csv,
            output_path=output_path,
            top_n=args.top_n,
            use_eval_data=use_eval_data,
        )
    else:
        parser.print_help()
        parser.exit(1)


if __name__ == "__main__":
    main()
