"""Analyze all trained RL models and generate comparison table."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.constants import (
    COL_EPISODE,
    COL_MEAN_REWARD,
    COL_STD_REWARD,
    COL_TIMESTEPS,
    DEFAULT_ENV_NAME,
    REWARD_THRESHOLD,
    STATUS_CONVERGED,
    STATUS_NOT_CONVERGED,
    STATUS_UNKNOWN,
)
from src.reporting.logging import get_logger, setup_logging
from src.reporting.types import ComparisonTable, HypothesisResult, ModelMetrics
from src.reporting.utils import (
    calculate_total_training_time,
    discover_experiments,
    extract_best_metrics,
    extract_final_metrics,
    get_model_path,
    read_config,
    read_eval_log,
    read_metrics_csv,
)

logger = get_logger(__name__)


# T013: Discover experiments
def discover_experiments(experiments_dir: Path) -> list[Path]:
    """Discover all valid experiments in directory.

    Args:
        experiments_dir: Path to experiments directory

    Returns:
        List of experiment directories containing config.json
    """
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    valid_experiments = []

    for item in experiments_dir.iterdir():
        if item.is_dir():
            config_path = item / "config.json"
            if config_path.exists():
                valid_experiments.append(item)
                logger.debug(f"Found valid experiment: {item.name}")

    logger.info(f"Found {len(valid_experiments)} valid experiments in {experiments_dir}")
    return valid_experiments


# T014: Read experiment configuration
def read_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    """Read experiment configuration from config.json.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config.json not found
    """
    config_path = experiment_dir / "config.json"
    config = read_config(config_path)

    # Validate required fields
    required_fields = ["algorithm", "environment"]
    for field in required_fields:
        if field not in config:
            logger.warning(f"Config missing required field: {field}")

    # Check seed (for reproducibility)
    if "seed" not in config:
        logger.warning(f"Config missing 'seed' field: {experiment_dir.name}")

    return config


# T015: Extract training metrics
def extract_training_metrics(experiment_dir: Path) -> tuple[float, float, float]:
    """Extract training metrics from metrics.csv.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Tuple of (final_mean_reward, final_std_reward, total_training_time)

    Raises:
        FileNotFoundError: If metrics.csv not found
    """
    metrics_path = experiment_dir / "metrics.csv"
    df = read_metrics_csv(metrics_path)

    # Extract final metrics (auto-detect column names)
    final_mean, final_std = extract_final_metrics(df)

    # Calculate total training time
    total_time = calculate_total_training_time(df)

    return final_mean, final_std, total_time


# T016: Extract evaluation metrics
def extract_eval_metrics(experiment_dir: Path) -> tuple[float, float, float, float]:
    """Extract evaluation metrics from eval_log.csv.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Tuple of (best_reward, best_std, final_reward, final_std)

    Raises:
        FileNotFoundError: If eval_log.csv not found
    """
    eval_path = experiment_dir / "eval_log.csv"

    # Check if eval log exists
    if not eval_path.exists():
        logger.warning(f"Eval log not found: {eval_path}")
        return 0.0, 0.0, 0.0, 0.0

    df = read_eval_log(eval_path)

    # Check number of eval episodes (should be 10-20)
    num_episodes = len(df)
    if num_episodes < 10 or num_episodes > 20:
        logger.warning(f"Eval episodes count {num_episodes} outside recommended range [10-20]")

    # Extract best metrics (auto-detect column names)
    best_reward, best_std, _ = extract_best_metrics(df)

    # Extract final metrics (auto-detect column names)
    final_reward, final_std = extract_final_metrics(df)

    return best_reward, best_std, final_reward, final_std


# T017: Validate reproducibility
def validate_reproducibility(experiment_dir: Path, config: dict[str, Any]) -> bool:
    """Validate that experiment is reproducible (seed documented).

    Args:
        experiment_dir: Path to experiment directory
        config: Experiment configuration

    Returns:
        True if reproducible (seed is set), False otherwise
    """
    # Check if seed is in config
    if "seed" not in config:
        logger.warning(f"Experiment missing seed: {experiment_dir.name}")
        return False

    seed = config["seed"]

    # Check if seed is valid (not None)
    if seed is None:
        logger.warning(f"Experiment has None seed: {experiment_dir.name}")
        return False

    logger.debug(f"Experiment reproducible with seed={seed}: {experiment_dir.name}")
    return True


# T018: Analyze hypothesis coverage
def analyze_hypothesis_coverage(table: ComparisonTable) -> list[HypothesisResult]:
    """Analyze which hypotheses are covered by experiments.

    Args:
        table: Comparison table with all model metrics

    Returns:
        List of hypothesis results
    """
    results = []

    # Hypothesis 1: PPO vs A2C
    has_ppo = any(m.algorithm == "PPO" for m in table.models)
    has_a2c = any(m.algorithm == "A2C" for m in table.models)

    if has_ppo and has_a2c:
        # Check which algorithm performs better
        ppo_rewards = [m.best_eval_reward for m in table.models if m.algorithm == "PPO"]
        a2c_rewards = [m.best_eval_reward for m in table.models if m.algorithm == "A2C"]

        avg_ppo = sum(ppo_rewards) / len(ppo_rewards) if ppo_rewards else 0
        avg_a2c = sum(a2c_rewards) / len(a2c_rewards) if a2c_rewards else 0

        supported = avg_ppo > avg_a2c
        evidence = f"PPO avg reward: {avg_ppo:.2f}, A2C avg reward: {avg_a2c:.2f}"

        results.append(
            HypothesisResult(
                hypothesis_id="H1",
                description="PPO will outperform A2C for LunarLander-v3",
                tested=True,
                supported=supported,
                evidence=evidence,
                recommendation="If hypothesis is false, consider investigating A2C hyperparameters" if not supported else None,
            )
        )
    else:
        results.append(
            HypothesisResult(
                hypothesis_id="H1",
                description="PPO will outperform A2C for LunarLander-v3",
                tested=False,
                supported=None,
                evidence=f"Missing experiments: PPO={has_ppo}, A2C={has_a2c}",
                recommendation="Run A2C experiments for comparison",
            )
        )

    # Hypothesis 2: Seed impact
    seeds = {m.seed for m in table.models if m.seed is not None}
    if len(seeds) >= 2:
        # Check variance across seeds
        rewards_by_seed = {}
        for seed in seeds:
            seed_rewards = [m.best_eval_reward for m in table.models if m.seed == seed]
            rewards_by_seed[seed] = seed_rewards

        evidence = f"Rewards by seed: {[(s, sum(r)/len(r)) for s, r in rewards_by_seed.items()]}"

        # Check if seed 42 (the fixed seed) performs well
        if 42 in rewards_by_seed:
            seed_42_reward = sum(rewards_by_seed[42]) / len(rewards_by_seed[42])
            other_seeds_reward = sum([sum(r)/len(r) for s, r in rewards_by_seed.items() if s != 42]) / len([s for s in seeds if s != 42])

            supported = seed_42_reward >= other_seeds_reward
            evidence += f". Seed 42: {seed_42_reward:.2f}, Other seeds: {other_seeds_reward:.2f}"
        else:
            supported = None

        results.append(
            HypothesisResult(
                hypothesis_id="H2",
                description="Different seeds will produce similar results (low variance)",
                tested=True,
                supported=supported,
                evidence=evidence,
                recommendation="If variance is high, increase n_steps or learning rate schedule" if supported is not None and not supported else None,
            )
        )
    else:
        results.append(
            HypothesisResult(
                hypothesis_id="H2",
                description="Different seeds will produce similar results (low variance)",
                tested=False,
                supported=None,
                evidence=f"Only {len(seeds)} seed(s) tested: {seeds}",
                recommendation="Run experiments with multiple seeds (42, 999, etc.)",
            )
        )

    return results


# T019: Generate experiment recommendations
def generate_experiment_recommendations(table: ComparisonTable, hypothesis_results: list[HypothesisResult]) -> list[str]:
    """Generate recommendations for additional experiments.

    Args:
        table: Comparison table with all model metrics
        hypothesis_results: List of hypothesis analysis results

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check for gaps in hypothesis coverage
    for result in hypothesis_results:
        if not result.tested:
            recommendations.append(result.recommendation)
        elif result.recommendation:
            recommendations.append(result.recommendation)

    # Check if any models converged
    converged_count = table.count_converged()
    if converged_count == 0:
        recommendations.append("No models achieved reward >=200. Consider increasing training timesteps or tuning hyperparameters.")

    # Check if converged models exist but few
    if 0 < converged_count < len(table.models) * 0.5:
        recommendations.append(f"Only {converged_count}/{len(table.models)} models converged. Consider hyperparameter tuning for non-converged models.")

    # Check for missing algorithms
    algorithms = {m.algorithm for m in table.models}
    if "PPO" not in algorithms:
        recommendations.append("No PPO experiments found. PPO is recommended for LunarLander-v3.")

    if "A2C" not in algorithms:
        recommendations.append("Consider running A2C experiments for algorithm comparison.")

    if not recommendations:
        recommendations.append("All hypotheses tested and sufficient experiments conducted. No additional experiments recommended.")

    return recommendations


# T020: Analyze all models
def analyze_all_models(
    experiments_dir: Path = Path("results/experiments"),
    output_dir: Path = Path("results/reports"),
    csv_output: Path = Path("results/reports/model_comparison.csv"),
    json_output: Path = Path("results/reports/model_comparison.json"),
) -> ComparisonTable:
    """Analyze all trained models and create comparison table.

    Args:
        experiments_dir: Directory with experiments
        output_dir: Directory for saving results
        csv_output: Path for CSV output
        json_output: Path for JSON output

    Returns:
        ComparisonTable with all model metrics
    """
    # Discover experiments
    experiment_dirs = discover_experiments(experiments_dir)

    if not experiment_dirs:
        raise ValueError(f"No valid experiments found in {experiments_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each experiment
    models = []
    for exp_dir in experiment_dirs:
        try:
            # Read config
            config = read_experiment_config(exp_dir)

            # Extract training metrics
            final_train_reward, final_train_std, total_time = extract_training_metrics(exp_dir)

            # Extract evaluation metrics
            best_reward, best_std, final_reward, final_std = extract_eval_metrics(exp_dir)

            # Get model path
            model_path = get_model_path(exp_dir)

            # Validate reproducibility
            is_reproducible = validate_reproducibility(exp_dir, config)

            # Determine convergence status
            if best_reward >= REWARD_THRESHOLD:
                convergence_status = STATUS_CONVERGED
            elif best_reward > 0:
                convergence_status = STATUS_NOT_CONVERGED
            else:
                convergence_status = STATUS_UNKNOWN

            # Create ModelMetrics
            metrics = ModelMetrics(
                experiment_id=exp_dir.name,
                algorithm=config.get("algorithm", "Unknown"),
                environment=config.get("environment", DEFAULT_ENV_NAME),
                seed=config.get("seed", None),
                timesteps=config.get("timesteps", None),
                gamma=config.get("gamma", None),
                ent_coef=config.get("ent_coef", None),
                learning_rate=config.get("learning_rate", None),
                model_path=model_path,
                final_train_reward=final_train_reward,
                final_train_std=final_train_std,
                best_eval_reward=best_reward,
                best_eval_std=best_std,
                final_eval_reward=final_reward,
                final_eval_std=final_std,
                total_training_time=total_time,
                convergence_status=convergence_status,
            )

            models.append(metrics)
            logger.info(f"Analyzed experiment: {exp_dir.name} - Reward: {best_reward:.2f}")

        except Exception as e:
            logger.error(f"Failed to analyze experiment {exp_dir.name}: {e}")
            continue

    # Create comparison table
    table = ComparisonTable(models=models)

    # Save to CSV
    save_comparison_csv(table, csv_output)

    # Save to JSON
    save_comparison_json(table, json_output)

    logger.info(f"Analysis complete. Total models: {len(table.models)}, Converged: {table.count_converged()}")

    return table


# T021: Save comparison to CSV
def save_comparison_csv(table: ComparisonTable, csv_path: Path) -> None:
    """Save comparison table to CSV file.

    Args:
        table: Comparison table to save
        csv_path: Path for CSV output
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_dict = table.to_dataframe_dict()
    df = pd.DataFrame(df_dict)

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to {csv_path}")


# T022: Save comparison to JSON
def save_comparison_json(table: ComparisonTable, json_path: Path) -> None:
    """Save comparison table to JSON file.

    Args:
        table: Comparison table to save
        json_path: Path for JSON output
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert ModelMetrics objects to dicts
    models_dict = []
    for m in table.models:
        models_dict.append({
            "experiment_id": m.experiment_id,
            "algorithm": m.algorithm,
            "environment": m.environment,
            "seed": m.seed,
            "timesteps": m.timesteps,
            "gamma": m.gamma,
            "ent_coef": m.ent_coef,
            "learning_rate": m.learning_rate,
            "model_path": str(m.model_path),
            "final_train_reward": m.final_train_reward,
            "final_train_std": m.final_train_std,
            "best_eval_reward": m.best_eval_reward,
            "best_eval_std": m.best_eval_std,
            "final_eval_reward": m.final_eval_reward,
            "final_eval_std": m.final_eval_std,
            "total_training_time": m.total_training_time,
            "convergence_status": m.convergence_status,
        })

    data = {
        "total_models": len(table.models),
        "converged_models": table.count_converged(),
        "models": models_dict,
        "top_models": [
            {"experiment_id": m.experiment_id, "best_eval_reward": m.best_eval_reward}
            for m in table.get_top_models()
        ],
        "generated_at": table.generated_at.isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved comparison table (JSON) to {json_path}")


# T023: Generate hypothesis coverage report
def generate_hypothesis_coverage_report(
    table: ComparisonTable,
    hypothesis_results: list[HypothesisResult],
    recommendations: list[str],
    output_path: Path = Path("results/reports/hypothesis_coverage.md"),
) -> None:
    """Generate markdown report on hypothesis coverage.

    Args:
        table: Comparison table
        hypothesis_results: List of hypothesis results
        recommendations: List of recommendations
        output_path: Path for markdown output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Hypothesis Coverage Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Models: {len(table.models)}",
        f"Converged Models: {table.count_converged()}\n",
        "## Hypotheses\n",
    ]

    for result in hypothesis_results:
        status_icon = "✅" if result.supported else ("❌" if result.supported is False else "⚠️")
        lines.append(f"### {result.hypothesis_id}: {result.description}")
        lines.append(f"- **Status**: {status_icon} Tested: {result.tested}")
        if result.tested:
            lines.append(f"- **Evidence**: {result.evidence}")
        if result.recommendation:
            lines.append(f"- **Recommendation**: {result.recommendation}")
        lines.append("")

    lines.append("## Recommendations\n")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")

    lines.append("\n## Top Models\n")
    for i, m in enumerate(table.get_top_models(), 1):
        lines.append(f"{i}. **{m.experiment_id}**: {m.best_eval_reward:.2f} ± {m.best_eval_std:.2f}")

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Saved hypothesis coverage report to {output_path}")


# T024: Main CLI entry point
def main() -> None:
    """CLI entry point for analyze_models."""
    parser = argparse.ArgumentParser(description="Analyze all trained RL models and generate comparison table")

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory with experiments (default: results/experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports"),
        help="Output directory for reports (default: results/reports)",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/reports/model_comparison.csv"),
        help="Path for CSV output (default: results/reports/model_comparison.csv)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/reports/model_comparison.json"),
        help="Path for JSON output (default: results/reports/model_comparison.json)",
    )
    parser.add_argument(
        "--check-hypotheses",
        action="store_true",
        help="Generate hypothesis coverage report",
    )
    parser.add_argument(
        "--suggest-experiments",
        action="store_true",
        help="Generate experiment recommendations",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    try:
        # Analyze all models
        table = analyze_all_models(
            experiments_dir=args.experiments_dir,
            output_dir=args.output_dir,
            csv_output=args.csv_output,
            json_output=args.json_output,
        )

        # Generate hypothesis report if requested
        if args.check_hypotheses or args.suggest_experiments:
            hypothesis_results = analyze_hypothesis_coverage(table)
            recommendations = generate_experiment_recommendations(table, hypothesis_results)

            if args.check_hypotheses:
                generate_hypothesis_coverage_report(
                    table,
                    hypothesis_results,
                    recommendations,
                    output_path=args.output_dir / "hypothesis_coverage.md",
                )

            if args.suggest_experiments:
                rec_path = args.output_dir / "experiment_recommendations.txt"
                with open(rec_path, "w") as f:
                    f.write("\n".join([f"{i+1}. {r}" for i, r in enumerate(recommendations)]))
                logger.info(f"Saved experiment recommendations to {rec_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
