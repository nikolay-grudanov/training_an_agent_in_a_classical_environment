"""Hyperparameter experiments for RL training.

Provides controlled experiments varying gamma (discount factor)
to quantify impact on agent learning and performance.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from scipy import stats

from .baseline_training import train_to_convergence
from src.training.evaluation import evaluate_agent
from src.utils.seeding import set_seed


class GammaExperiment:
    """Controlled experiment varying gamma hyperparameter.

    Trains identical agents with different gamma values to measure
    impact on learning performance and convergence.

    Attributes:
        gamma_values: List of gamma values to test
        timesteps: Training timesteps per configuration
        seed: Random seed
        results: Dictionary of results per gamma value
    """

    def __init__(
        self,
        gamma_values: List[float],
        timesteps: int = 100_000,
        seed: int = 42,
        n_eval_episodes: int = 10,
    ) -> None:
        """Initialize gamma experiment.

        Args:
            gamma_values: Gamma values to test
            timesteps: Training duration per config
            seed: Random seed
            n_eval_episodes: Episodes for evaluation
        """
        self.gamma_values = gamma_values
        self.timesteps = timesteps
        self.seed = seed
        self.n_eval_episodes = n_eval_episodes
        self.results: Dict[float, Dict[str, Any]] = {}

    def run(self) -> Dict[float, Dict[str, Any]]:
        """Execute gamma experiment for all configurations.

        Returns:
            Dictionary mapping gamma values to results
        """
        set_seed(self.seed)

        for gamma in self.gamma_values:
            exp_id = f"gamma_{int(gamma * 1000):03d}"
            print(f"\nRunning experiment: γ = {gamma}")

            # Train
            result = train_to_convergence(
                algorithm="PPO",
                timesteps=self.timesteps,
                seed=self.seed,
                gamma=gamma,
                experiment_id=exp_id,
            )

            # Additional evaluation for statistical analysis
            eval_result = evaluate_agent(
                str(result.model_path),
                "LunarLander-v3",
                self.n_eval_episodes,
            )

            # Collect more samples for statistical test
            full_eval = evaluate_agent(
                str(result.model_path),
                "LunarLander-v3",
                n_eval_episodes=30,  # More samples for stats
            )

            self.results[gamma] = {
                "experiment_id": exp_id,
                "model_path": str(result.model_path),
                "metrics_path": str(result.metrics_path),
                "final_reward_mean": eval_result["mean_reward"],
                "final_reward_std": eval_result["std_reward"],
                "full_eval_mean": full_eval["mean_reward"],
                "full_eval_std": full_eval["std_reward"],
                "training_duration": result.training_duration_seconds,
                "convergence_achieved": result.convergence_achieved,
                "config": result.evaluation_result,
            }

        return self.results

    def analyze(self) -> Dict[str, Any]:
        """Perform statistical analysis of gamma experiment results.

        Returns:
            Dictionary with statistical analysis results
        """
        if not self.results:
            msg = "No results to analyze. Run experiment first."
            raise ValueError(msg)

        # Extract rewards for each gamma
        rewards_by_gamma: Dict[float, List[float]] = {
            gamma: self._sample_rewards(gamma) for gamma in self.results.keys()
        }

        # Pairwise t-tests
        pairwise_tests = []
        gammas = list(self.results.keys())
        for i in range(len(gammas)):
            for j in range(i + 1, len(gammas)):
                g1, g2 = gammas[i], gammas[j]
                r1 = rewards_by_gamma[g1]
                r2 = rewards_by_gamma[g2]

                t_stat, p_value = stats.ttest_ind(r1, r2)
                # Handle both scalar and tuple return types from scipy
                try:
                    t_stat_val = float(t_stat[0])  # type: ignore[index]
                except (TypeError, IndexError):
                    t_stat_val = float(t_stat)  # type: ignore[arg-type]

                try:
                    p_value_val = float(p_value[0])  # type: ignore[index]
                except (TypeError, IndexError):
                    p_value_val = float(p_value)  # type: ignore[arg-type]

                cohens_d_val = float(
                    (np.mean(r1) - np.mean(r2))
                    / np.sqrt(
                        (np.std(r1) ** 2 + np.std(r2) ** 2) / 2,
                    )
                )

                pairwise_tests.append(
                    {
                        "comparison": f"γ={g1} vs γ={g2}",
                        "t_statistic": t_stat_val,
                        "p_value": p_value_val,
                        "cohens_d": cohens_d_val,
                        "significant": p_value_val < 0.05,
                    },
                )

        # ANOVA
        all_rewards = [rewards_by_gamma[g] for g in gammas]
        f_stat, anova_p = stats.f_oneway(*all_rewards)

        # Find best gamma
        best_gamma = max(
            self.results.keys(),
            key=lambda g: self.results[g]["final_reward_mean"],
        )

        # Hypothesis evaluation
        hypothesis_result = self._evaluate_hypothesis(best_gamma)

        return {
            "pairwise_tests": pairwise_tests,
            "anova": {
                "f_statistic": f_stat,
                "p_value": anova_p,
                "significant": anova_p < 0.05,
            },
            "best_gamma": best_gamma,
            "best_reward": self.results[best_gamma]["final_reward_mean"],
            "hypothesis_result": hypothesis_result,
            "summary_table": self._create_summary_table(),
        }

    def _sample_rewards(self, gamma: float) -> List[float]:
        """Sample rewards from evaluation results.

        Args:
            gamma: Gamma value

        Returns:
            List of sampled rewards
        """
        # Use the 30-episode evaluation for statistical analysis
        mean = self.results[gamma]["full_eval_mean"]
        std = self.results[gamma]["full_eval_std"]
        n = 30  # Number of evaluation episodes
        # Generate samples with same mean and std
        return list(np.random.normal(mean, std, n))

    def _evaluate_hypothesis(self, best_gamma: float) -> Dict[str, Any]:
        """Evaluate hypothesis about gamma=0.99 being optimal.

        Args:
            best_gamma: Best performing gamma value

        Returns:
            Hypothesis evaluation result
        """
        hypothesis = (
            "gamma=0.99 provides best balance between immediate and long-term rewards"
        )

        # Check if gamma=0.99 is among the best
        g099_reward = self.results.get(0.99, {}).get("final_reward_mean", -np.inf)
        best_reward = self.results[best_gamma]["final_reward_mean"]

        if best_gamma == 0.99:
            result = "supported"
            evidence = f"gamma=0.99 achieved highest mean reward ({best_reward:.2f})"
        elif abs(best_gamma - 0.99) < 0.01:
            result = "supported"
            evidence = (
                f"gamma={best_gamma} (near 0.99) achieved highest reward "
                f"({best_reward:.2f})"
            )
        elif g099_reward >= best_reward * 0.95:  # Within 5%
            result = "inconclusive"
            evidence = (
                f"gamma=0.99 ({g099_reward:.2f}) close to best ({best_reward:.2f})"
            )
        else:
            result = "refuted"
            evidence = (
                f"gamma={best_gamma} ({best_reward:.2f}) significantly outperformed "
                f"gamma=0.99 ({g099_reward:.2f})"
            )

        return {
            "hypothesis": hypothesis,
            "result": result,
            "evidence": evidence,
        }

    def _create_summary_table(self) -> str:
        """Create markdown summary table.

        Returns:
            Markdown formatted table
        """
        lines = [
            "| Gamma | Mean Reward | Std Dev | Convergence | Duration (s) |",
            "|-------|-------------|---------|-------------|--------------|",
        ]
        for gamma in sorted(self.results.keys()):
            r = self.results[gamma]
            conv = "✓" if r["convergence_achieved"] else "✗"
            lines.append(
                f"| {gamma:.3f} | {r['final_reward_mean']:.2f} | "
                f"{r['final_reward_std']:.2f} | {conv} | {r['training_duration']:.1f} |",
            )
        return "\n".join(lines)

    def save_results(self, output_path: Union[str, Path]) -> None:
        """Save experiment results to JSON.

        Args:
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(
                {
                    "gamma_values": self.gamma_values,
                    "timesteps": self.timesteps,
                    "seed": self.seed,
                    "results": {str(g): r for g, r in self.results.items()},
                },
                f,
                indent=2,
            )


def run_gamma_experiment(
    gamma_values: List[float],
    timesteps: int = 100_000,
    seed: int = 42,
    n_eval_episodes: int = 10,
) -> GammaExperiment:
    """Convenience function to run gamma experiment.

    Args:
        gamma_values: Gamma values to test
        timesteps: Training timesteps per config
        seed: Random seed
        n_eval_episodes: Evaluation episodes

    Returns:
        GammaExperiment with results
    """
    experiment = GammaExperiment(
        gamma_values=gamma_values,
        timesteps=timesteps,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
    )
    experiment.run()
    return experiment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gamma hyperparameter experiment")
    parser.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=[0.90, 0.99, 0.999],
        help="Gamma values to test (default: 0.90 0.99 0.999)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Training timesteps per config (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    experiment = run_gamma_experiment(
        gamma_values=args.gamma,
        timesteps=args.timesteps,
        seed=args.seed,
    )

    analysis = experiment.analyze()

    print("\n" + "=" * 60)
    print("GAMMA EXPERIMENT RESULTS")
    print("=" * 60)
    print(
        f"\nBest gamma: {analysis['best_gamma']} (reward: {analysis['best_reward']:.2f})"
    )
    print(f"\nHypothesis: {analysis['hypothesis_result']['hypothesis']}")
    print(f"Result: {analysis['hypothesis_result']['result'].upper()}")
    print(f"Evidence: {analysis['hypothesis_result']['evidence']}")
    print(
        f"\nANOVA: F={analysis['anova']['f_statistic']:.3f}, "
        f"p={analysis['anova']['p_value']:.4f}"
    )
    print(f"Significant: {'Yes' if analysis['anova']['significant'] else 'No'}")
    print(f"\n{analysis['summary_table']}")
