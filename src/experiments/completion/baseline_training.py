"""Baseline training experiments for RL agents.

Provides training pipelines for A2C and PPO algorithms with
checkpointing, metrics collection, and evaluation.
"""

import time
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
from stable_baselines3 import A2C, PPO

from .config import Algorithm, ExperimentConfig, ExperimentType
from .metrics_collector import MetricsCollector
from src.training.callbacks import CheckpointCallback, EvaluationCallback
from src.training.evaluation import evaluate_agent
from src.training.results import TrainingResult
from src.utils.seeding import set_seed


class BaselineExperiment:
    """Baseline training experiment for RL agents.

    Provides training pipeline with checkpointing, metrics collection,
    and evaluation for A2C and PPO algorithms.

    Attributes:
        experiment_id: Unique identifier for the experiment
        config: Experiment configuration
        results: Training results after completion
    """

    def __init__(
        self,
        experiment_id: str,
        algorithm: Algorithm,
        timesteps: int = 200_000,
        seed: int = 42,
        gamma: float = 0.99,
        checkpoint_freq: int = 50_000,
        eval_freq: int = 5_000,
    ) -> None:
        """Initialize baseline experiment.

        Args:
            experiment_id: Unique identifier
            algorithm: RL algorithm (A2C or PPO)
            timesteps: Training duration
            seed: Random seed
            gamma: Discount factor
            checkpoint_freq: Checkpoint save frequency
            eval_freq: Evaluation frequency
        """
        self.experiment_id = experiment_id
        self.config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_type=ExperimentType.BASELINE,
            algorithm=algorithm,
            environment="LunarLander-v3",
            timesteps=timesteps,
            seed=seed,
            gamma=gamma,
            checkpoint_freq=checkpoint_freq,
            eval_freq=eval_freq,
        )
        self.results: Optional[TrainingResult] = None

    def run_a2c(self) -> TrainingResult:
        """Run A2C training experiment.

        Returns:
            TrainingResult with metrics and model path
        """
        return self._run(Algorithm.A2C)

    def run_ppo(self) -> TrainingResult:
        """Run PPO training experiment.

        Returns:
            TrainingResult with metrics and model path
        """
        return self._run(Algorithm.PPO)

    def _run(self, algorithm: Algorithm) -> TrainingResult:
        """Internal training execution.

        Args:
            algorithm: Algorithm to use

        Returns:
            TrainingResult instance
        """
        # Set up directories
        results_dir = Path("results/experiments") / self.experiment_id
        checkpoints_dir = results_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        set_seed(self.config.seed)

        # Create environment
        env = gym.make(self.config.environment)

        # Create model
        if algorithm == Algorithm.A2C:
            model = A2C(
                "MlpPolicy",
                env,
                gamma=self.config.gamma,
                seed=self.config.seed,
                verbose=0,
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                gamma=self.config.gamma,
                seed=self.config.seed,
                verbose=0,
            )

        # Set up metrics collection
        metrics_path = results_dir / "metrics.csv"
        MetricsCollector(self.experiment_id, metrics_path)

        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            self.config.checkpoint_freq,
            str(checkpoints_dir),
            verbose=1,
        )
        eval_callback = EvaluationCallback(
            gym.make(self.config.environment),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            log_path=str(results_dir / "eval_log.csv"),
            verbose=0,
        )

        # Train
        start_time = time.time()
        model.learn(
            total_timesteps=self.config.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        training_duration = time.time() - start_time

        # Save model
        model_path = results_dir / f"{self.experiment_id}_model.zip"
        model.save(model_path)

        # Save config
        config_path = results_dir / "config.json"
        self.config.save_to_json(config_path)

        # Evaluate
        eval_result = evaluate_agent(
            str(model_path),
            self.config.environment,
            self.config.n_eval_episodes,
        )

        # Create results
        self.results = TrainingResult(
            experiment_id=self.experiment_id,
            model_path=model_path,
            metrics_path=metrics_path,
            evaluation_result=eval_result,
            training_duration_seconds=training_duration,
            convergence_achieved=eval_result["convergence_achieved"],
        )

        env.close()

        return self.results


def train_to_convergence(
    algorithm: Union[str, Algorithm],
    timesteps: int,
    seed: int = 42,
    gamma: float = 0.99,
    experiment_id: Optional[str] = None,
) -> TrainingResult:
    """Train an RL agent until convergence.

    Args:
        algorithm: Algorithm name ("A2C" or "PPO")
        timesteps: Maximum training timesteps
        seed: Random seed
        gamma: Discount factor
        experiment_id: Optional experiment ID (auto-generated if None)

    Returns:
        TrainingResult with final metrics
    """
    algo = Algorithm(algorithm.upper()) if isinstance(algorithm, str) else algorithm
    exp_id = experiment_id or f"{algo.value.lower()}_seed{seed}"

    experiment = BaselineExperiment(
        experiment_id=exp_id,
        algorithm=algo,
        timesteps=timesteps,
        seed=seed,
        gamma=gamma,
    )

    if algo == Algorithm.A2C:
        return experiment.run_a2c()
    else:
        return experiment.run_ppo()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL baseline agents")
    parser.add_argument(
        "--algo",
        choices=["a2c", "ppo"],
        required=True,
        help="Algorithm to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    args = parser.parse_args()

    result = train_to_convergence(
        algorithm=args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
    )

    print("\nTraining complete!")
    print(f"Experiment: {result.experiment_id}")
    print(f"Model: {result.model_path}")
    print(
        f"Mean reward: {result.evaluation_result['mean_reward']:.2f} ± "
        f"{result.evaluation_result['std_reward']:.2f}"
    )
    print(f"Convergence: {'YES' if result.convergence_achieved else 'NO'}")
    print(f"Duration: {result.training_duration_seconds:.1f}s")
