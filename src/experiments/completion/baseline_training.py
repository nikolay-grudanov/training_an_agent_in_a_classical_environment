"""Baseline training experiments for RL agents.

Provides training pipelines for A2C and PPO algorithms with
checkpointing, metrics collection, and evaluation.
"""

import os

# Force CPU by default if CUDA_VISIBLE_DEVICES is not set
# This must be done before importing torch/stable_baselines3
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    # We'll set it later based on the device argument
    pass

import time
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from .config import Algorithm, ExperimentConfig, ExperimentType
from src.training.callbacks import (
    CheckpointCallback,
    EvaluationCallback,
    MetricsLoggingCallback,
)
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
        n_envs: int = 1,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
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
            n_envs: Number of parallel environments (vectorization)
            ent_coef: Entropy coefficient for exploration
            gae_lambda: GAE lambda parameter
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            n_epochs: Number of epoch when optimizing the surrogate loss
            batch_size: Minibatch size
        """
        self.experiment_id = experiment_id
        self.n_envs = n_envs
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

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

        # Create environment (vectorized if n_envs > 1)
        if self.n_envs > 1:
            env = make_vec_env(self.config.environment, n_envs=self.n_envs)
        else:
            env = gym.make(self.config.environment)

        # Create model
        if algorithm == Algorithm.A2C:
            print(f"[DEBUG] Creating A2C model with device: {self.device}")
            model = A2C(
                "MlpPolicy",
                env,
                gamma=self.config.gamma,
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                n_steps=self.n_steps,
                seed=self.config.seed,
                verbose=0,
                device=self.device,
            )
        else:
            print(f"[DEBUG] Creating PPO model with device: {self.device}")
            model = PPO(
                "MlpPolicy",
                env,
                gamma=self.config.gamma,
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                gae_lambda=self.gae_lambda,
                n_steps=self.n_steps,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                seed=self.config.seed,
                verbose=0,
                device=self.device,
            )
        print(f"[DEBUG] Model created on device: {model.device}")

        # Set up metrics collection
        metrics_path = results_dir / "metrics.csv"
        metrics_callback = MetricsLoggingCallback(str(metrics_path), verbose=0)

        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            self.config.checkpoint_freq,
            str(checkpoints_dir),
            verbose=1,
        )
        # For eval, always use single environment
        eval_env = gym.make(self.config.environment)

        eval_callback = EvaluationCallback(
            eval_env,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            log_path=str(results_dir / "eval_log.csv"),
            verbose=0,
        )

        # Train
        start_time = time.time()
        model.learn(
            total_timesteps=self.config.timesteps,
            callback=[checkpoint_callback, eval_callback, metrics_callback],
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
    n_envs: int = 1,
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 64,
    device: str = "auto",
) -> TrainingResult:
    """Train an RL agent until convergence.

    Args:
        algorithm: Algorithm name ("A2C" or "PPO")
        timesteps: Maximum training timesteps
        seed: Random seed
        gamma: Discount factor
        experiment_id: Optional experiment ID (auto-generated if None)
        n_envs: Number of parallel environments (vectorization)
        ent_coef: Entropy coefficient for exploration
        gae_lambda: GAE lambda parameter
        learning_rate: Learning rate
        n_steps: Number of steps per env per update
        n_epochs: Number of optimization epochs
        batch_size: Minibatch size
        device: Device to use (auto, cpu, gpu, cuda, mps).
            Note: For ROCm systems, cpu will hide both CUDA and HIP devices.

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
        n_envs=n_envs,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
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
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient for exploration (default: 0.0)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per env per update (default: 2048)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of optimization epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Device to use (default: auto, recommended: cpu for MLP policies). "
        "Note: For ROCm systems, cpu will hide both CUDA and HIP devices.",
    )
    args = parser.parse_args()

    # Force CPU by hiding GPU if device='cpu' is requested
    # This must be done BEFORE any torch operations
    # For ROCm (AMD GPU), we need to hide both CUDA and HIP devices
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        print(
            "[INFO] Forcing CPU by hiding GPU (CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='')"
        )

    # Debug: print device argument
    print(f"[DEBUG] Command line args.device: {args.device}")
    print(f"[DEBUG] Command line args.algo: {args.algo}")
    print(f"[DEBUG] Command line args.timesteps: {args.timesteps}")
    print(f"[DEBUG] Command line args.gamma: {args.gamma}")
    print(f"[DEBUG] Command line args.ent_coef: {args.ent_coef}")

    result = train_to_convergence(
        algorithm=args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
        n_envs=args.n_envs,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
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
