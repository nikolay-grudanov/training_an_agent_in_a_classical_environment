"""Training script for PPO and A2C agents on LunarLander-v3.

This module provides a CLI interface for training RL agents with reproducible
settings. It implements the training pipeline per tasks.md T020 requirements:
- Seed=42 for reproducibility
- 50,000 timesteps training
- Checkpoint saving every 1,000 steps
- Metrics collection and export
- JSON experiment results

Usage:
    python -m src.training.train --algo ppo
    python -m src.training.train --algo a2c --verbose
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from src.training.checkpoint import CheckpointManager, CheckpointConfig
from src.training.metrics_collector import MetricsCollector, MetricsCollectorConfig
from src.utils.logging_setup import setup_logging
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer for LunarLander-v3 environment."""

    def __init__(
        self,
        algo: str = "ppo",
        seed: int = 42,
        total_timesteps: int = 50_000,
        verbose: bool = False,
    ) -> None:
        """Initialize PPO trainer.

        Args:
            algo: Algorithm name ("ppo" or "a2c")
            seed: Random seed for reproducibility
            total_timesteps: Total training timesteps
            verbose: Enable DEBUG-level logging

        Raises:
            ValueError: If algo not in ["ppo", "a2c"]
        """
        if algo not in ["ppo", "a2c"]:
            raise ValueError(f"Algorithm must be 'ppo' or 'a2c', got {algo}")

        self.algo = algo
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.verbose = verbose

        # Setup logging
        setup_logging()

        # Set seed FIRST before any random operations
        set_seed(seed)
        logger.info(f"Seed set to {seed}")

        # Create experiment directory
        self.exp_dir = Path(f"results/experiments/{algo}_seed{seed}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory: {self.exp_dir}")

        # Initialize environment
        self.env = gym.make("LunarLander-v3")
        logger.debug(f"Environment created: {self.env}")

        # Initialize model
        self.model = self._create_model()
        logger.info(f"Model created: {self.model}")

        # Initialize checkpoint manager
        checkpoint_config = CheckpointConfig(
            checkpoint_interval=1000,
            checkpoint_dir=Path("results/experiments"),
            keep_last_n=None,
        )
        self.checkpoint_manager = CheckpointManager(
            experiment_id=f"{algo}_seed{seed}",
            algorithm=algo.upper(),
            config=checkpoint_config,
        )
        logger.debug("Checkpoint manager initialized")

        # Initialize metrics collector
        metrics_config = MetricsCollectorConfig(
            experiment_id=f"{algo}_seed{seed}",
            algorithm=algo.upper(),
            environment="LunarLander-v3",
            seed=seed,
            recording_interval=100,
        )
        self.metrics_collector = MetricsCollector(metrics_config)
        logger.debug("Metrics collector initialized")

        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def _create_model(self) -> BaseAlgorithm:
        """Create PPO or A2C model with default hyperparameters.

        Returns:
            Initialized model (PPO or A2C)
        """
        if self.algo == "ppo":
            model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                seed=self.seed,
                verbose=1 if self.verbose else 0,
            )
        elif self.algo == "a2c":
            model = A2C(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                normalize_advantage=True,
                seed=self.seed,
                verbose=1 if self.verbose else 0,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")

        return model

    def train(self) -> None:
        """Execute training loop with checkpointing and metrics collection."""
        logger.info(
            f"Starting {self.algo.upper()} training for {self.total_timesteps} steps"
        )
        self.start_time = datetime.utcnow()

        try:
            # Training loop with checkpointing
            checkpoint_interval = 1000
            timesteps_completed = 0

            while timesteps_completed < self.total_timesteps:
                # Calculate steps for this batch
                remaining_steps = self.total_timesteps - timesteps_completed
                steps_this_batch = min(checkpoint_interval, remaining_steps)

                # Train for this batch
                logger.debug(
                    f"Training batch: {timesteps_completed} -> "
                    f"{timesteps_completed + steps_this_batch}"
                )
                self.model.learn(total_timesteps=steps_this_batch)

                timesteps_completed += steps_this_batch

                # Save checkpoint
                checkpoint_path = self.exp_dir / f"checkpoint_{timesteps_completed}.zip"
                self.model.save(str(checkpoint_path))
                logger.debug(f"Checkpoint saved: {checkpoint_path}")

                # Collect metrics (evaluate on environment)
                self._collect_metrics(timesteps_completed)

                # Log progress
                logger.info(
                    f"Progress: {timesteps_completed}/{self.total_timesteps} "
                    f"({100 * timesteps_completed / self.total_timesteps:.1f}%)"
                )

            self.end_time = datetime.utcnow()
            logger.info(
                f"Training completed in {self._get_training_time():.1f} seconds"
            )

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.end_time = datetime.utcnow()
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.end_time = datetime.utcnow()
            raise

    def _collect_metrics(self, timestep: int) -> None:
        """Collect metrics from current training state.

        Args:
            timestep: Current timestep in training
        """
        try:
            # Run evaluation episodes
            n_eval_episodes = 5
            episode_rewards = []
            episode_lengths = []

            for _ in range(n_eval_episodes):
                obs, _ = self.env.reset()
                done = False
                truncated = False
                episode_reward = 0.0
                episode_length = 0

                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            # Record metrics
            mean_reward = float(np.mean(episode_rewards))
            mean_length = float(np.mean(episode_lengths))

            self.metrics_collector.record(
                timestep=timestep,
                episode=timestep // 100,  # Approximate episode number
                reward=mean_reward,
                episode_length=mean_length,
                loss=None,
            )

            logger.debug(
                f"Metrics at {timestep}: reward={mean_reward:.2f}, "
                f"length={mean_length:.1f}"
            )

        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

    def _get_training_time(self) -> float:
        """Get training duration in seconds.

        Returns:
            Training duration in seconds
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def save_results(self) -> None:
        """Save trained model and experiment results."""
        try:
            # Save model
            model_path = self.exp_dir / f"{self.algo}_seed{self.seed}_model.zip"
            self.model.save(str(model_path))
            logger.info(f"Model saved: {model_path}")

            # Calculate final metrics
            final_stats = self.metrics_collector.calculate_statistics()

            # Create experiment results
            experiment_results = {
                "experiment_results": {
                    "metadata": {
                        "experiment_id": f"{self.algo}_seed{self.seed}",
                        "algorithm": self.algo.upper(),
                        "environment": "LunarLander-v3",
                        "seed": self.seed,
                        "start_time": self.start_time.isoformat() + "Z"
                        if self.start_time
                        else None,
                        "end_time": self.end_time.isoformat() + "Z"
                        if self.end_time
                        else None,
                        "total_timesteps": self.total_timesteps,
                        "conda_environment": "rocm",
                    },
                    "model": {
                        "algorithm": self.algo.upper(),
                        "policy": "MlpPolicy",
                        "model_file": f"{self.algo}_seed{self.seed}_model.zip",
                        "model_path": str(self.exp_dir),
                        "checkpoint_interval": 1000,
                        "checkpoints": [
                            f"checkpoint_{i}.zip"
                            for i in range(1000, self.total_timesteps + 1, 1000)
                        ],
                    },
                    "metrics": {
                        "final_reward_mean": final_stats.reward_mean,
                        "final_reward_std": final_stats.reward_std,
                        "episode_length_mean": final_stats.episode_length_mean,
                        "total_episodes": final_stats.total_episodes,
                        "training_time_seconds": self._get_training_time(),
                        "converged": final_stats.reward_mean > 200,
                    },
                    "hyperparameters": self._get_hyperparameters(),
                    "environment": {
                        "name": "LunarLander-v3",
                        "observation_space": "Box(8,)",
                        "action_space": "Discrete(4)",
                        "reward_threshold": 200.0,
                    },
                }
            }

            # Save experiment results
            results_path = self.exp_dir / f"{self.algo}_seed{self.seed}_results.json"
            with open(results_path, "w") as f:
                json.dump(experiment_results, f, indent=2)
            logger.info(f"Experiment results saved: {results_path}")

            # Save metrics
            metrics_path = self.exp_dir / f"{self.algo}_seed{self.seed}_metrics.json"
            self.metrics_collector.export_to_json(str(metrics_path))
            logger.info(f"Metrics saved: {metrics_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise

    def _get_hyperparameters(self) -> dict:
        """Get algorithm hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        if self.algo == "ppo":
            return {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            }
        elif self.algo == "a2c":
            return {
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "normalize_advantage": True,
            }
        else:
            return {}

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "env"):
                self.env.close()
                logger.debug("Environment closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train PPO or A2C agent on LunarLander-v3"
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "a2c"],
        default="ppo",
        help="Algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Total training timesteps (default: 50000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    trainer = PPOTrainer(
        algo=args.algo,
        seed=args.seed,
        total_timesteps=args.steps,
        verbose=args.verbose,
    )

    try:
        trainer.train()
        trainer.save_results()
        logger.info(f"{args.algo.upper()} training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
