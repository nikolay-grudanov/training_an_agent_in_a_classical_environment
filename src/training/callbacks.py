"""Training callbacks for RL experiments.

Provides callback classes for checkpoint saving, evaluation, and progress monitoring.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints at regular intervals.

    Attributes:
        save_freq: Frequency of checkpoint saves (in timesteps)
        save_path: Directory to save checkpoints
        verbose: Verbosity level
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0) -> None:
        """Initialize checkpoint callback.

        Args:
            save_freq: Frequency of checkpoint saves
            save_path: Directory to save checkpoints
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        if self.num_timesteps % self.save_freq == 0:
            self.checkpoint_count += 1
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(
                    f"Saved checkpoint {self.checkpoint_count} at {self.num_timesteps} timesteps"
                )
        return True


class EvaluationCallback(BaseCallback):
    """Callback for periodic evaluation during training.

    Attributes:
        eval_env: Evaluation environment
        eval_freq: Frequency of evaluations
        n_eval_episodes: Number of episodes per evaluation
        log_path: Path to save evaluation results
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        log_path: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        """Initialize evaluation callback.

        Args:
            eval_env: Environment for evaluation
            eval_freq: Frequency of evaluations
            n_eval_episodes: Episodes per evaluation
            log_path: Path to save evaluation results
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = Monitor(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = Path(log_path) if log_path else None
        self._last_eval_timesteps = 0
        self._results: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training
        """
        if self.num_timesteps - self._last_eval_timesteps >= self.eval_freq:
            self._last_eval_timesteps = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Run evaluation and log results."""
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=False,
        )

        result = {
            "timesteps": self.num_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
        self._results.append(result)

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                if self.log_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(result)

        if self.verbose > 0:
            print(
                f"Eval at {self.num_timesteps}: reward={mean_reward:.2f}±{std_reward:.2f}"
            )
