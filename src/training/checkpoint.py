"""Training module - Checkpoint management for SB3 models.

Per research.md decisions:
- Use SB3's built-in .save() and .load() methods
- Save checkpoints every 1,000 steps
- Implement resume capability from interruption point
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""

    checkpoint_interval: int = 1000  # Steps between checkpoints
    checkpoint_dir: Path = Path("results/experiments/")
    keep_last_n: Optional[int] = None  # None = keep all, 0 = keep none, N = keep N

    # Algorithm-specific configurations
    ppo_kwargs: dict = None
    a2c_kwargs: dict = None

    def __post_init__(self):
        if self.ppo_kwargs is None:
            self.ppo_kwargs = {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            }
        if self.a2c_kwargs is None:
            self.a2c_kwargs = {
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "normalize_advantage": True,
                "ent_coef": 0.01,
            }


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(
        self,
        experiment_id: str,
        algorithm: str,
        config: Optional[CheckpointConfig] = None,
    ):
        self.experiment_id = experiment_id
        self.algorithm = algorithm
        self.config = config or CheckpointConfig()

        # Build checkpoint directory path
        self.checkpoint_dir = self.config.checkpoint_dir / experiment_id / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self.checkpoints: list[dict] = []
        self.current_step = 0

        logger.info(f"Checkpoint manager initialized for {algorithm} ({experiment_id})")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def get_checkpoint_path(self, timestep: int) -> Path:
        """Get the file path for a checkpoint at given timestep.

        Args:
            timestep: Training timestep

        Returns:
            Path to checkpoint file
        """
        return self.checkpoint_dir / f"checkpoint_{timestep}.zip"

    def save_checkpoint(
        self, model: BaseAlgorithm, timestep: int, episode: int, reward: float
    ) -> dict:
        """Save a model checkpoint.

        Args:
            model: SB3 model to save
            timestep: Current training timestep
            episode: Current episode
            reward: Current reward

        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint_path = self.get_checkpoint_path(timestep)

        # Save using SB3's native method
        model.save(checkpoint_path)

        # Record checkpoint metadata
        checkpoint_info = {
            "timestep": timestep,
            "episode": episode,
            "reward": reward,
            "path": str(checkpoint_path),
            "timestamp": self._get_timestamp(),
        }
        self.checkpoints.append(checkpoint_info)
        self.current_step = timestep

        logger.info(f"💾 Checkpoint saved at {timestep} steps (reward: {reward:.1f})")

        return checkpoint_info

    def load_checkpoint(self, timestep: int) -> Optional[BaseAlgorithm]:
        """Load a model from checkpoint.

        Args:
            timestep: Timestep of checkpoint to load

        Returns:
            Loaded model or None if not found
        """
        checkpoint_path = self.get_checkpoint_path(timestep)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"📂 Loading checkpoint from {timestep} steps")

        # Load model based on algorithm
        model = self._load_model_by_algorithm(checkpoint_path)

        logger.info("✅ Checkpoint loaded successfully")
        return model

    def load_latest_checkpoint(self) -> tuple[Optional[BaseAlgorithm], int]:
        """Load the latest available checkpoint.

        Returns:
            Tuple of (model, last_timestep)
        """
        if not self.checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return None, 0

        # Get the latest checkpoint
        latest = max(self.checkpoints, key=lambda c: c["timestep"])
        timestep = latest["timestep"]

        model = self.load_checkpoint(timestep)

        return model, timestep

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            return None

        latest = max(self.checkpoints, key=lambda c: c["timestep"])
        return Path(latest["path"])

    def cleanup_old_checkpoints(self, keep_last_n: Optional[int] = None) -> int:
        """Remove old checkpoints, keeping only the most recent.

        Args:
            keep_last_n: Number of checkpoints to keep (None = keep all, 0 = keep none)

        Returns:
            Number of checkpoints removed
        """
        if keep_last_n is None:
            keep_last_n = self.config.keep_last_n

        if keep_last_n is None or keep_last_n >= len(self.checkpoints):
            logger.info("No checkpoints to clean up")
            return 0

        # Sort checkpoints by timestep
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c["timestep"])

        # Keep last N checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-keep_last_n]

        removed_count = 0
        for checkpoint in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                removed_count += 1

        # Update checkpoint list
        self.checkpoints = sorted_checkpoints[-keep_last_n:]

        logger.info(f"🧹 Cleaned up {removed_count} old checkpoints")
        return removed_count

    def get_checkpoint_summary(self) -> dict:
        """Get a summary of all checkpoints.

        Returns:
            Summary dictionary
        """
        if not self.checkpoints:
            return {
                "total_checkpoints": 0,
                "latest_timestep": 0,
                "latest_reward": None,
                "checkpoint_dir": str(self.checkpoint_dir),
            }

        latest = max(self.checkpoints, key=lambda c: c["timestep"])

        return {
            "total_checkpoints": len(self.checkpoints),
            "latest_timestep": latest["timestep"],
            "latest_reward": latest["reward"],
            "earliest_timestep": min(c["timestep"] for c in self.checkpoints),
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    def _load_model_by_algorithm(self, path: Path) -> BaseAlgorithm:
        """Load model based on algorithm type.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded SB3 model
        """
        algorithm_map = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3,
        }

        if self.algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        model_class = algorithm_map[self.algorithm]
        model = model_class.load(path)

        return model

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()


def create_model(
    algorithm: str,
    env,
    seed: int = 42,
    config: Optional[CheckpointConfig] = None,
) -> tuple[BaseAlgorithm, CheckpointManager]:
    """Create a new model and checkpoint manager.

    Args:
        algorithm: Algorithm name (PPO, A2C, SAC, TD3)
        env: Gymnasium environment
        seed: Random seed
        config: Checkpoint configuration

    Returns:
        Tuple of (model, checkpoint_manager)
    """
    config = config or CheckpointConfig()

    experiment_id = f"{algorithm.lower()}_seed{seed}"

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        experiment_id=experiment_id,
        algorithm=algorithm,
        config=config,
    )

    # Get algorithm-specific kwargs
    if algorithm.upper() == "PPO":
        kwargs = config.ppo_kwargs.copy()
    elif algorithm.upper() == "A2C":
        kwargs = config.a2c_kwargs.copy()
    else:
        kwargs = {}

    # Create model with SB3 defaults
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        **kwargs,
    )

    logger.info(f"Created {algorithm} model with seed={seed}")
    logger.info(f"Hyperparameters: {kwargs}")

    return model, checkpoint_manager


if __name__ == "__main__":
    from src.utils.logging_setup import setup_logging
    from src.utils.seeding import set_seed
    import gymnasium as gym

    setup_logging()
    set_seed(42)

    print("\n" + "=" * 60)
    print("Testing Checkpoint Manager")
    print("=" * 60)

    # Create test environment
    env = gym.make("LunarLander-v3")

    # Create PPO model and checkpoint manager
    model, checkpoint_manager = create_model("PPO", env, seed=42)

    print("\n✅ Model created: PPO with seed=42")
    print(f"📁 Checkpoint directory: {checkpoint_manager.checkpoint_dir}")

    # Save initial checkpoint
    print("\n💾 Saving initial checkpoint...")
    checkpoint_info = checkpoint_manager.save_checkpoint(model, 0, 0, 0.0)
    print(f"   Checkpoint: {checkpoint_info['path']}")

    # Simulate some training steps
    print("\n🏋️ Simulating training...")
    model.learn(total_timesteps=100, progress_bar=False)

    # Save another checkpoint
    print("💾 Saving checkpoint at 100 steps...")
    checkpoint_info = checkpoint_manager.save_checkpoint(model, 100, 5, 50.0)

    # Get summary
    summary = checkpoint_manager.get_checkpoint_summary()
    print("\n📊 Checkpoint Summary:")
    print(f"   Total checkpoints: {summary['total_checkpoints']}")
    print(f"   Latest timestep: {summary['latest_timestep']}")
    print(f"   Latest reward: {summary['latest_reward']}")

    # Test loading
    print("\n📂 Testing checkpoint loading...")
    loaded_model, last_timestep = checkpoint_manager.load_latest_checkpoint()
    if loaded_model:
        print(f"✅ Loaded checkpoint from {last_timestep} steps")
    else:
        print("❌ Failed to load checkpoint")

    env.close()
    print("\n✅ Checkpoint manager test complete!")
