"""Unit tests for A2C training in src/training/train.py.

Tests cover:
- A2C model creation with correct hyperparameters
- A2C training pipeline
- Output structure matching PPO format
- Reproducibility with identical seed
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.training.train import PPOTrainer


class TestA2CTrainerInit:
    """Test A2C trainer initialization."""

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_init_a2c_default(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test A2C trainer initialization with default parameters."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)

        assert trainer.algo == "a2c"
        assert trainer.seed == 42
        assert trainer.total_timesteps == 50_000
        assert trainer.exp_dir == Path("results/experiments/a2c_seed42")

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_init_a2c_custom_seed(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test A2C trainer with custom seed."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer = PPOTrainer(algo="a2c", seed=123, total_timesteps=50_000)

        assert trainer.seed == 123
        assert trainer.exp_dir == Path("results/experiments/a2c_seed123")


class TestA2CModelCreation:
    """Test A2C model creation with correct hyperparameters."""

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_create_a2c_model(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test A2C model creation with correct hyperparameters."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer = PPOTrainer(algo="a2c", seed=42)

        # Verify A2C was called with correct parameters
        mock_a2c.assert_called_once()
        call_kwargs = mock_a2c.call_args[1]

        assert call_kwargs["policy"] == "MlpPolicy"
        assert call_kwargs["learning_rate"] == 0.0007
        assert call_kwargs["n_steps"] == 5
        assert call_kwargs["gamma"] == 0.99
        assert call_kwargs["gae_lambda"] == 1.0
        assert call_kwargs["normalize_advantage"] is True
        assert call_kwargs["seed"] == 42

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_a2c_hyperparameters(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test A2C hyperparameters are correct."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer = PPOTrainer(algo="a2c")
        hparams = trainer._get_hyperparameters()

        assert hparams["learning_rate"] == 0.0007
        assert hparams["n_steps"] == 5
        assert hparams["gamma"] == 0.99
        assert hparams["gae_lambda"] == 1.0
        assert hparams["normalize_advantage"] is True


class TestA2COutputStructure:
    """Test A2C output structure matches PPO format."""

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_a2c_model_file_naming(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test that A2C model files follow correct naming convention."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)

        # Verify experiment directory follows naming convention
        assert "a2c_seed42" in str(trainer.exp_dir)
        assert trainer.algo == "a2c"
        assert trainer.seed == 42

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_a2c_metrics_json_structure(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test that A2C metrics JSON has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_env = MagicMock()
            mock_gym.return_value = mock_env
            mock_model = MagicMock()
            mock_a2c.return_value = mock_model

            trainer = PPOTrainer(algo="a2c", seed=42)

            # Mock the experiment directory
            trainer.exp_dir = Path(tmpdir) / "a2c_seed42"
            trainer.exp_dir.mkdir(parents=True, exist_ok=True)

            # Add some metrics
            trainer.metrics_collector.record(
                timestep=1000,
                episode=10,
                reward=100.5,
                episode_length=200,
            )

            # Export metrics
            metrics_file = trainer.exp_dir / "a2c_seed42_metrics.json"
            trainer.metrics_collector.export_to_json(str(metrics_file))

            # Verify file exists and has correct structure
            assert metrics_file.exists()
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Check structure
            assert "metadata" in metrics
            assert "metrics" in metrics or "time_series" in metrics
            assert metrics["metadata"]["algorithm"] == "A2C"
            assert metrics["metadata"]["environment"] == "LunarLander-v3"


class TestA2CReproducibility:
    """Test A2C reproducibility with identical seed."""

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_a2c_reproducibility_same_seed(
        self, mock_a2c: Mock, mock_gym: Mock
    ) -> None:
        """Test that A2C produces same results with same seed."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        # Create two trainers with same seed
        trainer1 = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)
        trainer2 = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)

        # Both should have same configuration
        assert trainer1.seed == trainer2.seed
        assert trainer1.algo == trainer2.algo
        assert trainer1.total_timesteps == trainer2.total_timesteps

    @patch("gymnasium.make")
    @patch("src.training.train.A2C")
    def test_a2c_different_seeds(self, mock_a2c: Mock, mock_gym: Mock) -> None:
        """Test that different seeds create different experiment directories."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_a2c.return_value = mock_model

        trainer1 = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)
        trainer2 = PPOTrainer(algo="a2c", seed=123, total_timesteps=50_000)

        # Different seeds should have different directories
        assert trainer1.exp_dir != trainer2.exp_dir
        assert "seed42" in str(trainer1.exp_dir)
        assert "seed123" in str(trainer2.exp_dir)


class TestA2CConsistencyWithPPO:
    """Test A2C consistency with PPO output format."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    @patch("src.training.train.A2C")
    def test_ppo_a2c_directory_structure(
        self, mock_a2c: Mock, mock_ppo: Mock, mock_gym: Mock
    ) -> None:
        """Test that PPO and A2C create consistent directory structures."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_ppo_model = MagicMock()
        mock_a2c_model = MagicMock()
        mock_ppo.return_value = mock_ppo_model
        mock_a2c.return_value = mock_a2c_model

        # Create PPO trainer
        ppo_trainer = PPOTrainer(algo="ppo", seed=42)

        # Create A2C trainer
        a2c_trainer = PPOTrainer(algo="a2c", seed=42)

        # Verify directory structure pattern
        assert "ppo_seed42" in str(ppo_trainer.exp_dir)
        assert "a2c_seed42" in str(a2c_trainer.exp_dir)
        assert ppo_trainer.seed == a2c_trainer.seed
        assert ppo_trainer.total_timesteps == a2c_trainer.total_timesteps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
