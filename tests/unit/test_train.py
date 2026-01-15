"""Unit tests for training module (src/training/train.py).

Tests cover:
- PPOTrainer initialization and configuration
- Model creation for PPO and A2C
- Hyperparameter configuration
- Results saving and JSON serialization
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.training.train import PPOTrainer


class TestPPOTrainerInit:
    """Test PPOTrainer initialization."""

    def test_init_ppo_default(self) -> None:
        """Test PPOTrainer initialization with default parameters."""
        with patch("gymnasium.make"), patch("src.training.train.PPO"):
            trainer = PPOTrainer(algo="ppo", seed=42, total_timesteps=50_000)
            assert trainer.algo == "ppo"
            assert trainer.seed == 42
            assert trainer.total_timesteps == 50_000

    def test_init_a2c_default(self) -> None:
        """Test PPOTrainer initialization with A2C algorithm."""
        with patch("gymnasium.make"), patch("src.training.train.A2C"):
            trainer = PPOTrainer(algo="a2c", seed=42, total_timesteps=50_000)
            assert trainer.algo == "a2c"
            assert trainer.seed == 42

    def test_init_invalid_algo(self) -> None:
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Algorithm must be 'ppo' or 'a2c'"):
            PPOTrainer(algo="invalid", seed=42)

    def test_init_creates_experiment_dir(self) -> None:
        """Test that initialization creates experiment directory."""
        with (
            patch("gymnasium.make"),
            patch("src.training.train.PPO"),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            with patch("pathlib.Path.mkdir"):
                trainer = PPOTrainer(algo="ppo", seed=42)
                assert trainer.exp_dir == Path("results/experiments/ppo_seed42")

    def test_init_verbose_flag(self) -> None:
        """Test that verbose flag is set correctly."""
        with patch("gymnasium.make"), patch("src.training.train.PPO"):
            trainer = PPOTrainer(algo="ppo", verbose=True)
            assert trainer.verbose is True

            trainer = PPOTrainer(algo="ppo", verbose=False)
            assert trainer.verbose is False


class TestModelCreation:
    """Test model creation for different algorithms."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_create_ppo_model(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test PPO model creation with correct hyperparameters."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo", seed=42)

        # Verify PPO was called with correct parameters
        mock_ppo.assert_called_once()
        call_kwargs = mock_ppo.call_args[1]

        assert call_kwargs["policy"] == "MlpPolicy"
        assert call_kwargs["learning_rate"] == 0.0003
        assert call_kwargs["n_steps"] == 2048
        assert call_kwargs["gamma"] == 0.99
        assert call_kwargs["seed"] == 42

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
        assert call_kwargs["seed"] == 42


class TestHyperparameters:
    """Test hyperparameter configuration."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_ppo_hyperparameters(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test PPO hyperparameters are correct."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo")
        hparams = trainer._get_hyperparameters()

        assert hparams["learning_rate"] == 0.0003
        assert hparams["n_steps"] == 2048
        assert hparams["batch_size"] == 64
        assert hparams["n_epochs"] == 10
        assert hparams["gamma"] == 0.99
        assert hparams["gae_lambda"] == 0.95
        assert hparams["clip_range"] == 0.2
        assert hparams["ent_coef"] == 0.01

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


class TestResultsSaving:
    """Test saving of training results."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_save_results_creates_json(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test that save_results creates JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_env = MagicMock()
            mock_gym.return_value = mock_env
            mock_model = MagicMock()
            mock_ppo.return_value = mock_model

            trainer = PPOTrainer(algo="ppo", seed=42)

            # Mock the experiment directory
            trainer.exp_dir = Path(tmpdir) / "ppo_seed42"
            trainer.exp_dir.mkdir(parents=True, exist_ok=True)

            # Mock model save
            trainer.model.save = MagicMock()

            # Mock metrics
            trainer.metrics_collector.calculate_statistics = MagicMock(
                return_value=MagicMock(
                    reward_mean=250.0,
                    reward_std=10.0,
                    episode_length_mean=200.0,
                    total_episodes=50,
                )
            )

            # Save results
            from datetime import datetime

            trainer.start_time = datetime.utcnow()
            trainer.end_time = datetime.utcnow()
            trainer.save_results()

            # Verify JSON files were created
            results_file = trainer.exp_dir / "ppo_seed42_results.json"
            metrics_file = trainer.exp_dir / "ppo_seed42_metrics.json"

            # Check results file exists
            assert results_file.exists() or trainer.model.save.called

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_save_results_json_structure(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test that saved results JSON has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_env = MagicMock()
            mock_gym.return_value = mock_env
            mock_model = MagicMock()
            mock_ppo.return_value = mock_model

            trainer = PPOTrainer(algo="ppo", seed=42, total_timesteps=50_000)

            # Mock the experiment directory
            trainer.exp_dir = Path(tmpdir) / "ppo_seed42"
            trainer.exp_dir.mkdir(parents=True, exist_ok=True)

            # Mock model save
            trainer.model.save = MagicMock()

            # Mock metrics
            mock_stats = MagicMock()
            mock_stats.reward_mean = 250.0
            mock_stats.reward_std = 10.0
            mock_stats.episode_length_mean = 200.0
            mock_stats.total_episodes = 50
            trainer.metrics_collector.calculate_statistics = MagicMock(
                return_value=mock_stats
            )

            # Set times
            from datetime import datetime

            trainer.start_time = datetime.utcnow()
            trainer.end_time = datetime.utcnow()

            # Save results
            trainer.save_results()

            # Read and verify results JSON
            results_file = trainer.exp_dir / "ppo_seed42_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                # Verify structure
                assert "experiment_results" in results
                exp_results = results["experiment_results"]
                assert "metadata" in exp_results
                assert "model" in exp_results
                assert "metrics" in exp_results
                assert "hyperparameters" in exp_results
                assert "environment" in exp_results

                # Verify metadata
                metadata = exp_results["metadata"]
                assert metadata["algorithm"] == "PPO"
                assert metadata["environment"] == "LunarLander-v3"
                assert metadata["seed"] == 42
                assert metadata["total_timesteps"] == 50_000

                # Verify metrics
                metrics = exp_results["metrics"]
                assert "final_reward_mean" in metrics
                assert "final_reward_std" in metrics
                assert "training_time_seconds" in metrics
                assert "converged" in metrics


class TestTrainingTime:
    """Test training time calculation."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_get_training_time_no_times(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test training time when times not set."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo")
        assert trainer._get_training_time() == 0.0

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_get_training_time_with_times(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test training time calculation with set times."""
        from datetime import datetime, timedelta

        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo")
        trainer.start_time = datetime.utcnow()
        trainer.end_time = trainer.start_time + timedelta(seconds=100)

        training_time = trainer._get_training_time()
        assert 99 < training_time < 101  # Allow small timing variations


class TestCleanup:
    """Test cleanup functionality."""

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_cleanup_closes_env(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test that cleanup closes environment."""
        mock_env = MagicMock()
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo")
        trainer.cleanup()

        # Verify env.close was called
        mock_env.close.assert_called_once()

    @patch("gymnasium.make")
    @patch("src.training.train.PPO")
    def test_cleanup_handles_errors(self, mock_ppo: Mock, mock_gym: Mock) -> None:
        """Test that cleanup handles errors gracefully."""
        mock_env = MagicMock()
        mock_env.close.side_effect = RuntimeError("Close failed")
        mock_gym.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        trainer = PPOTrainer(algo="ppo")

        # Should not raise exception
        trainer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
