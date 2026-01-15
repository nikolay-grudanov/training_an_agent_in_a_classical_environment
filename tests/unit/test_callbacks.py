"""Unit tests for training callbacks."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest

from src.training.callbacks import CheckpointCallback, EvaluationCallback


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_init(self) -> None:
        """Test CheckpointCallback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                verbose=1,
            )

            assert callback.save_freq == 1000
            assert callback.save_path == Path(tmpdir)
            assert callback.checkpoint_count == 0
            assert callback.verbose == 1

    def test_on_step_no_checkpoint(self) -> None:
        """Test that no checkpoint is saved when not at frequency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                verbose=0,
            )

            # Mock model
            callback.model = MagicMock()
            callback.num_timesteps = 500

            result = callback._on_step()

            assert result is True
            assert callback.checkpoint_count == 0
            callback.model.save.assert_not_called()

    def test_on_step_with_checkpoint(self) -> None:
        """Test that checkpoint is saved at frequency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                verbose=0,
            )

            # Mock model
            callback.model = MagicMock()
            callback.num_timesteps = 1000

            result = callback._on_step()

            assert result is True
            assert callback.checkpoint_count == 1
            callback.model.save.assert_called_once_with(
                Path(tmpdir) / "checkpoint_1000",
            )

    def test_on_step_multiple_checkpoints(self) -> None:
        """Test multiple checkpoint saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                verbose=0,
            )

            callback.model = MagicMock()

            # Simulate multiple timesteps
            for timesteps in [1000, 2000, 3000]:
                callback.num_timesteps = timesteps
                callback._on_step()

            assert callback.checkpoint_count == 3
            assert callback.model.save.call_count == 3


class TestEvaluationCallback:
    """Tests for EvaluationCallback."""

    def test_init(self) -> None:
        """Test EvaluationCallback initialization."""
        env = gym.make("CartPole-v1")
        callback = EvaluationCallback(
            eval_env=env,
            eval_freq=1000,
            n_eval_episodes=5,
            log_path=None,
            verbose=0,
        )

        assert callback.eval_freq == 1000
        assert callback.n_eval_episodes == 5
        assert callback.log_path is None
        assert callback._last_eval_timesteps == 0
        assert len(callback._results) == 0

        env.close()

    def test_on_step_no_evaluation(self) -> None:
        """Test that no evaluation occurs before frequency."""
        env = gym.make("CartPole-v1")
        callback = EvaluationCallback(
            eval_env=env,
            eval_freq=1000,
            n_eval_episodes=5,
            verbose=0,
        )

        callback.model = MagicMock()
        callback.num_timesteps = 500

        result = callback._on_step()

        assert result is True
        assert callback._last_eval_timesteps == 0
        assert len(callback._results) == 0

        env.close()

    @patch("stable_baselines3.common.evaluation.evaluate_policy")
    def test_on_step_with_evaluation(self, mock_evaluate: MagicMock) -> None:
        """Test that evaluation occurs at frequency."""
        mock_evaluate.return_value = (100.0, 10.0)

        env = gym.make("CartPole-v1")
        callback = EvaluationCallback(
            eval_env=env,
            eval_freq=1000,
            n_eval_episodes=5,
            verbose=0,
        )

        callback.model = MagicMock()
        callback.num_timesteps = 1000

        result = callback._on_step()

        assert result is True
        assert callback._last_eval_timesteps == 1000
        assert len(callback._results) == 1
        assert callback._results[0]["timesteps"] == 1000
        mock_evaluate.assert_called_once()

        env.close()

    @patch("stable_baselines3.common.evaluation.evaluate_policy")
    def test_evaluate_logs_to_file(self, mock_evaluate: MagicMock) -> None:
        """Test that evaluation results are logged to file."""
        mock_evaluate.return_value = (100.0, 10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "eval_log.csv"
            env = gym.make("CartPole-v1")
            callback = EvaluationCallback(
                eval_env=env,
                eval_freq=1000,
                n_eval_episodes=5,
                log_path=str(log_path),
                verbose=0,
            )

            callback.model = MagicMock()
            callback.num_timesteps = 1000
            callback._on_step()

            assert log_path.exists()
            content = log_path.read_text()
            assert "timesteps" in content
            assert "1000" in content

            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
