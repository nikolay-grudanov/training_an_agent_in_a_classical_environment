"""Unit tests for evaluation utilities."""

from unittest.mock import MagicMock, patch

import pytest

from src.training.evaluation import evaluate_agent


class TestEvaluateAgent:
    """Tests for evaluate_agent function."""

    @patch("src.training.evaluation.PPO")
    @patch("src.training.evaluation.gym")
    def test_evaluate_ppo_model(
        self,
        mock_gym: MagicMock,
        mock_ppo: MagicMock,
    ) -> None:
        """Test evaluation of PPO model."""
        # Setup mocks
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        # Mock evaluate_policy to return scalar values
        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            mock_eval.return_value = (150.0, 20.0)

            result = evaluate_agent(
                model_path="test_ppo_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=10,
                render=False,
                deterministic=True,
            )

            assert result["mean_reward"] == 150.0
            assert result["std_reward"] == 20.0
            assert result["n_episodes"] == 10
            assert result["convergence_achieved"] is False  # 150 < 200

            mock_ppo.load.assert_called_once_with("test_ppo_model.zip")
            mock_gym.make.assert_called_once_with(
                "LunarLander-v3", render_mode="rgb_array"
            )
            mock_eval.assert_called_once()

    @patch("src.training.evaluation.A2C")
    @patch("src.training.evaluation.gym")
    def test_evaluate_a2c_model(
        self,
        mock_gym: MagicMock,
        mock_a2c: MagicMock,
    ) -> None:
        """Test evaluation of A2C model."""
        # Setup mocks
        mock_model = MagicMock()
        mock_a2c.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            mock_eval.return_value = (250.0, 15.0)

            result = evaluate_agent(
                model_path="test_a2c_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=10,
            )

            assert result["mean_reward"] == 250.0
            assert result["std_reward"] == 15.0
            assert result["convergence_achieved"] is True  # 250 >= 200

            mock_a2c.load.assert_called_once_with("test_a2c_model.zip")

    @patch("src.training.evaluation.PPO")
    @patch("src.training.evaluation.gym")
    def test_evaluate_with_render(
        self,
        mock_gym: MagicMock,
        mock_ppo: MagicMock,
    ) -> None:
        """Test evaluation with rendering enabled."""
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            mock_eval.return_value = (100.0, 10.0)

            evaluate_agent(
                model_path="test_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=5,
                render=True,
            )

            mock_gym.make.assert_called_once_with("LunarLander-v3", render_mode="human")
            mock_eval.assert_called_once()
            # Check that render was passed to evaluate_policy
            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["render"] is True

    @patch("src.training.evaluation.PPO")
    @patch("src.training.evaluation.gym")
    def test_evaluate_with_list_return(
        self,
        mock_gym: MagicMock,
        mock_ppo: MagicMock,
    ) -> None:
        """Test evaluation when evaluate_policy returns list."""
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            # Simulate list return type (older SB3 versions)
            mock_eval.return_value = ([180.0], [15.0])

            result = evaluate_agent(
                model_path="test_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=10,
            )

            assert result["mean_reward"] == 180.0
            assert result["std_reward"] == 15.0

    @patch("src.training.evaluation.PPO")
    @patch("src.training.evaluation.gym")
    def test_evaluate_convergence_threshold(
        self,
        mock_gym: MagicMock,
        mock_ppo: MagicMock,
    ) -> None:
        """Test convergence threshold detection."""
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            # Test exactly at threshold
            mock_eval.return_value = (200.0, 10.0)

            result = evaluate_agent(
                model_path="test_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=10,
            )

            assert result["convergence_achieved"] is True

            # Test just below threshold
            mock_eval.return_value = (199.99, 10.0)
            result = evaluate_agent(
                model_path="test_model.zip",
                env_id="LunarLander-v3",
                n_eval_episodes=10,
            )

            assert result["convergence_achieved"] is False

    @patch("src.training.evaluation.PPO")
    @patch("src.training.evaluation.gym")
    def test_evaluate_default_parameters(
        self,
        mock_gym: MagicMock,
        mock_ppo: MagicMock,
    ) -> None:
        """Test evaluation with default parameters."""
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model

        mock_env = MagicMock()
        mock_gym.make.return_value = mock_env

        with patch("src.training.evaluation.evaluate_policy") as mock_eval:
            mock_eval.return_value = (150.0, 20.0)

            result = evaluate_agent(model_path="test_model.zip")

            assert result["n_episodes"] == 10  # Default
            mock_gym.make.assert_called_once_with(
                "LunarLander-v3", render_mode="rgb_array"
            )

            # Check that evaluate_policy was called
            assert mock_eval.called
            # Check call arguments include model and env
            call_args = mock_eval.call_args[0]
            assert call_args[0] == mock_model
            assert call_args[1] == mock_env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
