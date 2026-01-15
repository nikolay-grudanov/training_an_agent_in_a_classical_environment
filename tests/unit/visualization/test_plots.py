"""Tests for plotting utilities."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.plots import (
    PlotConfig,
    apply_smoothing,
    create_figure_grid,
    detect_convergence,
    plot_confidence_intervals,
    plot_convergence_analysis,
    plot_episode_lengths,
    plot_learning_curve,
    plot_loss_curves,
    plot_multiple_runs,
    plot_reward_distribution,
    save_plot,
    setup_matplotlib_style,
)


class TestPlotConfig:
    """Test PlotConfig class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        config = PlotConfig()
        assert config.get("figure_size") == (12, 8)
        assert config.get("dpi") == 300
        assert config.get("style") == "seaborn-v0_8-whitegrid"

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        config = PlotConfig(figure_size=(10, 6), dpi=150)
        assert config.get("figure_size") == (10, 6)
        assert config.get("dpi") == 150
        assert config.get("style") == "seaborn-v0_8-whitegrid"  # Default

    def test_update(self) -> None:
        """Test configuration update."""
        config = PlotConfig()
        config.update(figure_size=(8, 6), new_param="test")
        assert config.get("figure_size") == (8, 6)
        assert config.get("new_param") == "test"

    def test_get_default(self) -> None:
        """Test get with default value."""
        config = PlotConfig()
        assert config.get("nonexistent", "default") == "default"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_setup_matplotlib_style(self) -> None:
        """Test matplotlib style setup."""
        setup_matplotlib_style(font_size=14, dpi=200)
        assert plt.rcParams["font.size"] == 14
        assert plt.rcParams["figure.dpi"] == 200

    def test_create_figure_grid_single(self) -> None:
        """Test single subplot creation."""
        fig, axes = create_figure_grid(1, 1)
        assert len(axes) == 1
        plt.close(fig)

    def test_create_figure_grid_multiple(self) -> None:
        """Test multiple subplot creation."""
        fig, axes = create_figure_grid(2, 3)
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_create_figure_grid_row(self) -> None:
        """Test row subplot creation."""
        fig, axes = create_figure_grid(1, 3)
        assert len(axes) == 3
        plt.close(fig)

    def test_create_figure_grid_column(self) -> None:
        """Test column subplot creation."""
        fig, axes = create_figure_grid(3, 1)
        assert len(axes) == 3
        plt.close(fig)


class TestSmoothingFunctions:
    """Test data smoothing functions."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Generate sample noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)
        return y

    def test_moving_average_smoothing(self, sample_data: np.ndarray) -> None:
        """Test moving average smoothing."""
        smoothed = apply_smoothing(sample_data, method="moving_average", window=10)
        assert len(smoothed) == len(sample_data)
        assert np.var(smoothed) < np.var(sample_data)  # Should reduce variance

    def test_exponential_smoothing(self, sample_data: np.ndarray) -> None:
        """Test exponential smoothing."""
        smoothed = apply_smoothing(sample_data, method="exponential", alpha=0.1)
        assert len(smoothed) == len(sample_data)
        assert np.var(smoothed) < np.var(sample_data)

    def test_savgol_smoothing(self, sample_data: np.ndarray) -> None:
        """Test Savitzky-Golay smoothing."""
        smoothed = apply_smoothing(sample_data, method="savgol", window=11, polyorder=3)
        assert len(smoothed) == len(sample_data)

    def test_lowess_smoothing(self, sample_data: np.ndarray) -> None:
        """Test LOWESS smoothing."""
        smoothed = apply_smoothing(sample_data, method="lowess", frac=0.2)
        assert len(smoothed) == len(sample_data)

    def test_invalid_smoothing_method(self, sample_data: np.ndarray) -> None:
        """Test invalid smoothing method."""
        with pytest.raises(ValueError, match="Unsupported smoothing method"):
            apply_smoothing(sample_data, method="invalid")

    def test_short_data_warning(self) -> None:
        """Test warning for short data."""
        short_data = np.array([1, 2, 3])
        result = apply_smoothing(short_data, window=10)
        np.testing.assert_array_equal(result, short_data)


class TestConvergenceDetection:
    """Test convergence detection."""

    def test_detect_convergence_converged(self) -> None:
        """Test convergence detection on converged data."""
        # Create data that converges
        data = np.concatenate(
            [
                np.random.randn(100),  # Initial noisy period
                np.ones(200) + 0.01 * np.random.randn(200),  # Converged period
            ]
        )

        convergence_idx = detect_convergence(data, window=50, threshold=0.05)
        assert convergence_idx is not None
        assert convergence_idx > 50  # Should detect after initial period

    def test_detect_convergence_not_converged(self) -> None:
        """Test convergence detection on non-converged data."""
        # Create continuously changing data
        data = np.cumsum(np.random.randn(300))

        convergence_idx = detect_convergence(data, window=50, threshold=0.01)
        assert convergence_idx is None

    def test_detect_convergence_short_data(self) -> None:
        """Test convergence detection on short data."""
        data = np.random.randn(50)
        convergence_idx = detect_convergence(data, min_length=100)
        assert convergence_idx is None


class TestSavePlot:
    """Test plot saving functionality."""

    def test_save_matplotlib_plot(self) -> None:
        """Test saving matplotlib plot."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot"
            save_plot(fig, save_path, formats=["png"])

            assert (save_path.with_suffix(".png")).exists()

        plt.close(fig)

    def test_save_multiple_formats(self) -> None:
        """Test saving in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot"
            save_plot(fig, save_path, formats=["png", "svg"])

            assert (save_path.with_suffix(".png")).exists()
            assert (save_path.with_suffix(".svg")).exists()

        plt.close(fig)


class TestPlottingFunctions:
    """Test main plotting functions."""

    @pytest.fixture
    def sample_training_data(self) -> dict:
        """Generate sample training data."""
        np.random.seed(42)
        timesteps = np.arange(1000)
        rewards = np.cumsum(np.random.randn(1000) * 0.1) + np.sin(timesteps / 100) * 10
        episode_lengths = (
            100 + 50 * np.sin(timesteps / 200) + 10 * np.random.randn(1000)
        )

        return {
            "timesteps": timesteps,
            "rewards": rewards,
            "episode_lengths": episode_lengths,
        }

    def test_plot_learning_curve_matplotlib(self, sample_training_data: dict) -> None:
        """Test learning curve plotting with matplotlib."""
        fig = plot_learning_curve(
            timesteps=sample_training_data["timesteps"],
            rewards=sample_training_data["rewards"],
            backend="matplotlib",
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_learning_curve_plotly(self, sample_training_data: dict) -> None:
        """Test learning curve plotting with plotly."""
        fig = plot_learning_curve(
            timesteps=sample_training_data["timesteps"],
            rewards=sample_training_data["rewards"],
            backend="plotly",
        )
        assert fig is not None

    def test_plot_learning_curve_with_confidence(
        self, sample_training_data: dict
    ) -> None:
        """Test learning curve with confidence intervals."""
        fig = plot_learning_curve(
            timesteps=sample_training_data["timesteps"],
            rewards=sample_training_data["rewards"],
            confidence_interval=True,
            backend="matplotlib",
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_learning_curve_invalid_backend(
        self, sample_training_data: dict
    ) -> None:
        """Test invalid backend error."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            plot_learning_curve(
                timesteps=sample_training_data["timesteps"],
                rewards=sample_training_data["rewards"],
                backend="invalid",
            )

    def test_plot_learning_curve_mismatched_data(self) -> None:
        """Test error with mismatched data lengths."""
        with pytest.raises(ValueError, match="must have the same length"):
            plot_learning_curve(
                timesteps=[1, 2, 3],
                rewards=[1, 2],
                backend="matplotlib",
            )

    def test_plot_episode_lengths(self, sample_training_data: dict) -> None:
        """Test episode length plotting."""
        fig = plot_episode_lengths(
            episodes=sample_training_data["timesteps"],
            lengths=sample_training_data["episode_lengths"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_loss_curves(self, sample_training_data: dict) -> None:
        """Test loss curves plotting."""
        losses = {
            "policy_loss": np.random.exponential(1, 1000),
            "value_loss": np.random.exponential(0.5, 1000),
            "entropy_loss": np.random.exponential(0.1, 1000),
        }

        fig = plot_loss_curves(
            timesteps=sample_training_data["timesteps"],
            losses=losses,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_reward_distribution(self, sample_training_data: dict) -> None:
        """Test reward distribution plotting."""
        fig = plot_reward_distribution(
            rewards=sample_training_data["rewards"],
            show_stats=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_analysis(self, sample_training_data: dict) -> None:
        """Test convergence analysis plotting."""
        fig = plot_convergence_analysis(
            timesteps=sample_training_data["timesteps"],
            rewards=sample_training_data["rewards"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_multiple_runs(self, sample_training_data: dict) -> None:
        """Test multiple runs comparison plotting."""
        # Create data for multiple runs
        runs_data = {}
        for i, algorithm in enumerate(["PPO", "A2C", "SAC"]):
            noise_factor = (i + 1) * 0.1
            rewards = sample_training_data["rewards"] + noise_factor * np.random.randn(
                1000
            )
            runs_data[algorithm] = {
                "timesteps": sample_training_data["timesteps"],
                "reward": rewards,
            }

        fig = plot_multiple_runs(
            runs_data=runs_data,
            metric="reward",
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_confidence_intervals(self, sample_training_data: dict) -> None:
        """Test confidence intervals plotting."""
        # Create multiple runs data
        y_data_list = []
        for i in range(5):
            noise = 0.1 * np.random.randn(1000)
            y_data_list.append(sample_training_data["rewards"] + noise)

        fig = plot_confidence_intervals(
            x_data=sample_training_data["timesteps"],
            y_data_list=y_data_list,
            confidence_level=0.95,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_config(self, sample_training_data: dict) -> None:
        """Test plotting with custom configuration."""
        config = PlotConfig(
            figure_size=(10, 6),
            color_palette="colorblind",
            line_width=3,
        )

        fig = plot_learning_curve(
            timesteps=sample_training_data["timesteps"],
            rewards=sample_training_data["rewards"],
            config=config,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_save_path(self, sample_training_data: dict) -> None:
        """Test plotting with save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_learning_curve"

            fig = plot_learning_curve(
                timesteps=sample_training_data["timesteps"],
                rewards=sample_training_data["rewards"],
                save_path=save_path,
            )

            assert fig is not None
            assert save_path.with_suffix(".png").exists()
            plt.close(fig)


class TestIntegration:
    """Integration tests for plotting module."""

    def test_full_workflow(self) -> None:
        """Test complete plotting workflow."""
        # Generate realistic training data
        np.random.seed(42)
        timesteps = np.arange(0, 100000, 100)

        # Simulate learning progress
        base_reward = -200
        improvement = np.exp(-timesteps / 20000) * (-150)
        noise = 20 * np.random.randn(len(timesteps))
        rewards = base_reward - improvement + noise

        # Test multiple plotting functions
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Learning curve
            fig1 = plot_learning_curve(
                timesteps=timesteps,
                rewards=rewards,
                title="PPO Training Progress",
                save_path=save_dir / "learning_curve",
            )
            assert fig1 is not None
            plt.close(fig1)

            # Convergence analysis
            fig2 = plot_convergence_analysis(
                timesteps=timesteps,
                rewards=rewards,
                save_path=save_dir / "convergence",
            )
            assert fig2 is not None
            plt.close(fig2)

            # Reward distribution
            fig3 = plot_reward_distribution(
                rewards=rewards,
                save_path=save_dir / "distribution",
            )
            assert fig3 is not None
            plt.close(fig3)

            # Check all files were created
            assert (save_dir / "learning_curve.png").exists()
            assert (save_dir / "convergence.png").exists()
            assert (save_dir / "distribution.png").exists()

    @patch("src.visualization.plots.logger")
    def test_error_handling(self, mock_logger: Mock) -> None:
        """Test error handling and logging."""
        # Test with invalid data
        with pytest.raises(ValueError):
            plot_learning_curve(
                timesteps=[1, 2, 3],
                rewards=[1, 2],  # Mismatched length
            )

        # Test smoothing with very short data
        short_data = np.array([1, 2])
        result = apply_smoothing(short_data, window=10)
        mock_logger.warning.assert_called()
        np.testing.assert_array_equal(result, short_data)
