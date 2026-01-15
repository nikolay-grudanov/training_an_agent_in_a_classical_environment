"""Unit tests for graph generation utilities."""

import tempfile
from pathlib import Path

import matplotlib
import pytest

# Set non-interactive backend for testing
matplotlib.use("Agg")

from src.visualization.graphs import (
    ComparisonPlotGenerator,
    GammaComparisonPlotGenerator,
    LearningCurveGenerator,
)


class TestLearningCurveGenerator:
    """Tests for LearningCurveGenerator."""

    def test_init(self) -> None:
        """Test LearningCurveGenerator initialization."""
        generator = LearningCurveGenerator(width=12, height=8, dpi=200)

        assert generator.width == 12
        assert generator.height == 8
        assert generator.dpi == 200

    def test_init_defaults(self) -> None:
        """Test LearningCurveGenerator with default parameters."""
        generator = LearningCurveGenerator()

        assert generator.width == 10
        assert generator.height == 6
        assert generator.dpi == 150

    def test_generate_from_metrics_creates_file(self) -> None:
        """Test that graph generation creates PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock metrics CSV
            metrics_path = Path(tmpdir) / "metrics.csv"
            metrics_path.write_text(
                "timesteps,reward_mean,reward_std\n"
                "0,0.0,10.0\n"
                "50000,100.0,20.0\n"
                "100000,180.0,15.0\n"
            )

            output_path = Path(tmpdir) / "reward_curve.png"

            generator = LearningCurveGenerator()
            fig = generator.generate_from_metrics(
                metrics_path,
                output_path,
                title="Test Learning Curve",
            )

            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert fig is not None

    def test_generate_without_std(self) -> None:
        """Test generation without standard deviation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.csv"
            metrics_path.write_text("timesteps,reward_mean\n0,0.0\n50000,100.0\n")

            output_path = Path(tmpdir) / "curve.png"

            generator = LearningCurveGenerator()
            fig = generator.generate_from_metrics(
                metrics_path,
                output_path,
                show_ci=False,
            )

            assert output_path.exists()
            assert fig is not None

    def test_generate_creates_parent_directory(self) -> None:
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.csv"
            metrics_path.write_text(
                "timesteps,reward_mean,reward_std\n0,0.0,10.0\n50000,100.0,20.0\n"
            )

            output_path = Path(tmpdir) / "subdir" / "reward_curve.png"

            generator = LearningCurveGenerator()
            generator.generate_from_metrics(
                metrics_path,
                output_path,
            )

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_generate_with_custom_title(self) -> None:
        """Test generation with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.csv"
            metrics_path.write_text("timesteps,reward_mean\n0,0.0\n50000,100.0\n")

            output_path = Path(tmpdir) / "curve.png"

            generator = LearningCurveGenerator()
            generator.generate_from_metrics(
                metrics_path,
                output_path,
                title="Custom Title",
            )

            assert output_path.exists()


class TestComparisonPlotGenerator:
    """Tests for ComparisonPlotGenerator."""

    def test_init(self) -> None:
        """Test ComparisonPlotGenerator initialization."""
        generator = ComparisonPlotGenerator(width=14, height=9, dpi=200)

        assert generator.width == 14
        assert generator.height == 9
        assert generator.dpi == 200
        assert len(generator.colors) == 4

    def test_init_defaults(self) -> None:
        """Test ComparisonPlotGenerator with default parameters."""
        generator = ComparisonPlotGenerator()

        assert generator.width == 12
        assert generator.height == 7
        assert generator.dpi == 150

    def test_generate_comparison(self) -> None:
        """Test comparison plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two mock metrics files
            exp1_path = Path(tmpdir) / "exp1.csv"
            exp2_path = Path(tmpdir) / "exp2.csv"

            for path, label in [(exp1_path, "A2C"), (exp2_path, "PPO")]:
                path.write_text(
                    "timesteps,reward_mean,reward_std\n0,0.0,10.0\n100000,150.0,20.0\n"
                )

            output_path = Path(tmpdir) / "comparison.png"

            generator = ComparisonPlotGenerator()
            fig = generator.generate(
                experiment_paths=[str(exp1_path), str(exp2_path)],
                labels=["A2C", "PPO"],
                output_path=output_path,
            )

            assert output_path.exists()
            assert fig is not None

    def test_generate_multiple_experiments(self) -> None:
        """Test comparison with more than two experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create three mock metrics files
            paths = []
            for i in range(3):
                path = Path(tmpdir) / f"exp{i}.csv"
                path.write_text(
                    "timesteps,reward_mean,reward_std\n0,0.0,10.0\n100000,150.0,20.0\n"
                )
                paths.append(path)

            output_path = Path(tmpdir) / "comparison.png"

            generator = ComparisonPlotGenerator()
            fig = generator.generate(
                experiment_paths=[str(p) for p in paths],
                labels=["Exp1", "Exp2", "Exp3"],
                output_path=output_path,
            )

            assert output_path.exists()
            assert fig is not None

    def test_generate_without_std(self) -> None:
        """Test comparison without standard deviation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp1_path = Path(tmpdir) / "exp1.csv"
            exp2_path = Path(tmpdir) / "exp2.csv"

            for path in [exp1_path, exp2_path]:
                path.write_text("timesteps,reward_mean\n0,0.0\n100000,150.0\n")

            output_path = Path(tmpdir) / "comparison.png"

            generator = ComparisonPlotGenerator()
            generator.generate(
                experiment_paths=[str(exp1_path), str(exp2_path)],
                labels=["A2C", "PPO"],
                output_path=output_path,
            )

            assert output_path.exists()

    def test_generate_creates_parent_directory(self) -> None:
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp1_path = Path(tmpdir) / "exp1.csv"
            exp1_path.write_text("timesteps,reward_mean\n0,0.0\n100000,150.0\n")

            output_path = Path(tmpdir) / "subdir" / "comparison.png"

            generator = ComparisonPlotGenerator()
            generator.generate(
                experiment_paths=[str(exp1_path)],
                labels=["Exp1"],
                output_path=output_path,
            )

            assert output_path.exists()


class TestGammaComparisonPlotGenerator:
    """Tests for GammaComparisonPlotGenerator."""

    def test_init(self) -> None:
        """Test GammaComparisonPlotGenerator initialization."""
        generator = GammaComparisonPlotGenerator()

        assert isinstance(generator, ComparisonPlotGenerator)
        assert generator.width == 12
        assert generator.height == 7

    def test_generate_gamma_comparison(self) -> None:
        """Test gamma-specific comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock metrics files for different gamma values
            gamma_results = {
                0.90: str(Path(tmpdir) / "gamma_090.csv"),
                0.99: str(Path(tmpdir) / "gamma_099.csv"),
                0.999: str(Path(tmpdir) / "gamma_999.csv"),
            }

            for gamma, path in gamma_results.items():
                Path(path).write_text(
                    "timesteps,reward_mean,reward_std\n0,0.0,10.0\n100000,150.0,20.0\n"
                )

            output_path = Path(tmpdir) / "gamma_comparison.png"

            generator = GammaComparisonPlotGenerator()
            fig = generator.generate_gamma_comparison(
                gamma_results=gamma_results,
                output_path=str(output_path),
                title="Gamma Comparison",
            )

            assert output_path.exists()
            assert fig is not None

    def test_generate_gamma_comparison_default_title(self) -> None:
        """Test gamma comparison with default title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gamma_results = {
                0.99: str(Path(tmpdir) / "gamma_099.csv"),
            }

            Path(gamma_results[0.99]).write_text(
                "timesteps,reward_mean\n0,0.0\n100000,150.0\n"
            )

            output_path = Path(tmpdir) / "gamma_comparison.png"

            generator = GammaComparisonPlotGenerator()
            generator.generate_gamma_comparison(
                gamma_results=gamma_results,
                output_path=str(output_path),
            )

            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
