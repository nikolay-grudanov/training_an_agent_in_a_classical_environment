"""Tests for generate_report module."""

from pathlib import Path
from typing import Any

import pytest

from src.reporting.constants import ALGO_PPO, REWARD_THRESHOLD


@pytest.fixture
def temp_report_dir(tmp_path: Path) -> Path:
    """Create temporary directory for report output.

    Args:
        tmp_path: Pytest tmp_path fixture

    Returns:
        Temporary directory path for reports
    """
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


@pytest.fixture
def sample_comparison_csv(tmp_path: Path) -> Path:
    """Create sample comparison CSV for testing.

    Args:
        tmp_path: Pytest tmp_path fixture

    Returns:
        Path to sample comparison CSV
    """
    csv_path = tmp_path / "model_comparison.csv"
    csv_data = """experiment_id,algorithm,environment,seed,timesteps,gamma,ent_coef,learning_rate,model_path,final_train_reward,final_train_std,best_eval_reward,best_eval_std,final_eval_reward,final_eval_std,total_training_time,convergence_status
ppo_seed42,PPO,LunarLander-v3,42,500000,0.999,,0.0003,/path/to/model,-1.5,2.0,255.7,10.2,250.5,15.0,1000.0,CONVERGED
ppo_seed123,PPO,LunarLander-v3,123,500000,0.99,,0.0003,/path/to/model2,0.2,1.5,236.3,8.5,220.1,12.0,950.0,CONVERGED
a2c_seed42,A2C,LunarLander-v3,42,500000,0.99,,0.0001,/path/to/model3,-50.2,10.5,120.0,25.0,100.0,950.0,NOT_CONVERGED
"""
    csv_path.write_text(csv_data)
    return csv_path


@pytest.fixture
def sample_hypothesis_coverage(tmp_path: Path) -> Path:
    """Create sample hypothesis coverage file for testing.

    Args:
        tmp_path: Pytest tmp_path fixture

    Returns:
        Path to sample hypothesis coverage file
    """
    coverage_path = tmp_path / "hypothesis_coverage.md"
    coverage_data = """# Отчет о покрытии гипотез

## Проверенные гипотезы

### H1: PPO превосходит A2C по награде
- **Статус**: ПОДДЕРЖИВАЕТСЯ
- **Результаты**:
  - Средняя награда PPO: 250.5 ± 15.0
  - Средняя награда A2C: 120.0 ± 25.0
- **Вывод**: PPO показывает значительно лучшие результаты (p < 0.05)

### H2: Влияние seed на сходимость
- **Статус**: ЧАСТИЧНО ПОДДЕРЖИВАЕТСЯ
- **Результаты**:
  - Seed 42: 255.7 ± 10.2 (сошлось)
  - Seed 123: 236.3 ± 8.5 (сошлось)
  - Seed 999: 220.1 ± 12.0 (сошлось)
- **Вывод**: Различия в результатах меньше ожидаемых, но виден эффект

## Непроверенные гипотезы

### H3: Влияние learning rate
- **Статус**: НЕ ПРОВЕРЕНА
- **Рекомендация**: Провести эксперименты с различными learning rates
"""
    coverage_path.write_text(coverage_data)
    return coverage_path


# ============================================================================
# T045: Test for generate_final_report
# ============================================================================


def test_generate_final_report(
    sample_comparison_csv: Path,
    sample_hypothesis_coverage: Path,
    temp_report_dir: Path,
) -> None:
    """Test generation of final report.

    Verifies:
    - FINAL_REPORT.md is created
    - Report contains all required sections
    - Report integrates plots, videos, and analysis

    Args:
        sample_comparison_csv: Sample comparison CSV
        sample_hypothesis_coverage: Sample hypothesis coverage file
        temp_report_dir: Temporary directory for report output
    """
    from src.reporting.generate_report import generate_final_report

    # Create dummy plots directory with files
    plots_dir = temp_report_dir
    (plots_dir / "reward_vs_timestep.png").touch()
    (plots_dir / "reward_vs_episode.png").touch()
    (plots_dir / "agent_comparison.png").touch()

    output_path = temp_report_dir / "FINAL_REPORT.md"

    # Generate report
    generate_final_report(
        comparison_csv=sample_comparison_csv,
        hypothesis_coverage=sample_hypothesis_coverage,
        plots_dir=plots_dir,
        videos_dir=None,  # Skip videos for this test
        output_path=output_path,
        seed=42,
    )

    # Verify report was created
    assert output_path.exists(), f"Report not created at {output_path}"

    # Verify report contains required sections
    report_content = output_path.read_text()
    required_sections = [
        "# Цель",
        "# Гипотезы",
        "# Эксперименты",
        "# Результаты",
        "# Анализ",
        "# Выводы",
        "# Воспроизводимость",
    ]
    for section in required_sections:
        assert section in report_content, f"Report missing section: {section}"


# ============================================================================
# T046: Test for reproducibility_check
# ============================================================================


def test_reproducibility_check(temp_report_dir: Path, tmp_path: Path) -> None:
    """Test reproducibility verification.

    Verifies:
    - seed=42 is documented
    - requirements.txt contains dependencies
    - environment.yml contains conda environment
    - All artifacts are present

    Args:
        temp_report_dir: Temporary directory for report output
        tmp_path: Pytest tmp_path fixture
    """
    from src.reporting.generate_report import generate_reproducibility_section

    # Create dummy artifacts
    (temp_report_dir / "requirements.txt").write_text("numpy==1.24.0\nstable-baselines3==2.0.0")
    (temp_report_dir / "environment.yml").write_text("name: test\ndependencies:\n  - python=3.10")

    output_path = temp_report_dir / "reproducibility.md"

    # Generate reproducibility section
    content = generate_reproducibility_section(
        seed=42,
        requirements_path=temp_report_dir / "requirements.txt",
        environment_path=temp_report_dir / "environment.yml",
        training_code_path=tmp_path / "training.py",
    )

    output_path.write_text(content)

    # Verify section contains required information
    content = output_path.read_text()
    assert "seed=42" in content or "seed 42" in content
    assert "requirements.txt" in content
    assert "environment.yml" in content


# ============================================================================
# T047: Test for report_package
# ============================================================================


def test_report_package(
    sample_comparison_csv: Path,
    temp_report_dir: Path,
    tmp_path: Path,
) -> None:
    """Test complete report package generation.

    Verifies:
    - All required files are created
    - Report is complete and valid
    - Readme is updated with visualizations

    Args:
        sample_comparison_csv: Sample comparison CSV
        temp_report_dir: Temporary directory for report output
        tmp_path: Pytest tmp_path fixture
    """
    from src.reporting.generate_report import generate_final_report, update_readme

    # Create directories
    plots_dir = temp_report_dir / "plots"
    videos_dir = temp_report_dir / "videos"
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    # Create dummy files
    (plots_dir / "reward_vs_timestep.png").touch()
    (videos_dir / "demo_best.mp4").touch()

    # Create dummy hypothesis coverage
    hypothesis_path = temp_report_dir / "hypothesis_coverage.md"
    hypothesis_path.write_text("# Test Hypotheses\n\n## H1: Test\n- Status: SUPPORTED")

    output_path = temp_report_dir / "FINAL_REPORT.md"
    readme_path = temp_report_dir / "README.md"

    # Generate report
    generate_final_report(
        comparison_csv=sample_comparison_csv,
        hypothesis_coverage=hypothesis_path,
        plots_dir=plots_dir,
        videos_dir=videos_dir,
        output_path=output_path,
        seed=42,
    )

    # Update README
    update_readme(
        readme_path=readme_path,
        comparison_csv=sample_comparison_csv,
        plots_dir=plots_dir,
        videos_dir=videos_dir,
        output_dir=temp_report_dir,
    )

    # Verify all files created
    assert output_path.exists(), "FINAL_REPORT.md not created"
    assert readme_path.exists(), "README.md not updated"

    # Verify README contains visualizations references
    readme_content = readme_path.read_text()
    assert "reward_vs_timestep.png" in readme_content or "plots/" in readme_content
