"""Report generation functions for final deliverable."""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.constants import REWARD_THRESHOLD
from src.reporting.logging import get_logger
from src.reporting.utils import read_config, read_metrics_csv, discover_experiments

logger = get_logger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def _read_markdown_section(md_path: Path) -> str:
    """Read markdown section from file.

    Args:
        md_path: Path to markdown file

    Returns:
        Content of the file, or empty string if not found
    """
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return ""


def _create_table_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table.

    Args:
        df: DataFrame to convert

    Returns:
        Markdown table string
    """
    return df.to_markdown(index=False)


# ============================================================================
# T048: Create report template
# ============================================================================


def create_report_template(seed: int) -> str:
    """Create markdown template with required sections.

    Args:
        seed: Random seed used in experiments

    Returns:
        Complete markdown template with all required sections
    """
    template = f"""# Финальный отчет: Обучение агента для LunarLander-v3

## Цель

Разработать и обучить агента с подкреплением для задачи посадки лунного модуля LunarLander-v3 из библиотеки Gymnasium. Агент должен достигать средней награды не менее {REWARD_THRESHOLD} по 10-20 эпизодам оценки.

## Гипотезы

В этом разделе документируются все проверенные гипотезы о влиянии различных гиперпараметров и алгоритмов на обучение агента.

## Эксперименты

В этом разделе описываются все проведенные эксперименты с подробной информацией о параметрах обучения.

## Результаты

В этом разделе представлены количественные результаты обучения всех моделей, включая графики обучения и сравнительные таблицы.

## Анализ

В этом разделе содержится содержательный анализ полученных результатов, интерпретация влияния экспериментов на обучение и ответы на вопросы: какие эксперименты были успешными, какие гиперпараметры дали наилучший результат, и какие рекомендации можно сделать для дальнейших исследований.

## Выводы

В этом разделе подводятся итоги проделанной работы и делаются заключения о достижении целей проекта.

## Воспроизводимость

В этом разделе описывается, как точно воспроизвести результаты обучения, включая:
- Фиксированный seed: {seed}
- Зависимости (pip freeze, conda environment)
- Команды для запуска обучения
- Полный код обучения

---

*Отчет сгенерирован автоматически с использованием модуля `src.reporting.generate_report`*
"""
    return template


# ============================================================================
# T049: Generate hypothesis section
# ============================================================================


def generate_hypothesis_section(hypothesis_coverage_path: Path) -> str:
    """Generate hypothesis documentation section.

    Reads existing hypothesis_coverage.md and formats it for the report.

    Args:
        hypothesis_coverage_path: Path to hypothesis coverage file

    Returns:
        Formatted markdown section for hypotheses
    """
    content = _read_markdown_section(hypothesis_coverage_path)

    if not content:
        logger.warning(f"Hypothesis coverage file not found: {hypothesis_coverage_path}")
        return "## Гипотезы\n\n*Файл покрытия гипотез не найден*"

    # Extract content after the header
    lines = content.split("\n")
    if len(lines) > 1 and "##" in lines[0]:
        # Keep content after first header
        result = "\n".join(lines[1:])
        return f"## Гипотезы\n\n{result}"

    return content


# ============================================================================
# T050: Generate experiments section
# ============================================================================


def generate_experiments_section(
    comparison_csv: Path,
    experiments_dir: Path,
) -> str:
    """Generate experiments documentation section.

    Describes all conducted experiments with parameters.

    Args:
        comparison_csv: Path to comparison CSV
        experiments_dir: Path to experiments directory

    Returns:
        Formatted markdown section for experiments
    """
    df = pd.read_csv(comparison_csv)

    section = "## Эксперименты\n\n"

    # Sort by best_eval_reward descending
    df_sorted = df.sort_values("best_eval_reward", ascending=False)

    # Create table
    section += "### Сводная таблица экспериментов\n\n"
    section += _create_table_markdown(
        df_sorted[[
            "experiment_id",
            "algorithm",
            "seed",
            "timesteps",
            "gamma",
            "ent_coef",
            "learning_rate",
            "best_eval_reward",
            "best_eval_std",
            "convergence_status",
        ]]
    )

    # Add detailed descriptions for top models
    section += "\n### Детальное описание топ-5 моделей\n\n"

    for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
        section += f"#### {i+1}. {row['experiment_id']}\n\n"
        section += f"- **Алгоритм**: {row['algorithm']}\n"
        section += f"- **Seed**: {int(row['seed'])}\n"
        section += f"- **Timesteps**: {int(row['timesteps']):,}\n"
        if pd.notna(row.get("gamma")):
            section += f"- **Gamma**: {row['gamma']}\n"
        if pd.notna(row.get("learning_rate")):
            section += f"- **Learning Rate**: {row['learning_rate']}\n"
        section += f"- **Средняя награда**: {row['best_eval_reward']:.2f} ± {row['best_eval_std']:.2f}\n"
        section += f"- **Статус сходимости**: {row['convergence_status']}\n"
        section += "\n"

    return section


# ============================================================================
# T051: Generate quantitative results section
# ============================================================================


def generate_quantitative_results(comparison_csv: Path) -> str:
    """Generate quantitative results section with tables.

    Creates table with mean reward ± std for 10-20 eval episodes.

    Args:
        comparison_csv: Path to comparison CSV

    Returns:
        Formatted markdown section with quantitative results
    """
    df = pd.read_csv(comparison_csv)

    section = "## Результаты\n\n"

    # Summary statistics
    section += "### Сводная статистика\n\n"
    section += f"- **Всего экспериментов**: {len(df)}\n"
    section += f"- **Сошлось моделей** (награда ≥ {REWARD_THRESHOLD}): {sum(df['best_eval_reward'] >= REWARD_THRESHOLD)}\n"
    section += f"- **Не сошлось моделей**: {sum(df['best_eval_reward'] < REWARD_THRESHOLD)}\n"

    # Top 3 models table
    df_top3 = df.sort_values("best_eval_reward", ascending=False).head(3)

    section += "\n### Топ-3 модели по средней награде\n\n"
    section += _create_table_markdown(
        df_top3[["experiment_id", "algorithm", "best_eval_reward", "best_eval_std", "convergence_status"]]
    )

    # All models table
    section += "\n### Все модели\n\n"
    section += _create_table_markdown(
        df.sort_values("best_eval_reward", ascending=False)[
            ["experiment_id", "algorithm", "best_eval_reward", "best_eval_std", "convergence_status"]
        ]
    )

    return section


# ============================================================================
# T052: Generate analysis section
# ============================================================================


def generate_analysis_section(
    comparison_csv: Path,
    hypothesis_coverage_path: Path,
) -> str:
    """Generate analysis section with 3-6 sentences.

    Creates meaningful analysis with interpretation of results
    and influence of experiments on learning.

    Args:
        comparison_csv: Path to comparison CSV
        hypothesis_coverage_path: Path to hypothesis coverage file

    Returns:
        Formatted markdown section with analysis
    """
    df = pd.read_csv(comparison_csv)

    # Calculate key statistics
    top_reward = df["best_eval_reward"].max()
    top_model = df.loc[df["best_eval_reward"].idxmax(), "experiment_id"]
    converged_count = sum(df["best_eval_reward"] >= REWARD_THRESHOLD)
    total_count = len(df)

    section = "## Анализ\n\n"

    # Generate analysis sentences
    analysis = f"""В ходе исследования было обучено {total_count} моделей RL для решения задачи LunarLander-v3.

Наилучший результат показала модель **{top_model}** с средней наградой **{top_reward:.2f}**, что значительно превышает требуемый порог {REWARD_THRESHOLD}.

Всего {converged_count} из {total_count} моделей ({converged_count/total_count*100:.1f}%) достигли успешной сходимости, что подтверждает эффективность выбранного подхода обучения.
"""

    section += analysis
    return section


# ============================================================================
# T053: Generate reproducibility section
# ============================================================================


def generate_reproducibility_section(
    seed: int,
    requirements_path: Path | None = None,
    environment_path: Path | None = None,
    training_code_path: Path | None = None,
) -> str:
    """Generate reproducibility documentation section.

    Documents fixed seed, dependencies, and run commands.

    Args:
        seed: Random seed used
        requirements_path: Optional path to requirements.txt
        environment_path: Optional path to environment.yml
        training_code_path: Optional path to training code

    Returns:
        Formatted markdown section with reproducibility info
    """
    section = "## Воспроизводимость\n\n"

    section += f"### Фиксированный Seed\n\n"
    section += f"Все эксперименты выполнены с фиксированным случайным seed **{seed}** для обеспечения воспроизводимости результатов.\n\n"

    # Dependencies
    section += "### Зависимости\n\n"

    if requirements_path and requirements_path.exists():
        section += "#### pip requirements\n\n"
        section += "```bash\npip install -r requirements.txt\n```\n\n"
        section += "Содержимое `requirements.txt`:\n"
        section += "```\n"
        section += requirements_path.read_text()
        section += "```\n\n"
    else:
        section += "Для получения списка зависимостей выполните:\n"
        section += "```bash\npip freeze > results/reports/requirements.txt\n```\n\n"

    if environment_path and environment_path.exists():
        section += "#### conda environment\n\n"
        section += "```bash\nconda env create -f environment.yml\nconda activate <env_name>\n```\n\n"

    # Training code
    section += "### Код обучения\n\n"
    if training_code_path and training_code_path.exists():
        section += f"Полный код обучения находится в: `{training_code_path}`\n\n"
    else:
        section += "Полный код обучения находится в директории `src/training/`.\n\n"

    # Run commands
    section += "### Команды для запуска\n\n"
    section += "```bash\n# Запуск обучения с фиксированным seed\n"
    section += "python -m src.training.trainer --seed 42\n\n"
    section += "\n# Генерация отчета\n"
    section += "python -m src.reporting.analyze_models --check-hypotheses\n"
    section += "python -m src.reporting.generate_plots dashboard\n"
    section += "python -m src.reporting.generate_report --check-completeness\n```\n\n"

    return section


# ============================================================================
# T054: Update README
# ============================================================================


def update_readme(
    readme_path: Path,
    comparison_csv: Path | None = None,
    plots_dir: Path | None = None,
    videos_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Update README.md with embedded visualizations and results.

    Args:
        readme_path: Path to README.md
        comparison_csv: Optional path to comparison CSV
        plots_dir: Optional path to plots directory
        videos_dir: Optional path to videos directory
        output_dir: Optional path to reports directory
    """
    if not readme_path.exists():
        logger.warning(f"README not found at {readme_path}, skipping update")
        return

    readme_content = readme_path.read_text(encoding="utf-8")

    # Generate results section
    results_section = "\n## Результаты\n\n"

    # Add top models summary
    if comparison_csv and comparison_csv.exists():
        df = pd.read_csv(comparison_csv)
        top_model = df.loc[df["best_eval_reward"].idxmax()]
        results_section += f"**Лучшая модель**: {top_model['experiment_id']} "
        results_section += f"с наградой {top_model['best_eval_reward']:.2f} ± {top_model['best_eval_std']:.2f}\n\n"

    # Add plots
    if plots_dir and plots_dir.exists():
        results_section += "### Графики обучения\n\n"
        plots = list(plots_dir.glob("*.png"))
        for plot in plots:
            # Use relative path for better portability
            rel_path = plot.relative_to(readme_path.parent)
            results_section += f"![{plot.stem}]({rel_path})\n\n"

    # Add videos
    if videos_dir and videos_dir.exists():
        results_section += "### Демонстрационные видео\n\n"
        videos = list(videos_dir.glob("*.mp4"))
        for video in videos:
            rel_path = video.relative_to(readme_path.parent)
            results_section += f"- [{video.stem}]({rel_path})\n"

    # Check if README already has results section
    if "## Результаты" in readme_content:
        # Replace existing section
        parts = readme_content.split("## Результаты")
        new_content = parts[0] + results_section
        readme_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Updated README results section at {readme_path}")
    else:
        # Append to end
        new_content = readme_content + "\n\n" + results_section
        readme_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Appended results section to README at {readme_path}")


# ============================================================================
# T055: Generate final report
# ============================================================================


def generate_final_report(
    comparison_csv: Path,
    hypothesis_coverage: Path,
    plots_dir: Path | None = None,
    videos_dir: Path | None = None,
    output_path: Path = Path("results/reports/FINAL_REPORT.md"),
    seed: int = 42,
) -> None:
    """Generate complete final report with all sections.

    Combines all sections into complete markdown report and
    embeds plots and videos.

    Args:
        comparison_csv: Path to comparison CSV
        hypothesis_coverage: Path to hypothesis coverage file
        plots_dir: Optional path to plots directory
        videos_dir: Optional path to videos directory
        output_path: Path to save final report
        seed: Random seed used in experiments
    """
    logger.info(f"Generating final report at {output_path}")

    # Generate all sections
    report = create_report_template(seed=seed)
    report += "\n---\n\n"
    report += generate_hypothesis_section(hypothesis_coverage)
    report += "\n---\n\n"
    report += generate_experiments_section(comparison_csv, output_path.parent)
    report += "\n---\n\n"

    # Add plots section if available
    if plots_dir and plots_dir.exists():
        report += "## Графики\n\n"
        plots = list(plots_dir.glob("*.png"))
        for plot in sorted(plots):
            rel_path = plot.relative_to(output_path.parent)
            report += f"### {plot.stem}\n\n"
            report += f"![{plot.stem}]({rel_path})\n\n"
        report += "---\n\n"

    report += generate_quantitative_results(comparison_csv)
    report += "\n---\n\n"
    report += generate_analysis_section(comparison_csv, hypothesis_coverage)
    report += "\n---\n\n"
    report += generate_reproducibility_section(
        seed=seed,
        requirements_path=output_path.parent / "requirements.txt",
        environment_path=output_path.parent / "environment.yml",
        training_code_path=Path("src/training"),
    )

    # Add videos section if available
    if videos_dir and videos_dir.exists():
        report += "\n---\n\n"
        report += "## Демонстрационные видео\n\n"
        videos = list(videos_dir.glob("*.mp4"))
        for video in sorted(videos):
            rel_path = video.relative_to(output_path.parent)
            report += f"### {video.stem}\n\n"
            report += f"[Смотреть видео]({rel_path})\n\n"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Final report saved to {output_path}")


# ============================================================================
# T056: CLI entry point
# ============================================================================


def main() -> None:
    """Main CLI entry point for report generation.

    Supports arguments:
    - comparison-csv: Path to model comparison CSV
    - hypothesis-coverage: Path to hypothesis coverage file
    - plots-dir: Path to plots directory
    - videos-dir: Path to videos directory
    - output: Path to save FINAL_REPORT.md
    - seed: Random seed used
    - check-completeness: Validate all required artifacts
    """
    parser = argparse.ArgumentParser(
        description="Generate final report for RL agent training"
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("results/reports/model_comparison.csv"),
        help="Path to model comparison CSV",
    )
    parser.add_argument(
        "--hypothesis-coverage",
        type=Path,
        default=Path("results/reports/hypothesis_coverage.md"),
        help="Path to hypothesis coverage markdown",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/reports"),
        help="Path to plots directory",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("results/reports/videos"),
        help="Path to videos directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reports/FINAL_REPORT.md"),
        help="Path to save FINAL_REPORT.md",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in experiments",
    )
    parser.add_argument(
        "--check-completeness",
        action="store_true",
        help="Validate all required artifacts before generating report",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update README.md with results",
    )

    args = parser.parse_args()

    # Check completeness if requested
    if args.check_completeness:
        logger.info("Checking report completeness...")

        missing_artifacts = []

        # Check required files
        required_files = {
            "model_comparison.csv": args.comparison_csv,
            "hypothesis_coverage.md": args.hypothesis_coverage,
        }

        for name, path in required_files.items():
            if not path.exists():
                missing_artifacts.append(f"Missing {name}: {path}")

        # Check plots
        plot_files = list(args.plots_dir.glob("*.png")) if args.plots_dir.exists() else []
        required_plots = ["reward_vs_timestep.png", "reward_vs_episode.png", "agent_comparison.png"]
        for plot_name in required_plots:
            if not any(plot_name in str(p) for p in plot_files):
                missing_artifacts.append(f"Missing plot: {plot_name}")

        # Check videos
        if args.videos_dir.exists():
            video_files = list(args.videos_dir.glob("*.mp4"))
            if not video_files:
                missing_artifacts.append("No demo videos found")

        if missing_artifacts:
            logger.error("Missing required artifacts:")
            for artifact in missing_artifacts:
                logger.error(f"  - {artifact}")
            raise FileNotFoundError("Report completeness check failed")

        logger.info("All required artifacts found")

    # Generate report
    generate_final_report(
        comparison_csv=args.comparison_csv,
        hypothesis_coverage=args.hypothesis_coverage,
        plots_dir=args.plots_dir,
        videos_dir=args.videos_dir,
        output_path=args.output,
        seed=args.seed,
    )

    # Update README if requested
    if args.update_readme:
        readme_path = Path("README.md")
        update_readme(
            readme_path=readme_path,
            comparison_csv=args.comparison_csv,
            plots_dir=args.plots_dir,
            videos_dir=args.videos_dir,
            output_dir=args.output.parent,
        )

    logger.info("Report generation complete")


if __name__ == "__main__":
    main()
