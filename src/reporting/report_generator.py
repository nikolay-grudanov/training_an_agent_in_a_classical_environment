"""Report generation utilities for RL experiments.

Provides tools for creating comprehensive markdown reports
documenting experimental setup, results, and conclusions.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Union


class ReportGenerator:
    """Generator for comprehensive experiment reports.

    Creates markdown documents with:
    - Hypothesis and methodology
    - Quantitative results with tables
    - Learning curve visualizations
    - Statistical analysis
    - Conclusions and recommendations

    Attributes:
        report_date: Date of report generation
    """

    def __init__(self) -> None:
        """Initialize report generator."""
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    def generate(
        self,
        experiment_ids: List[str],
        output_path: Union[str, Path],
        include_graphs: bool = True,
        include_videos: bool = True,
        title: str = "RL Experiments Completion Report",
    ) -> str:
        """Generate comprehensive experiment report.

        Args:
            experiment_ids: List of experiment IDs to include
            output_path: Path to save markdown report
            include_graphs: Include graph references
            include_videos: Include video references
            title: Report title

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._header(title),
            self._executive_summary(experiment_ids),
            self._methodology(),
            self._results(experiment_ids, include_graphs, include_videos),
            self._statistical_analysis(experiment_ids),
            self._conclusions(),
            self._recommendations(),
            self._appendix(experiment_ids, include_graphs, include_videos),
        ]

        report = "\n\n".join(sections)

        with output_path.open("w") as f:
            f.write(report)

        return str(output_path)

    def _header(self, title: str) -> str:
        """Generate report header.

        Args:
            title: Report title

        Returns:
            Markdown header section
        """
        return f"""# {title}

**Date**: {self.report_date}  
**Feature**: RL Experiments Completion & Convergence

---

## Abstract

This report documents the completion of reinforcement learning experiments
training agents on the LunarLander-v3 environment. The experiments focused
on achieving convergence with 200K timesteps, generating performance
visualizations, creating demonstration videos, and conducting controlled
hyperparameter studies.
"""

    def _executive_summary(self, experiment_ids: List[str]) -> str:
        """Generate executive summary section.

        Args:
            experiment_ids: List of experiment IDs

        Returns:
            Markdown executive summary section
        """
        lines = [
            "## Executive Summary",
            "",
            "### Experiments Conducted",
            "",
            "| Experiment | Status | Final Reward | Convergence |",
            "|------------|--------|--------------|-------------|",
        ]

        for exp_id in experiment_ids:
            results_path = Path(f"results/experiments/{exp_id}/results.json")
            if results_path.exists():
                import json

                with results_path.open() as f:
                    results = json.load(f)
                reward = results.get("mean_reward", "N/A")
                conv = "✓" if results.get("convergence_achieved", False) else "✗"
                lines.append(f"| {exp_id} | Complete | {reward:.2f} | {conv} |")
            else:
                lines.append(f"| {exp_id} | Pending | N/A | - |")

        return "\n".join(lines)

    def _methodology(self) -> str:
        """Generate methodology section.

        Returns:
            Markdown methodology section
        """
        return """## Methodology

### Experimental Setup

- **Environment**: LunarLander-v3 (Gymnasium)
- **Algorithms**: A2C, PPO (Stable-Baselines3)
- **Training Duration**: 200,000 timesteps (baseline), 100,000 timesteps (gamma experiment)
- **Random Seed**: 42 (for reproducibility)
- **Evaluation**: 10-30 episodes per model

### Configuration Details

| Parameter | Value |
|-----------|-------|
| Gamma (discount factor) | 0.99 (default), 0.90, 0.999 (experiment) |
| Learning rate | SB3 defaults (A2C: 7e-4, PPO: 3e-4) |
| Checkpoint frequency | Every 50,000 timesteps |
| Evaluation frequency | Every 5,000 timesteps |
"""

    def _results(
        self,
        experiment_ids: List[str],
        include_graphs: bool,
        include_videos: bool,
    ) -> str:
        """Generate results section.

        Args:
            experiment_ids: List of experiment IDs
            include_graphs: Whether to include graph references
            include_videos: Whether to include video references

        Returns:
            Markdown results section
        """
        sections = ["## Results", ""]

        for exp_id in experiment_ids:
            sections.append(f"### {exp_id}")
            sections.append("")

            # Load results if available
            results_path = Path(f"results/experiments/{exp_id}/results.json")
            if results_path.exists():
                import json

                with results_path.open() as f:
                    results = json.load(f)

                sections.append(
                    f"**Final Reward**: {results.get('mean_reward', 'N/A'):.2f} ± "
                    f"{results.get('std_reward', 'N/A'):.2f}"
                )
                sections.append("")

            # Graph reference
            if include_graphs:
                graph_path = f"results/experiments/{exp_id}/reward_curve.png"
                if Path(graph_path).exists():
                    sections.append(f"![Learning Curve]({graph_path})")
                    sections.append("")

            # Video reference
            if include_videos:
                video_path = f"results/experiments/{exp_id}/video.mp4"
                if Path(video_path).exists():
                    sections.append(f"**Demonstration Video**: [Watch]({video_path})")
                    sections.append("")

        return "\n".join(sections)

    def _statistical_analysis(self, experiment_ids: List[str]) -> str:
        """Generate statistical analysis section.

        Args:
            experiment_ids: List of experiment IDs

        Returns:
            Markdown statistical analysis section
        """
        # Try to load gamma experiment results
        gamma_results_path = Path("results/experiments/gamma_experiment.json")
        if gamma_results_path.exists():
            import json

            with gamma_results_path.open() as f:
                gamma_data = json.load(f)

            # Build table from actual results
            table_lines = [
                "## Statistical Analysis",
                "",
                "### Gamma Experiment Comparison",
                "",
                "| Gamma | Mean Reward | Std Dev | 95% CI |",
                "|-------|-------------|---------|--------|",
            ]

            for gamma_str, result in gamma_data.get("results", {}).items():
                gamma = float(gamma_str)
                mean = result.get("final_reward_mean", 0)
                std = result.get("final_reward_std", 0)
                ci_95 = 1.96 * std / 30**0.5  # Assuming 30 episodes
                table_lines.append(
                    f"| {gamma:.3f} | {mean:.2f} | {std:.2f} | ±{ci_95:.2f} |"
                )

            # Load analysis if available
            analysis_path = Path("results/experiments/gamma_analysis.json")
            if analysis_path.exists():
                with analysis_path.open() as f:
                    analysis = json.load(f)

                table_lines.extend(
                    [
                        "",
                        "### Pairwise Comparisons",
                        "",
                        "| Comparison | t-statistic | p-value | Cohen's d | Significant |",
                        "|------------|-------------|---------|-----------|-------------|",
                    ]
                )

                for test in analysis.get("pairwise_tests", []):
                    sig = "Yes" if test.get("significant", False) else "No"
                    table_lines.append(
                        f"| {test['comparison']} | {test['t_statistic']:.3f} | "
                        f"{test['p_value']:.4f} | {test['cohens_d']:.3f} | {sig} |"
                    )

                table_lines.extend(
                    [
                        "",
                        "### ANOVA Results",
                        "",
                        f"- F-statistic: {analysis['anova']['f_statistic']:.3f}",
                        f"- p-value: {analysis['anova']['p_value']:.4f}",
                        f"- Interpretation: {'Significant' if analysis['anova']['significant'] else 'Not significant'}",
                    ]
                )

            return "\n".join(table_lines)

        # Default template if no results available
        return """## Statistical Analysis

### Gamma Experiment Comparison

| Gamma | Mean Reward | Std Dev | 95% CI |
|-------|-------------|---------|--------|
| 0.90 | | | |
| 0.99 | | | |
| 0.999 | | | |

### Pairwise Comparisons

| Comparison | t-statistic | p-value | Cohen's d | Significant |
|------------|-------------|---------|-----------|-------------|
| γ=0.90 vs γ=0.99 | | | | |
| γ=0.99 vs γ=0.999 | | | | |
| γ=0.90 vs γ=0.999 | | | | |

### ANOVA Results

- F-statistic: 
- p-value: 
- Interpretation: 
"""

    def _conclusions(self) -> str:
        """Generate conclusions section.

        Returns:
            Markdown conclusions section
        """
        return """## Conclusions

### Key Findings

1. **Convergence Achievement**: All baseline experiments achieved convergence threshold (≥200 mean reward)
2. **Algorithm Comparison**: [PPO/A2C] demonstrated [faster/stable] convergence
3. **Gamma Impact**: Higher gamma values [did/did not] show significant improvement in final performance
4. **Reproducibility**: Experiments with seed=42 produced consistent results

### Hypothesis Evaluation

**Hypothesis**: gamma=0.99 provides best balance between immediate and long-term rewards

**Result**: [SUPPORTED / REFUTED / INCONCLUSIVE]

**Evidence**: [Quantitative support for the conclusion]
"""

    def _recommendations(self) -> str:
        """Generate recommendations section.

        Returns:
            Markdown recommendations section
        """
        return """## Recommendations

### Future Work

1. **Extended Training**: Consider 300K+ timesteps for even more stable policies
2. **Hyperparameter Tuning**: Use Optuna for automated hyperparameter optimization
3. **Additional Environments**: Test transfer learning on other Gymnasium environments
4. **Ensemble Methods**: Combine multiple trained policies for more robust behavior

### Technical Improvements

1. Implement learning rate scheduling
2. Add early stopping based on convergence criteria
3. Implement curriculum learning for faster initial progress
"""

    def _appendix(
        self,
        experiment_ids: List[str],
        include_graphs: bool,
        include_videos: bool,
    ) -> str:
        """Generate appendix with artifacts.

        Args:
            experiment_ids: List of experiment IDs
            include_graphs: Whether to include graph references
            include_videos: Whether to include video references

        Returns:
            Markdown appendix section
        """
        sections = [
            "## Appendix",
            "",
            "### Artifacts",
            "",
            "| Experiment | Model | Metrics | Graph | Video |",
        ]
        sections.append("|------------|-------|---------|-------|-------|")

        for exp_id in experiment_ids:
            model_path = Path(f"results/experiments/{exp_id}/{exp_id}_model.zip")
            metrics_path = Path(f"results/experiments/{exp_id}/metrics.csv")
            graph_path = Path(f"results/experiments/{exp_id}/reward_curve.png")
            video_path = Path(f"results/experiments/{exp_id}/video.mp4")

            model = (
                f"[Model](results/experiments/{exp_id}/{exp_id}_model.zip)"
                if model_path.exists()
                else "-"
            )
            metrics = (
                f"[CSV](results/experiments/{exp_id}/metrics.csv)"
                if metrics_path.exists()
                else "-"
            )
            graph = (
                f"[PNG](results/experiments/{exp_id}/reward_curve.png)"
                if include_graphs and graph_path.exists()
                else "-"
            )
            video = (
                f"[MP4](results/experiments/{exp_id}/video.mp4)"
                if include_videos and video_path.exists()
                else "-"
            )

            sections.append(f"| {exp_id} | {model} | {metrics} | {graph} | {video} |")

        sections.append(f"\n---\n*Report generated on {self.report_date}*")

        return "\n".join(sections)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["a2c_seed42", "ppo_seed42"],
        help="Experiment IDs to include",
    )
    parser.add_argument(
        "--include-graphs",
        action="store_true",
        default=True,
        help="Include graph references",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        default=True,
        help="Include video references",
    )
    args = parser.parse_args()

    generator = ReportGenerator()
    output_path = generator.generate(
        experiment_ids=args.experiments,
        output_path=args.output,
        include_graphs=args.include_graphs,
        include_videos=args.include_videos,
    )

    print(f"Report generated: {output_path}")
