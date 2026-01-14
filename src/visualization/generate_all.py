"""Generate comprehensive visualization reports for RL training results.

This module provides functionality to automatically generate all relevant plots
and visualizations from training logs and experiment results.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .plots import (
    PlotConfig,
    plot_learning_curve,
    plot_episode_lengths,
    plot_loss_curves,
    plot_reward_distribution,
    plot_convergence_analysis,
    plot_multiple_runs,
    plot_confidence_intervals,
    save_plot,
)

logger = logging.getLogger(__name__)

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class VisualizationGenerator:
    """Generate comprehensive visualization reports from training data."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[PlotConfig] = None,
        formats: List[str] = ["png", "svg"],
    ) -> None:
        """Initialize visualization generator.
        
        Args:
            output_dir: Directory to save generated plots.
            config: Plot configuration settings.
            formats: Output formats for plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or PlotConfig()
        self.formats = formats
        
        logger.info(f"Initialized visualization generator, output: {self.output_dir}")
    
    def generate_single_run_report(
        self,
        data: Dict[str, Any],
        run_name: str = "training_run",
        include_losses: bool = True,
    ) -> Dict[str, Path]:
        """Generate complete visualization report for a single training run.
        
        Args:
            data: Training data dictionary with keys like 'timesteps', 'rewards', etc.
            run_name: Name for the training run (used in titles and filenames).
            include_losses: Whether to include loss plots if loss data is available.
            
        Returns:
            Dictionary mapping plot type to saved file path.
        """
        logger.info(f"Generating single run report for: {run_name}")
        
        saved_plots = {}
        
        # Validate required data
        required_keys = ["timesteps", "rewards"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Required data key '{key}' not found")
        
        timesteps = np.asarray(data["timesteps"])
        rewards = np.asarray(data["rewards"])
        
        # 1. Learning Curve
        try:
            fig = plot_learning_curve(
                timesteps=timesteps,
                rewards=rewards,
                title=f"{run_name} - Learning Curve",
                smooth=True,
                confidence_interval=True,
                config=self.config,
            )
            save_path = self.output_dir / f"{run_name}_learning_curve"
            save_plot(fig, save_path, formats=self.formats)
            saved_plots["learning_curve"] = save_path.with_suffix(f".{self.formats[0]}")
            logger.info(f"Generated learning curve: {save_path}")
        except Exception as e:
            logger.error(f"Failed to generate learning curve: {e}")
        
        # 2. Episode Lengths (if available)
        if "episode_lengths" in data:
            try:
                episodes = np.arange(len(data["episode_lengths"]))
                fig = plot_episode_lengths(
                    episodes=episodes,
                    lengths=data["episode_lengths"],
                    title=f"{run_name} - Episode Length Progression",
                    config=self.config,
                )
                save_path = self.output_dir / f"{run_name}_episode_lengths"
                save_plot(fig, save_path, formats=self.formats)
                saved_plots["episode_lengths"] = save_path.with_suffix(f".{self.formats[0]}")
                logger.info(f"Generated episode lengths plot: {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate episode lengths plot: {e}")
        
        # 3. Loss Curves (if available and requested)
        if include_losses and any(key.endswith("_loss") for key in data.keys()):
            try:
                losses = {
                    key.replace("_loss", ""): data[key]
                    for key in data.keys()
                    if key.endswith("_loss")
                }
                
                if losses:
                    fig = plot_loss_curves(
                        timesteps=timesteps,
                        losses=losses,
                        title=f"{run_name} - Training Losses",
                        config=self.config,
                    )
                    save_path = self.output_dir / f"{run_name}_losses"
                    save_plot(fig, save_path, formats=self.formats)
                    saved_plots["losses"] = save_path.with_suffix(f".{self.formats[0]}")
                    logger.info(f"Generated loss curves: {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate loss curves: {e}")
        
        # 4. Reward Distribution
        try:
            fig = plot_reward_distribution(
                rewards=rewards,
                title=f"{run_name} - Reward Distribution",
                show_stats=True,
                config=self.config,
            )
            save_path = self.output_dir / f"{run_name}_reward_distribution"
            save_plot(fig, save_path, formats=self.formats)
            saved_plots["reward_distribution"] = save_path.with_suffix(f".{self.formats[0]}")
            logger.info(f"Generated reward distribution: {save_path}")
        except Exception as e:
            logger.error(f"Failed to generate reward distribution: {e}")
        
        # 5. Convergence Analysis
        try:
            fig = plot_convergence_analysis(
                timesteps=timesteps,
                rewards=rewards,
                title=f"{run_name} - Convergence Analysis",
                config=self.config,
            )
            save_path = self.output_dir / f"{run_name}_convergence"
            save_plot(fig, save_path, formats=self.formats)
            saved_plots["convergence"] = save_path.with_suffix(f".{self.formats[0]}")
            logger.info(f"Generated convergence analysis: {save_path}")
        except Exception as e:
            logger.error(f"Failed to generate convergence analysis: {e}")
        
        logger.info(f"Generated {len(saved_plots)} plots for {run_name}")
        return saved_plots
    
    def generate_comparison_report(
        self,
        runs_data: Dict[str, Dict[str, Any]],
        comparison_name: str = "algorithm_comparison",
        metrics: List[str] = ["rewards"],
    ) -> Dict[str, Path]:
        """Generate comparison visualization report for multiple runs.
        
        Args:
            runs_data: Dictionary mapping run names to their data.
            comparison_name: Name for the comparison (used in titles and filenames).
            metrics: List of metrics to compare across runs.
            
        Returns:
            Dictionary mapping plot type to saved file path.
        """
        logger.info(f"Generating comparison report: {comparison_name}")
        
        saved_plots = {}
        
        # Validate data structure
        if not runs_data:
            raise ValueError("No runs data provided")
        
        # 1. Multiple Runs Comparison for each metric
        for metric in metrics:
            try:
                # Prepare data for comparison
                comparison_data = {}
                for run_name, data in runs_data.items():
                    if "timesteps" in data and metric in data:
                        comparison_data[run_name] = {
                            "timesteps": data["timesteps"],
                            metric: data[metric],
                        }
                
                if comparison_data:
                    fig = plot_multiple_runs(
                        runs_data=comparison_data,
                        metric=metric,
                        title=f"{comparison_name} - {metric.title()} Comparison",
                        ylabel=metric.title(),
                        config=self.config,
                    )
                    save_path = self.output_dir / f"{comparison_name}_{metric}_comparison"
                    save_plot(fig, save_path, formats=self.formats)
                    saved_plots[f"{metric}_comparison"] = save_path.with_suffix(f".{self.formats[0]}")
                    logger.info(f"Generated {metric} comparison: {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate {metric} comparison: {e}")
        
        # 2. Confidence Intervals (if multiple seeds/runs available)
        for metric in metrics:
            try:
                # Group runs by algorithm (assuming naming convention: algorithm_seed)
                algorithm_groups = {}
                for run_name, data in runs_data.items():
                    if "timesteps" in data and metric in data:
                        # Extract algorithm name (everything before last underscore)
                        parts = run_name.split("_")
                        if len(parts) > 1:
                            algorithm = "_".join(parts[:-1])
                        else:
                            algorithm = run_name
                        
                        if algorithm not in algorithm_groups:
                            algorithm_groups[algorithm] = []
                        
                        algorithm_groups[algorithm].append(data[metric])
                
                # Generate confidence interval plots for algorithms with multiple runs
                for algorithm, runs_list in algorithm_groups.items():
                    if len(runs_list) > 1:
                        # Use timesteps from first run (assuming all have same length)
                        first_run = next(iter(runs_data.values()))
                        timesteps = first_run["timesteps"]
                        
                        fig = plot_confidence_intervals(
                            x_data=timesteps,
                            y_data_list=runs_list,
                            title=f"{algorithm} - {metric.title()} Confidence Intervals",
                            xlabel="Timesteps",
                            ylabel=metric.title(),
                            config=self.config,
                        )
                        save_path = self.output_dir / f"{algorithm}_{metric}_confidence"
                        save_plot(fig, save_path, formats=self.formats)
                        saved_plots[f"{algorithm}_{metric}_confidence"] = save_path.with_suffix(f".{self.formats[0]}")
                        logger.info(f"Generated {algorithm} {metric} confidence intervals: {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate confidence intervals for {metric}: {e}")
        
        logger.info(f"Generated {len(saved_plots)} comparison plots")
        return saved_plots
    
    def generate_experiment_report(
        self,
        experiment_data: Dict[str, Any],
        experiment_name: str = "experiment",
    ) -> Dict[str, Path]:
        """Generate complete visualization report for an experiment.
        
        Args:
            experiment_data: Complete experiment data including multiple runs.
            experiment_name: Name of the experiment.
            
        Returns:
            Dictionary mapping plot type to saved file path.
        """
        logger.info(f"Generating experiment report: {experiment_name}")
        
        all_plots = {}
        
        # Check if this is single run or multiple runs data
        if "runs" in experiment_data:
            # Multiple runs experiment
            runs_data = experiment_data["runs"]
            
            # Generate individual run reports
            for run_name, run_data in runs_data.items():
                try:
                    run_plots = self.generate_single_run_report(
                        data=run_data,
                        run_name=f"{experiment_name}_{run_name}",
                    )
                    all_plots.update(run_plots)
                except Exception as e:
                    logger.error(f"Failed to generate report for run {run_name}: {e}")
            
            # Generate comparison report
            try:
                comparison_plots = self.generate_comparison_report(
                    runs_data=runs_data,
                    comparison_name=experiment_name,
                )
                all_plots.update(comparison_plots)
            except Exception as e:
                logger.error(f"Failed to generate comparison report: {e}")
        
        else:
            # Single run experiment
            single_run_plots = self.generate_single_run_report(
                data=experiment_data,
                run_name=experiment_name,
            )
            all_plots.update(single_run_plots)
        
        # Generate summary report
        self._generate_summary_report(all_plots, experiment_name)
        
        logger.info(f"Generated complete experiment report with {len(all_plots)} plots")
        return all_plots
    
    def _generate_summary_report(
        self,
        generated_plots: Dict[str, Path],
        experiment_name: str,
    ) -> None:
        """Generate HTML summary report with all plots.
        
        Args:
            generated_plots: Dictionary of generated plot paths.
            experiment_name: Name of the experiment.
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name} - Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        .plot-section {{ margin: 30px 0; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .plot-item {{ text-align: center; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{experiment_name} - Visualization Report</h1>
    <p class="timestamp">Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="plot-section">
        <h2>Generated Visualizations</h2>
        <div class="plot-grid">
"""
            
            for plot_type, plot_path in generated_plots.items():
                if plot_path.exists():
                    # Use relative path for HTML
                    relative_path = plot_path.relative_to(self.output_dir)
                    html_content += f"""
            <div class="plot-item">
                <h3>{plot_type.replace('_', ' ').title()}</h3>
                <img src="{relative_path}" alt="{plot_type}">
            </div>
"""
            
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            html_path = self.output_dir / f"{experiment_name}_report.html"
            html_path.write_text(html_content)
            logger.info(f"Generated HTML summary report: {html_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML summary report: {e}")


def load_training_data(log_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load training data from log directory.
    
    Args:
        log_dir: Directory containing training logs.
        
    Returns:
        Dictionary containing loaded training data.
        
    Raises:
        FileNotFoundError: If log directory or required files don't exist.
        ValueError: If data format is invalid.
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")
    
    data = {}
    
    # Look for common log file formats
    csv_files = list(log_path.glob("*.csv"))
    json_files = list(log_path.glob("*.json"))
    
    if csv_files:
        # Load CSV data (e.g., from stable-baselines3 monitor)
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Common column mappings
                column_mapping = {
                    "r": "rewards",
                    "l": "episode_lengths", 
                    "t": "timesteps",
                    "reward": "rewards",
                    "episode_reward": "rewards",
                    "timestep": "timesteps",
                    "step": "timesteps",
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        data[new_col] = df[old_col].values
                
                # Add any loss columns
                loss_columns = [col for col in df.columns if "loss" in col.lower()]
                for col in loss_columns:
                    data[col] = df[col].values
                
                logger.info(f"Loaded data from {csv_file}: {list(df.columns)}")
                
            except Exception as e:
                logger.warning(f"Failed to load CSV file {csv_file}: {e}")
    
    if json_files:
        # Load JSON data
        import json
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    json_data = json.load(f)
                
                # Merge JSON data
                if isinstance(json_data, dict):
                    data.update(json_data)
                
                logger.info(f"Loaded data from {json_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load JSON file {json_file}: {e}")
    
    if not data:
        raise ValueError(f"No valid training data found in {log_path}")
    
    return data


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RL training visualizations")
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save generated plots")
    parser.add_argument("--experiment-name", type=str, default="experiment",
                       help="Name of the experiment")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"],
                       help="Output formats for plots")
    parser.add_argument("--config-file", type=str,
                       help="YAML configuration file for plot settings")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Load configuration if provided
        config = None
        if args.config_file:
            import yaml
            with open(args.config_file) as f:
                config_dict = yaml.safe_load(f)
            config = PlotConfig(**config_dict)
        
        # Load training data
        logger.info(f"Loading training data from: {args.log_dir}")
        data = load_training_data(args.log_dir)
        
        # Generate visualizations
        generator = VisualizationGenerator(
            output_dir=args.output_dir,
            config=config,
            formats=args.formats,
        )
        
        plots = generator.generate_experiment_report(
            experiment_data=data,
            experiment_name=args.experiment_name,
        )
        
        logger.info(f"Successfully generated {len(plots)} visualizations")
        logger.info(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        raise


if __name__ == "__main__":
    main()