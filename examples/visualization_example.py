"""Example usage of visualization utilities for RL training analysis.

This script demonstrates how to use the visualization module to create
comprehensive plots and reports from training data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.visualization.plots import (
    plot_learning_curve,
    plot_multiple_runs,
    plot_convergence_analysis,
    PlotConfig,
)
from src.visualization.generate_all import VisualizationGenerator


def generate_sample_data() -> dict:
    """Generate realistic sample training data for demonstration."""
    np.random.seed(42)
    
    # Simulate PPO training on LunarLander
    timesteps = np.arange(0, 100000, 100)
    
    # Simulate learning progress: starts at -200, improves to +200
    base_reward = -200
    max_improvement = 400
    learning_rate = 0.00002
    
    # Exponential improvement with noise
    progress = 1 - np.exp(-learning_rate * timesteps)
    rewards = base_reward + max_improvement * progress + 30 * np.random.randn(len(timesteps))
    
    # Episode lengths: start high, decrease as agent learns
    base_length = 1000
    min_length = 200
    episode_lengths = base_length - (base_length - min_length) * progress + 50 * np.random.randn(len(timesteps))
    episode_lengths = np.clip(episode_lengths, min_length, base_length)
    
    # Simulate loss curves
    policy_loss = 0.1 * np.exp(-timesteps / 20000) + 0.01 * np.random.exponential(1, len(timesteps))
    value_loss = 0.5 * np.exp(-timesteps / 15000) + 0.05 * np.random.exponential(1, len(timesteps))
    entropy_loss = 0.05 * np.exp(-timesteps / 25000) + 0.005 * np.random.exponential(1, len(timesteps))
    
    return {
        "timesteps": timesteps,
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
    }


def example_single_plot():
    """Example: Create a single learning curve plot."""
    print("📊 Example 1: Single Learning Curve")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Create learning curve with custom styling
    config = PlotConfig(
        figure_size=(10, 6),
        color_palette="publication",
        line_width=2.5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fig = plot_learning_curve(
            timesteps=data["timesteps"],
            rewards=data["rewards"],
            title="PPO Training on LunarLander-v2",
            xlabel="Training Steps",
            ylabel="Episode Reward",
            smooth=True,
            confidence_interval=True,
            save_path=Path(tmpdir) / "learning_curve",
            config=config,
        )
        
        print(f"✅ Learning curve saved to: {tmpdir}/learning_curve.png")


def example_multiple_algorithms():
    """Example: Compare multiple algorithms."""
    print("\n📊 Example 2: Algorithm Comparison")
    
    # Generate data for different algorithms
    algorithms = ["PPO", "A2C", "SAC"]
    runs_data = {}
    
    for i, algorithm in enumerate(algorithms):
        # Slightly different performance for each algorithm
        base_data = generate_sample_data()
        
        # Modify performance characteristics
        if algorithm == "PPO":
            # PPO: Good final performance, stable learning
            performance_modifier = 1.0
            noise_modifier = 1.0
        elif algorithm == "A2C":
            # A2C: Slightly worse performance, more noise
            performance_modifier = 0.8
            noise_modifier = 1.5
        else:  # SAC
            # SAC: Better sample efficiency, less noise
            performance_modifier = 1.2
            noise_modifier = 0.7
        
        modified_rewards = base_data["rewards"] * performance_modifier
        modified_rewards += noise_modifier * 20 * np.random.randn(len(modified_rewards))
        
        runs_data[algorithm] = {
            "timesteps": base_data["timesteps"],
            "reward": modified_rewards,
        }
    
    # Create comparison plot
    with tempfile.TemporaryDirectory() as tmpdir:
        fig = plot_multiple_runs(
            runs_data=runs_data,
            metric="reward",
            title="Algorithm Comparison on LunarLander-v2",
            xlabel="Training Steps",
            ylabel="Episode Reward",
            save_path=Path(tmpdir) / "algorithm_comparison",
        )
        
        print(f"✅ Algorithm comparison saved to: {tmpdir}/algorithm_comparison.png")


def example_convergence_analysis():
    """Example: Analyze training convergence."""
    print("\n📊 Example 3: Convergence Analysis")
    
    data = generate_sample_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fig = plot_convergence_analysis(
            timesteps=data["timesteps"],
            rewards=data["rewards"],
            title="PPO Convergence Analysis",
            window=100,
            threshold=0.02,
            save_path=Path(tmpdir) / "convergence_analysis",
        )
        
        print(f"✅ Convergence analysis saved to: {tmpdir}/convergence_analysis.png")


def example_comprehensive_report():
    """Example: Generate comprehensive visualization report."""
    print("\n📊 Example 4: Comprehensive Report Generation")
    
    # Generate data for multiple runs of the same algorithm
    runs_data = {}
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        data = generate_sample_data()
        runs_data[f"ppo_seed_{seed}"] = data
    
    # Create experiment data structure
    experiment_data = {"runs": runs_data}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize visualization generator
        generator = VisualizationGenerator(
            output_dir=tmpdir,
            formats=["png", "svg"],
        )
        
        # Generate complete report
        plots = generator.generate_experiment_report(
            experiment_data=experiment_data,
            experiment_name="ppo_lunarlander_experiment",
        )
        
        print(f"✅ Generated {len(plots)} plots:")
        for plot_type, plot_path in plots.items():
            print(f"   - {plot_type}: {plot_path}")
        
        print(f"✅ HTML report available at: {tmpdir}/ppo_lunarlander_experiment_report.html")


def example_custom_styling():
    """Example: Custom plot styling and themes."""
    print("\n📊 Example 5: Custom Styling")
    
    data = generate_sample_data()
    
    # Publication-ready theme
    pub_config = PlotConfig(
        figure_size=(8, 5),
        dpi=600,
        font_size=14,
        line_width=2.5,
        color_palette="publication",
        style="seaborn-v0_8-whitegrid",
    )
    
    # Dark theme
    dark_config = PlotConfig(
        figure_size=(12, 8),
        font_size=16,
        line_width=3,
        color_palette="dark",
        style="dark_background",
        grid=False,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Publication style
        fig1 = plot_learning_curve(
            timesteps=data["timesteps"],
            rewards=data["rewards"],
            title="Publication Style",
            config=pub_config,
            save_path=Path(tmpdir) / "publication_style",
        )
        
        # Dark style
        fig2 = plot_learning_curve(
            timesteps=data["timesteps"],
            rewards=data["rewards"],
            title="Dark Theme",
            config=dark_config,
            save_path=Path(tmpdir) / "dark_theme",
        )
        
        print(f"✅ Custom styled plots saved to: {tmpdir}/")


def example_real_world_workflow():
    """Example: Real-world workflow with data loading."""
    print("\n📊 Example 6: Real-world Workflow")
    
    # Simulate loading data from training logs
    print("1. Loading training data...")
    
    # In real usage, you would load from actual log files:
    # from src.visualization.generate_all import load_training_data
    # data = load_training_data("results/logs/ppo_experiment/")
    
    # For demo, use generated data
    data = generate_sample_data()
    
    print("2. Preprocessing data...")
    # Add some preprocessing steps
    data["rewards_smoothed"] = pd.Series(data["rewards"]).rolling(window=50).mean().values
    
    print("3. Generating visualizations...")
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = VisualizationGenerator(
            output_dir=tmpdir,
            config=PlotConfig(color_palette="colorblind"),
        )
        
        # Generate single run report
        plots = generator.generate_single_run_report(
            data=data,
            run_name="ppo_lunarlander_final",
            include_losses=True,
        )
        
        print("4. Results:")
        print(f"   - Generated {len(plots)} visualizations")
        print(f"   - Output directory: {tmpdir}")
        print("   - Available plots:")
        for plot_type in plots.keys():
            print(f"     * {plot_type.replace('_', ' ').title()}")


def main():
    """Run all visualization examples."""
    print("🎨 RL Training Visualization Examples")
    print("=" * 50)
    
    try:
        example_single_plot()
        example_multiple_algorithms()
        example_convergence_analysis()
        example_comprehensive_report()
        example_custom_styling()
        example_real_world_workflow()
        
        print("\n✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("   - Use PlotConfig for consistent styling across plots")
        print("   - VisualizationGenerator automates report creation")
        print("   - Support for both matplotlib and plotly backends")
        print("   - Automatic smoothing and convergence detection")
        print("   - Multiple output formats (PNG, SVG, PDF, HTML)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()