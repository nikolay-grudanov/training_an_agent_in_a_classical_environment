"""Visualization utilities for RL training analysis."""

from .plots import (
    plot_learning_curve,
    plot_episode_lengths,
    plot_loss_curves,
    plot_reward_distribution,
    plot_convergence_analysis,
    plot_multiple_runs,
    plot_confidence_intervals,
    setup_matplotlib_style,
    create_figure_grid,
    save_plot,
    apply_smoothing,
    detect_convergence,
)

from .video_generator import (
    VideoConfig,
    VideoGenerationError,
    setup_recording_environment,
    add_metrics_overlay,
    record_agent_episode,
    record_multiple_episodes,
    create_training_montage,
    generate_comparison_video,
    compress_video,
)

from .performance_plots import (
    PlotStyle,
    DataLoader,
    PerformancePlotter,
    InteractivePlotter,
    export_plots_to_formats,
    create_performance_report,
    quick_reward_plot,
    quick_comparison_plot,
)

__all__ = [
    # Plotting functions
    "plot_learning_curve",
    "plot_episode_lengths", 
    "plot_loss_curves",
    "plot_reward_distribution",
    "plot_convergence_analysis",
    "plot_multiple_runs",
    "plot_confidence_intervals",
    "setup_matplotlib_style",
    "create_figure_grid",
    "save_plot",
    "apply_smoothing",
    "detect_convergence",
    
    # Video generation functions
    "VideoConfig",
    "VideoGenerationError",
    "setup_recording_environment",
    "add_metrics_overlay",
    "record_agent_episode",
    "record_multiple_episodes",
    "create_training_montage",
    "generate_comparison_video",
    "compress_video",
    
    # Performance plotting functions
    "PlotStyle",
    "DataLoader",
    "PerformancePlotter",
    "InteractivePlotter",
    "export_plots_to_formats",
    "create_performance_report",
    "quick_reward_plot",
    "quick_comparison_plot",
]