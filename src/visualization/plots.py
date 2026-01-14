"""Comprehensive plotting utilities for RL training visualization.

This module provides a complete suite of visualization tools for analyzing
reinforcement learning training progress, performance metrics, and comparisons.
Supports both Matplotlib and Plotly backends with publication-ready quality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure
from scipy import stats
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

logger = logging.getLogger(__name__)

# Type aliases
PlotData = Union[np.ndarray, pd.Series, List[float]]
PlotBackend = Union[str, None]
ColorPalette = Union[str, List[str]]

# Default configuration
DEFAULT_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8-whitegrid",
    "color_palette": "husl",
    "font_size": 12,
    "line_width": 2,
    "alpha": 0.7,
    "grid": True,
    "legend": True,
    "tight_layout": True,
}

# Color schemes
COLOR_SCHEMES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "husl": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],  # Same as default
    "colorblind": ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161"],
    "publication": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592941"],
    "dark": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"],
}


class PlotConfig:
    """Configuration class for plot settings."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize plot configuration.
        
        Args:
            **kwargs: Configuration parameters to override defaults.
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update(self, **kwargs: Any) -> None:
        """Update configuration."""
        self.config.update(kwargs)


def setup_matplotlib_style(
    style: str = "seaborn-v0_8-whitegrid",
    font_size: int = 12,
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 300,
) -> None:
    """Setup matplotlib style and defaults.
    
    Args:
        style: Matplotlib style name.
        font_size: Default font size.
        figure_size: Default figure size.
        dpi: Default DPI for high-quality output.
    """
    try:
        plt.style.use(style)
    except OSError:
        logger.warning(f"Style '{style}' not found, using default")
        plt.style.use("default")
    
    plt.rcParams.update({
        "font.size": font_size,
        "figure.figsize": figure_size,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def create_figure_grid(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Tuple[Figure, np.ndarray]:
    """Create a figure with subplot grid.
    
    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments for plt.subplots.
        
    Returns:
        Tuple of (figure, axes array).
    """
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Ensure axes is always an array
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(-1)
    
    return fig, axes


def apply_smoothing(
    data: PlotData,
    method: str = "moving_average",
    window: int = 10,
    **kwargs: Any,
) -> np.ndarray:
    """Apply smoothing to data.
    
    Args:
        data: Input data to smooth.
        method: Smoothing method ('moving_average', 'exponential', 'savgol', 'lowess').
        window: Window size for smoothing.
        **kwargs: Additional parameters for smoothing methods.
        
    Returns:
        Smoothed data array.
        
    Raises:
        ValueError: If smoothing method is not supported.
    """
    data_array = np.asarray(data)
    
    if len(data_array) < window:
        logger.warning(f"Data length ({len(data_array)}) < window ({window}), returning original")
        return data_array
    
    if method == "moving_average":
        series = pd.Series(data_array)
        smoothed_series = series.rolling(window=window, center=True).mean()
        # Fill NaN values
        filled_series = smoothed_series.bfill().ffill()  # type: ignore
        return np.asarray(filled_series.values)
    
    elif method == "exponential":
        alpha = kwargs.get("alpha", 0.1)
        series = pd.Series(data_array)
        smoothed_series = series.ewm(alpha=alpha).mean()
        return np.asarray(smoothed_series.values)
    
    elif method == "savgol":
        polyorder = min(kwargs.get("polyorder", 3), window - 1)
        if window % 2 == 0:
            window += 1  # Savgol requires odd window
        return savgol_filter(data_array, window, polyorder)
    
    elif method == "lowess":
        frac = kwargs.get("frac", 0.1)
        x = np.arange(len(data_array))
        smoothed = lowess(data_array, x, frac=frac)
        return smoothed[:, 1]
    
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")


def detect_convergence(
    data: PlotData,
    window: int = 100,
    threshold: float = 0.01,
    min_length: int = 200,
) -> Optional[int]:
    """Detect convergence point in training data.
    
    Args:
        data: Training metric data.
        window: Window size for convergence detection.
        threshold: Relative change threshold for convergence.
        min_length: Minimum data length before checking convergence.
        
    Returns:
        Index of convergence point or None if not converged.
    """
    data_array = np.asarray(data)
    
    if len(data_array) < min_length:
        return None
    
    # Calculate rolling statistics
    series = pd.Series(data_array)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Look for stable regions
    for i in range(window, len(data_array) - window):
        current_mean = rolling_mean.iloc[i]  # type: ignore
        future_mean = rolling_mean.iloc[i + window]  # type: ignore
        current_std = rolling_std.iloc[i]  # type: ignore
        
        # Check if mean is stable and variance is low
        if (
            abs(future_mean - current_mean) / abs(current_mean + 1e-8) < threshold
            and current_std < threshold * abs(current_mean)
        ):
            return i
    
    return None


def save_plot(
    fig: Union[Figure, PlotlyFigure],
    save_path: Union[str, Path],
    formats: List[str] = ["png"],
    dpi: int = 300,
    **kwargs: Any,
) -> None:
    """Save plot in multiple formats.
    
    Args:
        fig: Matplotlib or Plotly figure.
        save_path: Base path for saving (without extension).
        formats: List of formats to save ('png', 'svg', 'pdf', 'html').
        dpi: DPI for raster formats.
        **kwargs: Additional arguments for save functions.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = save_path.with_suffix(f".{fmt}")
        
        try:
            if isinstance(fig, Figure):  # Matplotlib
                if fmt in ["png", "jpg", "jpeg"]:
                    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", **kwargs)
                else:
                    fig.savefig(output_path, bbox_inches="tight", **kwargs)
            else:  # Plotly
                if fmt == "html":
                    fig.write_html(str(output_path), **kwargs)
                elif fmt == "png":
                    fig.write_image(str(output_path), **kwargs)
                elif fmt == "svg":
                    fig.write_image(str(output_path), format="svg", **kwargs)
                elif fmt == "pdf":
                    fig.write_image(str(output_path), format="pdf", **kwargs)
            
            logger.info(f"Saved plot to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save plot as {fmt}: {e}")


def plot_learning_curve(
    timesteps: PlotData,
    rewards: PlotData,
    title: str = "Learning Curve",
    xlabel: str = "Timesteps",
    ylabel: str = "Episode Reward",
    smooth: bool = True,
    smooth_method: str = "moving_average",
    smooth_window: int = 10,
    confidence_interval: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    backend: str = "matplotlib",
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Union[Figure, PlotlyFigure]:
    """Plot learning curve showing reward progression over time.
    
    Args:
        timesteps: X-axis data (timesteps or episodes).
        rewards: Y-axis data (rewards or other metrics).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        smooth: Whether to apply smoothing.
        smooth_method: Smoothing method to use.
        smooth_window: Window size for smoothing.
        confidence_interval: Whether to show confidence intervals.
        save_path: Path to save the plot.
        backend: Plotting backend ('matplotlib' or 'plotly').
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Figure object.
    """
    if config is None:
        config = PlotConfig()
    
    timesteps_array = np.asarray(timesteps)
    rewards_array = np.asarray(rewards)
    
    if len(timesteps_array) != len(rewards_array):
        raise ValueError("Timesteps and rewards must have the same length")
    
    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=config.get("figure_size"))
        
        if smooth and len(rewards_array) > smooth_window:
            smoothed_rewards = apply_smoothing(
                rewards_array, method=smooth_method, window=smooth_window
            )
            ax.plot(timesteps_array, smoothed_rewards, 
                   linewidth=config.get("line_width"), 
                   label="Smoothed", **kwargs)
            
            if confidence_interval:
                # Calculate confidence intervals
                window_half = smooth_window // 2
                ci_upper = []
                ci_lower = []
                
                for i in range(len(rewards_array)):
                    start_idx = max(0, i - window_half)
                    end_idx = min(len(rewards_array), i + window_half + 1)
                    window_data = rewards_array[start_idx:end_idx]
                    
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    ci_upper.append(mean_val + 1.96 * std_val / np.sqrt(len(window_data)))
                    ci_lower.append(mean_val - 1.96 * std_val / np.sqrt(len(window_data)))
                
                ax.fill_between(timesteps_array, ci_lower, ci_upper, 
                              alpha=0.3, label="95% CI")
        
        # Plot raw data
        ax.plot(timesteps_array, rewards_array, alpha=0.3, 
               color="gray", label="Raw data")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if config.get("legend"):
            ax.legend()
        
        if config.get("grid"):
            ax.grid(True, alpha=0.3)
        
        if config.get("tight_layout"):
            plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path, dpi=config.get("dpi"))
        
        return fig
    
    elif backend == "plotly":
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=timesteps_array,
            y=rewards_array,
            mode="lines",
            name="Raw data",
            opacity=0.3,
            line=dict(color="gray")
        ))
        
        if smooth and len(rewards_array) > smooth_window:
            smoothed_rewards = apply_smoothing(
                rewards_array, method=smooth_method, window=smooth_window
            )
            fig.add_trace(go.Scatter(
                x=timesteps_array,
                y=smoothed_rewards,
                mode="lines",
                name="Smoothed",
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=config.get("legend"),
            width=config.get("figure_size")[0] * 100,
            height=config.get("figure_size")[1] * 100,
        )
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def plot_episode_lengths(
    episodes: PlotData,
    lengths: PlotData,
    title: str = "Episode Length Progression",
    xlabel: str = "Episode",
    ylabel: str = "Episode Length",
    smooth: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot episode length progression over training.
    
    Args:
        episodes: Episode numbers.
        lengths: Episode lengths.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        smooth: Whether to apply smoothing.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    episodes_array = np.asarray(episodes)
    lengths_array = np.asarray(lengths)
    
    fig, ax = plt.subplots(figsize=config.get("figure_size"))
    
    # Plot raw data
    ax.plot(episodes_array, lengths_array, alpha=0.3, color="gray", label="Raw data")
    
    if smooth:
        smoothed_lengths = apply_smoothing(lengths_array, method="moving_average")
        ax.plot(episodes_array, smoothed_lengths, 
               linewidth=config.get("line_width"), label="Smoothed")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if config.get("legend"):
        ax.legend()
    
    if config.get("grid"):
        ax.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


def plot_loss_curves(
    timesteps: PlotData,
    losses: Dict[str, PlotData],
    title: str = "Training Loss Curves",
    xlabel: str = "Timesteps",
    ylabel: str = "Loss",
    smooth: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot multiple loss curves on the same plot.
    
    Args:
        timesteps: X-axis data.
        losses: Dictionary of loss name -> loss values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        smooth: Whether to apply smoothing.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    timesteps_array = np.asarray(timesteps)
    colors = COLOR_SCHEMES[config.get("color_palette", "default")]
    
    fig, ax = plt.subplots(figsize=config.get("figure_size"))
    
    for i, (loss_name, loss_values) in enumerate(losses.items()):
        loss_array = np.asarray(loss_values)
        color = colors[i % len(colors)]
        
        if smooth:
            smoothed_loss = apply_smoothing(loss_array, method="moving_average")
            ax.plot(timesteps_array, smoothed_loss, 
                   label=loss_name, color=color, 
                   linewidth=config.get("line_width"))
        else:
            ax.plot(timesteps_array, loss_array, 
                   label=loss_name, color=color, 
                   linewidth=config.get("line_width"))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")  # Log scale often better for losses
    
    if config.get("legend"):
        ax.legend()
    
    if config.get("grid"):
        ax.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


def plot_reward_distribution(
    rewards: PlotData,
    title: str = "Reward Distribution",
    xlabel: str = "Reward",
    ylabel: str = "Density",
    bins: int = 50,
    show_stats: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot reward distribution histogram with statistics.
    
    Args:
        rewards: Reward values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        bins: Number of histogram bins.
        show_stats: Whether to show statistics on plot.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    rewards_array = np.asarray(rewards)
    
    fig, ax = plt.subplots(figsize=config.get("figure_size"))
    
    # Histogram
    n, bins_edges, patches = ax.hist(
        rewards_array, bins=bins, density=True, 
        alpha=config.get("alpha"), **kwargs
    )
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(rewards_array)
    x = np.linspace(rewards_array.min(), rewards_array.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
           linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
    
    if show_stats:
        # Add statistics text
        stats_text = (
            f"Mean: {np.mean(rewards_array):.2f}\n"
            f"Std: {np.std(rewards_array):.2f}\n"
            f"Min: {np.min(rewards_array):.2f}\n"
            f"Max: {np.max(rewards_array):.2f}\n"
            f"Median: {np.median(rewards_array):.2f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if config.get("legend"):
        ax.legend()
    
    if config.get("grid"):
        ax.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


def plot_convergence_analysis(
    timesteps: PlotData,
    rewards: PlotData,
    title: str = "Convergence Analysis",
    window: int = 100,
    threshold: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot convergence analysis with convergence point detection.
    
    Args:
        timesteps: X-axis data.
        rewards: Y-axis data.
        title: Plot title.
        window: Window size for convergence detection.
        threshold: Convergence threshold.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    timesteps_array = np.asarray(timesteps)
    rewards_array = np.asarray(rewards)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config.get("figure_size")[0], 
                                                  config.get("figure_size")[1] * 1.5))
    
    # Top plot: Learning curve with convergence point
    smoothed_rewards = apply_smoothing(rewards_array, method="moving_average", window=window)
    ax1.plot(timesteps_array, rewards_array, alpha=0.3, color="gray", label="Raw data")
    ax1.plot(timesteps_array, smoothed_rewards, linewidth=2, label="Smoothed")
    
    # Detect and mark convergence point
    convergence_idx = detect_convergence(rewards_array, window=window, threshold=threshold)
    if convergence_idx is not None:
        ax1.axvline(timesteps_array[convergence_idx], color="red", linestyle="--", 
                   label=f"Convergence (step {timesteps_array[convergence_idx]})")
    
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"{title} - Learning Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Rolling statistics
    rolling_mean = pd.Series(rewards_array).rolling(window=window).mean()
    rolling_std = pd.Series(rewards_array).rolling(window=window).std()
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(timesteps_array, rolling_mean, color="blue", 
                    linewidth=2, label="Rolling Mean")
    line2 = ax2_twin.plot(timesteps_array, rolling_std, color="orange", 
                         linewidth=2, label="Rolling Std")
    
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Rolling Mean", color="blue")
    ax2_twin.set_ylabel("Rolling Std", color="orange")
    ax2.set_title("Rolling Statistics")
    
    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left")
    
    ax2.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


def plot_multiple_runs(
    runs_data: Dict[str, Dict[str, PlotData]],
    metric: str = "reward",
    title: str = "Multiple Runs Comparison",
    xlabel: str = "Timesteps",
    ylabel: str = "Reward",
    smooth: bool = True,
    confidence_interval: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot comparison of multiple training runs.
    
    Args:
        runs_data: Dictionary of run_name -> {timesteps, metric_values}.
        metric: Metric name to plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        smooth: Whether to apply smoothing.
        confidence_interval: Whether to show confidence intervals.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    colors = COLOR_SCHEMES[config.get("color_palette", "default")]
    
    fig, ax = plt.subplots(figsize=config.get("figure_size"))
    
    for i, (run_name, data) in enumerate(runs_data.items()):
        color = colors[i % len(colors)]
        
        timesteps = np.asarray(data["timesteps"])
        values = np.asarray(data[metric])
        
        if smooth:
            smoothed_values = apply_smoothing(values, method="moving_average")
            ax.plot(timesteps, smoothed_values, color=color, 
                   linewidth=config.get("line_width"), label=run_name)
        else:
            ax.plot(timesteps, values, color=color, 
                   linewidth=config.get("line_width"), label=run_name)
        
        if confidence_interval and smooth:
            # Calculate confidence intervals for smoothed data
            window = 50
            ci_upper = []
            ci_lower = []
            
            for j in range(len(values)):
                start_idx = max(0, j - window // 2)
                end_idx = min(len(values), j + window // 2 + 1)
                window_data = values[start_idx:end_idx]
                
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                ci_upper.append(mean_val + 1.96 * std_val / np.sqrt(len(window_data)))
                ci_lower.append(mean_val - 1.96 * std_val / np.sqrt(len(window_data)))
            
            ax.fill_between(timesteps, ci_lower, ci_upper, 
                          color=color, alpha=0.2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if config.get("legend"):
        ax.legend()
    
    if config.get("grid"):
        ax.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


def plot_confidence_intervals(
    x_data: PlotData,
    y_data_list: List[PlotData],
    labels: Optional[List[str]] = None,
    title: str = "Confidence Intervals",
    xlabel: str = "X",
    ylabel: str = "Y",
    confidence_level: float = 0.95,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs: Any,
) -> Figure:
    """Plot data with confidence intervals from multiple runs.
    
    Args:
        x_data: X-axis data.
        y_data_list: List of Y-axis data arrays (multiple runs).
        labels: Labels for each data series.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        confidence_level: Confidence level for intervals.
        save_path: Path to save the plot.
        config: Plot configuration.
        **kwargs: Additional plotting arguments.
        
    Returns:
        Matplotlib figure.
    """
    if config is None:
        config = PlotConfig()
    
    x_array = np.asarray(x_data)
    
    # Convert to numpy array and calculate statistics
    y_arrays = [np.asarray(y) for y in y_data_list]
    y_stacked = np.stack(y_arrays, axis=0)
    
    mean_y = np.mean(y_stacked, axis=0)
    std_y = np.std(y_stacked, axis=0)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    margin_error = z_score * std_y / np.sqrt(len(y_arrays))
    
    ci_lower = mean_y - margin_error
    ci_upper = mean_y + margin_error
    
    fig, ax = plt.subplots(figsize=config.get("figure_size"))
    
    # Plot mean line
    ax.plot(x_array, mean_y, linewidth=config.get("line_width"), 
           label=f"Mean (n={len(y_arrays)})")
    
    # Plot confidence interval
    ax.fill_between(x_array, ci_lower, ci_upper, alpha=0.3, 
                   label=f"{confidence_level*100:.0f}% CI")
    
    # Plot individual runs (optional)
    if len(y_arrays) <= 10:  # Only show individual runs if not too many
        for i, y_array in enumerate(y_arrays):
            label = labels[i] if labels and i < len(labels) else f"Run {i+1}"
            ax.plot(x_array, y_array, alpha=0.3, linewidth=1, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if config.get("legend"):
        ax.legend()
    
    if config.get("grid"):
        ax.grid(True, alpha=0.3)
    
    if config.get("tight_layout"):
        plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path, dpi=config.get("dpi"))
    
    return fig


# Initialize matplotlib style on import
setup_matplotlib_style()