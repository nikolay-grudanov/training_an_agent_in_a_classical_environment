"""RL Experiments Completion module.

This module provides training pipelines, experiments, and utilities
for training RL agents to convergence with 200K timesteps.

Modules:
- baseline_training: A2C and PPO baseline training
- gamma_experiment: Controlled hyperparameter experiments
- metrics_collector: Metrics logging and parsing
- config: Configuration dataclasses
"""

from .baseline_training import BaselineExperiment
from .config import ExperimentConfig
from .metrics_collector import MetricsCollector

__all__ = ["BaselineExperiment", "ExperimentConfig", "MetricsCollector"]
