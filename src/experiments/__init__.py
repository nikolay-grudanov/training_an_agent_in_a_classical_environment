"""Модуль для управления экспериментами RL.

Этот пакет содержит базовые классы и утилиты для создания, выполнения
и управления экспериментами по обучению RL агентов.
"""

from .base import (
    ExperimentManager,
    ExperimentResult,
    SimpleExperiment,
    create_experiment,
)

__all__ = [
    "ExperimentManager",
    "ExperimentResult",
    "SimpleExperiment",
    "create_experiment",
]
