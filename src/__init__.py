"""RL Agent Training System - Система обучения агентов подкрепляющего обучения.

Этот пакет предоставляет полную систему для обучения RL агентов с использованием
готовых алгоритмов в классических средах, проведения контролируемых экспериментов
и обеспечения воспроизводимости результатов.
"""

__version__ = "1.0.0"
__author__ = "RL Training Team"
__email__ = "team@rl-training.com"

# Основные компоненты системы
from .utils.seeding import set_seed, SeedManager
from .utils.config import RLConfig, load_config
from .utils.rl_logging import setup_logging, get_experiment_logger
from .utils.metrics import MetricsTracker, get_metrics_tracker
from .utils.checkpointing import CheckpointManager, create_checkpoint_metadata
from .experiments.base import ExperimentManager, create_experiment

__all__ = [
    # Версия и метаданные
    "__version__",
    "__author__",
    "__email__",
    # Воспроизводимость
    "set_seed",
    "SeedManager",
    # Конфигурация
    "RLConfig",
    "load_config",
    # Логирование
    "setup_logging",
    "get_experiment_logger",
    # Метрики
    "MetricsTracker",
    "get_metrics_tracker",
    # Чекпоинты
    "CheckpointManager",
    "create_checkpoint_metadata",
    # Эксперименты
    "ExperimentManager",
    "create_experiment",
]
