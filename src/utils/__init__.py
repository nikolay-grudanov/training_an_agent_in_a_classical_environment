"""Утилиты для системы обучения RL агентов.

Этот пакет содержит вспомогательные модули для обеспечения воспроизводимости,
логирования, управления конфигурацией, отслеживания метрик и создания чекпоинтов.
"""

from .seeding import set_seed, SeedManager, verify_reproducibility
from .config import RLConfig, ConfigLoader, load_config, get_config_loader
from .rl_logging import (
    setup_logging,
    get_experiment_logger,
    TrainingCallback,
    MetricsLogger,
    configure_default_logging,
)
from .metrics import (
    MetricsTracker,
    MetricPoint,
    MetricSummary,
    get_metrics_tracker,
    reset_metrics_tracker,
)
from .checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
    create_checkpoint_metadata,
    restore_training_state,
)
from .reproducibility_checker import (
    ReproducibilityChecker,
    StrictnessLevel,
    ReproducibilityIssueType,
    ReproducibilityIssue,
    ExperimentRun,
    ReproducibilityReport,
    create_simple_reproducibility_test,
    quick_reproducibility_check,
    validate_experiment_reproducibility,
)
from .dependency_tracker import (
    DependencyTracker,
    create_experiment_snapshot,
    validate_environment_for_experiment,
)

__all__ = [
    # Воспроизводимость
    "set_seed",
    "SeedManager",
    "verify_reproducibility",
    # Конфигурация
    "RLConfig",
    "ConfigLoader",
    "load_config",
    "get_config_loader",
    # Логирование
    "setup_logging",
    "get_experiment_logger",
    "TrainingCallback",
    "MetricsLogger",
    "configure_default_logging",
    # Метрики
    "MetricsTracker",
    "MetricPoint",
    "MetricSummary",
    "get_metrics_tracker",
    "reset_metrics_tracker",
    # Чекпоинты
    "CheckpointManager",
    "CheckpointMetadata",
    "create_checkpoint_metadata",
    "restore_training_state",
    # Проверка воспроизводимости
    "ReproducibilityChecker",
    "StrictnessLevel",
    "ReproducibilityIssueType",
    "ReproducibilityIssue",
    "ExperimentRun",
    "ReproducibilityReport",
    "create_simple_reproducibility_test",
    "quick_reproducibility_check",
    "validate_experiment_reproducibility",
    # Отслеживание зависимостей
    "DependencyTracker",
    "create_experiment_snapshot",
    "validate_environment_for_experiment",
]
