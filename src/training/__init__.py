"""Модуль обучения RL агентов.

Этот модуль предоставляет высокоуровневые инструменты для обучения
агентов обучения с подкреплением, включая:

- Trainer: Основной оркестратор обучения
- TrainerConfig: Конфигурация для обучения
- TrainingMode: Режимы обучения (train, resume, evaluate)
- TrainingResult: Результаты обучения
- TrainingLoop: Детальный тренировочный цикл
- TrainingProgress: Прогресс обучения
- TrainingStatistics: Статистика обучения
"""

from .trainer import (
    Trainer,
    TrainerConfig,
    TrainingMode,
    TrainingResult,
    create_trainer_from_config,
)

from .train_loop import (
    TrainingLoop,
    TrainingProgress,
    TrainingStatistics,
    TrainingStrategy,
    TrainingState,
    ProgressReporter,
    LoggingHook,
    EarlyStoppingHook,
    create_training_loop,
)

__all__ = [
    # Trainer components
    "Trainer",
    "TrainerConfig", 
    "TrainingMode",
    "TrainingResult",
    "create_trainer_from_config",
    
    # Training loop components
    "TrainingLoop",
    "TrainingProgress",
    "TrainingStatistics",
    "TrainingStrategy",
    "TrainingState",
    "ProgressReporter",
    "LoggingHook",
    "EarlyStoppingHook",
    "create_training_loop",
]