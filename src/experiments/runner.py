"""Комплексный оркестратор для выполнения контролируемых RL экспериментов.

Этот модуль предоставляет класс ExperimentRunner для управления полным жизненным
циклом экспериментов: от настройки до выполнения, мониторинга и сбора результатов.

Основные возможности:
- Оркестрация baseline и variant конфигураций
- Интеграция с существующей системой обучения
- Мониторинг прогресса в реальном времени
- Обработка ошибок и восстановление
- Управление ресурсами и чекпоинтами
- CLI интерфейс для запуска экспериментов
"""

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import psutil
import yaml
from tqdm import tqdm

from src.experiments.config import Configuration
from src.experiments.experiment import Experiment, ExperimentStatus
from src.training.trainer import Trainer, TrainerConfig, TrainingResult
from src.utils.checkpointing import CheckpointManager
from src.utils.rl_logging import get_experiment_logger
from src.utils.metrics import MetricsTracker
from src.utils.seeding import set_seed, SeedManager

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Режимы выполнения эксперимента."""
    
    SEQUENTIAL = "sequential"    # Последовательное выполнение
    PARALLEL = "parallel"       # Параллельное выполнение
    VALIDATION = "validation"   # Режим валидации (dry-run)


class RunnerStatus(Enum):
    """Статусы выполнения runner'а."""
    
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING_BASELINE = "running_baseline"
    RUNNING_VARIANT = "running_variant"
    COMPARING = "comparing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class ProgressInfo:
    """Информация о прогрессе выполнения."""
    
    current_step: int = 0
    total_steps: int = 0
    current_phase: str = "idle"
    baseline_progress: float = 0.0
    variant_progress: float = 0.0
    estimated_time_remaining: Optional[float] = None
    current_config: Optional[str] = None
    
    @property
    def overall_progress(self) -> float:
        """Общий прогресс выполнения."""
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100.0)


@dataclass
class ResourceUsage:
    """Информация об использовании ресурсов."""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_usage_mb: float = 0.0
    gpu_usage: Optional[float] = None
    
    @classmethod
    def current(cls) -> "ResourceUsage":
        """Получить текущее использование ресурсов."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return cls(
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_mb=memory_info.rss / 1024 / 1024,
            disk_usage_mb=0.0,  # Будет обновлено при необходимости
        )


class ExperimentRunner:
    """Комплексный оркестратор для выполнения контролируемых RL экспериментов.
    
    Управляет полным жизненным циклом эксперимента, включая:
    - Настройку и валидацию конфигураций
    - Выполнение baseline и variant обучения
    - Мониторинг прогресса и ресурсов
    - Обработку ошибок и восстановление
    - Сбор и анализ результатов
    """
    
    def __init__(
        self,
        experiment: Experiment,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: Optional[int] = None,
        enable_monitoring: bool = True,
        checkpoint_frequency: int = 10000,
        resource_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        """Инициализация ExperimentRunner.
        
        Args:
            experiment: Объект эксперимента для выполнения
            execution_mode: Режим выполнения (последовательный/параллельный)
            max_workers: Максимальное количество воркеров для параллельного выполнения
            enable_monitoring: Включить мониторинг ресурсов
            checkpoint_frequency: Частота создания чекпоинтов
            resource_limits: Ограничения ресурсов (memory_mb, cpu_percent)
            
        Raises:
            ValueError: При некорректных параметрах
        """
        self.experiment = experiment
        self.execution_mode = execution_mode
        self.enable_monitoring = enable_monitoring
        self.checkpoint_frequency = checkpoint_frequency
        
        # Валидация параметров
        if max_workers is not None and max_workers < 1:
            raise ValueError(f"max_workers должен быть >= 1, получен {max_workers}")
        
        self.max_workers = max_workers or min(2, mp.cpu_count())
        
        # Ограничения ресурсов
        self.resource_limits = resource_limits or {
            "memory_mb": 8192,  # 8GB по умолчанию
            "cpu_percent": 90.0,
        }
        
        # Состояние runner'а
        self.status = RunnerStatus.IDLE
        self.progress = ProgressInfo()
        self.resource_usage = ResourceUsage()
        
        # Результаты выполнения
        self.baseline_result: Optional[TrainingResult] = None
        self.variant_result: Optional[TrainingResult] = None
        self.execution_start_time: Optional[float] = None
        self.execution_end_time: Optional[float] = None
        
        # Компоненты системы
        self.logger = get_experiment_logger(
            experiment_id=experiment.experiment_id,
            base_logger=logger,
        )
        
        self.seed_manager = SeedManager(base_seed=42)
        
        # Менеджеры ресурсов
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=experiment.experiment_dir / "runner_checkpoints",
            experiment_id=experiment.experiment_id,
            max_checkpoints=5,
        )
        
        # Мониторинг
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._interrupt_requested = False
        
        # Настройка обработчиков сигналов
        self._setup_signal_handlers()
        
        self.logger.info(
            f"Инициализирован ExperimentRunner для эксперимента {experiment.experiment_id}",
            extra={
                "execution_mode": execution_mode.value,
                "max_workers": self.max_workers,
                "enable_monitoring": enable_monitoring,
            }
        )
    
    def run(self) -> bool:
        """Выполнить полный эксперимент.
        
        Returns:
            True если эксперимент выполнен успешно, False иначе
            
        Raises:
            RuntimeError: При критических ошибках выполнения
        """
        try:
            self.execution_start_time = time.time()
            self.status = RunnerStatus.INITIALIZING
            
            self.logger.info("Начало выполнения эксперимента")
            
            # Валидация и подготовка
            self._validate_experiment()
            self._setup_environment()
            
            # Запуск мониторинга
            if self.enable_monitoring:
                self._start_monitoring()
            
            # Запуск эксперимента
            self.experiment.start()
            
            # Выполнение конфигураций
            success = self._execute_configurations()
            
            if success:
                # Сравнение результатов
                self._compare_results()
                self.status = RunnerStatus.COMPLETED
                self.experiment.stop(failed=False)
            else:
                self.status = RunnerStatus.FAILED
                self.experiment.stop(failed=True, error_message="Ошибка выполнения конфигураций")
            
            return success
            
        except KeyboardInterrupt:
            self.logger.warning("Получен сигнал прерывания")
            self._handle_interruption()
            return False
            
        except Exception as e:
            error_msg = f"Критическая ошибка выполнения эксперимента: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.status = RunnerStatus.FAILED
            self.experiment.stop(failed=True, error_message=error_msg)
            return False
            
        finally:
            self.execution_end_time = time.time()
            self._cleanup()
    
    def run_configuration(
        self,
        config_type: str,
        config: Configuration,
        trainer_config: Optional[TrainerConfig] = None,
    ) -> Optional[TrainingResult]:
        """Выполнить обучение для одной конфигурации.
        
        Args:
            config_type: Тип конфигурации ('baseline' или 'variant')
            config: Конфигурация для выполнения
            trainer_config: Дополнительная конфигурация тренера
            
        Returns:
            Результат обучения или None при ошибке
        """
        if config_type not in ["baseline", "variant"]:
            raise ValueError(f"Неверный тип конфигурации: {config_type}")
        
        self.logger.info(f"Начало выполнения конфигурации {config_type}")
        
        try:
            # Установка seed для воспроизводимости
            seed = self.seed_manager.get_next_seed()
            set_seed(seed)
            
            # Создание конфигурации тренера
            if trainer_config is None:
                trainer_config = self._create_trainer_config(config, config_type)
            
            # Создание и настройка тренера
            trainer = Trainer(trainer_config)
            
            # Выполнение обучения с мониторингом
            with trainer:
                result = self._train_with_monitoring(trainer, config_type)
            
            # Добавление результатов в эксперимент
            if result and result.success:
                metrics = self._extract_metrics_from_result(result)
                self.experiment.add_result(
                    config_type=config_type,
                    results=result.to_dict(),
                    metrics=metrics,
                )
                
                self.logger.info(
                    f"Конфигурация {config_type} выполнена успешно",
                    extra={
                        "final_reward": result.final_mean_reward,
                        "training_time": result.training_time,
                    }
                )
            else:
                self.logger.error(f"Ошибка выполнения конфигурации {config_type}")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка выполнения конфигурации {config_type}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None
    
    def setup_environment(self) -> None:
        """Подготовить среду выполнения."""
        self.logger.info("Настройка среды выполнения")
        
        # Создание директорий
        self.experiment.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логирования
        log_dir = self.experiment.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Настройка чекпоинтов
        checkpoint_dir = self.experiment.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Проверка ресурсов
        self._check_resource_availability()
        
        self.logger.info("Среда выполнения настроена")
    
    def monitor_progress(self) -> ProgressInfo:
        """Получить текущую информацию о прогрессе.
        
        Returns:
            Актуальная информация о прогрессе
        """
        return self.progress
    
    def handle_failure(
        self,
        error: Exception,
        config_type: Optional[str] = None,
        recovery_strategy: str = "abort",
    ) -> bool:
        """Обработать ошибку выполнения.
        
        Args:
            error: Возникшая ошибка
            config_type: Тип конфигурации где произошла ошибка
            recovery_strategy: Стратегия восстановления ('abort', 'retry', 'skip')
            
        Returns:
            True если восстановление успешно, False иначе
        """
        error_msg = f"Ошибка в {config_type or 'общем выполнении'}: {error}"
        self.logger.error(error_msg, exc_info=True)
        
        if recovery_strategy == "abort":
            self.logger.info("Стратегия восстановления: прерывание эксперимента")
            return False
        
        elif recovery_strategy == "retry":
            self.logger.info("Стратегия восстановления: повторная попытка")
            # Реализация логики повторного выполнения
            return self._retry_configuration(config_type)
        
        elif recovery_strategy == "skip":
            self.logger.info("Стратегия восстановления: пропуск конфигурации")
            return True
        
        else:
            self.logger.warning(f"Неизвестная стратегия восстановления: {recovery_strategy}")
            return False
    
    def cleanup(self) -> None:
        """Очистить ресурсы после выполнения."""
        self.logger.info("Очистка ресурсов")
        
        # Остановка мониторинга
        if self._monitoring_active:
            self._stop_monitoring()
        
        # Сохранение финального состояния
        self._save_final_state()
        
        # Очистка временных файлов
        self._cleanup_temporary_files()
        
        self.logger.info("Очистка ресурсов завершена")
    
    def get_status(self) -> Dict[str, Any]:
        """Получить текущий статус runner'а.
        
        Returns:
            Словарь с информацией о статусе
        """
        execution_time = None
        if self.execution_start_time:
            end_time = self.execution_end_time or time.time()
            execution_time = end_time - self.execution_start_time
        
        return {
            "status": self.status.value,
            "execution_mode": self.execution_mode.value,
            "progress": {
                "overall": self.progress.overall_progress,
                "baseline": self.progress.baseline_progress,
                "variant": self.progress.variant_progress,
                "current_phase": self.progress.current_phase,
            },
            "resource_usage": {
                "cpu_percent": self.resource_usage.cpu_percent,
                "memory_mb": self.resource_usage.memory_mb,
                "memory_percent": self.resource_usage.memory_percent,
            },
            "execution_time": execution_time,
            "experiment_status": self.experiment.status.value,
            "results_available": {
                "baseline": self.baseline_result is not None,
                "variant": self.variant_result is not None,
            },
        }
    
    def _validate_experiment(self) -> None:
        """Валидировать эксперимент перед выполнением."""
        self.logger.info("Валидация эксперимента")
        
        # Проверка статуса эксперимента
        if self.experiment.status not in [ExperimentStatus.CREATED, ExperimentStatus.PAUSED]:
            raise RuntimeError(
                f"Эксперимент в неподходящем статусе для выполнения: {self.experiment.status}"
            )
        
        # Валидация конфигураций
        try:
            # Базовая валидация конфигураций уже выполнена в Experiment.__init__
            # Дополнительные проверки можно добавить здесь
            if not self.experiment.baseline_config.algorithm.name:
                raise ValueError("Алгоритм baseline не указан")
            if not self.experiment.variant_config.algorithm.name:
                raise ValueError("Алгоритм variant не указан")
        except Exception as e:
            raise RuntimeError(f"Ошибка валидации конфигураций: {e}")
        
        self.logger.info("Валидация эксперимента завершена успешно")
    
    def _setup_environment(self) -> None:
        """Настроить среду выполнения."""
        self.setup_environment()
    
    def _execute_configurations(self) -> bool:
        """Выполнить обучение для всех конфигураций.
        
        Returns:
            True если все конфигурации выполнены успешно
        """
        if self.execution_mode == ExecutionMode.VALIDATION:
            return self._validate_configurations()
        
        elif self.execution_mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential()
        
        elif self.execution_mode == ExecutionMode.PARALLEL:
            return self._execute_parallel()
        
        else:
            raise ValueError(f"Неподдерживаемый режим выполнения: {self.execution_mode}")
    
    def _execute_sequential(self) -> bool:
        """Последовательное выполнение конфигураций."""
        self.logger.info("Последовательное выполнение конфигураций")
        
        # Выполнение baseline
        self.status = RunnerStatus.RUNNING_BASELINE
        self.progress.current_phase = "baseline"
        
        self.baseline_result = self.run_configuration(
            config_type="baseline",
            config=self.experiment.baseline_config,
        )
        
        if not self.baseline_result or not self.baseline_result.success:
            self.logger.error("Ошибка выполнения baseline конфигурации")
            return False
        
        self.progress.baseline_progress = 100.0
        
        # Проверка прерывания
        if self._interrupt_requested:
            self.logger.warning("Получен запрос на прерывание")
            return False
        
        # Выполнение variant
        self.status = RunnerStatus.RUNNING_VARIANT
        self.progress.current_phase = "variant"
        
        self.variant_result = self.run_configuration(
            config_type="variant",
            config=self.experiment.variant_config,
        )
        
        if not self.variant_result or not self.variant_result.success:
            self.logger.error("Ошибка выполнения variant конфигурации")
            return False
        
        self.progress.variant_progress = 100.0
        
        return True
    
    def _execute_parallel(self) -> bool:
        """Параллельное выполнение конфигураций."""
        self.logger.info("Параллельное выполнение конфигураций")
        
        # Создание конфигураций тренеров
        baseline_trainer_config = self._create_trainer_config(
            self.experiment.baseline_config, "baseline"
        )
        variant_trainer_config = self._create_trainer_config(
            self.experiment.variant_config, "variant"
        )
        
        # Параллельное выполнение
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Запуск задач
            baseline_future = executor.submit(
                self._run_configuration_worker,
                "baseline",
                self.experiment.baseline_config,
                baseline_trainer_config,
            )
            
            variant_future = executor.submit(
                self._run_configuration_worker,
                "variant",
                self.experiment.variant_config,
                variant_trainer_config,
            )
            
            # Ожидание завершения
            futures = [baseline_future, variant_future]
            results = {}
            
            for future in as_completed(futures):
                try:
                    config_type, result = future.result()
                    results[config_type] = result
                    
                    if config_type == "baseline":
                        self.baseline_result = result
                        self.progress.baseline_progress = 100.0
                    else:
                        self.variant_result = result
                        self.progress.variant_progress = 100.0
                    
                    self.logger.info(f"Конфигурация {config_type} завершена")
                    
                except Exception as e:
                    self.logger.error(f"Ошибка в параллельном выполнении: {e}")
                    return False
        
        # Проверка результатов
        success = (
            self.baseline_result and self.baseline_result.success and
            self.variant_result and self.variant_result.success
        )
        
        return success
    
    def _validate_configurations(self) -> bool:
        """Валидация конфигураций без выполнения."""
        self.logger.info("Режим валидации: проверка конфигураций")
        
        try:
            # Создание тренеров для валидации
            baseline_trainer_config = self._create_trainer_config(
                self.experiment.baseline_config, "baseline"
            )
            variant_trainer_config = self._create_trainer_config(
                self.experiment.variant_config, "variant"
            )
            
            # Проверка создания тренеров
            baseline_trainer = Trainer(baseline_trainer_config)
            variant_trainer = Trainer(variant_trainer_config)
            
            # Проверка настройки без обучения
            baseline_trainer.setup()
            variant_trainer.setup()
            
            # Очистка
            baseline_trainer.cleanup()
            variant_trainer.cleanup()
            
            self.logger.info("Валидация конфигураций успешна")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации конфигураций: {e}")
            return False
    
    def _create_trainer_config(
        self,
        config: Configuration,
        config_type: str,
    ) -> TrainerConfig:
        """Создать конфигурацию тренера из конфигурации эксперимента.
        
        Args:
            config: Конфигурация эксперимента
            config_type: Тип конфигурации
            
        Returns:
            Конфигурация тренера
        """
        experiment_name = f"{self.experiment.experiment_id}_{config_type}"
        
        return TrainerConfig(
            experiment_name=experiment_name,
            algorithm=config.algorithm,
            environment_name=config.environment,
            total_timesteps=config.training_steps,
            seed=config.seed,
            eval_freq=config.evaluation_frequency,
            output_dir=str(self.experiment.experiment_dir),
            verbose=1,
            track_experiment=True,
        )
    
    def _train_with_monitoring(
        self,
        trainer: Trainer,
        config_type: str,
    ) -> Optional[TrainingResult]:
        """Выполнить обучение с мониторингом прогресса.
        
        Args:
            trainer: Тренер для выполнения обучения
            config_type: Тип конфигурации
            
        Returns:
            Результат обучения
        """
        self.progress.current_config = config_type
        
        # Создание прогресс-бара
        with tqdm(
            total=trainer.config.total_timesteps,
            desc=f"Обучение {config_type}",
            unit="steps",
        ) as pbar:
            
            # Callback для обновления прогресса
            class ProgressCallback:
                def __init__(self, pbar, runner):
                    self.pbar = pbar
                    self.runner = runner
                    self.last_update = 0
                
                def __call__(self, locals_, globals_):
                    current_step = locals_.get("self").num_timesteps
                    if current_step > self.last_update:
                        delta = current_step - self.last_update
                        self.pbar.update(delta)
                        self.last_update = current_step
                        
                        # Обновление прогресса runner'а
                        if config_type == "baseline":
                            self.runner.progress.baseline_progress = (
                                current_step / trainer.config.total_timesteps * 100
                            )
                        else:
                            self.runner.progress.variant_progress = (
                                current_step / trainer.config.total_timesteps * 100
                            )
            
            # Выполнение обучения
            try:
                result = trainer.train()
                return result
            except Exception as e:
                self.logger.error(f"Ошибка обучения {config_type}: {e}")
                return None
    
    def _compare_results(self) -> None:
        """Сравнить результаты baseline и variant."""
        self.status = RunnerStatus.COMPARING
        self.progress.current_phase = "comparing"
        
        self.logger.info("Сравнение результатов конфигураций")
        
        if not self.baseline_result or not self.variant_result:
            self.logger.error("Отсутствуют результаты для сравнения")
            return
        
        # Сравнение уже выполняется в experiment.add_result()
        # Здесь можем добавить дополнительную аналитику
        
        comparison = self.experiment.compare_results()
        
        self.logger.info(
            "Сравнение результатов завершено",
            extra={
                "baseline_reward": self.baseline_result.final_mean_reward,
                "variant_reward": self.variant_result.final_mean_reward,
                "improvement": comparison.get("performance_metrics", {}).get(
                    "mean_reward", {}
                ).get("improvement", 0),
            }
        )
    
    def _extract_metrics_from_result(self, result: TrainingResult) -> List[Dict[str, Any]]:
        """Извлечь метрики из результата обучения.
        
        Args:
            result: Результат обучения
            
        Returns:
            Список метрик по шагам
        """
        metrics = []
        
        # Извлечение истории обучения
        if result.training_history:
            timesteps = result.training_history.get("timesteps", [])
            rewards = result.training_history.get("mean_rewards", [])
            
            for i, (timestep, reward) in enumerate(zip(timesteps, rewards)):
                metrics.append({
                    "timestep": timestep,
                    "mean_reward": reward,
                    "episode": i,
                })
        
        return metrics
    
    def _start_monitoring(self) -> None:
        """Запустить мониторинг ресурсов."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self.logger.info("Запуск мониторинга ресурсов")
        
        # Запуск в отдельном потоке
        import threading
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self.resource_usage = ResourceUsage.current()
                    
                    # Проверка лимитов
                    self._check_resource_limits()
                    
                    time.sleep(5)  # Обновление каждые 5 секунд
                    
                except Exception as e:
                    self.logger.error(f"Ошибка мониторинга: {e}")
                    break
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def _stop_monitoring(self) -> None:
        """Остановить мониторинг ресурсов."""
        self._monitoring_active = False
        self.logger.info("Остановка мониторинга ресурсов")
    
    def _check_resource_limits(self) -> None:
        """Проверить ограничения ресурсов."""
        if self.resource_usage.memory_mb > self.resource_limits.get("memory_mb", float("inf")):
            self.logger.warning(
                f"Превышен лимит памяти: {self.resource_usage.memory_mb:.1f} MB"
            )
        
        if self.resource_usage.cpu_percent > self.resource_limits.get("cpu_percent", 100):
            self.logger.warning(
                f"Превышен лимит CPU: {self.resource_usage.cpu_percent:.1f}%"
            )
    
    def _check_resource_availability(self) -> None:
        """Проверить доступность ресурсов."""
        # Проверка свободной памяти
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        required_mb = self.resource_limits.get("memory_mb", 1024)
        
        if available_mb < required_mb:
            raise RuntimeError(
                f"Недостаточно свободной памяти: доступно {available_mb:.1f} MB, "
                f"требуется {required_mb:.1f} MB"
            )
        
        # Проверка свободного места на диске
        disk = psutil.disk_usage(self.experiment.experiment_dir)
        available_gb = disk.free / 1024 / 1024 / 1024
        
        if available_gb < 1.0:  # Минимум 1 GB
            raise RuntimeError(
                f"Недостаточно свободного места на диске: {available_gb:.1f} GB"
            )
    
    def _setup_signal_handlers(self) -> None:
        """Настроить обработчики сигналов."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Получен сигнал {signum}")
            self._interrupt_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _handle_interruption(self) -> None:
        """Обработать прерывание выполнения."""
        self.status = RunnerStatus.INTERRUPTED
        self.logger.info("Обработка прерывания выполнения")
        
        # Сохранение текущего состояния
        self._save_checkpoint()
        
        # Остановка эксперимента
        self.experiment.stop(failed=True, error_message="Прервано пользователем")
    
    def _save_checkpoint(self) -> None:
        """Сохранить чекпоинт текущего состояния."""
        checkpoint_data = {
            "experiment_id": self.experiment.experiment_id,
            "status": self.status.value,
            "progress": self.progress.__dict__,
            "baseline_completed": self.baseline_result is not None,
            "variant_completed": self.variant_result is not None,
            "execution_start_time": self.execution_start_time,
        }
        
        try:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                timestep=int(time.time()),
            )
            self.logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения чекпоинта: {e}")
    
    def _save_final_state(self) -> None:
        """Сохранить финальное состояние."""
        final_state = {
            "experiment_id": self.experiment.experiment_id,
            "status": self.status.value,
            "execution_time": self.execution_end_time - self.execution_start_time
            if self.execution_start_time and self.execution_end_time else None,
            "results": {
                "baseline": self.baseline_result.to_dict() if self.baseline_result else None,
                "variant": self.variant_result.to_dict() if self.variant_result else None,
            },
            "resource_usage": self.resource_usage.__dict__,
        }
        
        state_path = self.experiment.experiment_dir / "runner_final_state.yaml"
        
        try:
            with open(state_path, 'w', encoding='utf-8') as f:
                yaml.dump(final_state, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Финальное состояние сохранено: {state_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения финального состояния: {e}")
    
    def _cleanup_temporary_files(self) -> None:
        """Очистить временные файлы."""
        # Очистка временных файлов чекпоинтов
        temp_dir = self.experiment.experiment_dir / "temp"
        if temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(temp_dir)
                self.logger.info("Временные файлы очищены")
            except Exception as e:
                self.logger.warning(f"Ошибка очистки временных файлов: {e}")
    
    def _retry_configuration(self, config_type: Optional[str]) -> bool:
        """Повторить выполнение конфигурации.
        
        Args:
            config_type: Тип конфигурации для повтора
            
        Returns:
            True если повтор успешен
        """
        if not config_type:
            return False
        
        self.logger.info(f"Повторное выполнение конфигурации {config_type}")
        
        try:
            if config_type == "baseline":
                self.baseline_result = self.run_configuration(
                    config_type="baseline",
                    config=self.experiment.baseline_config,
                )
                return self.baseline_result is not None and self.baseline_result.success
            
            elif config_type == "variant":
                self.variant_result = self.run_configuration(
                    config_type="variant",
                    config=self.experiment.variant_config,
                )
                return self.variant_result is not None and self.variant_result.success
            
        except Exception as e:
            self.logger.error(f"Ошибка повторного выполнения {config_type}: {e}")
            return False
        
        return False
    
    def _run_configuration_worker(
        self,
        config_type: str,
        config: Configuration,
        trainer_config: TrainerConfig,
    ) -> Tuple[str, Optional[TrainingResult]]:
        """Воркер для параллельного выполнения конфигурации.
        
        Args:
            config_type: Тип конфигурации
            config: Конфигурация эксперимента
            trainer_config: Конфигурация тренера
            
        Returns:
            Кортеж (тип_конфигурации, результат)
        """
        # Установка seed в воркере
        set_seed(config.seed)
        
        # Создание тренера
        trainer = Trainer(trainer_config)
        
        try:
            with trainer:
                result = trainer.train()
            return config_type, result
        except Exception as e:
            logger.error(f"Ошибка в воркере {config_type}: {e}")
            return config_type, None
    
    def _cleanup(self) -> None:
        """Внутренняя очистка ресурсов."""
        # Остановка мониторинга
        if self._monitoring_active:
            self._stop_monitoring()
        
        # Вызов публичного метода очистки
        self.cleanup()


# CLI интерфейс

@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Путь к файлу конфигурации эксперимента",
)
@click.option(
    "--experiment-id",
    type=str,
    help="ID эксперимента для выполнения",
)
@click.option(
    "--mode",
    type=click.Choice(["sequential", "parallel", "validation"]),
    default="sequential",
    help="Режим выполнения эксперимента",
)
@click.option(
    "--max-workers",
    type=int,
    default=2,
    help="Максимальное количество воркеров для параллельного выполнения",
)
@click.option(
    "--no-monitoring",
    is_flag=True,
    help="Отключить мониторинг ресурсов",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/experiments",
    help="Директория для сохранения результатов",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Уровень детализации вывода",
)
def run_experiment_cli(
    config: Optional[str],
    experiment_id: Optional[str],
    mode: str,
    max_workers: int,
    no_monitoring: bool,
    output_dir: str,
    verbose: int,
) -> None:
    """Запустить эксперимент через командную строку."""
    # Настройка логирования
    log_level = logging.WARNING
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Загрузка эксперимента
        if config:
            # Загрузка из файла конфигурации
            experiment = Experiment.load(config)
        elif experiment_id:
            # Поиск эксперимента по ID
            experiment_path = Path(output_dir) / experiment_id / f"experiment_{experiment_id}.json"
            if not experiment_path.exists():
                click.echo(f"❌ Эксперимент {experiment_id} не найден в {experiment_path}")
                sys.exit(1)
            experiment = Experiment.load(experiment_path)
        else:
            click.echo("❌ Необходимо указать --config или --experiment-id")
            sys.exit(1)
        
        # Создание runner'а
        execution_mode = ExecutionMode(mode)
        runner = ExperimentRunner(
            experiment=experiment,
            execution_mode=execution_mode,
            max_workers=max_workers,
            enable_monitoring=not no_monitoring,
        )
        
        # Выполнение эксперимента
        click.echo(f"🚀 Запуск эксперимента {experiment.experiment_id}")
        click.echo(f"📋 Режим: {mode}")
        click.echo(f"🔧 Воркеров: {max_workers}")
        
        success = runner.run()
        
        if success:
            click.echo("✅ Эксперимент выполнен успешно!")
            
            # Вывод результатов
            status = runner.get_status()
            click.echo(f"📊 Время выполнения: {status['execution_time']:.1f} сек")
            
            if runner.baseline_result and runner.variant_result:
                click.echo(f"📈 Baseline награда: {runner.baseline_result.final_mean_reward:.2f}")
                click.echo(f"📈 Variant награда: {runner.variant_result.final_mean_reward:.2f}")
                
                improvement = runner.variant_result.final_mean_reward - runner.baseline_result.final_mean_reward
                click.echo(f"📊 Улучшение: {improvement:+.2f}")
        else:
            click.echo("❌ Эксперимент завершился с ошибкой")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Критическая ошибка: {e}")
        if verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_experiment_cli()