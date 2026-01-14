"""Комплексный класс для проведения контролируемых экспериментов в области RL.

Этот модуль предоставляет класс Experiment для управления жизненным циклом
экспериментов, сравнения конфигураций, сбора результатов и анализа производительности.
"""

import json
import pickle
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config import RLConfig
from src.utils.logging import get_experiment_logger


class ExperimentStatus(Enum):
    """Статусы эксперимента."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ExperimentError(Exception):
    """Базовое исключение для ошибок эксперимента."""

    pass


class InvalidStateTransitionError(ExperimentError):
    """Исключение для недопустимых переходов состояния."""

    pass


class ConfigurationError(ExperimentError):
    """Исключение для ошибок конфигурации."""

    pass


class Experiment:
    """Класс для проведения контролируемых RL экспериментов.

    Управляет жизненным циклом эксперимента, включая сравнение конфигураций,
    сбор результатов, анализ производительности и сериализацию данных.
    """

    def __init__(
        self,
        baseline_config: RLConfig,
        variant_config: RLConfig,
        hypothesis: str,
        experiment_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Инициализация эксперимента.

        Args:
            baseline_config: Базовая конфигурация для сравнения
            variant_config: Вариантная конфигурация для тестирования
            hypothesis: Гипотеза о предполагаемом результате
            experiment_id: Уникальный идентификатор (генерируется автоматически если None)
            output_dir: Директория для сохранения результатов

        Raises:
            ConfigurationError: Если конфигурации несовместимы
        """
        # Генерируем уникальный ID если не предоставлен
        self.experiment_id = experiment_id or str(uuid.uuid4())

        # Валидируем конфигурации
        self._validate_configurations(baseline_config, variant_config)

        # Основные поля
        self.baseline_config = baseline_config
        self.variant_config = variant_config
        self.hypothesis = hypothesis.strip()

        if not self.hypothesis:
            raise ConfigurationError("Гипотеза не может быть пустой")

        # Результаты и метрики
        self.results: Dict[str, Any] = {"baseline": {}, "variant": {}, "comparison": {}}

        # Статус и временные метки
        self.status = ExperimentStatus.CREATED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.paused_at: Optional[datetime] = None

        # Настройка директорий и логирования
        self.output_dir = (
            Path(output_dir) if output_dir else Path("results/experiments")
        )
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логгера
        self.logger = get_experiment_logger(self.experiment_id, base_logger=None)

        # Внутренние состояния
        self._baseline_completed = False
        self._variant_completed = False
        self._pause_requested = False

        self.logger.info(
            f"Создан эксперимент {self.experiment_id}",
            extra={
                "hypothesis": self.hypothesis,
                "baseline_algorithm": self.baseline_config.algorithm.name,
                "variant_algorithm": self.variant_config.algorithm.name,
                "output_dir": str(self.experiment_dir),
            },
        )

    def start(self) -> None:
        """Начать выполнение эксперимента.

        Raises:
            InvalidStateTransitionError: Если эксперимент уже запущен или завершен
        """
        if self.status not in [ExperimentStatus.CREATED, ExperimentStatus.PAUSED]:
            raise InvalidStateTransitionError(
                f"Нельзя запустить эксперимент в статусе {self.status.value}"
            )

        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()
        self._pause_requested = False

        self.logger.info(
            "Эксперимент запущен", extra={"started_at": self.started_at.isoformat()}
        )

    def pause(self) -> None:
        """Приостановить выполнение эксперимента.

        Raises:
            InvalidStateTransitionError: Если эксперимент не запущен
        """
        if self.status != ExperimentStatus.RUNNING:
            raise InvalidStateTransitionError(
                f"Нельзя приостановить эксперимент в статусе {self.status.value}"
            )

        self.status = ExperimentStatus.PAUSED
        self.paused_at = datetime.now()
        self._pause_requested = True

        self.logger.info(
            "Эксперимент приостановлен", extra={"paused_at": self.paused_at.isoformat()}
        )

    def resume(self) -> None:
        """Возобновить выполнение приостановленного эксперимента.

        Raises:
            InvalidStateTransitionError: Если эксперимент не приостановлен
        """
        if self.status != ExperimentStatus.PAUSED:
            raise InvalidStateTransitionError(
                f"Нельзя возобновить эксперимент в статусе {self.status.value}"
            )

        self.status = ExperimentStatus.RUNNING
        self._pause_requested = False

        self.logger.info("Эксперимент возобновлен")

    def stop(self, failed: bool = False, error_message: Optional[str] = None) -> None:
        """Остановить эксперимент и финализировать результаты.

        Args:
            failed: Указывает, завершился ли эксперимент с ошибкой
            error_message: Сообщение об ошибке (если failed=True)

        Raises:
            InvalidStateTransitionError: Если эксперимент уже завершен
        """
        if self.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
            raise InvalidStateTransitionError(
                f"Эксперимент уже завершен в статусе {self.status.value}"
            )

        self.completed_at = datetime.now()

        if failed:
            self.status = ExperimentStatus.FAILED
            self.results["error"] = error_message or "Неизвестная ошибка"
            self.logger.error(
                f"Эксперимент завершился с ошибкой: {error_message}",
                extra={"completed_at": self.completed_at.isoformat()},
            )
        else:
            self.status = ExperimentStatus.COMPLETED
            self._finalize_results()
            self.logger.info(
                "Эксперимент успешно завершен",
                extra={"completed_at": self.completed_at.isoformat()},
            )

        # Автоматическое сохранение при завершении
        try:
            self.save()
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении эксперимента: {e}")

    def add_result(
        self,
        config_type: str,
        results: Dict[str, Any],
        metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Добавить результаты обучения для конфигурации.

        Args:
            config_type: Тип конфигурации ('baseline' или 'variant')
            results: Словарь с результатами обучения
            metrics: Список метрик по шагам обучения

        Raises:
            ValueError: Если config_type неверный
        """
        if config_type not in ["baseline", "variant"]:
            raise ValueError(f"Неверный тип конфигурации: {config_type}")

        # Добавляем временную метку
        results["timestamp"] = datetime.now().isoformat()
        results["config_type"] = config_type

        # Сохраняем результаты
        self.results[config_type] = results

        # Сохраняем метрики если предоставлены
        if metrics:
            self.results[config_type]["metrics_history"] = metrics

        # Отмечаем завершение конфигурации
        if config_type == "baseline":
            self._baseline_completed = True
        else:
            self._variant_completed = True

        self.logger.info(
            f"Добавлены результаты для {config_type}",
            extra={
                "config_type": config_type,
                "results_keys": list(results.keys()),
                "metrics_count": len(metrics) if metrics else 0,
            },
        )

        # Проверяем, можно ли провести сравнение
        if self._baseline_completed and self._variant_completed:
            self._perform_comparison()

    def compare_results(self) -> Dict[str, Any]:
        """Сравнить результаты baseline и variant конфигураций.

        Returns:
            Словарь с результатами сравнения

        Raises:
            ValueError: Если результаты для сравнения недоступны
        """
        if not self._baseline_completed or not self._variant_completed:
            raise ValueError(
                "Невозможно провести сравнение: отсутствуют результаты для "
                f"baseline={self._baseline_completed}, variant={self._variant_completed}"
            )

        return self.results.get("comparison", {})

    def save(self, format_type: str = "json") -> Path:
        """Сериализовать эксперимент в файл.

        Args:
            format_type: Формат сохранения ('json' или 'pickle')

        Returns:
            Путь к сохраненному файлу

        Raises:
            ValueError: Если формат неподдерживаемый
            IOError: Если ошибка записи файла
        """
        if format_type not in ["json", "pickle"]:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")

        # Подготавливаем данные для сериализации
        data = self._to_dict()

        # Определяем имя файла
        filename = f"experiment_{self.experiment_id}.{format_type}"
        filepath = self.experiment_dir / filename

        try:
            if format_type == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:  # pickle
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)

            self.logger.info(f"Эксперимент сохранен в {filepath}")
            return filepath

        except Exception as e:
            error_msg = f"Ошибка при сохранении эксперимента: {e}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e

    @classmethod
    def load(
        cls, filepath: Union[str, Path], format_type: Optional[str] = None
    ) -> "Experiment":
        """Загрузить эксперимент из файла.

        Args:
            filepath: Путь к файлу эксперимента
            format_type: Формат файла (определяется автоматически если None)

        Returns:
            Загруженный объект эксперимента

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла неподдерживаемый
            IOError: Если ошибка чтения файла
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Файл эксперимента не найден: {filepath}")

        # Определяем формат по расширению если не указан
        if format_type is None:
            format_type = filepath.suffix.lstrip(".")

        if format_type not in ["json", "pickle"]:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")

        try:
            if format_type == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:  # pickle
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

            # Создаем объект эксперимента из данных
            experiment = cls._from_dict(data)

            # Настраиваем логгер для загруженного эксперимента
            experiment.logger = get_experiment_logger(experiment.experiment_id)
            experiment.logger.info(f"Эксперимент загружен из {filepath}")

            return experiment

        except Exception as e:
            error_msg = f"Ошибка при загрузке эксперимента: {e}"
            raise IOError(error_msg) from e

    def get_status(self) -> Dict[str, Any]:
        """Получить текущий статус эксперимента.

        Returns:
            Словарь с информацией о статусе
        """
        duration = None
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            duration = (end_time - self.started_at).total_seconds()

        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "hypothesis": self.hypothesis,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "duration_seconds": duration,
            "baseline_completed": self._baseline_completed,
            "variant_completed": self._variant_completed,
            "results_available": bool(self.results.get("comparison")),
            "output_dir": str(self.experiment_dir),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Получить краткую сводку эксперимента.

        Returns:
            Словарь с основной информацией об эксперименте
        """
        summary = {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "status": self.status.value,
            "configurations": {
                "baseline": {
                    "algorithm": self.baseline_config.algorithm.name,
                    "environment": self.baseline_config.environment.name,
                    "learning_rate": self.baseline_config.algorithm.learning_rate,
                    "total_timesteps": self.baseline_config.training.total_timesteps,
                },
                "variant": {
                    "algorithm": self.variant_config.algorithm.name,
                    "environment": self.variant_config.environment.name,
                    "learning_rate": self.variant_config.algorithm.learning_rate,
                    "total_timesteps": self.variant_config.training.total_timesteps,
                },
            },
        }

        # Добавляем результаты если доступны
        if self.results.get("comparison"):
            summary["results"] = self.results["comparison"]

        return summary

    def _validate_configurations(
        self, baseline_config: RLConfig, variant_config: RLConfig
    ) -> None:
        """Валидировать совместимость конфигураций.

        Args:
            baseline_config: Базовая конфигурация
            variant_config: Вариантная конфигурация

        Raises:
            ConfigurationError: Если конфигурации несовместимы
        """
        # Проверяем, что среды одинаковые
        if baseline_config.environment.name != variant_config.environment.name:
            raise ConfigurationError(
                f"Среды должны быть одинаковыми: "
                f"{baseline_config.environment.name} != {variant_config.environment.name}"
            )

        # Проверяем, что есть различия в конфигурациях
        baseline_dict = self._config_to_comparable_dict(baseline_config)
        variant_dict = self._config_to_comparable_dict(variant_config)

        if baseline_dict == variant_dict:
            raise ConfigurationError(
                "Конфигурации идентичны. Для эксперимента нужны различия."
            )

        # Проверяем валидность отдельных конфигураций
        self._validate_single_config(baseline_config, "baseline")
        self._validate_single_config(variant_config, "variant")

    def _validate_single_config(self, config: RLConfig, config_name: str) -> None:
        """Валидировать отдельную конфигурацию.

        Args:
            config: Конфигурация для валидации
            config_name: Имя конфигурации для сообщений об ошибках

        Raises:
            ConfigurationError: Если конфигурация невалидна
        """
        # Проверяем алгоритм
        supported_algorithms = ["PPO", "A2C", "SAC", "TD3"]
        if config.algorithm.name not in supported_algorithms:
            raise ConfigurationError(
                f"Неподдерживаемый алгоритм в {config_name}: {config.algorithm.name}"
            )

        # Проверяем параметры обучения
        if config.training.total_timesteps <= 0:
            raise ConfigurationError(
                f"total_timesteps должен быть положительным в {config_name}"
            )

        # Проверяем learning rate
        if config.algorithm.learning_rate <= 0:
            raise ConfigurationError(
                f"learning_rate должен быть положительным в {config_name}"
            )

    def _config_to_comparable_dict(self, config: RLConfig) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь для сравнения.

        Args:
            config: Конфигурация для преобразования

        Returns:
            Словарь с ключевыми параметрами конфигурации
        """
        return {
            "algorithm_name": config.algorithm.name,
            "learning_rate": config.algorithm.learning_rate,
            "n_steps": config.algorithm.n_steps,
            "batch_size": config.algorithm.batch_size,
            "gamma": config.algorithm.gamma,
            "total_timesteps": config.training.total_timesteps,
            "environment": config.environment.name,
        }

    def _perform_comparison(self) -> None:
        """Выполнить сравнение результатов baseline и variant."""
        baseline_results = self.results["baseline"]
        variant_results = self.results["variant"]

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis_confirmed": None,  # Будет определено на основе результатов
            "performance_metrics": {},
            "statistical_significance": {},
            "summary": {},
        }

        # Сравниваем основные метрики
        metrics_to_compare = [
            "mean_reward",
            "final_reward",
            "episode_length",
            "convergence_timesteps",
            "training_time",
        ]

        for metric in metrics_to_compare:
            baseline_value = baseline_results.get(metric)
            variant_value = variant_results.get(metric)

            if baseline_value is not None and variant_value is not None:
                improvement = variant_value - baseline_value
                improvement_percent = (
                    (improvement / baseline_value) * 100 if baseline_value != 0 else 0
                )

                comparison["performance_metrics"][metric] = {
                    "baseline": baseline_value,
                    "variant": variant_value,
                    "improvement": improvement,
                    "improvement_percent": improvement_percent,
                    "better": "variant" if improvement > 0 else "baseline",
                }

        # Определяем общий результат
        reward_improvement = (
            comparison["performance_metrics"]
            .get("mean_reward", {})
            .get("improvement", 0)
        )
        comparison["summary"] = {
            "overall_better": "variant" if reward_improvement > 0 else "baseline",
            "reward_improvement": reward_improvement,
            "significant_improvement": abs(reward_improvement) > 10,  # Порог значимости
        }

        self.results["comparison"] = comparison

        self.logger.info(
            "Выполнено сравнение результатов",
            extra={
                "reward_improvement": reward_improvement,
                "overall_better": comparison["summary"]["overall_better"],
            },
        )

    def _finalize_results(self) -> None:
        """Финализировать результаты эксперимента."""
        # Добавляем метаданные
        duration_seconds = None
        if self.started_at and self.completed_at:
            duration_seconds = (self.completed_at - self.started_at).total_seconds()

        self.results["metadata"] = {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "duration_seconds": duration_seconds,
            "baseline_algorithm": self.baseline_config.algorithm.name,
            "variant_algorithm": self.variant_config.algorithm.name,
            "environment": self.baseline_config.environment.name,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

        # Сохраняем конфигурации
        self.results["configurations"] = {
            "baseline": self._config_to_comparable_dict(self.baseline_config),
            "variant": self._config_to_comparable_dict(self.variant_config),
        }

    def _to_dict(self) -> Dict[str, Any]:
        """Преобразовать эксперимент в словарь для сериализации.

        Returns:
            Словарь с данными эксперимента
        """
        from dataclasses import asdict

        return {
            "experiment_id": self.experiment_id,
            "baseline_config": asdict(self.baseline_config),
            "variant_config": asdict(self.variant_config),
            "hypothesis": self.hypothesis,
            "results": self.results,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "output_dir": str(self.output_dir),
            "_baseline_completed": self._baseline_completed,
            "_variant_completed": self._variant_completed,
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Создать эксперимент из словаря данных.

        Args:
            data: Словарь с данными эксперимента

        Returns:
            Объект эксперимента
        """
        from src.utils.config import ConfigLoader

        # Восстанавливаем конфигурации
        loader = ConfigLoader()

        # Обрабатываем None значения в experiment поле
        baseline_config_data = data["baseline_config"].copy()
        if baseline_config_data.get("experiment") is None:
            baseline_config_data.pop("experiment", None)

        variant_config_data = data["variant_config"].copy()
        if variant_config_data.get("experiment") is None:
            variant_config_data.pop("experiment", None)

        baseline_config = loader._create_config_object(baseline_config_data)
        variant_config = loader._create_config_object(variant_config_data)

        # Создаем эксперимент
        experiment = cls(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis=data["hypothesis"],
            experiment_id=data["experiment_id"],
            output_dir=data["output_dir"],
        )

        # Восстанавливаем состояние
        experiment.results = data["results"]
        experiment.status = ExperimentStatus(data["status"])
        experiment.created_at = datetime.fromisoformat(data["created_at"])
        experiment.started_at = (
            datetime.fromisoformat(data["started_at"]) if data["started_at"] else None
        )
        experiment.completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if data["completed_at"]
            else None
        )
        experiment.paused_at = (
            datetime.fromisoformat(data["paused_at"]) if data["paused_at"] else None
        )
        experiment._baseline_completed = data.get("_baseline_completed", False)
        experiment._variant_completed = data.get("_variant_completed", False)

        return experiment

    def __repr__(self) -> str:
        """Строковое представление эксперимента."""
        return (
            f"Experiment(id={self.experiment_id}, "
            f"status={self.status.value}, "
            f"baseline={self.baseline_config.algorithm.name}, "
            f"variant={self.variant_config.algorithm.name})"
        )

    def __str__(self) -> str:
        """Человекочитаемое представление эксперимента."""
        status_info = self.get_status()
        return (
            f"Эксперимент {self.experiment_id}\n"
            f"Статус: {self.status.value}\n"
            f"Гипотеза: {self.hypothesis}\n"
            f"Baseline: {self.baseline_config.algorithm.name}\n"
            f"Variant: {self.variant_config.algorithm.name}\n"
            f"Создан: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Результаты доступны: {status_info['results_available']}"
        )
