"""Базовый класс для управления экспериментами RL.

Этот модуль предоставляет абстрактный базовый класс для экспериментов,
общие методы жизненного цикла эксперимента, интеграцию с управлением
конфигурацией, сбор и хранение результатов, обеспечение воспроизводимости
и обработку ошибок с очисткой ресурсов.
"""

import json
import logging
import os
import shutil
import subprocess
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..utils.config import RLConfig
from ..utils.rl_logging import get_experiment_logger
from ..utils.metrics import MetricsTracker
from ..utils.seeding import SeedManager

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Результат эксперимента."""

    experiment_id: str
    status: str  # 'success', 'failed', 'cancelled'
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]  # Пути к файлам результатов
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return asdict(self)


class ExperimentManager(ABC):
    """Абстрактный базовый класс для управления экспериментами RL."""

    def __init__(
        self,
        experiment_id: str,
        config: RLConfig,
        output_dir: Optional[Union[str, Path]] = None,
        cleanup_on_failure: bool = True,
    ):
        """Инициализация менеджера экспериментов.

        Args:
            experiment_id: Уникальный идентификатор эксперимента
            config: Конфигурация эксперимента
            output_dir: Директория для результатов
            cleanup_on_failure: Очищать ли ресурсы при ошибке
        """
        self.experiment_id = experiment_id
        self.config = config
        self.cleanup_on_failure = cleanup_on_failure

        # Настройка директорий
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(config.output_dir)

        self.experiment_dir = self.output_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Поддиректории
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        self.plots_dir = self.experiment_dir / "plots"
        self.videos_dir = self.experiment_dir / "videos"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"

        for directory in [
            self.models_dir,
            self.logs_dir,
            self.plots_dir,
            self.videos_dir,
            self.checkpoints_dir,
        ]:
            directory.mkdir(exist_ok=True)

        # Компоненты эксперимента
        self.logger = get_experiment_logger(experiment_id)
        self.metrics_tracker = MetricsTracker(
            experiment_id=experiment_id, save_dir=self.logs_dir
        )
        self.seed_manager = SeedManager(config.seed)

        # Состояние эксперимента
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "initialized"
        self.artifacts: Dict[str, str] = {}
        self.error_info: Optional[Dict[str, str]] = None

        # Сохраняем конфигурацию
        self._save_config()

        self.logger.info(f"Инициализирован эксперимент {experiment_id}")
        self.logger.info(f"Выходная директория: {self.experiment_dir}")

    def run(self) -> ExperimentResult:
        """Запустить эксперимент.

        Returns:
            Результат эксперимента
        """
        self.logger.info(f"Запуск эксперимента {self.experiment_id}")

        try:
            # Подготовка
            self._setup_experiment()

            # Выполнение
            self._execute_experiment()

            # Завершение
            self._finalize_experiment()

            # Создание результата
            result = self._create_result("success")

            self.logger.info(f"Эксперимент {self.experiment_id} успешно завершен")
            return result

        except KeyboardInterrupt:
            self.logger.warning(
                f"Эксперимент {self.experiment_id} прерван пользователем"
            )
            self._handle_cancellation()
            return self._create_result("cancelled")

        except Exception as e:
            self.logger.error(f"Ошибка в эксперименте {self.experiment_id}: {e}")
            self._handle_error(e)
            return self._create_result("failed")

    def _setup_experiment(self) -> None:
        """Подготовка эксперимента."""
        self.start_time = datetime.now()
        self.status = "running"

        # Устанавливаем seed
        self.seed_manager.set_experiment_seed(self.experiment_id)

        # Записываем информацию о среде
        self._save_environment_info()

        # Вызываем пользовательскую подготовку
        self.setup()

        self.logger.info("Подготовка эксперимента завершена")

    def _execute_experiment(self) -> None:
        """Выполнение основной части эксперимента."""
        self.logger.info("Начало выполнения эксперимента")

        # Вызываем пользовательскую реализацию
        self.execute()

        self.logger.info("Выполнение эксперимента завершено")

    def _finalize_experiment(self) -> None:
        """Завершение эксперимента."""
        self.end_time = datetime.now()
        self.status = "success"

        # Сохраняем метрики
        self.metrics_tracker.export_to_json()
        self.metrics_tracker.export_to_csv()

        # Вызываем пользовательскую финализацию
        self.finalize()

        # Сохраняем итоговый результат
        self._save_final_result()

        self.logger.info("Финализация эксперимента завершена")

    def _handle_error(self, error: Exception) -> None:
        """Обработка ошибки."""
        self.end_time = datetime.now()
        self.status = "failed"

        # Сохраняем информацию об ошибке
        self.error_info = {
            "message": str(error),
            "traceback": traceback.format_exc(),
            "type": type(error).__name__,
        }

        # Логируем ошибку
        self.logger.error(f"Ошибка в эксперименте: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Пытаемся сохранить частичные результаты
        try:
            self.metrics_tracker.export_to_json(
                f"partial_metrics_{self.experiment_id}.json"
            )
            self.logger.info("Частичные результаты сохранены")
        except Exception as save_error:
            self.logger.error(
                f"Не удалось сохранить частичные результаты: {save_error}"
            )

        # Очистка ресурсов при необходимости
        if self.cleanup_on_failure:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                self.logger.error(f"Ошибка при очистке ресурсов: {cleanup_error}")

    def _handle_cancellation(self) -> None:
        """Обработка отмены эксперимента."""
        self.end_time = datetime.now()
        self.status = "cancelled"

        # Сохраняем частичные результаты
        try:
            self.metrics_tracker.export_to_json(
                f"cancelled_metrics_{self.experiment_id}.json"
            )
            self.logger.info("Частичные результаты отмененного эксперимента сохранены")
        except Exception as e:
            self.logger.error(
                f"Не удалось сохранить результаты отмененного эксперимента: {e}"
            )

        # Очистка ресурсов
        try:
            self.cleanup()
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}")

    def _create_result(self, status: str) -> ExperimentResult:
        """Создать объект результата эксперимента."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        # Собираем метрики
        metrics_summary = self.metrics_tracker.get_all_summaries()
        metrics_dict = {
            name: summary.to_dict() for name, summary in metrics_summary.items()
        }

        return ExperimentResult(
            experiment_id=self.experiment_id,
            status=status,
            start_time=self.start_time.isoformat() if self.start_time else "",
            end_time=self.end_time.isoformat() if self.end_time else None,
            duration_seconds=duration,
            config=asdict(self.config),
            metrics=metrics_dict,
            artifacts=self.artifacts.copy(),
            error_message=self.error_info.get("message") if self.error_info else None,
            error_traceback=self.error_info.get("traceback")
            if self.error_info
            else None,
        )

    def _save_config(self) -> None:
        """Сохранить конфигурацию эксперимента."""
        config_file = self.experiment_dir / "config.json"

        try:
            config_dict = asdict(self.config)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            self.artifacts["config"] = str(config_file)
            self.logger.debug(f"Конфигурация сохранена: {config_file}")

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении конфигурации: {e}")

    def _save_environment_info(self) -> None:
        """Сохранить информацию о среде выполнения."""
        env_info = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "python_version": self._get_python_version(),
            "pip_freeze": self._get_pip_freeze(),
            "git_info": self._get_git_info(),
            "system_info": self._get_system_info(),
            "environment_variables": self._get_relevant_env_vars(),
        }

        env_file = self.experiment_dir / "environment.json"

        try:
            with open(env_file, "w", encoding="utf-8") as f:
                json.dump(env_info, f, ensure_ascii=False, indent=2)

            self.artifacts["environment"] = str(env_file)
            self.logger.debug(f"Информация о среде сохранена: {env_file}")

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении информации о среде: {e}")

    def _save_final_result(self) -> None:
        """Сохранить итоговый результат эксперимента."""
        result = self._create_result(self.status)
        result_file = self.experiment_dir / "result.json"

        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            self.artifacts["result"] = str(result_file)
            self.logger.info(f"Итоговый результат сохранен: {result_file}")

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении итогового результата: {e}")

    def _get_python_version(self) -> str:
        """Получить версию Python."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_pip_freeze(self) -> List[str]:
        """Получить список установленных пакетов."""
        try:
            result = subprocess.run(
                ["pip", "freeze"], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")
            else:
                self.logger.warning("Не удалось выполнить pip freeze")
                return []
        except Exception as e:
            self.logger.warning(f"Ошибка при выполнении pip freeze: {e}")
            return []

    def _get_git_info(self) -> Dict[str, str]:
        """Получить информацию о Git репозитории."""
        git_info = {}

        try:
            # Текущий коммит
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                git_info["commit"] = result.stdout.strip()

            # Текущая ветка
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Статус репозитория
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["dirty"] = bool(result.stdout.strip())

        except Exception as e:
            self.logger.debug(f"Не удалось получить информацию о Git: {e}")

        return git_info

    def _get_system_info(self) -> Dict[str, Any]:
        """Получить информацию о системе."""
        import platform
        import psutil

        try:
            return {
                "platform": platform.platform(),
                "python_implementation": platform.python_implementation(),
                "cpu_count": os.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_free_gb": round(shutil.disk_usage(".").free / (1024**3), 2),
            }
        except Exception as e:
            self.logger.debug(f"Не удалось получить информацию о системе: {e}")
            return {}

    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Получить релевантные переменные окружения."""
        relevant_vars = [
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONPATH",
            "CONDA_DEFAULT_ENV",
        ]

        env_vars = {}
        for var in relevant_vars:
            value = os.getenv(var)
            if value is not None:
                env_vars[var] = value

        return env_vars

    def add_artifact(self, name: str, path: Union[str, Path]) -> None:
        """Добавить артефакт эксперимента.

        Args:
            name: Название артефакта
            path: Путь к файлу артефакта
        """
        self.artifacts[name] = str(path)
        self.logger.debug(f"Добавлен артефакт {name}: {path}")

    def get_artifact_path(self, name: str) -> Optional[Path]:
        """Получить путь к артефакту.

        Args:
            name: Название артефакта

        Returns:
            Путь к артефакту или None если не найден
        """
        if name in self.artifacts:
            return Path(self.artifacts[name])
        return None

    def save_model(self, model: Any, name: str = "final_model") -> str:
        """Сохранить модель.

        Args:
            model: Модель для сохранения
            name: Название модели

        Returns:
            Путь к сохраненной модели
        """
        model_path = self.models_dir / f"{name}.zip"

        try:
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                # Для PyTorch моделей
                import torch

                torch.save(model.state_dict(), model_path)

            self.add_artifact(f"model_{name}", model_path)
            self.logger.info(f"Модель {name} сохранена: {model_path}")

            return str(model_path)

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели {name}: {e}")
            raise

    def save_plot(self, figure, name: str) -> str:
        """Сохранить график.

        Args:
            figure: Объект matplotlib figure
            name: Название графика

        Returns:
            Путь к сохраненному графику
        """
        plot_path = self.plots_dir / f"{name}.png"

        try:
            figure.savefig(plot_path, dpi=300, bbox_inches="tight")
            self.add_artifact(f"plot_{name}", plot_path)
            self.logger.info(f"График {name} сохранен: {plot_path}")

            return str(plot_path)

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении графика {name}: {e}")
            raise

    # Абстрактные методы для переопределения в подклассах

    @abstractmethod
    def setup(self) -> None:
        """Пользовательская подготовка эксперимента.

        Этот метод должен быть переопределен в подклассах для выполнения
        специфичной для эксперимента подготовки.
        """
        pass

    @abstractmethod
    def execute(self) -> None:
        """Выполнение основной части эксперимента.

        Этот метод должен быть переопределен в подклассах для выполнения
        основной логики эксперимента.
        """
        pass

    def finalize(self) -> None:
        """Пользовательская финализация эксперимента.

        Этот метод может быть переопределен в подклассах для выполнения
        специфичной для эксперимента финализации.
        """
        pass

    def cleanup(self) -> None:
        """Очистка ресурсов.

        Этот метод может быть переопределен в подклассах для очистки
        специфичных ресурсов.
        """
        pass


class SimpleExperiment(ExperimentManager):
    """Простая реализация эксперимента для быстрого прототипирования."""

    def __init__(
        self,
        experiment_id: str,
        config: RLConfig,
        setup_func: Optional[Callable] = None,
        execute_func: Optional[Callable] = None,
        finalize_func: Optional[Callable] = None,
        **kwargs,
    ):
        """Инициализация простого эксперимента.

        Args:
            experiment_id: Идентификатор эксперимента
            config: Конфигурация
            setup_func: Функция подготовки
            execute_func: Функция выполнения
            finalize_func: Функция финализации
            **kwargs: Дополнительные аргументы для базового класса
        """
        super().__init__(experiment_id, config, **kwargs)

        self.setup_func = setup_func
        self.execute_func = execute_func
        self.finalize_func = finalize_func

    def setup(self) -> None:
        """Выполнить пользовательскую функцию подготовки."""
        if self.setup_func:
            self.setup_func(self)

    def execute(self) -> None:
        """Выполнить пользовательскую функцию выполнения."""
        if self.execute_func:
            self.execute_func(self)
        else:
            raise NotImplementedError("execute_func должна быть предоставлена")

    def finalize(self) -> None:
        """Выполнить пользовательскую функцию финализации."""
        if self.finalize_func:
            self.finalize_func(self)


def create_experiment(
    experiment_id: str,
    config: RLConfig,
    execute_func: Callable,
    setup_func: Optional[Callable] = None,
    finalize_func: Optional[Callable] = None,
    **kwargs,
) -> SimpleExperiment:
    """Создать простой эксперимент.

    Args:
        experiment_id: Идентификатор эксперимента
        config: Конфигурация
        execute_func: Функция выполнения
        setup_func: Функция подготовки (опционально)
        finalize_func: Функция финализации (опционально)
        **kwargs: Дополнительные аргументы

    Returns:
        Экземпляр SimpleExperiment
    """
    return SimpleExperiment(
        experiment_id=experiment_id,
        config=config,
        setup_func=setup_func,
        execute_func=execute_func,
        finalize_func=finalize_func,
        **kwargs,
    )
