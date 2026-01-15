"""Коллектор метрик обучения для RL агентов.

Этот модуль предоставляет класс для сбора временных рядов метрик обучения,
агрегации статистики и экспорта в JSON формат согласно спецификации data-model.md.

Основные возможности:
- Сбор временных рядов: timestep, episode, reward, episode_length, loss
- Настраиваемый интервал записи (по умолчанию 100 шагов)
- Расчет агрегированной статистики
- Экспорт/импорт в JSON формат
- Метаданные эксперимента с ISO 8601 временными метками
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Сущность метрик обучения согласно data-model.md.

    Содержит агрегированную статистику за интервал записи.

    Attributes:
        timestep: Текущий временной шаг обучения
        episode: Номер эпизода
        reward: Награда за эпизод
        episode_length: Длина эпизода
        loss: Значение функции потерь (опционально)
        timestamp: Временная метка в формате ISO 8601
    """

    timestep: int
    episode: int
    reward: float
    episode_length: int
    loss: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для JSON сериализации.

        Returns:
            Словарь с полями метрики
        """
        return asdict(self)


@dataclass
class AggregatedStatistics:
    """Агрегированная статистика по собранным метрикам.

    Attributes:
        reward_mean: Средняя награда
        reward_std: Стандартное отклонение награды
        reward_min: Минимальная награда
        reward_max: Максимальная награда
        episode_length_mean: Средняя длина эпизода
        total_timesteps: Общее количество временных шагов
        total_episodes: Общее количество эпизодов
        loss_mean: Среднее значение функции потерь (опционально)
    """

    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    episode_length_mean: float
    total_timesteps: int
    total_episodes: int
    loss_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для JSON сериализации.

        Returns:
            Словарь с полями статистики
        """
        return asdict(self)


@dataclass
class MetricsCollectorConfig:
    """Конфигурация коллектора метрик.

    Attributes:
        experiment_id: Уникальный идентификатор эксперимента
        algorithm: Используемый алгоритм (например, "PPO", "A2C")
        environment: Название среды (например, "LunarLander-v3")
        seed: Seed для воспроизводимости
        recording_interval: Интервал записи в шагах (по умолчанию 100)
    """

    experiment_id: str
    algorithm: str
    environment: str
    seed: int
    recording_interval: int = 100

    def __post_init__(self) -> None:
        """Валидация конфигурации после инициализации."""
        if self.recording_interval <= 0:
            raise ValueError(
                f"recording_interval должен быть > 0, получено: {self.recording_interval}"
            )

        if self.seed < 0:
            raise ValueError(f"seed должен быть >= 0, получено: {self.seed}")


class MetricsCollector:
    """Коллектор метрик обучения RL агентов.

    Собирает временные ряды метрик обучения с настраиваемым интервалом записи,
    рассчитывает агрегированную статистику и экспортирует данные в JSON формат.

    Пример использования:
        >>> config = MetricsCollectorConfig(
        ...     experiment_id="exp_001",
        ...     algorithm="PPO",
        ...     environment="LunarLander-v3",
        ...     seed=42,
        ...     recording_interval=100
        ... )
        >>> collector = MetricsCollector(config)
        >>> collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        >>> collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)
        >>> stats = collector.calculate_statistics()
        >>> print(f"Средняя награда: {stats.reward_mean:.2f}")
        Средняя награда: 12.85
        >>> collector.export_to_json("metrics.json")
    """

    def __init__(self, config: MetricsCollectorConfig) -> None:
        """Инициализировать коллектор метрик.

        Args:
            config: Конфигурация коллектора

        Raises:
            ValueError: При некорректной конфигурации
        """
        self.config = config

        # Хранилище метрик
        self._metrics: List[TrainingMetrics] = []

        # Счетчики для интервальной записи
        self._last_recorded_timestep = 0

        logger.debug(
            f"Инициализирован MetricsCollector для эксперимента {config.experiment_id}",
            extra={
                "algorithm": config.algorithm,
                "environment": config.environment,
                "seed": config.seed,
                "recording_interval": config.recording_interval,
            },
        )

    def record(
        self,
        timestep: int,
        episode: int,
        reward: float,
        episode_length: int,
        loss: Optional[float] = None,
    ) -> None:
        """Записать метрики обучения.

        Запись происходит только если прошел интервал recording_interval
        с момента последней записи.

        Args:
            timestep: Текущий временной шаг обучения
            episode: Номер эпизода
            reward: Награда за эпизод
            episode_length: Длина эпизода
            loss: Значение функции потерь (опционально)

        Raises:
            ValueError: Если timestep или episode отрицательные
        """
        if timestep < 0:
            raise ValueError(f"timestep должен быть >= 0, получено: {timestep}")

        if episode < 0:
            raise ValueError(f"episode должен быть >= 0, получено: {episode}")

        # Проверяем интервал записи
        if timestep - self._last_recorded_timestep < self.config.recording_interval:
            logger.debug(
                f"Пропуск записи: timestep={timestep}, "
                f"last_recorded={self._last_recorded_timestep}, "
                f"interval={self.config.recording_interval}"
            )
            return

        # Создаем метрику
        metric = TrainingMetrics(
            timestep=timestep,
            episode=episode,
            reward=reward,
            episode_length=episode_length,
            loss=loss,
        )

        # Добавляем в хранилище
        self._metrics.append(metric)
        self._last_recorded_timestep = timestep

        logger.debug(
            f"Записана метрика: timestep={timestep}, episode={episode}, "
            f"reward={reward:.2f}, episode_length={episode_length}"
        )

    def calculate_statistics(self) -> AggregatedStatistics:
        """Рассчитать агрегированную статистику по собранным метрикам.

        Returns:
            Агрегированная статистика

        Raises:
            RuntimeError: Если нет собранных метрик
        """
        if not self._metrics:
            raise RuntimeError("Нет собранных метрик для расчета статистики")

        # Извлекаем данные
        rewards = [m.reward for m in self._metrics]
        episode_lengths = [m.episode_length for m in self._metrics]
        losses = [m.loss for m in self._metrics if m.loss is not None]

        # Рассчитываем статистику
        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        reward_min = float(np.min(rewards))
        reward_max = float(np.max(rewards))
        episode_length_mean = float(np.mean(episode_lengths))

        # Общая статистика
        total_timesteps = self._metrics[-1].timestep
        total_episodes = self._metrics[-1].episode

        # Статистика по loss если есть
        loss_mean = float(np.mean(losses)) if losses else None

        stats = AggregatedStatistics(
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_min=reward_min,
            reward_max=reward_max,
            episode_length_mean=episode_length_mean,
            total_timesteps=total_timesteps,
            total_episodes=total_episodes,
            loss_mean=loss_mean,
        )

        logger.debug(
            f"Рассчитана статистика: reward_mean={reward_mean:.2f}, "
            f"total_timesteps={total_timesteps}, total_episodes={total_episodes}"
        )

        return stats

    def get_metrics(self) -> List[TrainingMetrics]:
        """Получить все собранные метрики.

        Returns:
            Список метрик обучения
        """
        return self._metrics.copy()

    def get_metrics_count(self) -> int:
        """Получить количество собранных метрик.

        Returns:
            Количество метрик
        """
        return len(self._metrics)

    def clear_metrics(self) -> None:
        """Очистить все собранные метрики."""
        self._metrics.clear()
        self._last_recorded_timestep = 0
        logger.debug("Метрики очищены")

    def export_to_json(self, filepath: Union[str, Path]) -> str:
        """Экспортировать метрики и статистику в JSON файл.

        Формат соответствует спецификации data-model.md:
        - metadata: информация об эксперименте
        - metrics: временные ряды метрик
        - statistics: агрегированная статистика
        - export_timestamp: время экспорта в ISO 8601

        Args:
            filepath: Путь к файлу для экспорта

        Returns:
            Путь к созданному файлу

        Raises:
            RuntimeError: Если нет собранных метрик
            IOError: При ошибке записи файла

        Пример:
            >>> collector.export_to_json("results/metrics.json")
            'results/metrics.json'
        """
        if not self._metrics:
            raise RuntimeError("Нет собранных метрик для экспорта")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Рассчитываем статистику
        statistics = self.calculate_statistics()

        # Формируем данные для экспорта
        export_data = {
            "metadata": {
                "experiment_id": self.config.experiment_id,
                "algorithm": self.config.algorithm,
                "environment": self.config.environment,
                "seed": self.config.seed,
                "recording_interval": self.config.recording_interval,
            },
            "metrics": [metric.to_dict() for metric in self._metrics],
            "statistics": statistics.to_dict(),
            "export_timestamp": datetime.now().isoformat(),
        }

        # Записываем в файл
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Метрики экспортированы в JSON: {filepath}")
            return str(filepath)

        except IOError as e:
            error_msg = f"Ошибка записи в файл {filepath}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "MetricsCollector":
        """Загрузить коллектор метрик из JSON файла.

        Args:
            filepath: Путь к файлу для загрузки

        Returns:
            Экземпляр MetricsCollector с загруженными данными

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: При некорректном формате файла
            IOError: При ошибке чтения файла

        Пример:
            >>> collector = MetricsCollector.load_from_json("results/metrics.json")
            >>> print(collector.get_metrics_count())
            42
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        except IOError as e:
            error_msg = f"Ошибка чтения файла {filepath}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except json.JSONDecodeError as e:
            error_msg = f"Некорректный JSON формат в файле {filepath}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Валидация структуры
        required_keys = ["metadata", "metrics", "statistics"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Отсутствует обязательное поле: {key}")

        # Создаем конфигурацию
        metadata = data["metadata"]
        config = MetricsCollectorConfig(
            experiment_id=metadata["experiment_id"],
            algorithm=metadata["algorithm"],
            environment=metadata["environment"],
            seed=metadata["seed"],
            recording_interval=metadata["recording_interval"],
        )

        # Создаем коллектор
        collector = cls(config)

        # Загружаем метрики
        for metric_data in data["metrics"]:
            metric = TrainingMetrics(**metric_data)
            collector._metrics.append(metric)

        # Обновляем последний записанный timestep
        if collector._metrics:
            collector._last_recorded_timestep = collector._metrics[-1].timestep

        logger.info(
            f"Метрики загружены из JSON: {filepath}, "
            f"количество метрик: {len(collector._metrics)}"
        )

        return collector

    def __len__(self) -> int:
        """Получить количество собранных метрик.

        Returns:
            Количество метрик
        """
        return len(self._metrics)

    def __repr__(self) -> str:
        """Строковое представление коллектора.

        Returns:
            Строковое представление
        """
        return (
            f"MetricsCollector(experiment_id={self.config.experiment_id!r}, "
            f"algorithm={self.config.algorithm!r}, "
            f"environment={self.config.environment!r}, "
            f"metrics_count={len(self._metrics)})"
        )
