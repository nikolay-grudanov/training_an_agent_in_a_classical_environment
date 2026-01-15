"""Утилиты для отслеживания и анализа метрик в RL экспериментах.

Этот модуль предоставляет сбор метрик в реальном времени, интеграцию с
Stable-Baselines3 callbacks, агрегацию статистики, экспорт в JSON/CSV,
утилиты для построения графиков и поддержку пользовательских метрик.
"""

import csv
import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Точка метрики с временной меткой."""

    timestep: int
    episode: int
    value: float
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return asdict(self)


@dataclass
class MetricSummary:
    """Сводная статистика по метрике."""

    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    last_value: float
    last_timestep: int

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return asdict(self)


class MetricsTracker:
    """Трекер для сбора и анализа метрик обучения."""

    def __init__(
        self,
        experiment_id: str,
        buffer_size: int = 10000,
        auto_save: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """Инициализация трекера метрик.

        Args:
            experiment_id: Идентификатор эксперимента
            buffer_size: Размер буфера для метрик
            auto_save: Автоматически сохранять метрики
            save_dir: Директория для сохранения
        """
        self.experiment_id = experiment_id
        self.buffer_size = buffer_size
        self.auto_save = auto_save

        # Директория для сохранения
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path("results") / "metrics"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Хранилища метрик
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.custom_metrics: Dict[str, Callable] = {}
        self.aggregators: Dict[str, List[Callable]] = defaultdict(list)

        # Счетчики
        self.total_timesteps = 0
        self.total_episodes = 0

        # Файлы для сохранения
        self.csv_file = self.save_dir / f"metrics_{experiment_id}.csv"
        self.json_file = self.save_dir / f"metrics_{experiment_id}.json"

        logger.info(f"Инициализирован MetricsTracker для эксперимента {experiment_id}")

    def add_metric(
        self,
        name: str,
        value: float,
        timestep: Optional[int] = None,
        episode: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Добавить значение метрики.

        Args:
            name: Название метрики
            value: Значение метрики
            timestep: Временной шаг (опционально)
            episode: Номер эпизода (опционально)
            metadata: Дополнительные метаданные
        """
        if timestep is None:
            timestep = self.total_timesteps
        if episode is None:
            episode = self.total_episodes
        if metadata is None:
            metadata = {}

        # Создаем точку метрики
        point = MetricPoint(
            timestep=timestep,
            episode=episode,
            value=value,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
        )

        # Добавляем в буфер
        self.metrics[name].append(point)

        # Обновляем счетчики
        self.total_timesteps = max(self.total_timesteps, timestep)
        self.total_episodes = max(self.total_episodes, episode)

        # Применяем агрегаторы
        for aggregator in self.aggregators[name]:
            try:
                aggregator(point)
            except Exception as e:
                logger.warning(f"Ошибка в агрегаторе для метрики {name}: {e}")

        # Автосохранение
        if self.auto_save and len(self.metrics[name]) % 100 == 0:
            self._auto_save_metrics(name)

        logger.debug(f"Добавлена метрика {name}: {value} (timestep={timestep})")

    def add_episode_metrics(
        self, episode: int, timestep: int, reward: float, length: int, **kwargs: Any
    ) -> None:
        """Добавить метрики эпизода.

        Args:
            episode: Номер эпизода
            timestep: Временной шаг
            reward: Награда за эпизод
            length: Длина эпизода
            **kwargs: Дополнительные метрики
        """
        self.add_metric("episode_reward", reward, timestep, episode)
        self.add_metric("episode_length", length, timestep, episode)

        # Добавляем дополнительные метрики
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.add_metric(name, value, timestep, episode)

    def add_training_metrics(
        self,
        timestep: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Добавить метрики обучения.

        Args:
            timestep: Временной шаг
            loss: Значение функции потерь
            learning_rate: Текущая скорость обучения
            **kwargs: Дополнительные метрики обучения
        """
        if loss is not None:
            self.add_metric("training_loss", loss, timestep)
        if learning_rate is not None:
            self.add_metric("learning_rate", learning_rate, timestep)

        # Добавляем дополнительные метрики
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.add_metric(f"training_{name}", value, timestep)

    def register_custom_metric(
        self, name: str, calculator: Callable[[List[MetricPoint]], float]
    ) -> None:
        """Зарегистрировать пользовательскую метрику.

        Args:
            name: Название метрики
            calculator: Функция для вычисления метрики
        """
        self.custom_metrics[name] = calculator
        logger.info(f"Зарегистрирована пользовательская метрика: {name}")

    def register_aggregator(
        self, metric_name: str, aggregator: Callable[[MetricPoint], None]
    ) -> None:
        """Зарегистрировать агрегатор для метрики.

        Args:
            metric_name: Название метрики
            aggregator: Функция агрегации
        """
        self.aggregators[metric_name].append(aggregator)
        logger.info(f"Зарегистрирован агрегатор для метрики: {metric_name}")

    def get_metric_values(self, name: str, last_n: Optional[int] = None) -> List[float]:
        """Получить значения метрики.

        Args:
            name: Название метрики
            last_n: Количество последних значений

        Returns:
            Список значений метрики
        """
        if name not in self.metrics:
            return []

        points = list(self.metrics[name])
        if last_n:
            points = points[-last_n:]

        return [point.value for point in points]

    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Получить сводную статистику по метрике.

        Args:
            name: Название метрики

        Returns:
            Сводная статистика или None если метрика не найдена
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        values = self.get_metric_values(name)
        if not values:
            return None

        last_point = self.metrics[name][-1]

        return MetricSummary(
            name=name,
            count=len(values),
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            q25=float(np.percentile(values, 25)),
            q75=float(np.percentile(values, 75)),
            last_value=last_point.value,
            last_timestep=last_point.timestep,
        )

    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Получить сводную статистику по всем метрикам.

        Returns:
            Словарь с сводной статистикой
        """
        summaries = {}
        for name in self.metrics:
            summary = self.get_metric_summary(name)
            if summary:
                summaries[name] = summary

        return summaries

    def calculate_moving_average(self, name: str, window: int = 100) -> List[float]:
        """Вычислить скользящее среднее для метрики.

        Args:
            name: Название метрики
            window: Размер окна

        Returns:
            Список значений скользящего среднего
        """
        values = self.get_metric_values(name)
        if len(values) < window:
            return values

        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx : i + 1]
            moving_avg.append(sum(window_values) / len(window_values))

        return moving_avg

    def calculate_custom_metrics(self) -> Dict[str, float]:
        """Вычислить все пользовательские метрики.

        Returns:
            Словарь с вычисленными метриками
        """
        custom_values = {}

        for name, calculator in self.custom_metrics.items():
            try:
                # Получаем все точки метрик
                all_points = []
                for metric_points in self.metrics.values():
                    all_points.extend(list(metric_points))

                # Вычисляем пользовательскую метрику
                value = calculator(all_points)
                custom_values[name] = value

            except Exception as e:
                logger.error(
                    f"Ошибка при вычислении пользовательской метрики {name}: {e}"
                )

        return custom_values

    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """Экспортировать метрики в CSV файл.

        Args:
            filename: Имя файла (опционально)

        Returns:
            Путь к созданному файлу
        """
        if filename:
            csv_path = self.save_dir / filename
        else:
            csv_path = self.csv_file

        try:
            # Подготавливаем данные
            rows = []
            for metric_name, points in self.metrics.items():
                for point in points:
                    row = {
                        "experiment_id": self.experiment_id,
                        "metric_name": metric_name,
                        "timestep": point.timestep,
                        "episode": point.episode,
                        "value": point.value,
                        "timestamp": point.timestamp,
                    }
                    # Добавляем метаданные как отдельные колонки
                    for key, value in point.metadata.items():
                        row[f"meta_{key}"] = value
                    rows.append(row)

            # Записываем в CSV
            if rows:
                fieldnames = rows[0].keys()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            logger.info(f"Метрики экспортированы в CSV: {csv_path}")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Ошибка при экспорте в CSV: {e}")
            raise

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Экспортировать метрики в JSON файл.

        Args:
            filename: Имя файла (опционально)

        Returns:
            Путь к созданному файлу
        """
        if filename:
            json_path = self.save_dir / filename
        else:
            json_path = self.json_file

        try:
            # Подготавливаем данные
            export_data = {
                "experiment_id": self.experiment_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_timesteps": self.total_timesteps,
                "total_episodes": self.total_episodes,
                "metrics": {},
                "summaries": {},
            }

            # Добавляем метрики
            for name, points in self.metrics.items():
                export_data["metrics"][name] = [point.to_dict() for point in points]

            # Добавляем сводную статистику
            summaries = self.get_all_summaries()
            for name, summary in summaries.items():
                export_data["summaries"][name] = summary.to_dict()

            # Добавляем пользовательские метрики
            custom_metrics = self.calculate_custom_metrics()
            if custom_metrics:
                export_data["custom_metrics"] = custom_metrics

            # Записываем в JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Метрики экспортированы в JSON: {json_path}")
            return str(json_path)

        except Exception as e:
            logger.error(f"Ошибка при экспорте в JSON: {e}")
            raise

    def plot_metric(
        self,
        name: str,
        save_path: Optional[Union[str, Path]] = None,
        show_moving_average: bool = True,
        ma_window: int = 100,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Optional[str]:
        """Построить график метрики.

        Args:
            name: Название метрики
            save_path: Путь для сохранения графика
            show_moving_average: Показать скользящее среднее
            ma_window: Размер окна для скользящего среднего
            figsize: Размер фигуры

        Returns:
            Путь к сохраненному файлу или None
        """
        if name not in self.metrics or not self.metrics[name]:
            logger.warning(f"Метрика {name} не найдена или пуста")
            return None

        try:
            # Получаем данные
            points = list(self.metrics[name])
            timesteps = [p.timestep for p in points]
            values = [p.value for p in points]

            # Создаем график
            plt.figure(figsize=figsize)
            plt.plot(timesteps, values, alpha=0.7, label=f"{name} (raw)")

            # Добавляем скользящее среднее
            if show_moving_average and len(values) >= ma_window:
                ma_values = self.calculate_moving_average(name, ma_window)
                plt.plot(
                    timesteps, ma_values, linewidth=2, label=f"{name} (MA-{ma_window})"
                )

            plt.xlabel("Timesteps")
            plt.ylabel(name.replace("_", " ").title())
            plt.title(f"{name.replace('_', ' ').title()} - {self.experiment_id}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Сохраняем график
            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.save_dir / f"{name}_{self.experiment_id}.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"График метрики {name} сохранен: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Ошибка при построении графика метрики {name}: {e}")
            return None

    def plot_multiple_metrics(
        self,
        metric_names: List[str],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> Optional[str]:
        """Построить графики нескольких метрик.

        Args:
            metric_names: Список названий метрик
            save_path: Путь для сохранения
            figsize: Размер фигуры

        Returns:
            Путь к сохраненному файлу или None
        """
        available_metrics = [name for name in metric_names if name in self.metrics]
        if not available_metrics:
            logger.warning("Ни одна из запрошенных метрик не найдена")
            return None

        try:
            # Создаем подграфики
            n_metrics = len(available_metrics)
            rows = (n_metrics + 1) // 2
            cols = 2 if n_metrics > 1 else 1

            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, metric_name in enumerate(available_metrics):
                ax = axes[i]

                # Получаем данные
                points = list(self.metrics[metric_name])
                timesteps = [p.timestep for p in points]
                values = [p.value for p in points]

                # Строим график
                ax.plot(timesteps, values, alpha=0.7)

                # Добавляем скользящее среднее
                if len(values) >= 50:
                    ma_values = self.calculate_moving_average(metric_name, 50)
                    ax.plot(timesteps, ma_values, linewidth=2)

                ax.set_xlabel("Timesteps")
                ax.set_ylabel(metric_name.replace("_", " ").title())
                ax.set_title(metric_name.replace("_", " ").title())
                ax.grid(True, alpha=0.3)

            # Скрываем лишние подграфики
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f"Training Metrics - {self.experiment_id}", fontsize=16)
            plt.tight_layout()

            # Сохраняем
            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.save_dir / f"multiple_metrics_{self.experiment_id}.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"График множественных метрик сохранен: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Ошибка при построении графиков метрик: {e}")
            return None

    def get_convergence_info(
        self,
        metric_name: str,
        threshold: Optional[float] = None,
        window: int = 100,
        stability_episodes: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """Определить информацию о сходимости метрики.

        Args:
            metric_name: Название метрики
            threshold: Пороговое значение для сходимости
            window: Размер окна для анализа
            stability_episodes: Количество эпизодов для проверки стабильности

        Returns:
            Информация о сходимости или None
        """
        if metric_name not in self.metrics:
            return None

        values = self.get_metric_values(metric_name)
        if len(values) < window:
            return None

        try:
            # Вычисляем скользящее среднее
            ma_values = self.calculate_moving_average(metric_name, window)

            # Определяем пороговое значение если не задано
            if threshold is None:
                # Используем 90% от максимального значения
                threshold = max(values) * 0.9

            # Ищем точку сходимости
            convergence_timestep = None
            for i, ma_value in enumerate(ma_values):
                if ma_value >= threshold:
                    # Проверяем стабильность
                    if i + stability_episodes < len(ma_values):
                        stable = all(
                            v >= threshold
                            for v in ma_values[i : i + stability_episodes]
                        )
                        if stable:
                            points = list(self.metrics[metric_name])
                            convergence_timestep = points[i].timestep
                            break

            return {
                "metric_name": metric_name,
                "threshold": threshold,
                "convergence_timestep": convergence_timestep,
                "converged": convergence_timestep is not None,
                "final_value": values[-1],
                "max_value": max(values),
                "mean_value": statistics.mean(values),
                "stability_window": stability_episodes,
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе сходимости метрики {metric_name}: {e}")
            return None

    def _auto_save_metrics(self, metric_name: str) -> None:
        """Автоматическое сохранение метрик."""
        try:
            # Сохраняем только если накопилось достаточно данных
            if len(self.metrics[metric_name]) >= 100:
                self.export_to_json()
        except Exception as e:
            logger.warning(f"Ошибка автосохранения метрик: {e}")

    def clear_metrics(self, metric_name: Optional[str] = None) -> None:
        """Очистить метрики.

        Args:
            metric_name: Название конкретной метрики (если None, очищаются все)
        """
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
                logger.info(f"Метрика {metric_name} очищена")
        else:
            self.metrics.clear()
            self.total_timesteps = 0
            self.total_episodes = 0
            logger.info("Все метрики очищены")


# Предопределенные пользовательские метрики
def sample_efficiency_metric(points: List[MetricPoint]) -> float:
    """Вычислить метрику эффективности выборки."""
    reward_points = [p for p in points if "reward" in p.metadata]
    if len(reward_points) < 2:
        return 0.0

    # Простая метрика: награда на единицу времени
    total_reward = sum(p.value for p in reward_points)
    total_timesteps = max(p.timestep for p in reward_points) - min(
        p.timestep for p in reward_points
    )

    return total_reward / max(total_timesteps, 1)


def stability_metric(points: List[MetricPoint]) -> float:
    """Вычислить метрику стабильности обучения."""
    if len(points) < 10:
        return 0.0

    values = [p.value for p in points[-100:]]  # Последние 100 значений
    if not values:
        return 0.0

    # Коэффициент вариации (обратный показатель стабильности)
    mean_val = statistics.mean(values)
    if mean_val == 0:
        return 0.0

    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = std_val / abs(mean_val)

    # Преобразуем в показатель стабильности (чем выше, тем стабильнее)
    return 1.0 / (1.0 + cv)


# Глобальный трекер метрик
_global_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker(experiment_id: Optional[str] = None) -> MetricsTracker:
    """Получить глобальный трекер метрик.

    Args:
        experiment_id: Идентификатор эксперимента

    Returns:
        Экземпляр MetricsTracker
    """
    global _global_tracker

    if _global_tracker is None or (
        experiment_id and _global_tracker.experiment_id != experiment_id
    ):
        if experiment_id is None:
            experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _global_tracker = MetricsTracker(experiment_id)

    return _global_tracker


def reset_metrics_tracker() -> None:
    """Сбросить глобальный трекер метрик."""
    global _global_tracker
    _global_tracker = None
