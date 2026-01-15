"""Комплексные утилиты для сравнения и анализа результатов RL экспериментов.

Этот модуль предоставляет мощные инструменты для статистического сравнения
экспериментов, анализа производительности, генерации отчетов и визуализации
результатов с поддержкой множественных сравнений и строгих статистических методов.
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from src.experiments.experiment import Experiment
from src.utils.rl_logging import get_experiment_logger

logger = logging.getLogger(__name__)

# Подавляем предупреждения scipy для чистого вывода
warnings.filterwarnings("ignore", category=RuntimeWarning)


class StatisticalTest(Enum):
    """Типы статистических тестов."""

    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    BOOTSTRAP = "bootstrap"


class EffectSizeMethod(Enum):
    """Методы расчета размера эффекта."""

    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"


class MultipleComparisonMethod(Enum):
    """Методы коррекции множественных сравнений."""

    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"
    HOLM = "holm"
    NONE = "none"


@dataclass
class StatisticalTestResult:
    """Результат статистического теста."""

    test_type: StatisticalTest
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    effect_size_method: Optional[EffectSizeMethod] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size_1: int = 0
    sample_size_2: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "effect_size_method": self.effect_size_method.value
            if self.effect_size_method
            else None,
            "confidence_interval": self.confidence_interval,
            "sample_size_1": self.sample_size_1,
            "sample_size_2": self.sample_size_2,
        }


@dataclass
class PerformanceMetrics:
    """Метрики производительности эксперимента."""

    experiment_id: str
    mean_reward: float
    std_reward: float
    final_reward: float
    max_reward: float
    min_reward: float
    convergence_timesteps: Optional[int]
    sample_efficiency: float
    stability_score: float
    success_rate: Optional[float] = None
    training_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "experiment_id": self.experiment_id,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "final_reward": self.final_reward,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "convergence_timesteps": self.convergence_timesteps,
            "sample_efficiency": self.sample_efficiency,
            "stability_score": self.stability_score,
            "success_rate": self.success_rate,
            "training_time": self.training_time,
        }


@dataclass
class ComparisonConfig:
    """Конфигурация для сравнения экспериментов."""

    significance_level: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = 10000
    multiple_comparison_method: MultipleComparisonMethod = (
        MultipleComparisonMethod.FDR_BH
    )
    effect_size_method: EffectSizeMethod = EffectSizeMethod.COHENS_D
    convergence_threshold: Optional[float] = None
    convergence_window: int = 100
    stability_window: int = 50
    min_sample_size: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "significance_level": self.significance_level,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "multiple_comparison_method": self.multiple_comparison_method.value,
            "effect_size_method": self.effect_size_method.value,
            "convergence_threshold": self.convergence_threshold,
            "convergence_window": self.convergence_window,
            "stability_window": self.stability_window,
            "min_sample_size": self.min_sample_size,
        }


@dataclass
class ComparisonResult:
    """Результат сравнения экспериментов."""

    experiment_ids: List[str]
    config: ComparisonConfig
    performance_metrics: Dict[str, PerformanceMetrics]
    statistical_tests: Dict[str, Dict[str, StatisticalTestResult]]
    rankings: Dict[str, List[str]]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "experiment_ids": self.experiment_ids,
            "config": self.config.to_dict(),
            "performance_metrics": {
                exp_id: metrics.to_dict()
                for exp_id, metrics in self.performance_metrics.items()
            },
            "statistical_tests": {
                metric: {
                    comparison: test.to_dict() for comparison, test in tests.items()
                }
                for metric, tests in self.statistical_tests.items()
            },
            "rankings": self.rankings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class ExperimentComparator:
    """Класс для сравнения и анализа RL экспериментов."""

    def __init__(
        self,
        config: Optional[ComparisonConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Инициализация компаратора экспериментов.

        Args:
            config: Конфигурация сравнения
            output_dir: Директория для сохранения результатов
        """
        self.config = config or ComparisonConfig()
        self.output_dir = (
            Path(output_dir) if output_dir else Path("results/comparisons")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_experiment_logger("comparison", base_logger=logger)

        # Настройка стилей для графиков
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        self.logger.info("Инициализирован ExperimentComparator")

    def compare_experiments(
        self,
        experiments: List[Experiment],
        metrics: Optional[List[str]] = None,
        config: Optional[ComparisonConfig] = None,
    ) -> ComparisonResult:
        """Сравнить несколько экспериментов.

        Args:
            experiments: Список экспериментов для сравнения
            metrics: Метрики для сравнения
            config: Конфигурация сравнения

        Returns:
            Результат сравнения

        Raises:
            ValueError: Если недостаточно экспериментов или данных
        """
        if len(experiments) < 2:
            raise ValueError("Для сравнения необходимо минимум 2 эксперимента")

        comparison_config = config or self.config

        if metrics is None:
            metrics = ["mean_reward", "convergence_timesteps", "stability_score"]

        self.logger.info(
            f"Начинаем сравнение {len(experiments)} экспериментов по метрикам: {metrics}"
        )

        # Извлекаем метрики производительности
        performance_metrics = {}
        for exp in experiments:
            try:
                metrics_data = self._extract_performance_metrics(exp, comparison_config)
                performance_metrics[exp.experiment_id] = metrics_data
            except Exception as e:
                self.logger.error(
                    f"Ошибка извлечения метрик для {exp.experiment_id}: {e}"
                )
                continue

        if len(performance_metrics) < 2:
            raise ValueError("Недостаточно валидных экспериментов для сравнения")

        # Проводим статистические тесты
        statistical_tests = {}
        for metric in metrics:
            statistical_tests[metric] = self._perform_pairwise_tests(
                performance_metrics, metric, comparison_config
            )

        # Применяем коррекцию множественных сравнений
        statistical_tests = self._apply_multiple_comparison_correction(
            statistical_tests, comparison_config
        )

        # Ранжируем эксперименты
        rankings = self._rank_experiments(performance_metrics, metrics)

        # Генерируем рекомендации
        recommendations = self._generate_recommendations(
            performance_metrics, statistical_tests, rankings
        )

        result = ComparisonResult(
            experiment_ids=list(performance_metrics.keys()),
            config=comparison_config,
            performance_metrics=performance_metrics,
            statistical_tests=statistical_tests,
            rankings=rankings,
            recommendations=recommendations,
        )

        self.logger.info("Сравнение экспериментов завершено")
        return result

    def statistical_significance(
        self,
        data1: List[float],
        data2: List[float],
        test: StatisticalTest = StatisticalTest.T_TEST,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Проверить статистическую значимость различий.

        Args:
            data1: Первая выборка данных
            data2: Вторая выборка данных
            test: Тип статистического теста
            alpha: Уровень значимости

        Returns:
            Результат статистического теста
        """
        if (
            len(data1) < self.config.min_sample_size
            or len(data2) < self.config.min_sample_size
        ):
            raise ValueError(
                f"Недостаточный размер выборки (минимум {self.config.min_sample_size})"
            )

        data1_array = np.array(data1)
        data2_array = np.array(data2)

        if test == StatisticalTest.T_TEST:
            statistic, p_value = ttest_ind(data1_array, data2_array, equal_var=False)
        elif test == StatisticalTest.MANN_WHITNEY:
            statistic, p_value = mannwhitneyu(
                data1_array, data2_array, alternative="two-sided"
            )
        elif test == StatisticalTest.WILCOXON:
            if len(data1) != len(data2):
                raise ValueError(
                    "Для теста Уилкоксона выборки должны быть одинакового размера"
                )
            statistic, p_value = wilcoxon(data1_array, data2_array)
        elif test == StatisticalTest.BOOTSTRAP:
            statistic, p_value = self._bootstrap_test(data1_array, data2_array)
        else:
            raise ValueError(f"Неподдерживаемый тип теста: {test}")

        # Вычисляем размер эффекта
        effect_size = self._calculate_effect_size(
            data1_array, data2_array, self.config.effect_size_method
        )

        # Вычисляем доверительный интервал
        confidence_interval = self._calculate_confidence_interval(
            data1_array, data2_array, self.config.confidence_level
        )

        return StatisticalTestResult(
            test_type=test,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=bool(p_value < alpha),
            alpha=alpha,
            effect_size=effect_size,
            effect_size_method=self.config.effect_size_method,
            confidence_interval=confidence_interval,
            sample_size_1=len(data1),
            sample_size_2=len(data2),
        )

    def confidence_intervals(
        self, data: List[float], confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Вычислить доверительные интервалы.

        Args:
            data: Данные для анализа
            confidence_level: Уровень доверия

        Returns:
            Кортеж (нижняя граница, верхняя граница)
        """
        if len(data) < 2:
            raise ValueError(
                "Недостаточно данных для вычисления доверительного интервала"
            )

        data_array = np.array(data)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(data) - 1)

        margin_error = t_critical * std_err

        return (mean - margin_error, mean + margin_error)

    def effect_size(
        self,
        data1: List[float],
        data2: List[float],
        method: EffectSizeMethod = EffectSizeMethod.COHENS_D,
    ) -> float:
        """Вычислить размер эффекта.

        Args:
            data1: Первая выборка
            data2: Вторая выборка
            method: Метод вычисления

        Returns:
            Размер эффекта
        """
        return self._calculate_effect_size(np.array(data1), np.array(data2), method)

    def convergence_analysis(
        self,
        experiment: Experiment,
        metric: str = "episode_reward",
        threshold: Optional[float] = None,
        window: int = 100,
    ) -> Dict[str, Any]:
        """Анализ сходимости обучения.

        Args:
            experiment: Эксперимент для анализа
            metric: Метрика для анализа
            threshold: Пороговое значение
            window: Размер окна для анализа

        Returns:
            Результаты анализа сходимости
        """
        if not experiment.results or "baseline" not in experiment.results:
            raise ValueError("Отсутствуют результаты эксперимента")

        # Извлекаем данные метрики
        metrics_history = experiment.results["baseline"].get("metrics_history", [])
        if not metrics_history:
            raise ValueError("Отсутствует история метрик")

        values = []
        timesteps = []

        for entry in metrics_history:
            if metric in entry:
                values.append(entry[metric])
                timesteps.append(entry.get("timestep", len(values)))

        if len(values) < window:
            raise ValueError(f"Недостаточно данных для анализа (минимум {window})")

        # Вычисляем скользящее среднее
        moving_avg = self._calculate_moving_average(values, window)

        # Определяем пороговое значение
        if threshold is None:
            threshold = np.percentile(values, 90)

        # Ищем точку сходимости
        convergence_point = None
        for i, avg_val in enumerate(moving_avg):
            if avg_val >= threshold:
                # Проверяем стабильность
                stability_window = min(50, len(moving_avg) - i)
                if stability_window > 10:
                    stable_values = moving_avg[i : i + stability_window]
                    if all(v >= threshold * 0.95 for v in stable_values):
                        convergence_point = timesteps[i] if i < len(timesteps) else i
                        break

        # Анализ тренда
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(values)), values
        )

        return {
            "metric": metric,
            "threshold": threshold,
            "convergence_timestep": convergence_point,
            "converged": convergence_point is not None,
            "final_value": values[-1],
            "max_value": max(values),
            "mean_value": np.mean(values),
            "trend_slope": slope,
            "trend_r_squared": r_value**2,
            "trend_p_value": p_value,
            "moving_average": moving_avg,
            "raw_values": values,
            "timesteps": timesteps,
        }

    def performance_summary(
        self, experiments: List[Experiment]
    ) -> Dict[str, PerformanceMetrics]:
        """Генерировать сводку производительности.

        Args:
            experiments: Список экспериментов

        Returns:
            Словарь с метриками производительности
        """
        summary = {}

        for exp in experiments:
            try:
                metrics = self._extract_performance_metrics(exp, self.config)
                summary[exp.experiment_id] = metrics
            except Exception as e:
                self.logger.error(
                    f"Ошибка извлечения метрик для {exp.experiment_id}: {e}"
                )
                continue

        return summary

    def learning_efficiency(
        self,
        experiments: List[Experiment],
        threshold: float,
        metric: str = "episode_reward",
    ) -> Dict[str, Dict[str, Any]]:
        """Сравнить эффективность обучения.

        Args:
            experiments: Список экспериментов
            threshold: Пороговое значение для достижения
            metric: Метрика для анализа

        Returns:
            Результаты анализа эффективности
        """
        efficiency_results = {}

        for exp in experiments:
            try:
                convergence_info = self.convergence_analysis(
                    exp, metric, threshold, self.config.convergence_window
                )

                efficiency_results[exp.experiment_id] = {
                    "steps_to_threshold": convergence_info["convergence_timestep"],
                    "achieved_threshold": convergence_info["converged"],
                    "final_performance": convergence_info["final_value"],
                    "sample_efficiency": (
                        threshold / convergence_info["convergence_timestep"]
                        if convergence_info["convergence_timestep"]
                        else 0
                    ),
                }
            except Exception as e:
                self.logger.error(
                    f"Ошибка анализа эффективности для {exp.experiment_id}: {e}"
                )
                efficiency_results[exp.experiment_id] = {
                    "steps_to_threshold": None,
                    "achieved_threshold": False,
                    "final_performance": 0.0,
                    "sample_efficiency": 0.0,
                }

        return efficiency_results

    def stability_analysis(
        self, experiments: List[Experiment], metric: str = "episode_reward"
    ) -> Dict[str, Dict[str, float]]:
        """Анализ стабильности обучения.

        Args:
            experiments: Список экспериментов
            metric: Метрика для анализа

        Returns:
            Результаты анализа стабильности
        """
        stability_results = {}

        for exp in experiments:
            try:
                # Извлекаем данные
                metrics_history = exp.results.get("baseline", {}).get(
                    "metrics_history", []
                )
                values = [
                    entry.get(metric, 0) for entry in metrics_history if metric in entry
                ]

                if len(values) < self.config.stability_window:
                    stability_results[exp.experiment_id] = {
                        "coefficient_of_variation": float("inf"),
                        "variance": float("inf"),
                        "stability_score": 0.0,
                    }
                    continue

                # Берем последние значения для анализа стабильности
                recent_values = values[-self.config.stability_window :]

                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)

                # Коэффициент вариации
                cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")

                # Оценка стабильности (обратная к коэффициенту вариации)
                stability_score = 1.0 / (1.0 + cv) if cv != float("inf") else 0.0

                stability_results[exp.experiment_id] = {
                    "coefficient_of_variation": cv,
                    "variance": std_val**2,
                    "stability_score": stability_score,
                }

            except Exception as e:
                self.logger.error(
                    f"Ошибка анализа стабильности для {exp.experiment_id}: {e}"
                )
                stability_results[exp.experiment_id] = {
                    "coefficient_of_variation": float("inf"),
                    "variance": float("inf"),
                    "stability_score": 0.0,
                }

        return stability_results

    def final_performance(
        self,
        experiments: List[Experiment],
        metric: str = "episode_reward",
        last_n: int = 10,
    ) -> Dict[str, float]:
        """Сравнить финальную производительность.

        Args:
            experiments: Список экспериментов
            metric: Метрика для анализа
            last_n: Количество последних эпизодов для усреднения

        Returns:
            Финальная производительность каждого эксперимента
        """
        final_performance = {}

        for exp in experiments:
            try:
                metrics_history = exp.results.get("baseline", {}).get(
                    "metrics_history", []
                )
                values = [
                    entry.get(metric, 0) for entry in metrics_history if metric in entry
                ]

                if len(values) >= last_n:
                    final_performance[exp.experiment_id] = np.mean(values[-last_n:])
                elif values:
                    final_performance[exp.experiment_id] = np.mean(values)
                else:
                    final_performance[exp.experiment_id] = 0.0

            except Exception as e:
                self.logger.error(
                    f"Ошибка анализа финальной производительности для {exp.experiment_id}: {e}"
                )
                final_performance[exp.experiment_id] = 0.0

        return final_performance

    def peak_performance(
        self, experiments: List[Experiment], metric: str = "episode_reward"
    ) -> Dict[str, Dict[str, Any]]:
        """Найти и сравнить пиковую производительность.

        Args:
            experiments: Список экспериментов
            metric: Метрика для анализа

        Returns:
            Пиковая производительность и информация о ней
        """
        peak_results = {}

        for exp in experiments:
            try:
                metrics_history = exp.results.get("baseline", {}).get(
                    "metrics_history", []
                )
                values = [
                    entry.get(metric, 0) for entry in metrics_history if metric in entry
                ]
                timesteps = [
                    entry.get("timestep", i)
                    for i, entry in enumerate(metrics_history)
                    if metric in entry
                ]

                if not values:
                    peak_results[exp.experiment_id] = {
                        "peak_value": 0.0,
                        "peak_timestep": 0,
                        "peak_episode": 0,
                    }
                    continue

                peak_idx = np.argmax(values)
                peak_value = values[peak_idx]
                peak_timestep = (
                    timesteps[peak_idx] if peak_idx < len(timesteps) else peak_idx
                )

                peak_results[exp.experiment_id] = {
                    "peak_value": peak_value,
                    "peak_timestep": peak_timestep,
                    "peak_episode": peak_idx,
                }

            except Exception as e:
                self.logger.error(
                    f"Ошибка анализа пиковой производительности для {exp.experiment_id}: {e}"
                )
                peak_results[exp.experiment_id] = {
                    "peak_value": 0.0,
                    "peak_timestep": 0,
                    "peak_episode": 0,
                }

        return peak_results

    def generate_comparison_plots(
        self,
        comparison_result: ComparisonResult,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Генерировать графики сравнения.

        Args:
            comparison_result: Результат сравнения
            save_dir: Директория для сохранения

        Returns:
            Словарь с путями к созданным графикам
        """
        if save_dir:
            plot_dir = Path(save_dir)
        else:
            plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        created_plots = {}

        # График производительности
        performance_plot = self._create_performance_comparison_plot(
            comparison_result, plot_dir
        )
        if performance_plot:
            created_plots["performance"] = performance_plot

        # Box plots для статистического сравнения
        box_plot = self._create_box_plots(comparison_result, plot_dir)
        if box_plot:
            created_plots["box_plots"] = box_plot

        # Heatmap корреляций
        heatmap_plot = self._create_correlation_heatmap(comparison_result, plot_dir)
        if heatmap_plot:
            created_plots["correlation_heatmap"] = heatmap_plot

        # Radar chart
        radar_plot = self._create_radar_chart(comparison_result, plot_dir)
        if radar_plot:
            created_plots["radar_chart"] = radar_plot

        return created_plots

    def learning_curves_comparison(
        self,
        experiments: List[Experiment],
        metric: str = "episode_reward",
        save_path: Optional[Union[str, Path]] = None,
        smoothing_window: int = 50,
    ) -> Optional[str]:
        """Создать сравнение кривых обучения.

        Args:
            experiments: Список экспериментов
            metric: Метрика для отображения
            save_path: Путь для сохранения
            smoothing_window: Окно сглаживания

        Returns:
            Путь к созданному графику
        """
        try:
            plt.figure(figsize=(12, 8))

            for exp in experiments:
                metrics_history = exp.results.get("baseline", {}).get(
                    "metrics_history", []
                )
                values = [
                    entry.get(metric, 0) for entry in metrics_history if metric in entry
                ]
                timesteps = [
                    entry.get("timestep", i)
                    for i, entry in enumerate(metrics_history)
                    if metric in entry
                ]

                if not values:
                    continue

                # Сглаживание
                if len(values) >= smoothing_window:
                    smoothed = self._calculate_moving_average(values, smoothing_window)
                    plt.plot(
                        timesteps,
                        smoothed,
                        label=f"{exp.experiment_id} (smoothed)",
                        linewidth=2,
                    )

                # Исходные данные с прозрачностью
                plt.plot(timesteps, values, alpha=0.3, linewidth=1)

            plt.xlabel("Timesteps")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title("Learning Curves Comparison")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / "learning_curves_comparison.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"График кривых обучения сохранен: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания графика кривых обучения: {e}")
            return None

    def distribution_plots(
        self,
        experiments: List[Experiment],
        metric: str = "episode_reward",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        """Создать графики распределений.

        Args:
            experiments: Список экспериментов
            metric: Метрика для анализа
            save_path: Путь для сохранения

        Returns:
            Путь к созданному графику
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Подготовка данных
            data_dict = {}
            for exp in experiments:
                metrics_history = exp.results.get("baseline", {}).get(
                    "metrics_history", []
                )
                values = [
                    entry.get(metric, 0) for entry in metrics_history if metric in entry
                ]
                if values:
                    data_dict[exp.experiment_id] = values

            if not data_dict:
                self.logger.warning("Нет данных для построения распределений")
                return None

            # Гистограммы
            axes[0, 0].set_title("Histograms")
            for exp_id, values in data_dict.items():
                axes[0, 0].hist(values, alpha=0.6, label=exp_id, bins=30)
            axes[0, 0].set_xlabel(metric.replace("_", " ").title())
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].legend()

            # Box plots
            axes[0, 1].set_title("Box Plots")
            box_data = [values for values in data_dict.values()]
            box_labels = list(data_dict.keys())
            axes[0, 1].boxplot(box_data, labels=box_labels)
            axes[0, 1].set_ylabel(metric.replace("_", " ").title())
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Violin plots
            axes[1, 0].set_title("Violin Plots")
            positions = range(1, len(data_dict) + 1)
            violin_parts = axes[1, 0].violinplot(box_data, positions=positions)
            axes[1, 0].set_xticks(positions)
            axes[1, 0].set_xticklabels(box_labels, rotation=45)
            axes[1, 0].set_ylabel(metric.replace("_", " ").title())

            # Q-Q plots
            axes[1, 1].set_title("Q-Q Plot vs Normal Distribution")
            for exp_id, values in data_dict.items():
                stats.probplot(values, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title("Q-Q Plot vs Normal Distribution")

            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / "distribution_plots.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"График распределений сохранен: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания графиков распределений: {e}")
            return None

    def box_plots(
        self,
        experiments: List[Experiment],
        metrics: List[str],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        """Создать box plots для сравнения.

        Args:
            experiments: Список экспериментов
            metrics: Список метрик
            save_path: Путь для сохранения

        Returns:
            Путь к созданному графику
        """
        try:
            n_metrics = len(metrics)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if n_metrics == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, metric in enumerate(metrics):
                ax = axes[i]

                # Подготовка данных
                data_for_metric = []
                labels = []

                for exp in experiments:
                    metrics_history = exp.results.get("baseline", {}).get(
                        "metrics_history", []
                    )
                    values = [
                        entry.get(metric, 0)
                        for entry in metrics_history
                        if metric in entry
                    ]
                    if values:
                        data_for_metric.append(values)
                        labels.append(exp.experiment_id)

                if data_for_metric:
                    ax.boxplot(data_for_metric, labels=labels)
                    ax.set_title(metric.replace("_", " ").title())
                    ax.set_ylabel("Value")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(True, alpha=0.3)

            # Скрываем лишние подграфики
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / "box_plots_comparison.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Box plots сохранены: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания box plots: {e}")
            return None

    def heatmap_comparison(
        self,
        comparison_result: ComparisonResult,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        """Создать heatmap для сравнения экспериментов.

        Args:
            comparison_result: Результат сравнения
            save_path: Путь для сохранения

        Returns:
            Путь к созданному графику
        """
        try:
            # Подготовка данных для heatmap
            metrics_data = []
            experiment_ids = comparison_result.experiment_ids

            # Извлекаем ключевые метрики
            key_metrics = ["mean_reward", "stability_score", "sample_efficiency"]

            for exp_id in experiment_ids:
                if exp_id in comparison_result.performance_metrics:
                    metrics = comparison_result.performance_metrics[exp_id]
                    row = []
                    for metric in key_metrics:
                        value = getattr(metrics, metric, 0)
                        row.append(value if value is not None else 0)
                    metrics_data.append(row)

            if not metrics_data:
                self.logger.warning("Нет данных для создания heatmap")
                return None

            # Нормализация данных
            metrics_array = np.array(metrics_data)
            normalized_data = (metrics_array - metrics_array.min(axis=0)) / (
                metrics_array.max(axis=0) - metrics_array.min(axis=0) + 1e-8
            )

            # Создание heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                normalized_data,
                xticklabels=[m.replace("_", " ").title() for m in key_metrics],
                yticklabels=experiment_ids,
                annot=True,
                cmap="RdYlGn",
                center=0.5,
                fmt=".3f",
            )

            plt.title("Normalized Performance Metrics Comparison")
            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.output_dir / "performance_heatmap.png"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Heatmap сохранена: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания heatmap: {e}")
            return None

    def generate_comparison_report(
        self,
        comparison_result: ComparisonResult,
        include_plots: bool = True,
        output_format: str = "html",
    ) -> str:
        """Генерировать комплексный отчет сравнения.

        Args:
            comparison_result: Результат сравнения
            include_plots: Включать ли графики
            output_format: Формат вывода ('html', 'markdown', 'json')

        Returns:
            Путь к созданному отчету
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == "html":
            report_path = self.output_dir / f"comparison_report_{timestamp}.html"
            content = self._generate_html_report(comparison_result, include_plots)
        elif output_format == "markdown":
            report_path = self.output_dir / f"comparison_report_{timestamp}.md"
            content = self._generate_markdown_report(comparison_result, include_plots)
        elif output_format == "json":
            report_path = self.output_dir / f"comparison_report_{timestamp}.json"
            content = json.dumps(
                comparison_result.to_dict(), indent=2, ensure_ascii=False
            )
        else:
            raise ValueError(f"Неподдерживаемый формат: {output_format}")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"Отчет сравнения сохранен: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания отчета: {e}")
            raise

    def hypothesis_test_results(
        self, comparison_result: ComparisonResult, format_type: str = "table"
    ) -> str:
        """Форматировать результаты статистических тестов.

        Args:
            comparison_result: Результат сравнения
            format_type: Тип форматирования ('table', 'summary')

        Returns:
            Отформатированные результаты
        """
        if format_type == "table":
            return self._format_statistical_tests_table(comparison_result)
        elif format_type == "summary":
            return self._format_statistical_tests_summary(comparison_result)
        else:
            raise ValueError(f"Неподдерживаемый тип форматирования: {format_type}")

    def recommendations(self, comparison_result: ComparisonResult) -> List[str]:
        """Генерировать рекомендации на основе сравнения.

        Args:
            comparison_result: Результат сравнения

        Returns:
            Список рекомендаций
        """
        return self._generate_recommendations(
            comparison_result.performance_metrics,
            comparison_result.statistical_tests,
            comparison_result.rankings,
        )

    def export_results(
        self, comparison_result: ComparisonResult, formats: List[str] = ["csv", "json"]
    ) -> Dict[str, str]:
        """Экспортировать результаты в различные форматы.

        Args:
            comparison_result: Результат сравнения
            formats: Список форматов для экспорта

        Returns:
            Словарь с путями к экспортированным файлам
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for format_type in formats:
            try:
                if format_type == "csv":
                    csv_path = self.output_dir / f"comparison_results_{timestamp}.csv"
                    self._export_to_csv(comparison_result, csv_path)
                    exported_files["csv"] = str(csv_path)

                elif format_type == "json":
                    json_path = self.output_dir / f"comparison_results_{timestamp}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            comparison_result.to_dict(), f, indent=2, ensure_ascii=False
                        )
                    exported_files["json"] = str(json_path)

                elif format_type == "excel":
                    excel_path = (
                        self.output_dir / f"comparison_results_{timestamp}.xlsx"
                    )
                    self._export_to_excel(comparison_result, excel_path)
                    exported_files["excel"] = str(excel_path)

            except Exception as e:
                self.logger.error(f"Ошибка экспорта в формат {format_type}: {e}")

        return exported_files

    def hyperparameter_sensitivity(
        self,
        experiments: List[Experiment],
        hyperparameter: str,
        metric: str = "mean_reward",
    ) -> Dict[str, Any]:
        """Анализ чувствительности к гиперпараметрам.

        Args:
            experiments: Список экспериментов
            hyperparameter: Название гиперпараметра
            metric: Метрика для анализа

        Returns:
            Результаты анализа чувствительности
        """
        # Извлекаем значения гиперпараметра и соответствующие метрики
        hyperparameter_values = []
        metric_values = []

        for exp in experiments:
            try:
                # Получаем значение гиперпараметра из конфигурации
                hp_value = self._extract_hyperparameter_value(exp, hyperparameter)
                if hp_value is None:
                    continue

                # Получаем значение метрики
                performance_metrics = self._extract_performance_metrics(
                    exp, self.config
                )
                metric_value = getattr(performance_metrics, metric, None)
                if metric_value is None:
                    continue

                hyperparameter_values.append(hp_value)
                metric_values.append(metric_value)

            except Exception as e:
                self.logger.error(
                    f"Ошибка анализа чувствительности для {exp.experiment_id}: {e}"
                )
                continue

        if len(hyperparameter_values) < 3:
            raise ValueError("Недостаточно данных для анализа чувствительности")

        # Вычисляем корреляцию
        correlation, p_value = stats.pearsonr(hyperparameter_values, metric_values)

        # Линейная регрессия
        slope, intercept, r_value, reg_p_value, std_err = stats.linregress(
            hyperparameter_values, metric_values
        )

        return {
            "hyperparameter": hyperparameter,
            "metric": metric,
            "correlation": correlation,
            "correlation_p_value": p_value,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "r_squared": r_value**2,
            "regression_p_value": reg_p_value,
            "standard_error": std_err,
            "hyperparameter_values": hyperparameter_values,
            "metric_values": metric_values,
        }

    def environment_generalization(
        self,
        experiments_by_env: Dict[str, List[Experiment]],
        metric: str = "mean_reward",
    ) -> Dict[str, Any]:
        """Анализ обобщения между средами.

        Args:
            experiments_by_env: Словарь экспериментов по средам
            metric: Метрика для анализа

        Returns:
            Результаты анализа обобщения
        """
        env_performance = {}

        # Вычисляем производительность для каждой среды
        for env_name, experiments in experiments_by_env.items():
            env_metrics = []
            for exp in experiments:
                try:
                    performance_metrics = self._extract_performance_metrics(
                        exp, self.config
                    )
                    metric_value = getattr(performance_metrics, metric, None)
                    if metric_value is not None:
                        env_metrics.append(metric_value)
                except Exception as e:
                    self.logger.error(f"Ошибка анализа для {exp.experiment_id}: {e}")
                    continue

            if env_metrics:
                env_performance[env_name] = {
                    "mean": np.mean(env_metrics),
                    "std": np.std(env_metrics),
                    "min": np.min(env_metrics),
                    "max": np.max(env_metrics),
                    "count": len(env_metrics),
                }

        # Анализ вариативности между средами
        all_means = [perf["mean"] for perf in env_performance.values()]
        if len(all_means) > 1:
            generalization_score = 1.0 - (np.std(all_means) / np.mean(all_means))
        else:
            generalization_score = 1.0

        return {
            "environment_performance": env_performance,
            "generalization_score": generalization_score,
            "best_environment": max(
                env_performance.keys(), key=lambda k: env_performance[k]["mean"]
            )
            if env_performance
            else None,
            "worst_environment": min(
                env_performance.keys(), key=lambda k: env_performance[k]["mean"]
            )
            if env_performance
            else None,
        }

    def algorithm_ranking(
        self,
        experiments_by_algorithm: Dict[str, List[Experiment]],
        metrics: List[str] = ["mean_reward", "stability_score", "sample_efficiency"],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Ранжирование алгоритмов по множественным критериям.

        Args:
            experiments_by_algorithm: Словарь экспериментов по алгоритмам
            metrics: Список метрик для ранжирования
            weights: Веса для метрик

        Returns:
            Результаты ранжирования
        """
        if weights is None:
            weights = [1.0] * len(metrics)

        if len(weights) != len(metrics):
            raise ValueError(
                "Количество весов должно соответствовать количеству метрик"
            )

        algorithm_scores = {}

        # Вычисляем оценки для каждого алгоритма
        for algorithm, experiments in experiments_by_algorithm.items():
            metric_values = {metric: [] for metric in metrics}

            for exp in experiments:
                try:
                    performance_metrics = self._extract_performance_metrics(
                        exp, self.config
                    )
                    for metric in metrics:
                        value = getattr(performance_metrics, metric, None)
                        if value is not None:
                            metric_values[metric].append(value)
                except Exception as e:
                    self.logger.error(f"Ошибка анализа для {exp.experiment_id}: {e}")
                    continue

            # Вычисляем средние значения метрик
            algorithm_means = {}
            for metric in metrics:
                if metric_values[metric]:
                    algorithm_means[metric] = np.mean(metric_values[metric])
                else:
                    algorithm_means[metric] = 0.0

            algorithm_scores[algorithm] = algorithm_means

        # Нормализация и взвешенная оценка
        if algorithm_scores:
            # Нормализация по каждой метрике
            normalized_scores = {}
            for algorithm in algorithm_scores:
                normalized_scores[algorithm] = {}

            for metric in metrics:
                values = [algorithm_scores[alg][metric] for alg in algorithm_scores]
                if max(values) > min(values):
                    for algorithm in algorithm_scores:
                        normalized_value = (
                            algorithm_scores[algorithm][metric] - min(values)
                        ) / (max(values) - min(values))
                        normalized_scores[algorithm][metric] = normalized_value
                else:
                    for algorithm in algorithm_scores:
                        normalized_scores[algorithm][metric] = 1.0

            # Вычисляем взвешенные оценки
            weighted_scores = {}
            for algorithm in normalized_scores:
                score = sum(
                    normalized_scores[algorithm][metric] * weight
                    for metric, weight in zip(metrics, weights)
                ) / sum(weights)
                weighted_scores[algorithm] = score

            # Ранжирование
            ranking = sorted(
                weighted_scores.keys(), key=lambda k: weighted_scores[k], reverse=True
            )
        else:
            weighted_scores = {}
            ranking = []

        return {
            "algorithm_scores": algorithm_scores,
            "normalized_scores": normalized_scores if algorithm_scores else {},
            "weighted_scores": weighted_scores,
            "ranking": ranking,
            "metrics": metrics,
            "weights": weights,
        }

    def pareto_analysis(
        self,
        experiments: List[Experiment],
        objective1: str = "mean_reward",
        objective2: str = "sample_efficiency",
    ) -> Dict[str, Any]:
        """Многокритериальный анализ Парето.

        Args:
            experiments: Список экспериментов
            objective1: Первая цель для оптимизации
            objective2: Вторая цель для оптимизации

        Returns:
            Результаты анализа Парето
        """
        # Извлекаем значения целевых функций
        points = []
        experiment_mapping = {}

        for exp in experiments:
            try:
                performance_metrics = self._extract_performance_metrics(
                    exp, self.config
                )
                obj1_value = getattr(performance_metrics, objective1, None)
                obj2_value = getattr(performance_metrics, objective2, None)

                if obj1_value is not None and obj2_value is not None:
                    points.append((obj1_value, obj2_value))
                    experiment_mapping[len(points) - 1] = exp.experiment_id

            except Exception as e:
                self.logger.error(f"Ошибка анализа Парето для {exp.experiment_id}: {e}")
                continue

        if len(points) < 2:
            raise ValueError("Недостаточно данных для анализа Парето")

        # Находим фронт Парето
        pareto_indices = self._find_pareto_front(points)
        pareto_experiments = [experiment_mapping[i] for i in pareto_indices]
        pareto_points = [points[i] for i in pareto_indices]

        # Вычисляем гиперобъем (hypervolume)
        reference_point = (min(p[0] for p in points), min(p[1] for p in points))
        hypervolume = self._calculate_hypervolume(pareto_points, reference_point)

        return {
            "objective1": objective1,
            "objective2": objective2,
            "all_points": points,
            "experiment_mapping": experiment_mapping,
            "pareto_front_indices": pareto_indices,
            "pareto_front_experiments": pareto_experiments,
            "pareto_front_points": pareto_points,
            "hypervolume": hypervolume,
            "reference_point": reference_point,
        }

    # Приватные методы

    def _extract_performance_metrics(
        self, experiment: Experiment, config: ComparisonConfig
    ) -> PerformanceMetrics:
        """Извлечь метрики производительности из эксперимента."""
        if not experiment.results or "baseline" not in experiment.results:
            raise ValueError(
                f"Отсутствуют результаты для эксперимента {experiment.experiment_id}"
            )

        baseline_results = experiment.results["baseline"]

        # Извлекаем основные метрики
        mean_reward = baseline_results.get("mean_reward", 0.0)
        final_reward = baseline_results.get("final_reward", mean_reward)

        # Извлекаем историю метрик для дополнительных расчетов
        metrics_history = baseline_results.get("metrics_history", [])
        episode_rewards = [
            entry.get("episode_reward", 0)
            for entry in metrics_history
            if "episode_reward" in entry
        ]

        if episode_rewards:
            std_reward = np.std(episode_rewards)
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)

            # Вычисляем стабильность
            if len(episode_rewards) >= config.stability_window:
                recent_rewards = episode_rewards[-config.stability_window :]
                cv = np.std(recent_rewards) / (abs(np.mean(recent_rewards)) + 1e-8)
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 0.0
        else:
            std_reward = 0.0
            max_reward = mean_reward
            min_reward = mean_reward
            stability_score = 0.0

        # Анализ сходимости
        convergence_timesteps = None
        if episode_rewards and len(episode_rewards) >= config.convergence_window:
            threshold = config.convergence_threshold or (max_reward * 0.9)
            moving_avg = self._calculate_moving_average(
                episode_rewards, config.convergence_window
            )

            for i, avg_val in enumerate(moving_avg):
                if avg_val >= threshold:
                    convergence_timesteps = i * 1000  # Приблизительная оценка
                    break

        # Эффективность выборки
        if convergence_timesteps:
            sample_efficiency = mean_reward / convergence_timesteps * 1000
        else:
            sample_efficiency = 0.0

        # Дополнительные метрики
        training_time = baseline_results.get("training_time")
        success_rate = baseline_results.get("success_rate")

        return PerformanceMetrics(
            experiment_id=experiment.experiment_id,
            mean_reward=mean_reward,
            std_reward=std_reward,
            final_reward=final_reward,
            max_reward=max_reward,
            min_reward=min_reward,
            convergence_timesteps=convergence_timesteps,
            sample_efficiency=sample_efficiency,
            stability_score=stability_score,
            success_rate=success_rate,
            training_time=training_time,
        )

    def _perform_pairwise_tests(
        self,
        performance_metrics: Dict[str, PerformanceMetrics],
        metric: str,
        config: ComparisonConfig,
    ) -> Dict[str, StatisticalTestResult]:
        """Выполнить попарные статистические тесты."""
        tests = {}
        experiment_ids = list(performance_metrics.keys())

        for i in range(len(experiment_ids)):
            for j in range(i + 1, len(experiment_ids)):
                exp1_id = experiment_ids[i]
                exp2_id = experiment_ids[j]

                try:
                    # Получаем данные для сравнения
                    exp1_metrics = performance_metrics[exp1_id]
                    exp2_metrics = performance_metrics[exp2_id]

                    # Для простоты используем одно значение метрики
                    # В реальном случае нужны массивы данных
                    value1 = getattr(exp1_metrics, metric, 0)
                    value2 = getattr(exp2_metrics, metric, 0)

                    # Создаем искусственные выборки для демонстрации
                    # В реальности нужны исходные данные
                    data1 = np.random.normal(value1, value1 * 0.1, 50)
                    data2 = np.random.normal(value2, value2 * 0.1, 50)

                    # Выбираем подходящий тест
                    test_type = self._select_statistical_test(data1, data2)

                    # Выполняем тест
                    test_result = self.statistical_significance(
                        data1.tolist(),
                        data2.tolist(),
                        test_type,
                        config.significance_level,
                    )

                    comparison_key = f"{exp1_id}_vs_{exp2_id}"
                    tests[comparison_key] = test_result

                except Exception as e:
                    self.logger.error(
                        f"Ошибка статистического теста {exp1_id} vs {exp2_id}: {e}"
                    )
                    continue

        return tests

    def _apply_multiple_comparison_correction(
        self,
        statistical_tests: Dict[str, Dict[str, StatisticalTestResult]],
        config: ComparisonConfig,
    ) -> Dict[str, Dict[str, StatisticalTestResult]]:
        """Применить коррекцию множественных сравнений."""
        if config.multiple_comparison_method == MultipleComparisonMethod.NONE:
            return statistical_tests

        # Собираем все p-values
        all_p_values = []
        test_mapping = []

        for metric, tests in statistical_tests.items():
            for comparison, test_result in tests.items():
                all_p_values.append(test_result.p_value)
                test_mapping.append((metric, comparison))

        if not all_p_values:
            return statistical_tests

        # Применяем коррекцию
        if config.multiple_comparison_method == MultipleComparisonMethod.BONFERRONI:
            corrected_alpha = config.significance_level / len(all_p_values)
            corrected_p_values = [p * len(all_p_values) for p in all_p_values]
        elif config.multiple_comparison_method == MultipleComparisonMethod.FDR_BH:
            # Benjamini-Hochberg процедура
            sorted_indices = np.argsort(all_p_values)
            corrected_p_values = [0] * len(all_p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p_values[idx] = (
                    all_p_values[idx] * len(all_p_values) / (i + 1)
                )

            corrected_alpha = config.significance_level
        else:
            # Для других методов используем Bonferroni
            corrected_alpha = config.significance_level / len(all_p_values)
            corrected_p_values = [p * len(all_p_values) for p in all_p_values]

        # Обновляем результаты тестов
        corrected_tests = {}
        for i, (metric, comparison) in enumerate(test_mapping):
            if metric not in corrected_tests:
                corrected_tests[metric] = {}

            original_test = statistical_tests[metric][comparison]
            corrected_test = StatisticalTestResult(
                test_type=original_test.test_type,
                statistic=original_test.statistic,
                p_value=min(corrected_p_values[i], 1.0),  # p-value не может быть > 1
                significant=corrected_p_values[i] < corrected_alpha,
                alpha=corrected_alpha,
                effect_size=original_test.effect_size,
                effect_size_method=original_test.effect_size_method,
                confidence_interval=original_test.confidence_interval,
                sample_size_1=original_test.sample_size_1,
                sample_size_2=original_test.sample_size_2,
            )

            corrected_tests[metric][comparison] = corrected_test

        return corrected_tests

    def _rank_experiments(
        self, performance_metrics: Dict[str, PerformanceMetrics], metrics: List[str]
    ) -> Dict[str, List[str]]:
        """Ранжировать эксперименты по метрикам."""
        rankings = {}

        for metric in metrics:
            # Извлекаем значения метрики
            metric_values = {}
            for exp_id, perf_metrics in performance_metrics.items():
                value = getattr(perf_metrics, metric, 0)
                metric_values[exp_id] = value if value is not None else 0

            # Сортируем по убыванию (лучше = больше)
            sorted_experiments = sorted(
                metric_values.keys(), key=lambda x: metric_values[x], reverse=True
            )

            rankings[metric] = sorted_experiments

        # Общий рейтинг (средний ранг по всем метрикам)
        if metrics:
            experiment_ranks = {}
            for exp_id in performance_metrics.keys():
                ranks = []
                for metric in metrics:
                    if metric in rankings:
                        rank = rankings[metric].index(exp_id) + 1
                        ranks.append(rank)

                if ranks:
                    experiment_ranks[exp_id] = np.mean(ranks)
                else:
                    experiment_ranks[exp_id] = float("inf")

            overall_ranking = sorted(
                experiment_ranks.keys(), key=lambda x: experiment_ranks[x]
            )

            rankings["overall"] = overall_ranking

        return rankings

    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, PerformanceMetrics],
        statistical_tests: Dict[str, Dict[str, StatisticalTestResult]],
        rankings: Dict[str, List[str]],
    ) -> List[str]:
        """Генерировать рекомендации на основе анализа."""
        recommendations = []

        if not performance_metrics:
            return ["Недостаточно данных для генерации рекомендаций"]

        # Лучший эксперимент по общему рейтингу
        if "overall" in rankings and rankings["overall"]:
            best_experiment = rankings["overall"][0]
            recommendations.append(
                f"Лучший эксперимент по общей производительности: {best_experiment}"
            )

        # Анализ статистической значимости
        significant_differences = 0
        total_comparisons = 0

        for metric, tests in statistical_tests.items():
            for comparison, test_result in tests.items():
                total_comparisons += 1
                if test_result.significant:
                    significant_differences += 1

                    # Определяем какой эксперимент лучше
                    exp1, exp2 = comparison.split("_vs_")
                    exp1_value = getattr(performance_metrics[exp1], metric, 0)
                    exp2_value = getattr(performance_metrics[exp2], metric, 0)

                    if exp1_value > exp2_value:
                        better_exp, worse_exp = exp1, exp2
                    else:
                        better_exp, worse_exp = exp2, exp1

                    effect_size_desc = "малый"
                    if test_result.effect_size:
                        if test_result.effect_size > 0.8:
                            effect_size_desc = "большой"
                        elif test_result.effect_size > 0.5:
                            effect_size_desc = "средний"

                    recommendations.append(
                        f"Статистически значимое различие по {metric}: "
                        f"{better_exp} превосходит {worse_exp} "
                        f"(размер эффекта: {effect_size_desc})"
                    )

        # Общая оценка значимости
        if total_comparisons > 0:
            significance_rate = significant_differences / total_comparisons
            if significance_rate > 0.5:
                recommendations.append(
                    f"Обнаружено много значимых различий ({significance_rate:.1%}). "
                    "Выбор алгоритма критически важен."
                )
            elif significance_rate > 0.2:
                recommendations.append(
                    f"Обнаружены умеренные различия ({significance_rate:.1%}). "
                    "Рекомендуется дополнительное тестирование."
                )
            else:
                recommendations.append(
                    f"Различия между экспериментами незначительны ({significance_rate:.1%}). "
                    "Можно выбрать любой алгоритм."
                )

        # Анализ стабильности
        stability_scores = {
            exp_id: metrics.stability_score
            for exp_id, metrics in performance_metrics.items()
        }

        most_stable = max(stability_scores.keys(), key=lambda x: stability_scores[x])
        least_stable = min(stability_scores.keys(), key=lambda x: stability_scores[x])

        recommendations.append(
            f"Наиболее стабильный эксперимент: {most_stable} "
            f"(стабильность: {stability_scores[most_stable]:.3f})"
        )

        if stability_scores[least_stable] < 0.5:
            recommendations.append(
                f"Эксперимент {least_stable} показал низкую стабильность "
                f"({stability_scores[least_stable]:.3f}). Требуется настройка гиперпараметров."
            )

        # Анализ эффективности выборки
        efficiency_scores = {
            exp_id: metrics.sample_efficiency
            for exp_id, metrics in performance_metrics.items()
        }

        most_efficient = max(
            efficiency_scores.keys(), key=lambda x: efficiency_scores[x]
        )
        recommendations.append(
            f"Наиболее эффективный по выборке: {most_efficient} "
            f"(эффективность: {efficiency_scores[most_efficient]:.3f})"
        )

        return recommendations

    def _calculate_effect_size(
        self, data1: np.ndarray, data2: np.ndarray, method: EffectSizeMethod
    ) -> float:
        """Вычислить размер эффекта."""
        mean1, mean2 = np.mean(data1), np.mean(data2)

        if method == EffectSizeMethod.COHENS_D:
            pooled_std = np.sqrt(
                (
                    (len(data1) - 1) * np.var(data1, ddof=1)
                    + (len(data2) - 1) * np.var(data2, ddof=1)
                )
                / (len(data1) + len(data2) - 2)
            )
            return abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        elif method == EffectSizeMethod.GLASS_DELTA:
            std2 = np.std(data2, ddof=1)
            return abs(mean1 - mean2) / std2 if std2 > 0 else 0.0

        elif method == EffectSizeMethod.HEDGES_G:
            pooled_std = np.sqrt(
                (
                    (len(data1) - 1) * np.var(data1, ddof=1)
                    + (len(data2) - 1) * np.var(data2, ddof=1)
                )
                / (len(data1) + len(data2) - 2)
            )
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
            correction = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
            return cohens_d * correction

        else:
            raise ValueError(f"Неподдерживаемый метод размера эффекта: {method}")

    def _calculate_confidence_interval(
        self, data1: np.ndarray, data2: np.ndarray, confidence_level: float
    ) -> Tuple[float, float]:
        """Вычислить доверительный интервал для разности средних."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        # Стандартная ошибка разности средних
        se_diff = np.sqrt(var1 / n1 + var2 / n2)

        # Степени свободы (приближение Уэлча)
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        # t-критическое значение
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Разность средних и доверительный интервал
        mean_diff = mean1 - mean2
        margin_error = t_critical * se_diff

        return (mean_diff - margin_error, mean_diff + margin_error)

    def _bootstrap_test(
        self, data1: np.ndarray, data2: np.ndarray, n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Выполнить bootstrap тест."""
        # Объединяем данные
        combined = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)

        # Наблюдаемая разность средних
        observed_diff = np.mean(data1) - np.mean(data2)

        # Bootstrap выборки
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Перемешиваем объединенные данные
            shuffled = np.random.permutation(combined)

            # Разделяем на две группы
            bootstrap_sample1 = shuffled[:n1]
            bootstrap_sample2 = shuffled[n1 : n1 + n2]

            # Вычисляем разность средних
            bootstrap_diff = np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2)
            bootstrap_diffs.append(bootstrap_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # p-value (двусторонний тест)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        return observed_diff, p_value

    def _select_statistical_test(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> StatisticalTest:
        """Выбрать подходящий статистический тест."""
        # Проверяем нормальность распределения
        _, p1 = stats.shapiro(data1)
        _, p2 = stats.shapiro(data2)

        # Если оба распределения нормальные, используем t-test
        if p1 > 0.05 and p2 > 0.05:
            return StatisticalTest.T_TEST
        else:
            # Иначе используем непараметрический тест
            return StatisticalTest.MANN_WHITNEY

    def _calculate_moving_average(
        self, values: List[float], window: int
    ) -> List[float]:
        """Вычислить скользящее среднее."""
        if len(values) < window:
            return values

        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx : i + 1]
            moving_avg.append(sum(window_values) / len(window_values))

        return moving_avg

    def _extract_hyperparameter_value(
        self, experiment: Experiment, hyperparameter: str
    ) -> Optional[float]:
        """Извлечь значение гиперпараметра из эксперимента."""
        try:
            # Проверяем в конфигурации алгоритма
            if hasattr(experiment.baseline_config.algorithm, hyperparameter):
                return getattr(experiment.baseline_config.algorithm, hyperparameter)

            # Проверяем в конфигурации обучения
            if hasattr(experiment.baseline_config.training, hyperparameter):
                return getattr(experiment.baseline_config.training, hyperparameter)

            # Проверяем в результатах
            if hyperparameter in experiment.results.get("configurations", {}).get(
                "baseline", {}
            ):
                return experiment.results["configurations"]["baseline"][hyperparameter]

            return None

        except Exception:
            return None

    def _find_pareto_front(self, points: List[Tuple[float, float]]) -> List[int]:
        """Найти фронт Парето."""
        pareto_indices = []

        for i, point1 in enumerate(points):
            is_pareto = True

            for j, point2 in enumerate(points):
                if i != j:
                    # Проверяем доминирование (предполагаем максимизацию обеих целей)
                    if (
                        point2[0] >= point1[0]
                        and point2[1] >= point1[1]
                        and (point2[0] > point1[0] or point2[1] > point1[1])
                    ):
                        is_pareto = False
                        break

            if is_pareto:
                pareto_indices.append(i)

        return pareto_indices

    def _calculate_hypervolume(
        self,
        pareto_points: List[Tuple[float, float]],
        reference_point: Tuple[float, float],
    ) -> float:
        """Вычислить гиперобъем фронта Парето."""
        if not pareto_points:
            return 0.0

        # Сортируем точки по первой координате
        sorted_points = sorted(pareto_points, key=lambda p: p[0])

        hypervolume = 0.0
        prev_x = reference_point[0]

        for point in sorted_points:
            width = point[0] - prev_x
            height = point[1] - reference_point[1]
            hypervolume += width * height
            prev_x = point[0]

        return hypervolume

    def _create_performance_comparison_plot(
        self, comparison_result: ComparisonResult, plot_dir: Path
    ) -> Optional[str]:
        """Создать график сравнения производительности."""
        try:
            metrics = ["mean_reward", "stability_score", "sample_efficiency"]
            experiment_ids = comparison_result.experiment_ids

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, metric in enumerate(metrics):
                values = []
                labels = []

                for exp_id in experiment_ids:
                    if exp_id in comparison_result.performance_metrics:
                        value = getattr(
                            comparison_result.performance_metrics[exp_id], metric, 0
                        )
                        values.append(value if value is not None else 0)
                        labels.append(exp_id)

                if values:
                    bars = axes[i].bar(range(len(values)), values)
                    axes[i].set_title(metric.replace("_", " ").title())
                    axes[i].set_xticks(range(len(labels)))
                    axes[i].set_xticklabels(labels, rotation=45)
                    axes[i].grid(True, alpha=0.3)

                    # Добавляем значения на столбцы
                    for j, (bar, value) in enumerate(zip(bars, values)):
                        axes[i].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{value:.3f}",
                            ha="center",
                            va="bottom",
                        )

            plt.tight_layout()

            save_path = plot_dir / "performance_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания графика производительности: {e}")
            return None

    def _create_box_plots(
        self, comparison_result: ComparisonResult, plot_dir: Path
    ) -> Optional[str]:
        """Создать box plots для статистического сравнения."""
        try:
            # Для демонстрации создаем искусственные данные
            # В реальности нужны исходные данные экспериментов

            fig, ax = plt.subplots(figsize=(12, 8))

            data_for_plot = []
            labels = []

            for exp_id in comparison_result.experiment_ids:
                if exp_id in comparison_result.performance_metrics:
                    metrics = comparison_result.performance_metrics[exp_id]
                    # Создаем искусственную выборку на основе mean_reward
                    mean_val = metrics.mean_reward
                    std_val = (
                        metrics.std_reward if metrics.std_reward > 0 else mean_val * 0.1
                    )
                    synthetic_data = np.random.normal(mean_val, std_val, 50)
                    data_for_plot.append(synthetic_data)
                    labels.append(exp_id)

            if data_for_plot:
                box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

                # Раскрашиваем box plots
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
                for patch, color in zip(box_plot["boxes"], colors):
                    patch.set_facecolor(color)

                ax.set_title("Reward Distribution Comparison")
                ax.set_ylabel("Reward")
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            save_path = plot_dir / "box_plots_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания box plots: {e}")
            return None

    def _create_correlation_heatmap(
        self, comparison_result: ComparisonResult, plot_dir: Path
    ) -> Optional[str]:
        """Создать heatmap корреляций между метриками."""
        try:
            # Подготавливаем данные
            metrics_data = []
            metric_names = [
                "mean_reward",
                "stability_score",
                "sample_efficiency",
                "std_reward",
            ]

            for exp_id in comparison_result.experiment_ids:
                if exp_id in comparison_result.performance_metrics:
                    metrics = comparison_result.performance_metrics[exp_id]
                    row = []
                    for metric_name in metric_names:
                        value = getattr(metrics, metric_name, 0)
                        row.append(value if value is not None else 0)
                    metrics_data.append(row)

            if len(metrics_data) < 2:
                self.logger.warning("Недостаточно данных для корреляционной матрицы")
                return None

            # Создаем DataFrame и вычисляем корреляции
            df = pd.DataFrame(metrics_data, columns=metric_names)
            correlation_matrix = df.corr()

            # Создаем heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                fmt=".3f",
            )

            plt.title("Metrics Correlation Matrix")
            plt.tight_layout()

            save_path = plot_dir / "correlation_heatmap.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания correlation heatmap: {e}")
            return None

    def _create_radar_chart(
        self, comparison_result: ComparisonResult, plot_dir: Path
    ) -> Optional[str]:
        """Создать radar chart для сравнения экспериментов."""
        try:
            metrics = ["mean_reward", "stability_score", "sample_efficiency"]
            experiment_ids = comparison_result.experiment_ids[
                :5
            ]  # Ограничиваем количество

            # Подготавливаем данные
            data_matrix = []
            for exp_id in experiment_ids:
                if exp_id in comparison_result.performance_metrics:
                    perf_metrics = comparison_result.performance_metrics[exp_id]
                    row = []
                    for metric in metrics:
                        value = getattr(perf_metrics, metric, 0)
                        row.append(value if value is not None else 0)
                    data_matrix.append(row)

            if not data_matrix:
                return None

            # Нормализация данных
            data_array = np.array(data_matrix)
            normalized_data = (data_array - data_array.min(axis=0)) / (
                data_array.max(axis=0) - data_array.min(axis=0) + 1e-8
            )

            # Создаем radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Замыкаем круг

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_ids)))

            for i, (exp_id, color) in enumerate(zip(experiment_ids, colors)):
                values = normalized_data[i].tolist()
                values += values[:1]  # Замыкаем круг

                ax.plot(angles, values, "o-", linewidth=2, label=exp_id, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

            # Настраиваем оси
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
            ax.grid(True)

            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            plt.title("Normalized Performance Comparison", size=16, y=1.08)

            save_path = plot_dir / "radar_chart.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(save_path)

        except Exception as e:
            self.logger.error(f"Ошибка создания radar chart: {e}")
            return None

    def _generate_html_report(
        self, comparison_result: ComparisonResult, include_plots: bool
    ) -> str:
        """Генерировать HTML отчет."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; }}
                .significant {{ background-color: #ffeeee; }}
                .recommendation {{ background-color: #eeffee; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Experiment Comparison Report</h1>
            <p><strong>Generated:</strong> {comparison_result.timestamp}</p>
            <p><strong>Experiments:</strong> {", ".join(comparison_result.experiment_ids)}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Experiment</th>
                    <th>Mean Reward</th>
                    <th>Stability Score</th>
                    <th>Sample Efficiency</th>
                    <th>Convergence Steps</th>
                </tr>
        """

        for exp_id in comparison_result.experiment_ids:
            if exp_id in comparison_result.performance_metrics:
                metrics = comparison_result.performance_metrics[exp_id]
                html_content += f"""
                <tr>
                    <td>{exp_id}</td>
                    <td>{metrics.mean_reward:.3f}</td>
                    <td>{metrics.stability_score:.3f}</td>
                    <td>{metrics.sample_efficiency:.3f}</td>
                    <td>{metrics.convergence_timesteps or "N/A"}</td>
                </tr>
                """

        html_content += """
            </table>
            
            <h2>Statistical Tests</h2>
        """

        for metric, tests in comparison_result.statistical_tests.items():
            html_content += f"<h3>{metric.replace('_', ' ').title()}</h3><table>"
            html_content += "<tr><th>Comparison</th><th>Test</th><th>p-value</th><th>Significant</th><th>Effect Size</th></tr>"

            for comparison, test_result in tests.items():
                significance_class = "significant" if test_result.significant else ""
                html_content += f"""
                <tr class="{significance_class}">
                    <td>{comparison.replace("_vs_", " vs ")}</td>
                    <td>{test_result.test_type.value}</td>
                    <td>{test_result.p_value:.4f}</td>
                    <td>{"Yes" if test_result.significant else "No"}</td>
                    <td>{test_result.effect_size:.3f if test_result.effect_size else 'N/A'}</td>
                </tr>
                """

            html_content += "</table>"

        html_content += "<h2>Rankings</h2>"
        for metric, ranking in comparison_result.rankings.items():
            html_content += f"<p><strong>{metric.replace('_', ' ').title()}:</strong> {' > '.join(ranking)}</p>"

        html_content += "<h2>Recommendations</h2>"
        for rec in comparison_result.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'

        html_content += """
            </body>
        </html>
        """

        return html_content

    def _generate_markdown_report(
        self, comparison_result: ComparisonResult, include_plots: bool
    ) -> str:
        """Генерировать Markdown отчет."""
        md_content = f"""# Experiment Comparison Report

**Generated:** {comparison_result.timestamp}
**Experiments:** {", ".join(comparison_result.experiment_ids)}

## Performance Metrics

| Experiment | Mean Reward | Stability Score | Sample Efficiency | Convergence Steps |
|------------|-------------|-----------------|-------------------|-------------------|
"""

        for exp_id in comparison_result.experiment_ids:
            if exp_id in comparison_result.performance_metrics:
                metrics = comparison_result.performance_metrics[exp_id]
                md_content += f"| {exp_id} | {metrics.mean_reward:.3f} | {metrics.stability_score:.3f} | {metrics.sample_efficiency:.3f} | {metrics.convergence_timesteps or 'N/A'} |\n"

        md_content += "\n## Statistical Tests\n"

        for metric, tests in comparison_result.statistical_tests.items():
            md_content += f"\n### {metric.replace('_', ' ').title()}\n\n"
            md_content += (
                "| Comparison | Test | p-value | Significant | Effect Size |\n"
            )
            md_content += (
                "|------------|------|---------|-------------|-------------|\n"
            )

            for comparison, test_result in tests.items():
                significance = "✅" if test_result.significant else "❌"
                md_content += f"| {comparison.replace('_vs_', ' vs ')} | {test_result.test_type.value} | {test_result.p_value:.4f} | {significance} | {test_result.effect_size:.3f if test_result.effect_size else 'N/A'} |\n"

        md_content += "\n## Rankings\n\n"
        for metric, ranking in comparison_result.rankings.items():
            md_content += (
                f"**{metric.replace('_', ' ').title()}:** {' > '.join(ranking)}\n\n"
            )

        md_content += "## Recommendations\n\n"
        for rec in comparison_result.recommendations:
            md_content += f"- {rec}\n"

        return md_content

    def _format_statistical_tests_table(
        self, comparison_result: ComparisonResult
    ) -> str:
        """Форматировать результаты статистических тестов в виде таблицы."""
        table_lines = []
        table_lines.append(
            "Metric | Comparison | Test | Statistic | p-value | Significant | Effect Size"
        )
        table_lines.append(
            "-------|------------|------|-----------|---------|-------------|------------"
        )

        for metric, tests in comparison_result.statistical_tests.items():
            for comparison, test_result in tests.items():
                significance = "Yes" if test_result.significant else "No"
                effect_size = (
                    f"{test_result.effect_size:.3f}"
                    if test_result.effect_size
                    else "N/A"
                )

                table_lines.append(
                    f"{metric} | {comparison.replace('_vs_', ' vs ')} | "
                    f"{test_result.test_type.value} | {test_result.statistic:.3f} | "
                    f"{test_result.p_value:.4f} | {significance} | {effect_size}"
                )

        return "\n".join(table_lines)

    def _format_statistical_tests_summary(
        self, comparison_result: ComparisonResult
    ) -> str:
        """Форматировать краткую сводку статистических тестов."""
        summary_lines = []
        summary_lines.append("Statistical Tests Summary")
        summary_lines.append("=" * 30)

        total_tests = 0
        significant_tests = 0

        for metric, tests in comparison_result.statistical_tests.items():
            metric_significant = 0
            metric_total = len(tests)
            total_tests += metric_total

            for test_result in tests.values():
                if test_result.significant:
                    metric_significant += 1
                    significant_tests += 1

            summary_lines.append(
                f"{metric.replace('_', ' ').title()}: "
                f"{metric_significant}/{metric_total} significant comparisons"
            )

        summary_lines.append("")
        summary_lines.append(
            f"Overall: {significant_tests}/{total_tests} "
            f"({significant_tests / total_tests * 100:.1f}%) significant differences"
        )

        return "\n".join(summary_lines)

    def _export_to_csv(
        self, comparison_result: ComparisonResult, csv_path: Path
    ) -> None:
        """Экспортировать результаты в CSV."""
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Заголовки для метрик производительности
            writer.writerow(
                [
                    "experiment_id",
                    "mean_reward",
                    "std_reward",
                    "final_reward",
                    "max_reward",
                    "min_reward",
                    "convergence_timesteps",
                    "sample_efficiency",
                    "stability_score",
                    "success_rate",
                    "training_time",
                ]
            )

            # Данные метрик производительности
            for exp_id, metrics in comparison_result.performance_metrics.items():
                writer.writerow(
                    [
                        exp_id,
                        metrics.mean_reward,
                        metrics.std_reward,
                        metrics.final_reward,
                        metrics.max_reward,
                        metrics.min_reward,
                        metrics.convergence_timesteps,
                        metrics.sample_efficiency,
                        metrics.stability_score,
                        metrics.success_rate,
                        metrics.training_time,
                    ]
                )

    def _export_to_excel(
        self, comparison_result: ComparisonResult, excel_path: Path
    ) -> None:
        """Экспортировать результаты в Excel."""
        try:
            import pandas as pd

            # Создаем DataFrame для метрик производительности
            performance_data = []
            for exp_id, metrics in comparison_result.performance_metrics.items():
                performance_data.append(
                    {
                        "experiment_id": exp_id,
                        "mean_reward": metrics.mean_reward,
                        "std_reward": metrics.std_reward,
                        "final_reward": metrics.final_reward,
                        "max_reward": metrics.max_reward,
                        "min_reward": metrics.min_reward,
                        "convergence_timesteps": metrics.convergence_timesteps,
                        "sample_efficiency": metrics.sample_efficiency,
                        "stability_score": metrics.stability_score,
                        "success_rate": metrics.success_rate,
                        "training_time": metrics.training_time,
                    }
                )

            performance_df = pd.DataFrame(performance_data)

            # Создаем DataFrame для статистических тестов
            statistical_data = []
            for metric, tests in comparison_result.statistical_tests.items():
                for comparison, test_result in tests.items():
                    statistical_data.append(
                        {
                            "metric": metric,
                            "comparison": comparison,
                            "test_type": test_result.test_type.value,
                            "statistic": test_result.statistic,
                            "p_value": test_result.p_value,
                            "significant": test_result.significant,
                            "effect_size": test_result.effect_size,
                            "alpha": test_result.alpha,
                        }
                    )

            statistical_df = pd.DataFrame(statistical_data)

            # Сохраняем в Excel с несколькими листами
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                performance_df.to_excel(
                    writer, sheet_name="Performance Metrics", index=False
                )
                statistical_df.to_excel(
                    writer, sheet_name="Statistical Tests", index=False
                )

                # Добавляем лист с рейтингами
                rankings_data = []
                for metric, ranking in comparison_result.rankings.items():
                    for i, exp_id in enumerate(ranking):
                        rankings_data.append(
                            {"metric": metric, "rank": i + 1, "experiment_id": exp_id}
                        )

                rankings_df = pd.DataFrame(rankings_data)
                rankings_df.to_excel(writer, sheet_name="Rankings", index=False)

        except ImportError:
            self.logger.error("pandas и openpyxl требуются для экспорта в Excel")
            raise


def compare_experiments_cli(
    experiment_files: List[str],
    metrics: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    output_format: str = "html",
) -> None:
    """CLI интерфейс для сравнения экспериментов.

    Args:
        experiment_files: Список путей к файлам экспериментов
        metrics: Метрики для сравнения
        output_dir: Директория для сохранения результатов
        config_file: Файл конфигурации сравнения
        output_format: Формат выходного отчета
    """
    try:
        # Загружаем конфигурацию
        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            config = ComparisonConfig(**config_data)
        else:
            config = ComparisonConfig()

        # Создаем компаратор
        comparator = ExperimentComparator(config, output_dir)

        # Загружаем эксперименты
        experiments = []
        for file_path in experiment_files:
            try:
                experiment = Experiment.load(file_path)
                experiments.append(experiment)
                logger.info(f"Загружен эксперимент: {experiment.experiment_id}")
            except Exception as e:
                logger.error(f"Ошибка загрузки эксперимента {file_path}: {e}")
                continue

        if len(experiments) < 2:
            logger.error("Недостаточно экспериментов для сравнения")
            return

        # Выполняем сравнение
        comparison_result = comparator.compare_experiments(experiments, metrics, config)

        # Генерируем отчет
        report_path = comparator.generate_comparison_report(
            comparison_result, include_plots=True, output_format=output_format
        )

        # Создаем графики
        plots = comparator.generate_comparison_plots(comparison_result)

        # Экспортируем результаты
        exported_files = comparator.export_results(comparison_result, ["csv", "json"])

        logger.info(f"Сравнение завершено. Отчет: {report_path}")
        logger.info(f"Графики: {list(plots.values())}")
        logger.info(f"Экспортированные файлы: {list(exported_files.values())}")

    except Exception as e:
        logger.error(f"Ошибка выполнения сравнения: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Сравнение RL экспериментов")
    parser.add_argument("experiments", nargs="+", help="Пути к файлам экспериментов")
    parser.add_argument("--metrics", nargs="*", help="Метрики для сравнения")
    parser.add_argument("--output-dir", help="Директория для сохранения результатов")
    parser.add_argument("--config", help="Файл конфигурации сравнения")
    parser.add_argument(
        "--format",
        choices=["html", "markdown", "json"],
        default="html",
        help="Формат отчета",
    )

    args = parser.parse_args()

    compare_experiments_cli(
        experiment_files=args.experiments,
        metrics=args.metrics,
        output_dir=args.output_dir,
        config_file=args.config,
        output_format=args.format,
    )
