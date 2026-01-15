"""
Модуль для количественной оценки и статистического анализа RL агентов.

Предоставляет высокоуровневые функции для стандартизированной оценки агентов,
статистического анализа результатов, сравнения с базовыми показателями,
генерации отчетов и визуализации распределений.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import Agent
from src.evaluation.evaluator import (
    ComparisonResult,
    EvaluationMetrics,
    Evaluator,
)
from src.utils.rl_logging import get_logger
from src.utils.seeding import set_seed

logger = get_logger(__name__)


@dataclass
class QuantitativeMetrics:
    """Расширенные количественные метрики оценки агента."""

    # Базовые метрики из EvaluationMetrics
    base_metrics: EvaluationMetrics

    # Статистические показатели
    reward_median: float
    reward_q25: float  # 25-й квартиль
    reward_q75: float  # 75-й квартиль
    reward_iqr: float  # Межквартильный размах
    reward_skewness: float  # Асимметрия
    reward_kurtosis: float  # Эксцесс

    length_median: float
    length_q25: float
    length_q75: float
    length_iqr: float

    # Показатели стабильности
    reward_cv: float  # Коэффициент вариации
    reward_stability_score: float  # Оценка стабильности (0-1)
    consecutive_success_rate: float  # Доля последовательных успехов

    # Дополнительные метрики
    reward_trend_slope: float  # Тренд изменения награды
    learning_efficiency: float  # Эффективность обучения
    outlier_count: int  # Количество выбросов


@dataclass
class BaselineComparison:
    """Результат сравнения с базовыми показателями."""

    agent_name: str
    baseline_name: str

    # Метрики агента и базовой линии
    agent_metrics: QuantitativeMetrics
    baseline_metrics: QuantitativeMetrics

    # Статистические тесты
    reward_improvement: float  # Процентное улучшение
    reward_ttest_pvalue: float
    reward_wilcoxon_pvalue: float  # Непараметрический тест
    reward_significant: bool

    # Практическая значимость
    effect_size: float  # Cohen's d
    practical_significance: bool  # Превышает ли минимальный порог

    # Сводка
    is_better: bool
    confidence_level: float


@dataclass
class BatchEvaluationResult:
    """Результат пакетной оценки нескольких агентов."""

    agents_metrics: Dict[str, QuantitativeMetrics]
    comparison_matrix: Dict[Tuple[str, str], ComparisonResult]
    ranking: List[Tuple[str, float]]  # (имя_агента, средняя_награда)
    best_agent: str
    statistical_summary: Dict[str, Any]


class QuantitativeEvaluator:
    """
    Класс для количественной оценки и статистического анализа RL агентов.

    Предоставляет высокоуровневые функции для:
    - Стандартизированной оценки агентов на 10-20 эпизодах
    - Расширенного статистического анализа результатов
    - Сравнения с базовыми показателями
    - Пакетной оценки нескольких агентов
    - Генерации отчетов в различных форматах
    - Визуализации распределений результатов
    """

    def __init__(
        self,
        env: gym.Env,
        baseline_threshold: Optional[float] = None,
        success_threshold: Optional[float] = None,
        min_effect_size: float = 0.5,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        """
        Инициализация количественного оценщика.

        Args:
            env: Среда для оценки агентов
            baseline_threshold: Пороговое значение для базовой линии
            success_threshold: Пороговое значение для определения успеха
            min_effect_size: Минимальный размер эффекта для практической значимости
            confidence_level: Уровень доверия для статистических тестов
            random_seed: Семя для воспроизводимости
        """
        self.env = env
        self.baseline_threshold = baseline_threshold
        self.min_effect_size = min_effect_size
        self.random_seed = random_seed

        # Базовый оценщик
        self.evaluator = Evaluator(
            env=env,
            success_threshold=success_threshold,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )

        # Кэш для результатов
        self._metrics_cache: Dict[str, QuantitativeMetrics] = {}
        self._baseline_cache: Dict[str, QuantitativeMetrics] = {}

        logger.info(
            f"Создан количественный оценщик для среды "
            f"{getattr(env, 'spec', {}).get('id', 'unknown')}, "
            f"базовая линия: {baseline_threshold}, "
            f"минимальный размер эффекта: {min_effect_size}"
        )

    def evaluate_agent_quantitative(
        self,
        agent: Agent,
        num_episodes: int = 20,
        agent_name: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> QuantitativeMetrics:
        """
        Количественная оценка агента с расширенным статистическим анализом.

        Args:
            agent: Агент для оценки
            num_episodes: Количество эпизодов (рекомендуется 10-20)
            agent_name: Имя агента для кэширования
            use_cache: Использовать ли кэш результатов
            **kwargs: Дополнительные параметры для базового оценщика

        Returns:
            Расширенные количественные метрики

        Raises:
            ValueError: При некорректных параметрах
            RuntimeError: При ошибках во время оценки
        """
        if num_episodes < 5:
            raise ValueError(
                f"num_episodes должно быть >= 5 для статистической значимости, "
                f"получено: {num_episodes}"
            )

        # Проверка кэша
        cache_key = f"{agent_name or id(agent)}_{num_episodes}"
        if use_cache and cache_key in self._metrics_cache:
            logger.info(f"Использование кэшированных количественных метрик: {cache_key}")
            return self._metrics_cache[cache_key]

        logger.info(
            f"Начало количественной оценки агента {agent_name or 'unnamed'}: "
            f"{num_episodes} эпизодов"
        )

        # Установка семени
        set_seed(self.random_seed)

        # Базовая оценка
        base_metrics = self.evaluator.evaluate_agent(
            agent=agent,
            num_episodes=num_episodes,
            agent_name=agent_name,
            use_cache=use_cache,
            **kwargs,
        )

        # Расчет расширенных метрик
        quantitative_metrics = self._calculate_quantitative_metrics(base_metrics)

        # Сохранение в кэш
        if use_cache:
            self._metrics_cache[cache_key] = quantitative_metrics

        logger.info(
            f"Количественная оценка агента {agent_name or 'unnamed'} завершена: "
            f"награда {quantitative_metrics.base_metrics.mean_reward:.3f} ± "
            f"{quantitative_metrics.base_metrics.std_reward:.3f}, "
            f"стабильность {quantitative_metrics.reward_stability_score:.3f}"
        )

        return quantitative_metrics

    def compare_with_baseline(
        self,
        agent: Agent,
        baseline_agent: Optional[Agent] = None,
        baseline_metrics: Optional[QuantitativeMetrics] = None,
        num_episodes: int = 20,
        agent_name: str = "Agent",
        baseline_name: str = "Baseline",
        alpha: float = 0.05,
    ) -> BaselineComparison:
        """
        Сравнение агента с базовой линией.

        Args:
            agent: Агент для сравнения
            baseline_agent: Базовый агент (если None, используется baseline_metrics)
            baseline_metrics: Готовые метрики базовой линии
            num_episodes: Количество эпизодов для оценки
            agent_name: Имя агента
            baseline_name: Имя базовой линии
            alpha: Уровень значимости

        Returns:
            Результат сравнения с базовой линией

        Raises:
            ValueError: Если не предоставлен ни baseline_agent, ни baseline_metrics
        """
        if baseline_agent is None and baseline_metrics is None:
            raise ValueError(
                "Необходимо предоставить либо baseline_agent, либо baseline_metrics"
            )

        logger.info(f"Начало сравнения {agent_name} с базовой линией {baseline_name}")

        # Оценка агента
        agent_metrics = self.evaluate_agent_quantitative(
            agent=agent, num_episodes=num_episodes, agent_name=agent_name
        )

        # Получение метрик базовой линии
        if baseline_metrics is None:
            baseline_metrics = self.evaluate_agent_quantitative(
                agent=baseline_agent,  # type: ignore
                num_episodes=num_episodes,
                agent_name=baseline_name,
            )

        # Статистические тесты
        agent_rewards = agent_metrics.base_metrics.episode_rewards
        baseline_rewards = baseline_metrics.base_metrics.episode_rewards

        # t-тест
        ttest_result = stats.ttest_ind(
            agent_rewards, baseline_rewards, equal_var=False
        )

        # Непараметрический тест Вилкоксона
        wilcoxon_result = stats.mannwhitneyu(
            agent_rewards, baseline_rewards, alternative="two-sided"
        )

        # Расчет улучшения и размера эффекта
        reward_improvement = (
            (agent_metrics.base_metrics.mean_reward - baseline_metrics.base_metrics.mean_reward)
            / abs(baseline_metrics.base_metrics.mean_reward)
            * 100
        )

        effect_size = self._calculate_cohens_d(agent_rewards, baseline_rewards)

        # Определение практической значимости
        practical_significance = abs(effect_size) >= self.min_effect_size

        # Определение лучшего агента
        is_better = agent_metrics.base_metrics.mean_reward > baseline_metrics.base_metrics.mean_reward

        result = BaselineComparison(
            agent_name=agent_name,
            baseline_name=baseline_name,
            agent_metrics=agent_metrics,
            baseline_metrics=baseline_metrics,
            reward_improvement=reward_improvement,
            reward_ttest_pvalue=ttest_result.pvalue,  # type: ignore
            reward_wilcoxon_pvalue=wilcoxon_result.pvalue,  # type: ignore
            reward_significant=ttest_result.pvalue < alpha,  # type: ignore
            effect_size=effect_size,
            practical_significance=practical_significance,
            is_better=is_better,
            confidence_level=1 - alpha,
        )

        logger.info(
            f"Сравнение с базовой линией завершено: "
            f"улучшение {reward_improvement:.1f}%, "
            f"размер эффекта {effect_size:.3f}, "
            f"практически значимо: {practical_significance}"
        )

        return result

    def evaluate_multiple_agents_batch(
        self,
        agents: Dict[str, Agent],
        num_episodes: int = 20,
        include_pairwise_comparison: bool = True,
    ) -> BatchEvaluationResult:
        """
        Пакетная оценка нескольких агентов с полным статистическим анализом.

        Args:
            agents: Словарь агентов {имя: агент}
            num_episodes: Количество эпизодов для каждого агента
            include_pairwise_comparison: Включить попарное сравнение агентов

        Returns:
            Результат пакетной оценки
        """
        logger.info(
            f"Начало пакетной оценки {len(agents)} агентов: {list(agents.keys())}"
        )

        start_time = time.time()

        # Оценка всех агентов
        agents_metrics = {}
        for name, agent in agents.items():
            logger.info(f"Оценка агента: {name}")
            agents_metrics[name] = self.evaluate_agent_quantitative(
                agent=agent, num_episodes=num_episodes, agent_name=name
            )

        # Попарное сравнение (если требуется)
        comparison_matrix = {}
        if include_pairwise_comparison and len(agents) > 1:
            agent_names = list(agents.keys())
            for i, name1 in enumerate(agent_names):
                for j, name2 in enumerate(agent_names[i + 1 :], i + 1):
                    logger.debug(f"Сравнение {name1} vs {name2}")
                    comparison = self.evaluator.compare_agents(
                        agent1=agents[name1],
                        agent2=agents[name2],
                        num_episodes=num_episodes,
                        agent1_name=name1,
                        agent2_name=name2,
                    )
                    comparison_matrix[(name1, name2)] = comparison

        # Ранжирование по средней награде
        ranking = sorted(
            [(name, metrics.base_metrics.mean_reward) for name, metrics in agents_metrics.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        best_agent = ranking[0][0]

        # Статистическая сводка
        all_rewards = [
            metrics.base_metrics.mean_reward for metrics in agents_metrics.values()
        ]
        statistical_summary = {
            "total_agents": len(agents),
            "evaluation_time": time.time() - start_time,
            "mean_performance": float(np.mean(all_rewards)),
            "std_performance": float(np.std(all_rewards)),
            "performance_range": float(np.max(all_rewards) - np.min(all_rewards)),
            "best_agent": best_agent,
            "best_performance": ranking[0][1],
        }

        result = BatchEvaluationResult(
            agents_metrics=agents_metrics,
            comparison_matrix=comparison_matrix,
            ranking=ranking,
            best_agent=best_agent,
            statistical_summary=statistical_summary,
        )

        logger.info(
            f"Пакетная оценка завершена за {statistical_summary['evaluation_time']:.1f}с, "
            f"лучший агент: {best_agent} ({ranking[0][1]:.3f})"
        )

        return result

    def generate_comprehensive_report(
        self,
        metrics: Union[QuantitativeMetrics, Dict[str, QuantitativeMetrics], BatchEvaluationResult],
        save_path: Optional[Path] = None,
        format_type: str = "text",
    ) -> str:
        """
        Генерация комплексного отчета об оценке.

        Args:
            metrics: Метрики для отчета
            save_path: Путь для сохранения отчета
            format_type: Формат отчета ('text', 'json', 'csv')

        Returns:
            Содержимое отчета

        Raises:
            ValueError: При неподдерживаемом формате
        """
        if format_type not in ["text", "json", "csv"]:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")

        if format_type == "text":
            report = self._generate_text_report(metrics)
        elif format_type == "json":
            report = self._generate_json_report(metrics)
        else:  # csv
            report = self._generate_csv_report(metrics)

        if save_path:
            save_path.write_text(report, encoding="utf-8")
            logger.info(f"Комплексный отчет сохранен: {save_path}")

        return report

    def visualize_results(
        self,
        metrics: Union[QuantitativeMetrics, Dict[str, QuantitativeMetrics]],
        save_path: Optional[Path] = None,
        show_plots: bool = True,
    ) -> None:
        """
        Визуализация распределений результатов.

        Args:
            metrics: Метрики для визуализации
            save_path: Путь для сохранения графиков
            show_plots: Показывать ли графики
        """
        if isinstance(metrics, QuantitativeMetrics):
            self._plot_single_agent_distribution(metrics, save_path, show_plots)
        else:
            self._plot_multiple_agents_comparison(metrics, save_path, show_plots)

    def _calculate_quantitative_metrics(
        self, base_metrics: EvaluationMetrics
    ) -> QuantitativeMetrics:
        """Расчет расширенных количественных метрик."""
        rewards = np.array(base_metrics.episode_rewards)
        lengths = np.array(base_metrics.episode_lengths)

        # Квартили и статистики распределения
        reward_q25, reward_median, reward_q75 = np.percentile(rewards, [25, 50, 75])
        reward_iqr = reward_q75 - reward_q25
        reward_skewness = float(stats.skew(rewards))
        reward_kurtosis = float(stats.kurtosis(rewards))

        length_q25, length_median, length_q75 = np.percentile(lengths, [25, 50, 75])
        length_iqr = length_q75 - length_q25

        # Коэффициент вариации
        reward_cv = base_metrics.std_reward / abs(base_metrics.mean_reward) if base_metrics.mean_reward != 0 else 0

        # Оценка стабильности (обратная к CV, нормализованная)
        reward_stability_score = max(0, 1 - min(reward_cv, 1))

        # Доля последовательных успехов
        successes = base_metrics.episode_successes
        consecutive_success_rate = self._calculate_consecutive_success_rate(successes)

        # Тренд изменения награды
        episodes = np.arange(len(rewards))
        if len(rewards) > 1:
            slope, _, _, _, _ = stats.linregress(episodes, rewards)
            reward_trend_slope = float(slope)
        else:
            reward_trend_slope = 0.0

        # Эффективность обучения (награда на единицу времени)
        learning_efficiency = (
            base_metrics.mean_reward / base_metrics.evaluation_time
            if base_metrics.evaluation_time > 0
            else 0.0
        )

        # Количество выбросов (метод IQR)
        outlier_threshold = 1.5 * reward_iqr
        outliers = rewards[(rewards < reward_q25 - outlier_threshold) | 
                          (rewards > reward_q75 + outlier_threshold)]
        outlier_count = len(outliers)

        return QuantitativeMetrics(
            base_metrics=base_metrics,
            reward_median=float(reward_median),
            reward_q25=float(reward_q25),
            reward_q75=float(reward_q75),
            reward_iqr=float(reward_iqr),
            reward_skewness=reward_skewness,
            reward_kurtosis=reward_kurtosis,
            length_median=float(length_median),
            length_q25=float(length_q25),
            length_q75=float(length_q75),
            length_iqr=float(length_iqr),
            reward_cv=float(reward_cv),
            reward_stability_score=float(reward_stability_score),
            consecutive_success_rate=float(consecutive_success_rate),
            reward_trend_slope=reward_trend_slope,
            learning_efficiency=float(learning_efficiency),
            outlier_count=outlier_count,
        )

    def _calculate_consecutive_success_rate(self, successes: List[bool]) -> float:
        """Расчет доли последовательных успехов."""
        if not successes or len(successes) < 2:
            return 0.0

        consecutive_pairs = 0
        total_pairs = len(successes) - 1

        for i in range(total_pairs):
            if successes[i] and successes[i + 1]:
                consecutive_pairs += 1

        return consecutive_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Расчет размера эффекта Cohen's d."""
        n1, n2 = len(group1), len(group2)

        # Объединенное стандартное отклонение
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
            / (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _generate_text_report(
        self,
        metrics: Union[QuantitativeMetrics, Dict[str, QuantitativeMetrics], BatchEvaluationResult],
    ) -> str:
        """Генерация текстового отчета."""
        if isinstance(metrics, QuantitativeMetrics):
            return self._generate_single_agent_text_report(metrics)
        elif isinstance(metrics, dict):
            return self._generate_multiple_agents_text_report(metrics)
        else:  # BatchEvaluationResult
            return self._generate_batch_text_report(metrics)

    def _generate_single_agent_text_report(self, metrics: QuantitativeMetrics) -> str:
        """Генерация текстового отчета для одного агента."""
        base = metrics.base_metrics
        return f"""
=== КОМПЛЕКСНЫЙ ОТЧЕТ КОЛИЧЕСТВЕННОЙ ОЦЕНКИ АГЕНТА ===

Общая информация:
- Количество эпизодов: {base.total_episodes}
- Общее количество шагов: {base.total_timesteps}
- Время оценки: {base.evaluation_time:.2f} сек
- Эффективность обучения: {metrics.learning_efficiency:.3f} награда/сек

Основные статистики награды:
- Среднее: {base.mean_reward:.3f}
- Медиана: {metrics.reward_median:.3f}
- Стандартное отклонение: {base.std_reward:.3f}
- Минимум: {base.min_reward:.3f}
- Максимум: {base.max_reward:.3f}
- Доверительный интервал (95%): [{base.reward_ci_lower:.3f}, {base.reward_ci_upper:.3f}]

Квартили и распределение:
- Q1 (25%): {metrics.reward_q25:.3f}
- Q3 (75%): {metrics.reward_q75:.3f}
- Межквартильный размах: {metrics.reward_iqr:.3f}
- Асимметрия: {metrics.reward_skewness:.3f}
- Эксцесс: {metrics.reward_kurtosis:.3f}
- Количество выбросов: {metrics.outlier_count}

Показатели стабильности:
- Коэффициент вариации: {metrics.reward_cv:.3f}
- Оценка стабильности: {metrics.reward_stability_score:.3f}
- Тренд изменения: {metrics.reward_trend_slope:.6f}

Длина эпизодов:
- Среднее: {base.mean_episode_length:.1f}
- Медиана: {metrics.length_median:.1f}
- Стандартное отклонение: {base.std_episode_length:.1f}
- Q1-Q3: {metrics.length_q25:.1f} - {metrics.length_q75:.1f}

Успешность:
- Доля успешных эпизодов: {base.success_rate:.1%}
- Доля последовательных успехов: {metrics.consecutive_success_rate:.1%}
"""

    def _generate_multiple_agents_text_report(
        self, metrics_dict: Dict[str, QuantitativeMetrics]
    ) -> str:
        """Генерация текстового отчета для нескольких агентов."""
        report = "=== СРАВНИТЕЛЬНЫЙ КОЛИЧЕСТВЕННЫЙ ОТЧЕТ АГЕНТОВ ===\n\n"

        # Ранжирование
        ranking = sorted(
            metrics_dict.items(),
            key=lambda x: x[1].base_metrics.mean_reward,
            reverse=True,
        )

        report += "Рейтинг по средней награде:\n"
        for i, (name, metrics) in enumerate(ranking, 1):
            base = metrics.base_metrics
            report += (
                f"{i}. {name}: {base.mean_reward:.3f} ± {base.std_reward:.3f} "
                f"(стабильность: {metrics.reward_stability_score:.3f})\n"
            )

        # Сравнительная таблица
        report += "\nСравнительная таблица ключевых метрик:\n"
        report += f"{'Агент':<15} {'Награда':<12} {'Медиана':<10} {'Стабильность':<12} {'Успешность':<12}\n"
        report += "-" * 70 + "\n"

        for name, metrics in ranking:
            base = metrics.base_metrics
            report += (
                f"{name:<15} {base.mean_reward:<12.3f} {metrics.reward_median:<10.3f} "
                f"{metrics.reward_stability_score:<12.3f} {base.success_rate:<12.1%}\n"
            )

        return report

    def _generate_batch_text_report(self, result: BatchEvaluationResult) -> str:
        """Генерация текстового отчета для пакетной оценки."""
        report = "=== ОТЧЕТ ПАКЕТНОЙ ОЦЕНКИ АГЕНТОВ ===\n\n"

        summary = result.statistical_summary
        report += "Общая информация:\n"
        report += f"- Количество агентов: {summary['total_agents']}\n"
        report += f"- Время оценки: {summary['evaluation_time']:.1f} сек\n"
        report += f"- Лучший агент: {summary['best_agent']} ({summary['best_performance']:.3f})\n"
        report += f"- Средняя производительность: {summary['mean_performance']:.3f} ± {summary['std_performance']:.3f}\n"
        report += f"- Диапазон производительности: {summary['performance_range']:.3f}\n\n"

        # Рейтинг
        report += "Рейтинг агентов:\n"
        for i, (name, reward) in enumerate(result.ranking, 1):
            metrics = result.agents_metrics[name]
            report += (
                f"{i}. {name}: {reward:.3f} "
                f"(стабильность: {metrics.reward_stability_score:.3f})\n"
            )

        return report

    def _generate_json_report(
        self,
        metrics: Union[QuantitativeMetrics, Dict[str, QuantitativeMetrics], BatchEvaluationResult],
    ) -> str:
        """Генерация JSON отчета."""
        if isinstance(metrics, QuantitativeMetrics):
            data = self._metrics_to_dict(metrics)
        elif isinstance(metrics, dict):
            data = {name: self._metrics_to_dict(m) for name, m in metrics.items()}
        else:  # BatchEvaluationResult
            data = {
                "agents_metrics": {
                    name: self._metrics_to_dict(m) for name, m in metrics.agents_metrics.items()
                },
                "ranking": metrics.ranking,
                "best_agent": metrics.best_agent,
                "statistical_summary": metrics.statistical_summary,
            }

        return json.dumps(data, indent=2, ensure_ascii=False)

    def _generate_csv_report(
        self,
        metrics: Union[QuantitativeMetrics, Dict[str, QuantitativeMetrics], BatchEvaluationResult],
    ) -> str:
        """Генерация CSV отчета."""
        if isinstance(metrics, QuantitativeMetrics):
            df = pd.DataFrame([self._metrics_to_dict(metrics)])
        elif isinstance(metrics, dict):
            df = pd.DataFrame([self._metrics_to_dict(m) for m in metrics.values()])
            df.insert(0, "agent_name", list(metrics.keys()))
        else:  # BatchEvaluationResult
            data = []
            for name, m in metrics.agents_metrics.items():
                row = self._metrics_to_dict(m)
                row["agent_name"] = name
                data.append(row)
            df = pd.DataFrame(data)

        return df.to_csv(index=False)

    def _metrics_to_dict(self, metrics: QuantitativeMetrics) -> Dict[str, Any]:
        """Преобразование метрик в словарь."""
        base = metrics.base_metrics
        return {
            "mean_reward": base.mean_reward,
            "std_reward": base.std_reward,
            "median_reward": metrics.reward_median,
            "q25_reward": metrics.reward_q25,
            "q75_reward": metrics.reward_q75,
            "iqr_reward": metrics.reward_iqr,
            "skewness": metrics.reward_skewness,
            "kurtosis": metrics.reward_kurtosis,
            "cv": metrics.reward_cv,
            "stability_score": metrics.reward_stability_score,
            "success_rate": base.success_rate,
            "consecutive_success_rate": metrics.consecutive_success_rate,
            "mean_episode_length": base.mean_episode_length,
            "total_episodes": base.total_episodes,
            "evaluation_time": base.evaluation_time,
            "learning_efficiency": metrics.learning_efficiency,
            "trend_slope": metrics.reward_trend_slope,
            "outlier_count": metrics.outlier_count,
        }

    def _plot_single_agent_distribution(
        self,
        metrics: QuantitativeMetrics,
        save_path: Optional[Path] = None,
        show_plots: bool = True,
    ) -> None:
        """Визуализация распределения для одного агента."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Распределение результатов агента", fontsize=16)

        rewards = metrics.base_metrics.episode_rewards
        lengths = metrics.base_metrics.episode_lengths

        # Гистограмма наград
        axes[0, 0].hist(rewards, bins=min(15, len(rewards) // 2), alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(metrics.base_metrics.mean_reward, color="red", linestyle="--", label="Среднее")
        axes[0, 0].axvline(metrics.reward_median, color="green", linestyle="--", label="Медиана")
        axes[0, 0].set_xlabel("Награда")
        axes[0, 0].set_ylabel("Частота")
        axes[0, 0].set_title("Распределение наград")
        axes[0, 0].legend()

        # Box plot наград
        axes[0, 1].boxplot(rewards, vert=True)
        axes[0, 1].set_ylabel("Награда")
        axes[0, 1].set_title("Box plot наград")

        # Временной ряд наград
        episodes = range(1, len(rewards) + 1)
        axes[1, 0].plot(episodes, rewards, marker="o", markersize=4, alpha=0.7)
        axes[1, 0].axhline(metrics.base_metrics.mean_reward, color="red", linestyle="--", alpha=0.7)
        axes[1, 0].set_xlabel("Эпизод")
        axes[1, 0].set_ylabel("Награда")
        axes[1, 0].set_title("Динамика наград по эпизодам")

        # Гистограмма длин эпизодов
        axes[1, 1].hist(lengths, bins=min(15, len(lengths) // 2), alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(metrics.base_metrics.mean_episode_length, color="red", linestyle="--", label="Среднее")
        axes[1, 1].axvline(metrics.length_median, color="green", linestyle="--", label="Медиана")
        axes[1, 1].set_xlabel("Длина эпизода")
        axes[1, 1].set_ylabel("Частота")
        axes[1, 1].set_title("Распределение длин эпизодов")
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График сохранен: {save_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_multiple_agents_comparison(
        self,
        metrics_dict: Dict[str, QuantitativeMetrics],
        save_path: Optional[Path] = None,
        show_plots: bool = True,
    ) -> None:
        """Визуализация сравнения нескольких агентов."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Сравнение агентов", fontsize=16)

        agent_names = list(metrics_dict.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))  # type: ignore

        # Box plot сравнение наград
        rewards_data = [metrics_dict[name].base_metrics.episode_rewards for name in agent_names]
        bp = axes[0, 0].boxplot(rewards_data, labels=agent_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        axes[0, 0].set_ylabel("Награда")
        axes[0, 0].set_title("Сравнение распределений наград")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Барный график средних наград
        mean_rewards = [metrics_dict[name].base_metrics.mean_reward for name in agent_names]
        std_rewards = [metrics_dict[name].base_metrics.std_reward for name in agent_names]
        axes[0, 1].bar(agent_names, mean_rewards, yerr=std_rewards, capsize=5, color=colors)
        axes[0, 1].set_ylabel("Средняя награда")
        axes[0, 1].set_title("Средние награды с доверительными интервалами")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Scatter plot стабильность vs производительность
        stability_scores = [metrics_dict[name].reward_stability_score for name in agent_names]
        axes[1, 0].scatter(mean_rewards, stability_scores, c=colors, s=100, alpha=0.7)
        for i, name in enumerate(agent_names):
            axes[1, 0].annotate(name, (mean_rewards[i], stability_scores[i]), 
                               xytext=(5, 5), textcoords="offset points")
        axes[1, 0].set_xlabel("Средняя награда")
        axes[1, 0].set_ylabel("Оценка стабильности")
        axes[1, 0].set_title("Производительность vs Стабильность")

        # Радарная диаграмма (упрощенная)
        success_rates = [metrics_dict[name].base_metrics.success_rate for name in agent_names]
        
        # Нормализация метрик для радарной диаграммы
        norm_rewards = np.array(mean_rewards) / max(mean_rewards) if max(mean_rewards) > 0 else np.zeros_like(mean_rewards)
        norm_stability = np.array(stability_scores)
        norm_success = np.array(success_rates)
        
        x = np.arange(len(agent_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, norm_rewards, width, label="Награда (норм.)", alpha=0.7)
        axes[1, 1].bar(x, norm_stability, width, label="Стабильность", alpha=0.7)
        axes[1, 1].bar(x + width, norm_success, width, label="Успешность", alpha=0.7)
        
        axes[1, 1].set_xlabel("Агенты")
        axes[1, 1].set_ylabel("Нормализованные метрики")
        axes[1, 1].set_title("Сравнение ключевых метрик")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(agent_names, rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График сравнения сохранен: {save_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()


# Удобные функции для быстрого использования

def evaluate_agent_standard(
    agent: Agent,
    env: gym.Env,
    num_episodes: int = 20,
    agent_name: Optional[str] = None,
    **kwargs: Any,
) -> QuantitativeMetrics:
    """
    Стандартная количественная оценка агента на 10-20 эпизодах.

    Args:
        agent: Агент для оценки
        env: Среда для оценки
        num_episodes: Количество эпизодов (рекомендуется 10-20)
        agent_name: Имя агента
        **kwargs: Дополнительные параметры

    Returns:
        Количественные метрики оценки
    """
    evaluator = QuantitativeEvaluator(env=env)
    return evaluator.evaluate_agent_quantitative(
        agent=agent, num_episodes=num_episodes, agent_name=agent_name, **kwargs
    )


def compare_agents_statistical(
    agents: Dict[str, Agent],
    env: gym.Env,
    num_episodes: int = 20,
    save_report: Optional[Path] = None,
    save_plots: Optional[Path] = None,
) -> BatchEvaluationResult:
    """
    Статистическое сравнение нескольких агентов.

    Args:
        agents: Словарь агентов для сравнения
        env: Среда для оценки
        num_episodes: Количество эпизодов для каждого агента
        save_report: Путь для сохранения текстового отчета
        save_plots: Путь для сохранения графиков

    Returns:
        Результат пакетной оценки
    """
    evaluator = QuantitativeEvaluator(env=env)
    
    # Пакетная оценка
    result = evaluator.evaluate_multiple_agents_batch(
        agents=agents, num_episodes=num_episodes
    )
    
    # Сохранение отчета
    if save_report:
        evaluator.generate_comprehensive_report(
            metrics=result, save_path=save_report, format_type="text"
        )
    
    # Сохранение графиков
    if save_plots:
        evaluator.visualize_results(
            metrics=result.agents_metrics, save_path=save_plots, show_plots=False
        )
    
    return result


def analyze_agent_stability(
    agent: Agent,
    env: gym.Env,
    num_runs: int = 5,
    episodes_per_run: int = 20,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Анализ стабильности агента через несколько запусков.

    Args:
        agent: Агент для анализа
        env: Среда для оценки
        num_runs: Количество независимых запусков
        episodes_per_run: Количество эпизодов в каждом запуске
        agent_name: Имя агента

    Returns:
        Словарь с анализом стабильности
    """
    evaluator = QuantitativeEvaluator(env=env)
    
    run_metrics = []
    mean_rewards = []
    stability_scores = []
    
    for run in range(num_runs):
        # Используем разные семена для каждого запуска
        evaluator.random_seed = 42 + run * 100
        
        metrics = evaluator.evaluate_agent_quantitative(
            agent=agent,
            num_episodes=episodes_per_run,
            agent_name=f"{agent_name}_run_{run}" if agent_name else f"agent_run_{run}",
            use_cache=False,  # Не используем кэш для независимых запусков
        )
        
        run_metrics.append(metrics)
        mean_rewards.append(metrics.base_metrics.mean_reward)
        stability_scores.append(metrics.reward_stability_score)
    
    # Анализ стабильности между запусками
    inter_run_stability = {
        "mean_reward_across_runs": float(np.mean(mean_rewards)),
        "std_reward_across_runs": float(np.std(mean_rewards)),
        "cv_across_runs": float(np.std(mean_rewards) / np.mean(mean_rewards)) if np.mean(mean_rewards) != 0 else 0,
        "min_reward": float(np.min(mean_rewards)),
        "max_reward": float(np.max(mean_rewards)),
        "reward_range": float(np.max(mean_rewards) - np.min(mean_rewards)),
        "mean_stability_score": float(np.mean(stability_scores)),
        "stability_consistency": float(1 - np.std(stability_scores)),  # Чем меньше разброс, тем лучше
        "num_runs": num_runs,
        "episodes_per_run": episodes_per_run,
    }
    
    return {
        "inter_run_stability": inter_run_stability,
        "run_metrics": run_metrics,
        "agent_name": agent_name or "unnamed",
    }