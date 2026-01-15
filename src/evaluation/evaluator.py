"""
Модуль для оценки производительности обученных RL агентов.

Предоставляет инструменты для количественной оценки агентов,
статистического анализа результатов и сравнения различных моделей.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import Agent
from src.utils.rl_logging import get_logger
from src.utils.seeding import set_seed


logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Метрики оценки производительности агента."""

    # Основные метрики
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float

    mean_episode_length: float
    std_episode_length: float
    min_episode_length: int
    max_episode_length: int

    # Статистические метрики
    reward_ci_lower: float  # Нижняя граница доверительного интервала
    reward_ci_upper: float  # Верхняя граница доверительного интервала
    success_rate: float  # Доля успешных эпизодов

    # Дополнительные метрики
    total_episodes: int
    total_timesteps: int
    evaluation_time: float  # Время оценки в секундах

    # Детальные данные
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_successes: List[bool] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Результат сравнения двух агентов."""

    agent1_name: str
    agent2_name: str

    # Статистические тесты
    reward_ttest_statistic: float
    reward_ttest_pvalue: float
    reward_significant: bool

    length_ttest_statistic: float
    length_ttest_pvalue: float
    length_significant: bool

    # Практические различия
    reward_effect_size: float  # Cohen's d
    length_effect_size: float

    # Сводка
    better_agent: str
    confidence_level: float


class EvaluationCallback(Protocol):
    """Протокол для callback'ов во время оценки."""

    def on_episode_start(self, episode: int) -> None:
        """Вызывается в начале каждого эпизода."""
        ...

    def on_episode_end(
        self, episode: int, reward: float, length: int, success: bool
    ) -> None:
        """Вызывается в конце каждого эпизода."""
        ...

    def on_evaluation_end(self, metrics: EvaluationMetrics) -> None:
        """Вызывается в конце оценки."""
        ...


class Evaluator:
    """
    Класс для оценки производительности обученных RL агентов.

    Предоставляет методы для:
    - Количественной оценки производительности агента
    - Статистического анализа результатов
    - Сравнения нескольких агентов
    - Расчета различных метрик
    """

    def __init__(
        self,
        env: gym.Env,
        success_threshold: Optional[float] = None,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        """
        Инициализация оценщика.

        Args:
            env: Среда для оценки агентов
            success_threshold: Пороговое значение награды для определения успеха
            confidence_level: Уровень доверия для доверительных интервалов
            random_seed: Семя для воспроизводимости
        """
        self.env = env
        self.success_threshold = success_threshold
        self.confidence_level = confidence_level
        self.random_seed = random_seed

        # Кэш для результатов оценки
        self._evaluation_cache: Dict[str, EvaluationMetrics] = {}

        logger.info(
            f"Создан оценщик агентов для среды {getattr(env, 'spec', {}).get('id', 'unknown')}, "
            f"порог успеха: {success_threshold}, уровень доверия: {confidence_level}"
        )

    def evaluate_agent(
        self,
        agent: Agent,
        num_episodes: int = 100,
        max_steps_per_episode: Optional[int] = None,
        render: bool = False,
        callback: Optional[EvaluationCallback] = None,
        agent_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> EvaluationMetrics:
        """
        Оценка производительности агента.

        Args:
            agent: Агент для оценки
            num_episodes: Количество эпизодов для оценки
            max_steps_per_episode: Максимальное количество шагов в эпизоде
            render: Отображать ли среду во время оценки
            callback: Callback для отслеживания прогресса
            agent_name: Имя агента для кэширования
            use_cache: Использовать ли кэш результатов

        Returns:
            Метрики оценки агента

        Raises:
            ValueError: При некорректных параметрах
            RuntimeError: При ошибках во время оценки
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes должно быть > 0, получено: {num_episodes}")

        # Проверка кэша
        cache_key = f"{agent_name or id(agent)}_{num_episodes}_{max_steps_per_episode}"
        if use_cache and cache_key in self._evaluation_cache:
            logger.info(f"Использование кэшированных результатов: {cache_key}")
            return self._evaluation_cache[cache_key]

        logger.info(
            f"Начало оценки агента {agent_name or 'unnamed'}: "
            f"{num_episodes} эпизодов, макс. шагов: {max_steps_per_episode}"
        )

        # Установка семени для воспроизводимости
        set_seed(self.random_seed)

        import time

        start_time = time.time()

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_successes: List[bool] = []
        total_timesteps = 0

        try:
            for episode in range(num_episodes):
                if callback:
                    callback.on_episode_start(episode)

                obs, _ = self.env.reset(seed=self.random_seed + episode)
                episode_reward = 0.0
                episode_length = 0
                done = False
                truncated = False

                while not (done or truncated):
                    if (
                        max_steps_per_episode
                        and episode_length >= max_steps_per_episode
                    ):
                        truncated = True
                        break

                    action, _ = agent.predict(obs)
                    # Извлекаем скалярное действие для дискретных сред
                    if hasattr(action, "item"):
                        action_scalar = action.item()
                    else:
                        action_scalar = action[0] if len(action) > 0 else action
                    obs, reward, done, truncated, _ = self.env.step(action_scalar)

                    episode_reward += float(reward)
                    episode_length += 1
                    total_timesteps += 1

                    if render:
                        self.env.render()

                # Определение успешности эпизода
                success = (
                    self.success_threshold is not None
                    and episode_reward >= self.success_threshold
                )

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_successes.append(success)

                if callback:
                    callback.on_episode_end(
                        episode, episode_reward, episode_length, success
                    )

                if (episode + 1) % 10 == 0:
                    logger.debug(
                        f"Прогресс оценки: {episode + 1}/{num_episodes}, "
                        f"средняя награда: {np.mean(episode_rewards):.2f}"
                    )

        except Exception as e:
            logger.error(f"Ошибка во время оценки агента: {e}")
            raise RuntimeError(f"Ошибка оценки агента: {e}") from e

        evaluation_time = time.time() - start_time

        # Расчет статистик
        metrics = self._calculate_metrics(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episode_successes=episode_successes,
            total_timesteps=total_timesteps,
            evaluation_time=evaluation_time,
        )

        if callback:
            callback.on_evaluation_end(metrics)

        # Сохранение в кэш
        if use_cache:
            self._evaluation_cache[cache_key] = metrics

        logger.info(
            f"Оценка агента {agent_name or 'unnamed'} завершена: "
            f"награда {metrics.mean_reward:.3f}, успешность {metrics.success_rate:.1%}, "
            f"время {evaluation_time:.1f}с"
        )

        return metrics

    def compare_agents(
        self,
        agent1: Agent,
        agent2: Agent,
        num_episodes: int = 100,
        agent1_name: str = "Agent1",
        agent2_name: str = "Agent2",
        alpha: float = 0.05,
    ) -> ComparisonResult:
        """
        Сравнение двух агентов с использованием статистических тестов.

        Args:
            agent1: Первый агент для сравнения
            agent2: Второй агент для сравнения
            num_episodes: Количество эпизодов для оценки каждого агента
            agent1_name: Имя первого агента
            agent2_name: Имя второго агента
            alpha: Уровень значимости для статистических тестов

        Returns:
            Результат сравнения агентов
        """
        logger.info(
            f"Начало сравнения агентов {agent1_name} vs {agent2_name}, "
            f"{num_episodes} эпизодов каждый"
        )

        # Оценка обоих агентов
        metrics1 = self.evaluate_agent(
            agent1, num_episodes=num_episodes, agent_name=agent1_name
        )
        metrics2 = self.evaluate_agent(
            agent2, num_episodes=num_episodes, agent_name=agent2_name
        )

        # Статистические тесты для наград
        reward_ttest = stats.ttest_ind(
            metrics1.episode_rewards,
            metrics2.episode_rewards,
            equal_var=False,  # Welch's t-test
        )

        # Статистические тесты для длин эпизодов
        length_ttest = stats.ttest_ind(
            metrics1.episode_lengths, metrics2.episode_lengths, equal_var=False
        )

        # Расчет размера эффекта (Cohen's d)
        reward_effect_size = self._calculate_cohens_d(
            metrics1.episode_rewards, metrics2.episode_rewards
        )
        length_effect_size = self._calculate_cohens_d(
            metrics1.episode_lengths, metrics2.episode_lengths
        )

        # Определение лучшего агента
        better_agent = (
            agent1_name if metrics1.mean_reward > metrics2.mean_reward else agent2_name
        )

        result = ComparisonResult(
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            reward_ttest_statistic=reward_ttest.statistic,  # type: ignore
            reward_ttest_pvalue=reward_ttest.pvalue,  # type: ignore
            reward_significant=reward_ttest.pvalue < alpha,  # type: ignore
            length_ttest_statistic=length_ttest.statistic,  # type: ignore
            length_ttest_pvalue=length_ttest.pvalue,  # type: ignore
            length_significant=length_ttest.pvalue < alpha,  # type: ignore
            reward_effect_size=reward_effect_size,
            length_effect_size=length_effect_size,
            better_agent=better_agent,
            confidence_level=1 - alpha,
        )

        logger.info(
            f"Сравнение агентов завершено: лучший {better_agent}, "
            f"значимость наград: {result.reward_significant}, "
            f"размер эффекта: {reward_effect_size:.3f}"
        )

        return result

    def evaluate_multiple_agents(
        self,
        agents: Dict[str, Agent],
        num_episodes: int = 100,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Оценка нескольких агентов.

        Args:
            agents: Словарь агентов {имя: агент}
            num_episodes: Количество эпизодов для каждого агента

        Returns:
            Словарь результатов {имя: метрики}
        """
        logger.info(f"Начало оценки {len(agents)} агентов: {list(agents.keys())}")

        results = {}
        for name, agent in agents.items():
            logger.info(f"Оценка агента: {name}")
            results[name] = self.evaluate_agent(
                agent, num_episodes=num_episodes, agent_name=name
            )

        return results

    def generate_report(
        self,
        metrics: Union[EvaluationMetrics, Dict[str, EvaluationMetrics]],
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Генерация текстового отчета об оценке.

        Args:
            metrics: Метрики одного агента или словарь метрик нескольких агентов
            save_path: Путь для сохранения отчета

        Returns:
            Текст отчета
        """
        if isinstance(metrics, EvaluationMetrics):
            report = self._generate_single_agent_report(metrics)
        else:
            report = self._generate_multiple_agents_report(metrics)

        if save_path:
            save_path.write_text(report, encoding="utf-8")
            logger.info(f"Отчет сохранен: {save_path}")

        return report

    def export_to_dataframe(
        self,
        metrics: Union[EvaluationMetrics, Dict[str, EvaluationMetrics]],
    ) -> pd.DataFrame:
        """
        Экспорт метрик в pandas DataFrame.

        Args:
            metrics: Метрики для экспорта

        Returns:
            DataFrame с метриками
        """
        if isinstance(metrics, EvaluationMetrics):
            data = {
                "agent_name": ["agent"],
                "mean_reward": [metrics.mean_reward],
                "std_reward": [metrics.std_reward],
                "mean_episode_length": [metrics.mean_episode_length],
                "success_rate": [metrics.success_rate],
                "total_episodes": [metrics.total_episodes],
            }
        else:
            data = {
                "agent_name": list(metrics.keys()),
                "mean_reward": [m.mean_reward for m in metrics.values()],
                "std_reward": [m.std_reward for m in metrics.values()],
                "mean_episode_length": [
                    m.mean_episode_length for m in metrics.values()
                ],
                "success_rate": [m.success_rate for m in metrics.values()],
                "total_episodes": [m.total_episodes for m in metrics.values()],
            }

        return pd.DataFrame(data)

    def _calculate_metrics(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        episode_successes: List[bool],
        total_timesteps: int,
        evaluation_time: float,
    ) -> EvaluationMetrics:
        """Расчет метрик оценки."""
        rewards_array = np.array(episode_rewards)
        lengths_array = np.array(episode_lengths)

        # Основные статистики
        mean_reward = float(np.mean(rewards_array))
        std_reward = float(np.std(rewards_array, ddof=1))
        min_reward = float(np.min(rewards_array))
        max_reward = float(np.max(rewards_array))

        mean_length = float(np.mean(lengths_array))
        std_length = float(np.std(lengths_array, ddof=1))
        min_length = int(np.min(lengths_array))
        max_length = int(np.max(lengths_array))

        # Доверительный интервал для награды
        alpha = 1 - self.confidence_level
        n = len(episode_rewards)
        se = std_reward / np.sqrt(n)
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        ci_lower = mean_reward - t_critical * se
        ci_upper = mean_reward + t_critical * se

        # Доля успешных эпизодов
        success_rate = float(np.mean(episode_successes))

        return EvaluationMetrics(
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            mean_episode_length=mean_length,
            std_episode_length=std_length,
            min_episode_length=min_length,
            max_episode_length=max_length,
            reward_ci_lower=ci_lower,
            reward_ci_upper=ci_upper,
            success_rate=success_rate,
            total_episodes=len(episode_rewards),
            total_timesteps=total_timesteps,
            evaluation_time=evaluation_time,
            episode_rewards=episode_rewards.copy(),
            episode_lengths=episode_lengths.copy(),
            episode_successes=episode_successes.copy(),
        )

    def _calculate_cohens_d(
        self, group1: Sequence[Union[float, int]], group2: Sequence[Union[float, int]]
    ) -> float:
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

    def _generate_single_agent_report(self, metrics: EvaluationMetrics) -> str:
        """Генерация отчета для одного агента."""
        return f"""
=== ОТЧЕТ ОЦЕНКИ АГЕНТА ===

Общая информация:
- Количество эпизодов: {metrics.total_episodes}
- Общее количество шагов: {metrics.total_timesteps}
- Время оценки: {metrics.evaluation_time:.2f} сек

Награды:
- Среднее: {metrics.mean_reward:.3f}
- Стандартное отклонение: {metrics.std_reward:.3f}
- Минимум: {metrics.min_reward:.3f}
- Максимум: {metrics.max_reward:.3f}
- Доверительный интервал ({self.confidence_level * 100:.0f}%): [{metrics.reward_ci_lower:.3f}, {metrics.reward_ci_upper:.3f}]

Длина эпизодов:
- Среднее: {metrics.mean_episode_length:.1f}
- Стандартное отклонение: {metrics.std_episode_length:.1f}
- Минимум: {metrics.min_episode_length}
- Максимум: {metrics.max_episode_length}

Успешность:
- Доля успешных эпизодов: {metrics.success_rate:.1%}
"""

    def _generate_multiple_agents_report(
        self, metrics_dict: Dict[str, EvaluationMetrics]
    ) -> str:
        """Генерация отчета для нескольких агентов."""
        report = "=== СРАВНИТЕЛЬНЫЙ ОТЧЕТ АГЕНТОВ ===\n\n"

        # Сортировка по средней награде
        sorted_agents = sorted(
            metrics_dict.items(), key=lambda x: x[1].mean_reward, reverse=True
        )

        report += "Рейтинг по средней награде:\n"
        for i, (name, metrics) in enumerate(sorted_agents, 1):
            report += (
                f"{i}. {name}: {metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}\n"
            )

        report += "\nДетальная статистика:\n"
        for name, metrics in sorted_agents:
            report += f"\n--- {name} ---\n"
            report += f"Награда: {metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}\n"
            report += f"Длина эпизода: {metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}\n"
            report += f"Успешность: {metrics.success_rate:.1%}\n"

        return report
