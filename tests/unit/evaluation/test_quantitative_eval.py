"""
Тесты для модуля количественной оценки RL агентов.

Проверяет корректность расчета расширенных метрик, статистических тестов,
сравнения с базовыми показателями и генерации отчетов.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest

from src.agents.base import Agent
from src.evaluation.quantitative_eval import (
    BaselineComparison,
    BatchEvaluationResult,
    QuantitativeEvaluator,
    QuantitativeMetrics,
    analyze_agent_stability,
    compare_agents_statistical,
    evaluate_agent_standard,
)


class MockAgent(Agent):
    """Мок-агент для тестирования."""

    def __init__(self, rewards: list[float], episode_lengths: list[int] | None = None):
        """
        Инициализация мок-агента.

        Args:
            rewards: Список наград для возврата
            episode_lengths: Список длин эпизодов (опционально)
        """
        self.rewards = rewards
        self.episode_lengths = episode_lengths or [100] * len(rewards)
        self.current_episode = 0
        self.is_trained = True
        self.model = Mock()

    def predict(self, observation, deterministic=True):
        """Предсказание действия."""
        return np.array([0]), None

    def _create_model(self):
        """Создание модели."""
        return Mock()

    def train(self, **kwargs):
        """Обучение агента."""
        pass

    @classmethod
    def load(cls, path, env=None, **kwargs):
        """Загрузка агента."""
        return cls([100.0, 150.0, 200.0])


@pytest.fixture
def mock_env():
    """Создание мок-среды."""
    env = Mock(spec=gym.Env)
    env.reset.return_value = (np.array([0.0, 0.0]), {})
    env.step.return_value = (np.array([0.0, 0.0]), 100.0, True, False, {})
    env.action_space = gym.spaces.Discrete(2)
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    env.spec = Mock()
    env.spec.id = "TestEnv-v1"
    return env


@pytest.fixture
def stable_agent():
    """Агент со стабильными результатами."""
    rewards = [100.0] * 10  # Одинаковые награды
    return MockAgent(rewards)


@pytest.fixture
def variable_agent():
    """Агент с переменными результатами."""
    rewards = [50.0, 100.0, 150.0, 75.0, 125.0, 90.0, 110.0, 80.0, 120.0, 95.0]
    return MockAgent(rewards)


@pytest.fixture
def improving_agent():
    """Агент с улучшающимися результатами."""
    rewards = list(range(50, 150, 10))  # 50, 60, 70, ..., 140
    return MockAgent(rewards)


@pytest.fixture
def quantitative_evaluator(mock_env):
    """Создание количественного оценщика."""
    return QuantitativeEvaluator(
        env=mock_env,
        baseline_threshold=100.0,
        success_threshold=90.0,
        min_effect_size=0.5,
        random_seed=42,
    )


class TestQuantitativeEvaluator:
    """Тесты для класса QuantitativeEvaluator."""

    def test_init(self, mock_env):
        """Тест инициализации оценщика."""
        evaluator = QuantitativeEvaluator(
            env=mock_env,
            baseline_threshold=100.0,
            success_threshold=90.0,
            min_effect_size=0.5,
            random_seed=42,
        )

        assert evaluator.env == mock_env
        assert evaluator.baseline_threshold == 100.0
        assert evaluator.min_effect_size == 0.5
        assert evaluator.random_seed == 42
        assert evaluator.evaluator.success_threshold == 90.0

    def test_evaluate_agent_quantitative_basic(self, quantitative_evaluator, stable_agent):
        """Тест базовой количественной оценки агента."""
        with patch.object(
            quantitative_evaluator.evaluator, "evaluate_agent"
        ) as mock_evaluate:
            # Мокаем базовые метрики
            from src.evaluation.evaluator import EvaluationMetrics

            mock_metrics = EvaluationMetrics(
                mean_reward=100.0,
                std_reward=0.0,
                min_reward=100.0,
                max_reward=100.0,
                mean_episode_length=100.0,
                std_episode_length=0.0,
                min_episode_length=100,
                max_episode_length=100,
                reward_ci_lower=100.0,
                reward_ci_upper=100.0,
                success_rate=1.0,
                total_episodes=10,
                total_timesteps=1000,
                evaluation_time=10.0,
                episode_rewards=[100.0] * 10,
                episode_lengths=[100] * 10,
                episode_successes=[True] * 10,
            )
            mock_evaluate.return_value = mock_metrics

            result = quantitative_evaluator.evaluate_agent_quantitative(
                agent=stable_agent, num_episodes=10, agent_name="stable"
            )

            assert isinstance(result, QuantitativeMetrics)
            assert result.base_metrics == mock_metrics
            assert result.reward_median == 100.0
            assert result.reward_cv == 0.0  # Нет вариации
            assert result.reward_stability_score == 1.0  # Максимальная стабильность
            assert result.consecutive_success_rate == 1.0  # Все успешные

    def test_evaluate_agent_quantitative_invalid_episodes(self, quantitative_evaluator, stable_agent):
        """Тест с некорректным количеством эпизодов."""
        with pytest.raises(ValueError, match="num_episodes должно быть >= 5"):
            quantitative_evaluator.evaluate_agent_quantitative(
                agent=stable_agent, num_episodes=3
            )

    def test_calculate_quantitative_metrics_stable(self, quantitative_evaluator):
        """Тест расчета метрик для стабильного агента."""
        from src.evaluation.evaluator import EvaluationMetrics

        # Создаем базовые метрики для стабильного агента
        base_metrics = EvaluationMetrics(
            mean_reward=100.0,
            std_reward=0.0,
            min_reward=100.0,
            max_reward=100.0,
            mean_episode_length=50.0,
            std_episode_length=0.0,
            min_episode_length=50,
            max_episode_length=50,
            reward_ci_lower=100.0,
            reward_ci_upper=100.0,
            success_rate=1.0,
            total_episodes=10,
            total_timesteps=500,
            evaluation_time=5.0,
            episode_rewards=[100.0] * 10,
            episode_lengths=[50] * 10,
            episode_successes=[True] * 10,
        )

        result = quantitative_evaluator._calculate_quantitative_metrics(base_metrics)

        assert result.reward_median == 100.0
        assert result.reward_q25 == 100.0
        assert result.reward_q75 == 100.0
        assert result.reward_iqr == 0.0
        assert result.reward_cv == 0.0
        assert result.reward_stability_score == 1.0
        assert result.consecutive_success_rate == 1.0
        assert result.outlier_count == 0
        assert result.learning_efficiency == 20.0  # 100 / 5

    def test_calculate_quantitative_metrics_variable(self, quantitative_evaluator):
        """Тест расчета метрик для агента с переменными результатами."""
        from src.evaluation.evaluator import EvaluationMetrics

        rewards = [50.0, 100.0, 150.0, 75.0, 125.0]
        successes = [False, True, True, False, True]

        base_metrics = EvaluationMetrics(
            mean_reward=100.0,
            std_reward=np.std(rewards, ddof=1),
            min_reward=50.0,
            max_reward=150.0,
            mean_episode_length=100.0,
            std_episode_length=10.0,
            min_episode_length=90,
            max_episode_length=110,
            reward_ci_lower=80.0,
            reward_ci_upper=120.0,
            success_rate=0.6,
            total_episodes=5,
            total_timesteps=500,
            evaluation_time=10.0,
            episode_rewards=rewards,
            episode_lengths=[100, 95, 110, 90, 105],
            episode_successes=successes,
        )

        result = quantitative_evaluator._calculate_quantitative_metrics(base_metrics)

        assert result.reward_median == 100.0
        assert result.reward_q25 == 75.0
        assert result.reward_q75 == 125.0
        assert result.reward_iqr == 50.0
        assert result.reward_cv > 0  # Есть вариация
        assert result.reward_stability_score < 1.0  # Не максимальная стабильность
        assert 0 <= result.consecutive_success_rate <= 1
        assert result.learning_efficiency == 10.0  # 100 / 10

    def test_consecutive_success_rate_calculation(self, quantitative_evaluator):
        """Тест расчета доли последовательных успехов."""
        # Все успешные
        assert quantitative_evaluator._calculate_consecutive_success_rate([True, True, True]) == 1.0

        # Все неуспешные
        assert quantitative_evaluator._calculate_consecutive_success_rate([False, False, False]) == 0.0

        # Чередующиеся
        assert quantitative_evaluator._calculate_consecutive_success_rate([True, False, True, False]) == 0.0

        # Частично последовательные
        result = quantitative_evaluator._calculate_consecutive_success_rate([True, True, False, True])
        assert result == 1/3  # 1 последовательная пара из 3 возможных

        # Пустой список
        assert quantitative_evaluator._calculate_consecutive_success_rate([]) == 0.0

        # Один элемент
        assert quantitative_evaluator._calculate_consecutive_success_rate([True]) == 0.0

    def test_cohens_d_calculation(self, quantitative_evaluator):
        """Тест расчета размера эффекта Cohen's d."""
        group1 = [100.0, 110.0, 90.0, 105.0, 95.0]
        group2 = [80.0, 85.0, 75.0, 90.0, 70.0]

        d = quantitative_evaluator._calculate_cohens_d(group1, group2)
        
        # Проверяем, что размер эффекта положительный (group1 > group2)
        assert d > 0
        
        # Проверяем разумность значения (должно быть большим эффектом)
        assert d > 0.8

        # Тест с одинаковыми группами
        d_same = quantitative_evaluator._calculate_cohens_d(group1, group1)
        assert d_same == 0.0

        # Тест с нулевым стандартным отклонением
        d_zero_std = quantitative_evaluator._calculate_cohens_d([100.0] * 5, [100.0] * 5)
        assert d_zero_std == 0.0

    def test_compare_with_baseline_agent(self, quantitative_evaluator, stable_agent, variable_agent):
        """Тест сравнения с базовым агентом."""
        with patch.object(
            quantitative_evaluator, "evaluate_agent_quantitative"
        ) as mock_evaluate:
            # Мокаем метрики для обоих агентов
            from src.evaluation.evaluator import EvaluationMetrics

            stable_base = EvaluationMetrics(
                mean_reward=100.0, std_reward=0.0, min_reward=100.0, max_reward=100.0,
                mean_episode_length=100.0, std_episode_length=0.0,
                min_episode_length=100, max_episode_length=100,
                reward_ci_lower=100.0, reward_ci_upper=100.0, success_rate=1.0,
                total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
                episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
                episode_successes=[True] * 10
            )

            variable_base = EvaluationMetrics(
                mean_reward=120.0, std_reward=20.0, min_reward=80.0, max_reward=160.0,
                mean_episode_length=100.0, std_episode_length=10.0,
                min_episode_length=90, max_episode_length=110,
                reward_ci_lower=110.0, reward_ci_upper=130.0, success_rate=0.8,
                total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
                episode_rewards=[120.0, 100.0, 140.0, 110.0, 130.0, 90.0, 150.0, 105.0, 125.0, 95.0],
                episode_lengths=[100] * 10, episode_successes=[True] * 8 + [False] * 2
            )

            stable_metrics = QuantitativeMetrics(
                base_metrics=stable_base, reward_median=100.0, reward_q25=100.0,
                reward_q75=100.0, reward_iqr=0.0, reward_skewness=0.0,
                reward_kurtosis=0.0, length_median=100.0, length_q25=100.0,
                length_q75=100.0, length_iqr=0.0, reward_cv=0.0,
                reward_stability_score=1.0, consecutive_success_rate=1.0,
                reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
            )

            variable_metrics = QuantitativeMetrics(
                base_metrics=variable_base, reward_median=120.0, reward_q25=105.0,
                reward_q75=135.0, reward_iqr=30.0, reward_skewness=0.0,
                reward_kurtosis=0.0, length_median=100.0, length_q25=100.0,
                length_q75=100.0, length_iqr=0.0, reward_cv=0.167,
                reward_stability_score=0.833, consecutive_success_rate=0.7,
                reward_trend_slope=0.0, learning_efficiency=12.0, outlier_count=0
            )

            mock_evaluate.side_effect = [variable_metrics, stable_metrics]

            result = quantitative_evaluator.compare_with_baseline(
                agent=variable_agent,
                baseline_agent=stable_agent,
                agent_name="variable",
                baseline_name="stable"
            )

            assert isinstance(result, BaselineComparison)
            assert result.agent_name == "variable"
            assert result.baseline_name == "stable"
            assert result.is_better  # 120 > 100
            assert result.reward_improvement == 20.0  # (120-100)/100 * 100

    def test_compare_with_baseline_metrics(self, quantitative_evaluator, variable_agent):
        """Тест сравнения с готовыми метриками базовой линии."""
        from src.evaluation.evaluator import EvaluationMetrics

        # Создаем готовые метрики базовой линии
        baseline_base = EvaluationMetrics(
            mean_reward=80.0, std_reward=10.0, min_reward=60.0, max_reward=100.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=75.0, reward_ci_upper=85.0, success_rate=0.6,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[80.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 6 + [False] * 4
        )

        baseline_metrics = QuantitativeMetrics(
            base_metrics=baseline_base, reward_median=80.0, reward_q25=75.0,
            reward_q75=85.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=100.0,
            length_q75=100.0, length_iqr=0.0, reward_cv=0.125,
            reward_stability_score=0.875, consecutive_success_rate=0.5,
            reward_trend_slope=0.0, learning_efficiency=8.0, outlier_count=0
        )

        with patch.object(
            quantitative_evaluator, "evaluate_agent_quantitative"
        ) as mock_evaluate:
            # Мокаем только метрики агента
            agent_base = EvaluationMetrics(
                mean_reward=100.0, std_reward=15.0, min_reward=70.0, max_reward=130.0,
                mean_episode_length=100.0, std_episode_length=8.0,
                min_episode_length=85, max_episode_length=115,
                reward_ci_lower=90.0, reward_ci_upper=110.0, success_rate=0.8,
                total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
                episode_rewards=[100.0, 95.0, 105.0, 90.0, 110.0, 85.0, 115.0, 92.0, 108.0, 98.0],
                episode_lengths=[100] * 10,
                episode_successes=[True] * 8 + [False] * 2
            )

            agent_metrics = QuantitativeMetrics(
                base_metrics=agent_base, reward_median=100.0, reward_q25=90.0,
                reward_q75=110.0, reward_iqr=20.0, reward_skewness=0.0,
                reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
                length_q75=105.0, length_iqr=10.0, reward_cv=0.15,
                reward_stability_score=0.85, consecutive_success_rate=0.7,
                reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
            )

            mock_evaluate.return_value = agent_metrics

            result = quantitative_evaluator.compare_with_baseline(
                agent=variable_agent,
                baseline_metrics=baseline_metrics,
                agent_name="test_agent",
                baseline_name="baseline"
            )

            assert result.is_better  # 100 > 80
            assert result.reward_improvement == 25.0  # (100-80)/80 * 100
            assert result.effect_size > 0  # Положительный эффект

    def test_compare_with_baseline_no_baseline(self, quantitative_evaluator, variable_agent):
        """Тест сравнения без предоставления базовой линии."""
        with pytest.raises(ValueError, match="Необходимо предоставить либо baseline_agent"):
            quantitative_evaluator.compare_with_baseline(agent=variable_agent)

    def test_evaluate_multiple_agents_batch(self, quantitative_evaluator):
        """Тест пакетной оценки нескольких агентов."""
        agent1 = MockAgent([100.0] * 10)
        agent2 = MockAgent([120.0] * 10)
        agent3 = MockAgent([80.0] * 10)

        agents = {"agent1": agent1, "agent2": agent2, "agent3": agent3}

        with patch.object(
            quantitative_evaluator, "evaluate_agent_quantitative"
        ) as mock_evaluate:
            # Мокаем метрики для всех агентов
            from src.evaluation.evaluator import EvaluationMetrics

            def create_mock_metrics(mean_reward):
                base = EvaluationMetrics(
                    mean_reward=mean_reward, std_reward=5.0, min_reward=mean_reward-10,
                    max_reward=mean_reward+10, mean_episode_length=100.0, std_episode_length=5.0,
                    min_episode_length=95, max_episode_length=105,
                    reward_ci_lower=mean_reward-5, reward_ci_upper=mean_reward+5, success_rate=0.8,
                    total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
                    episode_rewards=[mean_reward] * 10, episode_lengths=[100] * 10,
                    episode_successes=[True] * 8 + [False] * 2
                )
                return QuantitativeMetrics(
                    base_metrics=base, reward_median=mean_reward, reward_q25=mean_reward-5,
                    reward_q75=mean_reward+5, reward_iqr=10.0, reward_skewness=0.0,
                    reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
                    length_q75=105.0, length_iqr=10.0, reward_cv=0.05,
                    reward_stability_score=0.95, consecutive_success_rate=0.7,
                    reward_trend_slope=0.0, learning_efficiency=mean_reward/10, outlier_count=0
                )

            mock_evaluate.side_effect = [
                create_mock_metrics(100.0),  # agent1
                create_mock_metrics(120.0),  # agent2
                create_mock_metrics(80.0),   # agent3
            ]

            with patch.object(quantitative_evaluator.evaluator, "compare_agents") as mock_compare:
                from src.evaluation.evaluator import ComparisonResult

                mock_compare.return_value = ComparisonResult(
                    agent1_name="agent1", agent2_name="agent2",
                    reward_ttest_statistic=2.0, reward_ttest_pvalue=0.05,
                    reward_significant=True, length_ttest_statistic=1.0,
                    length_ttest_pvalue=0.3, length_significant=False,
                    reward_effect_size=0.8, length_effect_size=0.2,
                    better_agent="agent2", confidence_level=0.95
                )

                result = quantitative_evaluator.evaluate_multiple_agents_batch(
                    agents=agents, num_episodes=10, include_pairwise_comparison=True
                )

                assert isinstance(result, BatchEvaluationResult)
                assert len(result.agents_metrics) == 3
                assert result.best_agent == "agent2"
                assert result.ranking[0] == ("agent2", 120.0)
                assert result.ranking[1] == ("agent1", 100.0)
                assert result.ranking[2] == ("agent3", 80.0)
                assert result.statistical_summary["total_agents"] == 3
                assert result.statistical_summary["best_agent"] == "agent2"

    def test_generate_comprehensive_report_text(self, quantitative_evaluator):
        """Тест генерации текстового отчета."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        report = quantitative_evaluator.generate_comprehensive_report(
            metrics=metrics, format_type="text"
        )

        assert "КОМПЛЕКСНЫЙ ОТЧЕТ КОЛИЧЕСТВЕННОЙ ОЦЕНКИ АГЕНТА" in report
        assert "Среднее: 100.000" in report
        assert "Медиана: 100.000" in report
        assert "Оценка стабильности: 0.900" in report
        assert "Эффективность обучения: 10.000" in report

    def test_generate_comprehensive_report_json(self, quantitative_evaluator):
        """Тест генерации JSON отчета."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        report = quantitative_evaluator.generate_comprehensive_report(
            metrics=metrics, format_type="json"
        )

        data = json.loads(report)
        assert data["mean_reward"] == 100.0
        assert data["median_reward"] == 100.0
        assert data["stability_score"] == 0.9
        assert data["learning_efficiency"] == 10.0

    def test_generate_comprehensive_report_csv(self, quantitative_evaluator):
        """Тест генерации CSV отчета."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        report = quantitative_evaluator.generate_comprehensive_report(
            metrics=metrics, format_type="csv"
        )

        # Проверяем, что это валидный CSV
        lines = report.strip().split('\n')
        assert len(lines) >= 2  # Заголовок + данные
        assert "mean_reward" in lines[0]
        assert "100.0" in lines[1]

    def test_generate_comprehensive_report_invalid_format(self, quantitative_evaluator):
        """Тест с неподдерживаемым форматом отчета."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            quantitative_evaluator.generate_comprehensive_report(
                metrics=metrics, format_type="xml"
            )

    def test_visualize_results_single_agent(self, quantitative_evaluator):
        """Тест визуализации для одного агента."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0, 95.0, 105.0, 90.0, 110.0, 85.0, 115.0, 92.0, 108.0, 98.0],
            episode_lengths=[100, 95, 105, 90, 110, 85, 115, 92, 108, 98],
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        # Тестируем без показа графиков
        with patch("matplotlib.pyplot.show") as mock_show:
            quantitative_evaluator.visualize_results(
                metrics=metrics, show_plots=False
            )
            mock_show.assert_not_called()

    def test_save_report_to_file(self, quantitative_evaluator):
        """Тест сохранения отчета в файл."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_report.txt"
            
            report = quantitative_evaluator.generate_comprehensive_report(
                metrics=metrics, save_path=save_path, format_type="text"
            )

            assert save_path.exists()
            saved_content = save_path.read_text(encoding="utf-8")
            assert saved_content == report
            assert "КОМПЛЕКСНЫЙ ОТЧЕТ" in saved_content


class TestConvenienceFunctions:
    """Тесты для удобных функций."""

    def test_evaluate_agent_standard(self, mock_env):
        """Тест стандартной оценки агента."""
        agent = MockAgent([100.0] * 20)

        with patch("src.evaluation.quantitative_eval.QuantitativeEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator
            
            # Создаем мок-метрики
            from src.evaluation.evaluator import EvaluationMetrics
            base_metrics = EvaluationMetrics(
                mean_reward=100.0, std_reward=5.0, min_reward=90.0, max_reward=110.0,
                mean_episode_length=100.0, std_episode_length=5.0,
                min_episode_length=90, max_episode_length=110,
                reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=1.0,
                total_episodes=20, total_timesteps=2000, evaluation_time=20.0,
                episode_rewards=[100.0] * 20, episode_lengths=[100] * 20,
                episode_successes=[True] * 20
            )

            mock_metrics = QuantitativeMetrics(
                base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
                reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
                reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
                length_q75=105.0, length_iqr=10.0, reward_cv=0.05,
                reward_stability_score=0.95, consecutive_success_rate=1.0,
                reward_trend_slope=0.0, learning_efficiency=5.0, outlier_count=0
            )

            mock_evaluator.evaluate_agent_quantitative.return_value = mock_metrics

            result = evaluate_agent_standard(
                agent=agent, env=mock_env, num_episodes=20, agent_name="test"
            )

            assert isinstance(result, QuantitativeMetrics)
            mock_evaluator.evaluate_agent_quantitative.assert_called_once_with(
                agent=agent, num_episodes=20, agent_name="test"
            )

    def test_compare_agents_statistical(self, mock_env):
        """Тест статистического сравнения агентов."""
        agents = {
            "agent1": MockAgent([100.0] * 20),
            "agent2": MockAgent([120.0] * 20),
        }

        with patch("src.evaluation.quantitative_eval.QuantitativeEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator
            
            mock_result = BatchEvaluationResult(
                agents_metrics={},
                comparison_matrix={},
                ranking=[("agent2", 120.0), ("agent1", 100.0)],
                best_agent="agent2",
                statistical_summary={"total_agents": 2}
            )

            mock_evaluator.evaluate_multiple_agents_batch.return_value = mock_result

            result = compare_agents_statistical(
                agents=agents, env=mock_env, num_episodes=20
            )

            assert isinstance(result, BatchEvaluationResult)
            assert result.best_agent == "agent2"
            mock_evaluator.evaluate_multiple_agents_batch.assert_called_once_with(
                agents=agents, num_episodes=20
            )

    def test_analyze_agent_stability(self, mock_env):
        """Тест анализа стабильности агента."""
        agent = MockAgent([100.0] * 20)

        with patch("src.evaluation.quantitative_eval.QuantitativeEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator
            
            # Создаем мок-метрики для разных запусков
            from src.evaluation.evaluator import EvaluationMetrics

            def create_mock_metrics(mean_reward):
                base_metrics = EvaluationMetrics(
                    mean_reward=mean_reward, std_reward=5.0, min_reward=mean_reward-10,
                    max_reward=mean_reward+10, mean_episode_length=100.0, std_episode_length=5.0,
                    min_episode_length=90, max_episode_length=110,
                    reward_ci_lower=mean_reward-5, reward_ci_upper=mean_reward+5, success_rate=1.0,
                    total_episodes=20, total_timesteps=2000, evaluation_time=20.0,
                    episode_rewards=[mean_reward] * 20, episode_lengths=[100] * 20,
                    episode_successes=[True] * 20
                )

                return QuantitativeMetrics(
                    base_metrics=base_metrics, reward_median=mean_reward, reward_q25=mean_reward-5,
                    reward_q75=mean_reward+5, reward_iqr=10.0, reward_skewness=0.0,
                    reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
                    length_q75=105.0, length_iqr=10.0, reward_cv=0.05,
                    reward_stability_score=0.95, consecutive_success_rate=1.0,
                    reward_trend_slope=0.0, learning_efficiency=mean_reward/20, outlier_count=0
                )

            # Мокаем разные результаты для разных запусков
            mock_evaluator.evaluate_agent_quantitative.side_effect = [
                create_mock_metrics(100.0),  # run 0
                create_mock_metrics(105.0),  # run 1
                create_mock_metrics(95.0),   # run 2
            ]

            result = analyze_agent_stability(
                agent=agent, env=mock_env, num_runs=3, episodes_per_run=20, agent_name="test"
            )

            assert "inter_run_stability" in result
            assert "run_metrics" in result
            assert result["agent_name"] == "test"
            
            stability = result["inter_run_stability"]
            assert stability["num_runs"] == 3
            assert stability["episodes_per_run"] == 20
            assert "mean_reward_across_runs" in stability
            assert "cv_across_runs" in stability

            # Проверяем, что было 3 вызова с разными семенами
            assert mock_evaluator.evaluate_agent_quantitative.call_count == 3


class TestDataClasses:
    """Тесты для dataclass'ов."""

    def test_quantitative_metrics_creation(self):
        """Тест создания QuantitativeMetrics."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics,
            reward_median=100.0,
            reward_q25=95.0,
            reward_q75=105.0,
            reward_iqr=10.0,
            reward_skewness=0.0,
            reward_kurtosis=0.0,
            length_median=100.0,
            length_q25=95.0,
            length_q75=105.0,
            length_iqr=10.0,
            reward_cv=0.1,
            reward_stability_score=0.9,
            consecutive_success_rate=0.7,
            reward_trend_slope=0.0,
            learning_efficiency=10.0,
            outlier_count=0,
        )

        assert metrics.base_metrics == base_metrics
        assert metrics.reward_median == 100.0
        assert metrics.reward_stability_score == 0.9
        assert metrics.learning_efficiency == 10.0

    def test_baseline_comparison_creation(self):
        """Тест создания BaselineComparison."""
        from src.evaluation.evaluator import EvaluationMetrics

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        agent_metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        baseline_metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=80.0, reward_q25=75.0,
            reward_q75=85.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.15,
            reward_stability_score=0.85, consecutive_success_rate=0.6,
            reward_trend_slope=0.0, learning_efficiency=8.0, outlier_count=0
        )

        comparison = BaselineComparison(
            agent_name="test_agent",
            baseline_name="baseline",
            agent_metrics=agent_metrics,
            baseline_metrics=baseline_metrics,
            reward_improvement=25.0,
            reward_ttest_pvalue=0.01,
            reward_wilcoxon_pvalue=0.02,
            reward_significant=True,
            effect_size=1.2,
            practical_significance=True,
            is_better=True,
            confidence_level=0.95,
        )

        assert comparison.agent_name == "test_agent"
        assert comparison.baseline_name == "baseline"
        assert comparison.is_better
        assert comparison.practical_significance
        assert comparison.reward_improvement == 25.0

    def test_batch_evaluation_result_creation(self):
        """Тест создания BatchEvaluationResult."""
        from src.evaluation.evaluator import EvaluationMetrics, ComparisonResult

        base_metrics = EvaluationMetrics(
            mean_reward=100.0, std_reward=10.0, min_reward=80.0, max_reward=120.0,
            mean_episode_length=100.0, std_episode_length=5.0,
            min_episode_length=90, max_episode_length=110,
            reward_ci_lower=95.0, reward_ci_upper=105.0, success_rate=0.8,
            total_episodes=10, total_timesteps=1000, evaluation_time=10.0,
            episode_rewards=[100.0] * 10, episode_lengths=[100] * 10,
            episode_successes=[True] * 8 + [False] * 2
        )

        metrics = QuantitativeMetrics(
            base_metrics=base_metrics, reward_median=100.0, reward_q25=95.0,
            reward_q75=105.0, reward_iqr=10.0, reward_skewness=0.0,
            reward_kurtosis=0.0, length_median=100.0, length_q25=95.0,
            length_q75=105.0, length_iqr=10.0, reward_cv=0.1,
            reward_stability_score=0.9, consecutive_success_rate=0.7,
            reward_trend_slope=0.0, learning_efficiency=10.0, outlier_count=0
        )

        comparison = ComparisonResult(
            agent1_name="agent1", agent2_name="agent2",
            reward_ttest_statistic=2.0, reward_ttest_pvalue=0.05,
            reward_significant=True, length_ttest_statistic=1.0,
            length_ttest_pvalue=0.3, length_significant=False,
            reward_effect_size=0.8, length_effect_size=0.2,
            better_agent="agent2", confidence_level=0.95
        )

        result = BatchEvaluationResult(
            agents_metrics={"agent1": metrics, "agent2": metrics},
            comparison_matrix={("agent1", "agent2"): comparison},
            ranking=[("agent2", 120.0), ("agent1", 100.0)],
            best_agent="agent2",
            statistical_summary={"total_agents": 2, "evaluation_time": 30.0},
        )

        assert len(result.agents_metrics) == 2
        assert len(result.comparison_matrix) == 1
        assert result.best_agent == "agent2"
        assert result.ranking[0][0] == "agent2"
        assert result.statistical_summary["total_agents"] == 2