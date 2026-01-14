"""
Тесты для модуля оценки агентов.

Проверяет корректность работы Evaluator и связанных классов.
"""

import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from src.agents.base import Agent, TrainingResult
from src.evaluation.evaluator import (
    ComparisonResult,
    EvaluationMetrics,
    Evaluator,
)


class MockAgent(Agent):
    """Мок-агент для тестирования."""
    
    def __init__(self, reward_pattern: str = "constant") -> None:
        """
        Инициализация мок-агента.
        
        Args:
            reward_pattern: Паттерн наград ("constant", "random", "increasing")
        """
        self.reward_pattern = reward_pattern
        self.step_count = 0
        # Не вызываем super().__init__() чтобы избежать создания модели
    
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True, 
        **kwargs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание действия."""
        self.step_count += 1
        
        if self.reward_pattern == "constant":
            action = 0  # Всегда одно действие
        elif self.reward_pattern == "random":
            action = np.random.randint(0, 2)
        else:  # increasing
            action = self.step_count % 2
        
        return np.array([action]), None
    
    def _create_model(self) -> Any:
        """Заглушка для создания модели."""
        return None
    
    def train(self, *args, **kwargs) -> TrainingResult:
        """Заглушка для обучения."""
        return TrainingResult(
            total_timesteps=1000,
            training_time=1.0,
            final_mean_reward=100.0,
            final_std_reward=10.0,
        )
    
    def save(self, path: str) -> None:
        """Заглушка для сохранения."""
        pass
    
    @classmethod
    def load(cls, path: str, env: Optional[Any] = None, **kwargs: Any) -> "MockAgent":
        """Заглушка для загрузки."""
        return cls()


class MockEnv:
    """Мок-среда для тестирования."""
    
    def __init__(self, episode_length: int = 10, base_reward: float = 1.0) -> None:
        self.episode_length = episode_length
        self.base_reward = base_reward
        self.current_step = 0
        self.spec = Mock()
        self.spec.id = "MockEnv-v0"
    
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Сброс среды."""
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
        return np.array([0.0, 0.0]), {}  # type: ignore
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Шаг в среде."""
        self.current_step += 1
        
        # Простая логика награды
        reward = self.base_reward + np.random.normal(0, 0.1)
        
        # Эпизод заканчивается через episode_length шагов
        done = self.current_step >= self.episode_length
        truncated = False
        
        obs = np.array([float(self.current_step), float(action)])
        
        return obs, reward, done, truncated, {}
    
    def render(self) -> None:
        """Заглушка для рендеринга."""
        pass


@pytest.fixture
def mock_env() -> MockEnv:
    """Фикстура мок-среды."""
    return MockEnv(episode_length=5, base_reward=10.0)


@pytest.fixture
def evaluator(mock_env: MockEnv) -> Evaluator:
    """Фикстура оценщика."""
    return Evaluator(
        env=mock_env,  # type: ignore  # type: ignore
        success_threshold=50.0,
        confidence_level=0.95,
        random_seed=42,
    )


@pytest.fixture
def mock_agent() -> MockAgent:
    """Фикстура мок-агента."""
    return MockAgent(reward_pattern="constant")


class TestEvaluationMetrics:
    """Тесты для класса EvaluationMetrics."""
    
    def test_metrics_creation(self) -> None:
        """Тест создания метрик."""
        metrics = EvaluationMetrics(
            mean_reward=100.0,
            std_reward=10.0,
            min_reward=80.0,
            max_reward=120.0,
            mean_episode_length=50.0,
            std_episode_length=5.0,
            min_episode_length=40,
            max_episode_length=60,
            reward_ci_lower=95.0,
            reward_ci_upper=105.0,
            success_rate=0.8,
            total_episodes=100,
            total_timesteps=5000,
            evaluation_time=60.0,
        )
        
        assert metrics.mean_reward == 100.0
        assert metrics.success_rate == 0.8
        assert metrics.total_episodes == 100
        assert len(metrics.episode_rewards) == 0  # По умолчанию пустой


class TestEvaluator:
    """Тесты для класса Evaluator."""
    
    def test_evaluator_creation(self, mock_env: MockEnv) -> None:
        """Тест создания оценщика."""
        evaluator = Evaluator(
            env=mock_env,  # type: ignore
            success_threshold=50.0,
            confidence_level=0.95,
            random_seed=42,
        )
        
        assert evaluator.env == mock_env
        assert evaluator.success_threshold == 50.0
        assert evaluator.confidence_level == 0.95
        assert evaluator.random_seed == 42
    
    def test_evaluate_agent_basic(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест базовой оценки агента."""
        metrics = evaluator.evaluate_agent(
            agent=mock_agent,
            num_episodes=5,
            agent_name="test_agent",
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_episodes == 5
        assert metrics.mean_reward > 0
        assert len(metrics.episode_rewards) == 5
        assert len(metrics.episode_lengths) == 5
        assert len(metrics.episode_successes) == 5
    
    def test_evaluate_agent_with_max_steps(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест оценки агента с ограничением шагов."""
        metrics = evaluator.evaluate_agent(
            agent=mock_agent,
            num_episodes=3,
            max_steps_per_episode=3,
        )
        
        assert metrics.total_episodes == 3
        assert all(length <= 3 for length in metrics.episode_lengths)
    
    def test_evaluate_agent_invalid_episodes(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест оценки с некорректным количеством эпизодов."""
        with pytest.raises(ValueError, match="num_episodes должно быть > 0"):
            evaluator.evaluate_agent(mock_agent, num_episodes=0)
        
        with pytest.raises(ValueError, match="num_episodes должно быть > 0"):
            evaluator.evaluate_agent(mock_agent, num_episodes=-1)
    
    def test_evaluate_agent_caching(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест кэширования результатов оценки."""
        # Первая оценка
        metrics1 = evaluator.evaluate_agent(
            agent=mock_agent,
            num_episodes=3,
            agent_name="cached_agent",
            use_cache=True,
        )
        
        # Вторая оценка (должна использовать кэш)
        metrics2 = evaluator.evaluate_agent(
            agent=mock_agent,
            num_episodes=3,
            agent_name="cached_agent",
            use_cache=True,
        )
        
        # Результаты должны быть идентичными
        assert metrics1.episode_rewards == metrics2.episode_rewards
        assert metrics1.mean_reward == metrics2.mean_reward
    
    def test_compare_agents(self, evaluator: Evaluator) -> None:
        """Тест сравнения агентов."""
        agent1 = MockAgent(reward_pattern="constant")
        agent2 = MockAgent(reward_pattern="random")
        
        result = evaluator.compare_agents(
            agent1=agent1,
            agent2=agent2,
            num_episodes=10,
            agent1_name="ConstantAgent",
            agent2_name="RandomAgent",
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.agent1_name == "ConstantAgent"
        assert result.agent2_name == "RandomAgent"
        assert result.better_agent in ["ConstantAgent", "RandomAgent"]
        assert 0 <= result.reward_ttest_pvalue <= 1
        assert result.reward_significant in [True, False]
    
    def test_evaluate_multiple_agents(self, evaluator: Evaluator) -> None:
        """Тест оценки нескольких агентов."""
        agents = {
            "agent1": MockAgent(reward_pattern="constant"),
            "agent2": MockAgent(reward_pattern="random"),
            "agent3": MockAgent(reward_pattern="increasing"),
        }
        
        results = evaluator.evaluate_multiple_agents(
            agents=agents,  # type: ignore
            num_episodes=5,
        )
        
        assert len(results) == 3
        assert "agent1" in results
        assert "agent2" in results
        assert "agent3" in results
        
        for metrics in results.values():
            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.total_episodes == 5
    
    def test_generate_report_single_agent(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест генерации отчета для одного агента."""
        metrics = evaluator.evaluate_agent(mock_agent, num_episodes=5)
        report = evaluator.generate_report(metrics)
        
        assert "ОТЧЕТ ОЦЕНКИ АГЕНТА" in report
        assert "Количество эпизодов: 5" in report
        assert "Среднее:" in report
        assert "Доверительный интервал" in report
    
    def test_generate_report_multiple_agents(self, evaluator: Evaluator) -> None:
        """Тест генерации отчета для нескольких агентов."""
        agents = {
            "agent1": MockAgent(reward_pattern="constant"),
            "agent2": MockAgent(reward_pattern="random"),
        }
        
        results = evaluator.evaluate_multiple_agents(agents, num_episodes=3)
        report = evaluator.generate_report(results)
        
        assert "СРАВНИТЕЛЬНЫЙ ОТЧЕТ АГЕНТОВ" in report
        assert "Рейтинг по средней награде:" in report
        assert "agent1" in report
        assert "agent2" in report
    
    def test_export_to_dataframe_single(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест экспорта метрик одного агента в DataFrame."""
        metrics = evaluator.evaluate_agent(mock_agent, num_episodes=3)
        df = evaluator.export_to_dataframe(metrics)
        
        assert len(df) == 1
        assert "agent_name" in df.columns
        assert "mean_reward" in df.columns
        assert "success_rate" in df.columns
        assert df.iloc[0]["agent_name"] == "agent"
    
    def test_export_to_dataframe_multiple(self, evaluator: Evaluator) -> None:
        """Тест экспорта метрик нескольких агентов в DataFrame."""
        agents = {
            "agent1": MockAgent(reward_pattern="constant"),
            "agent2": MockAgent(reward_pattern="random"),
        }
        
        results = evaluator.evaluate_multiple_agents(agents, num_episodes=3)
        df = evaluator.export_to_dataframe(results)
        
        assert len(df) == 2
        assert set(df["agent_name"]) == {"agent1", "agent2"}
        assert all(col in df.columns for col in [
            "mean_reward", "std_reward", "success_rate"
        ])
    
    def test_save_report_to_file(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест сохранения отчета в файл."""
        metrics = evaluator.evaluate_agent(mock_agent, num_episodes=3)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            save_path = Path(f.name)
        
        try:
            report = evaluator.generate_report(metrics, save_path=save_path)
            
            # Проверяем, что файл создан и содержит отчет
            assert save_path.exists()
            saved_content = save_path.read_text(encoding='utf-8')
            assert saved_content == report
            assert "ОТЧЕТ ОЦЕНКИ АГЕНТА" in saved_content
        finally:
            save_path.unlink()  # Удаляем временный файл
    
    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_confidence_intervals(
        self, 
        mock_env: MockEnv, 
        mock_agent: MockAgent,
        confidence_level: float
    ) -> None:
        """Тест расчета доверительных интервалов."""
        evaluator = Evaluator(
            env=mock_env,  # type: ignore
            confidence_level=confidence_level,
            random_seed=42,
        )
        
        metrics = evaluator.evaluate_agent(mock_agent, num_episodes=10)
        
        # Проверяем, что доверительный интервал корректен
        assert metrics.reward_ci_lower <= metrics.mean_reward
        assert metrics.reward_ci_upper >= metrics.mean_reward
        assert metrics.reward_ci_lower < metrics.reward_ci_upper
    
    def test_success_rate_calculation(self, mock_env: MockEnv) -> None:
        """Тест расчета доли успешных эпизодов."""
        evaluator = Evaluator(
            env=mock_env,  # type: ignore
            success_threshold=45.0,  # Порог ниже базовой награды
            random_seed=42,
        )
        
        agent = MockAgent(reward_pattern="constant")
        metrics = evaluator.evaluate_agent(agent, num_episodes=5)
        
        # При базовой награде 10.0 * 5 шагов = 50.0, что выше порога 45.0
        assert metrics.success_rate > 0
        assert 0 <= metrics.success_rate <= 1
    
    def test_cohens_d_calculation(self, evaluator: Evaluator) -> None:
        """Тест расчета размера эффекта Cohen's d."""
        # Тестируем приватный метод
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        effect_size = evaluator._calculate_cohens_d(group1, group2)
        
        # Ожидаем отрицательный эффект (group1 < group2)
        assert effect_size < 0
        assert abs(effect_size) > 0  # Должен быть ненулевой
    
    def test_cohens_d_identical_groups(self, evaluator: Evaluator) -> None:
        """Тест Cohen's d для идентичных групп."""
        group1 = [1.0, 2.0, 3.0]
        group2 = [1.0, 2.0, 3.0]
        
        effect_size = evaluator._calculate_cohens_d(group1, group2)
        
        assert effect_size == 0.0


class TestEvaluationCallback:
    """Тесты для callback'ов оценки."""
    
    def test_callback_integration(
        self, 
        evaluator: Evaluator, 
        mock_agent: MockAgent
    ) -> None:
        """Тест интеграции callback'ов."""
        callback_calls = {
            "episode_start": 0,
            "episode_end": 0,
            "evaluation_end": 0,
        }
        
        class TestCallback:
            def on_episode_start(self, episode: int) -> None:
                callback_calls["episode_start"] += 1
            
            def on_episode_end(
                self, 
                episode: int, 
                reward: float, 
                length: int, 
                success: bool
            ) -> None:
                callback_calls["episode_end"] += 1
            
            def on_evaluation_end(self, metrics: EvaluationMetrics) -> None:
                callback_calls["evaluation_end"] += 1
        
        callback = TestCallback()
        
        evaluator.evaluate_agent(
            agent=mock_agent,
            num_episodes=3,
            callback=callback,
        )
        
        assert callback_calls["episode_start"] == 3
        assert callback_calls["episode_end"] == 3
        assert callback_calls["evaluation_end"] == 1


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_reproducibility(mock_env: MockEnv, seed: int) -> None:
    """Тест воспроизводимости результатов."""
    evaluator1 = Evaluator(env=mock_env,  # type: ignore
                           random_seed=seed)
    evaluator2 = Evaluator(env=mock_env,  # type: ignore
                           random_seed=seed)
    
    agent1 = MockAgent(reward_pattern="random")
    agent2 = MockAgent(reward_pattern="random")
    
    metrics1 = evaluator1.evaluate_agent(agent1, num_episodes=5, use_cache=False)
    metrics2 = evaluator2.evaluate_agent(agent2, num_episodes=5, use_cache=False)
    
    # При одинаковом семени результаты должны быть близкими
    # (точное равенство может не достигаться из-за особенностей мок-среды)
    assert abs(metrics1.mean_reward - metrics2.mean_reward) < 1.0