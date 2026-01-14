"""Тесты для базового класса Agent.

Проверяет корректность инициализации, валидации конфигурации,
создания среды и основных методов базового класса.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np

from src.agents.base import Agent, AgentConfig, TrainingResult, ModelProtocol


class MockModel:
    """Мок-модель для тестирования."""
    
    def __init__(self):
        self.saved_path = None
    
    def learn(self, total_timesteps, callback=None, **kwargs):
        return self
    
    def predict(self, observation, deterministic=True):
        # Возвращаем случайное действие для тестирования
        action = np.array([0])  # Простое действие для дискретной среды
        return action, None
    
    def save(self, path):
        self.saved_path = path
        # Создаем файл для тестирования
        from pathlib import Path
        Path(path).touch()
    
    @classmethod
    def load(cls, path, env=None):
        instance = cls()
        instance.saved_path = path
        return instance


class MockAgent(Agent):
    """Тестовая реализация абстрактного класса Agent."""
    
    def _create_model(self):
        return MockModel()
    
    def train(self, total_timesteps=None, callback=None, **kwargs):
        if self.model is None:
            self.model = self._create_model()
        
        timesteps = total_timesteps or self.config.total_timesteps
        
        # Симуляция обучения
        self.model.learn(timesteps, callback=callback, **kwargs)
        self.is_trained = True
        
        # Создание результата обучения
        self.training_result = TrainingResult(
            total_timesteps=timesteps,
            training_time=10.0,
            final_mean_reward=100.0,
            final_std_reward=10.0,
            episode_rewards=[90, 95, 100, 105, 110],
            episode_lengths=[200, 195, 200, 205, 190],
            best_mean_reward=110.0,
        )
        
        return self.training_result
    
    def predict(self, observation, deterministic=True, **kwargs):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    @classmethod
    def load(cls, path, env=None, **kwargs):
        # Загрузка конфигурации из YAML файла
        config_path = Path(path).with_suffix('.yaml')
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            config = AgentConfig(**config_dict)
        else:
            # Конфигурация по умолчанию для тестов
            config = AgentConfig(
                algorithm="PPO",
                env_name="CartPole-v1",
                total_timesteps=1000,
            )
        
        agent = cls(config=config, env=env)
        agent.model = MockModel.load(path, env)
        agent.is_trained = True
        
        return agent


class MockAgentConfig:
    """Тесты для класса AgentConfig."""
    
    def test_valid_config(self):
        """Тест создания валидной конфигурации."""
        config = AgentConfig(
            algorithm="PPO",
            env_name="CartPole-v1",
            total_timesteps=10000,
            learning_rate=3e-4,
            seed=42,
        )
        
        assert config.algorithm == "PPO"
        assert config.env_name == "CartPole-v1"
        assert config.total_timesteps == 10000
        assert config.learning_rate == 3e-4
        assert config.seed == 42
        assert config.policy_kwargs == {}  # Значение по умолчанию
    
    def test_invalid_total_timesteps(self):
        """Тест валидации total_timesteps."""
        with pytest.raises(ValueError, match="total_timesteps должен быть > 0"):
            AgentConfig(
                algorithm="PPO",
                env_name="CartPole-v1",
                total_timesteps=0,
            )
    
    def test_invalid_learning_rate(self):
        """Тест валидации learning_rate."""
        with pytest.raises(ValueError, match="learning_rate должен быть > 0"):
            AgentConfig(
                algorithm="PPO",
                env_name="CartPole-v1",
                total_timesteps=1000,
                learning_rate=0,
            )
    
    def test_invalid_gamma(self):
        """Тест валидации gamma."""
        with pytest.raises(ValueError, match="gamma должен быть в \\[0, 1\\]"):
            AgentConfig(
                algorithm="PPO",
                env_name="CartPole-v1",
                total_timesteps=1000,
                gamma=1.5,
            )
    
    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = AgentConfig(
            algorithm="PPO",
            env_name="CartPole-v1",
        )
        
        assert config.total_timesteps == 100_000
        assert config.seed == 42
        assert config.learning_rate == 3e-4
        assert config.policy_kwargs == {}


class TestTrainingResult:
    """Тесты для класса TrainingResult."""
    
    def test_training_result_creation(self):
        """Тест создания результата обучения."""
        result = TrainingResult(
            total_timesteps=10000,
            training_time=120.5,
            final_mean_reward=95.5,
            final_std_reward=5.2,
            episode_rewards=[90, 95, 100],
            best_mean_reward=100.0,
        )
        
        assert result.total_timesteps == 10000
        assert result.training_time == 120.5
        assert result.final_mean_reward == 95.5
        assert result.best_mean_reward == 100.0
        assert len(result.episode_rewards) == 3


class MockAgentBase:
    """Тесты для базового класса Agent."""
    
    @pytest.fixture
    def config(self):
        """Конфигурация для тестов."""
        return AgentConfig(
            algorithm="PPO",
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
        )
    
    @pytest.fixture
    def mock_env(self):
        """Мок-среда для тестов."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.__class__.__name__ = "Discrete"
        env.observation_space = Mock()
        env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        env.step.return_value = (
            np.array([0, 0, 0, 0]),  # observation
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}  # info
        )
        return env
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_agent_initialization(
        self, 
        mock_metrics, 
        mock_logger, 
        mock_seed, 
        mock_gym_make, 
        config, 
        mock_env
    ):
        """Тест инициализации агента."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        
        # Проверяем вызовы
        mock_seed.assert_called_once_with(42)
        mock_gym_make.assert_called_once_with("CartPole-v1")
        mock_logger.assert_called_once()
        
        # Проверяем состояние агента
        assert agent.config == config
        assert agent.env == mock_env
        assert not agent.is_trained
        assert agent.training_result is None
    
    @patch('src.agents.base.gym.make')
    def test_environment_creation_error(self, mock_gym_make, config):
        """Тест обработки ошибки создания среды."""
        mock_gym_make.side_effect = Exception("Среда не найдена")
        
        with pytest.raises(RuntimeError, match="Ошибка создания среды"):
            MockAgent(config=config)
    
    @pytest.mark.parametrize("algorithm,action_space_type,should_raise", [
        ("PPO", "Discrete", False),
        ("A2C", "Discrete", False),
        ("DQN", "Discrete", False),
        ("SAC", "Discrete", True),  # SAC не поддерживает дискретные действия
        ("TD3", "Discrete", True),  # TD3 не поддерживает дискретные действия
        ("PPO", "Box", False),
        ("SAC", "Box", False),
        ("DQN", "Box", True),  # DQN не поддерживает непрерывные действия
    ])
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_env_algorithm_compatibility(
        self,
        mock_metrics,
        mock_logger,
        mock_seed,
        mock_gym_make,
        algorithm,
        action_space_type,
        should_raise,
    ):
        """Тест проверки совместимости среды и алгоритма."""
        # Настройка мок-среды
        mock_env = Mock()
        if action_space_type == "Discrete":
            mock_env.action_space = Mock(spec=['n'])
            mock_env.action_space.__class__.__name__ = "Discrete"
        else:  # Box
            mock_env.action_space = Mock(spec=['shape', 'low', 'high'])
            mock_env.action_space.__class__.__name__ = "Box"
        
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        config = AgentConfig(
            algorithm=algorithm,
            env_name="TestEnv-v1",
            total_timesteps=1000,
        )
        
        if should_raise:
            with pytest.raises(ValueError, match="не поддерживает"):
                MockAgent(config=config)
        else:
            agent = MockAgent(config=config)
            assert agent.config.algorithm == algorithm
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_training(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест обучения агента."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        result = agent.train()
        
        assert agent.is_trained
        assert isinstance(result, TrainingResult)
        assert result.total_timesteps == 1000
        assert result.final_mean_reward == 100.0
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_prediction_before_training(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест предсказания до обучения."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        observation = np.array([0, 0, 0, 0])
        
        with pytest.raises(RuntimeError, match="Модель не обучена"):
            agent.predict(observation)
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_prediction_after_training(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест предсказания после обучения."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        agent.train()
        
        observation = np.array([0, 0, 0, 0])
        action, state = agent.predict(observation)
        
        assert isinstance(action, np.ndarray)
        assert state is None  # Для нашего мок-объекта
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_save_and_load(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест сохранения и загрузки модели."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        agent.train()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.zip"
            
            # Сохранение
            agent.save(str(model_path))
            assert model_path.exists()
            
            # Проверка сохранения конфигурации
            config_path = model_path.with_suffix('.yaml')
            assert config_path.exists()
            
            # Загрузка
            loaded_agent = MockAgent.load(str(model_path))
            assert loaded_agent.is_trained
            assert loaded_agent.config.algorithm == config.algorithm
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_evaluation(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест оценки производительности агента."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        # Настройка мок-среды для оценки
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.side_effect = [
            (np.array([0, 0, 0, 0]), 1.0, True, False, {}),  # Завершение эпизода
        ]
        
        agent = MockAgent(config=config)
        agent.train()
        
        metrics = agent.evaluate(n_episodes=1)
        
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "mean_length" in metrics
        assert isinstance(metrics["mean_reward"], float)
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_get_model_info(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест получения информации о модели."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        info = agent.get_model_info()
        
        assert info["algorithm"] == "PPO"
        assert info["env_name"] == "CartPole-v1"
        assert not info["is_trained"]
        
        # После обучения
        agent.train()
        info = agent.get_model_info()
        assert info["is_trained"]
        assert "final_mean_reward" in info
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_reset_model(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест сброса модели."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        agent.train()
        
        assert agent.is_trained
        assert agent.training_result is not None
        
        agent.reset_model()
        
        assert not agent.is_trained
        assert agent.training_result is None
        assert agent.model is not None  # Новая модель создана
    
    @patch('src.agents.base.gym.make')
    @patch('src.agents.base.set_seed')
    @patch('src.agents.base.get_experiment_logger')
    @patch('src.agents.base.get_metrics_tracker')
    def test_repr(self, mock_metrics, mock_logger, mock_seed, mock_gym_make, config, mock_env):
        """Тест строкового представления агента."""
        mock_gym_make.return_value = mock_env
        mock_logger.return_value = Mock()
        mock_metrics.return_value = Mock()
        
        agent = MockAgent(config=config)
        repr_str = repr(agent)
        
        assert "MockAgent" in repr_str
        assert "algorithm=PPO" in repr_str
        assert "env=CartPole-v1" in repr_str
        assert "trained=False" in repr_str