"""Тесты для TD3 агента.

Этот модуль содержит комплексные тесты для TD3Agent, включая:
- Инициализацию агента с различными конфигурациями
- Валидацию пространств действий
- Создание и настройку модели TD3
- Обучение агента
- Предсказание действий
- Сохранение и загрузку модели
- Обработку ошибок
- Интеграцию с системой метрик
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.td3_agent import (
    TD3Agent,
    TD3Config,
    TD3MetricsCallback,
    TD3EarlyStoppingCallback,
)
from src.agents.base import TrainingResult


class TestTD3Config:
    """Тесты для конфигурации TD3 агента."""

    def test_default_config(self) -> None:
        """Тест создания конфигурации с параметрами по умолчанию."""
        config = TD3Config(env_name="LunarLander-v3")

        assert config.algorithm == "TD3"
        assert config.env_name == "LunarLander-v3"
        assert config.learning_rate == 1e-3
        assert config.buffer_size == 1_000_000
        assert config.batch_size == 256
        assert config.tau == 0.005
        assert config.policy_delay == 2
        assert config.target_policy_noise == 0.2
        assert config.action_noise_type == "normal"
        assert config.normalize_env is True
        assert config.early_stopping is True

    def test_custom_config(self) -> None:
        """Тест создания конфигурации с кастомными параметрами."""
        config = TD3Config(
            env_name="Pendulum-v1",
            learning_rate=5e-4,
            buffer_size=500_000,
            batch_size=128,
            tau=0.01,
            policy_delay=3,
            action_noise_type="ornstein_uhlenbeck",
            normalize_env=False,
        )

        assert config.env_name == "Pendulum-v1"
        assert config.learning_rate == 5e-4
        assert config.buffer_size == 500_000
        assert config.batch_size == 128
        assert config.tau == 0.01
        assert config.policy_delay == 3
        assert config.action_noise_type == "ornstein_uhlenbeck"
        assert config.normalize_env is False

    def test_config_validation_tau(self) -> None:
        """Тест валидации параметра tau."""
        # Валидные значения
        config = TD3Config(env_name="Test-v1", tau=0.005)
        assert config.tau == 0.005

        config = TD3Config(env_name="Test-v1", tau=1.0)
        assert config.tau == 1.0

        # Невалидные значения
        with pytest.raises(ValueError, match="tau должен быть в \\(0, 1\\]"):
            TD3Config(env_name="Test-v1", tau=0.0)

        with pytest.raises(ValueError, match="tau должен быть в \\(0, 1\\]"):
            TD3Config(env_name="Test-v1", tau=1.5)

    def test_config_validation_policy_delay(self) -> None:
        """Тест валидации параметра policy_delay."""
        # Валидные значения
        config = TD3Config(env_name="Test-v1", policy_delay=1)
        assert config.policy_delay == 1

        config = TD3Config(env_name="Test-v1", policy_delay=5)
        assert config.policy_delay == 5

        # Невалидные значения
        with pytest.raises(ValueError, match="policy_delay должен быть >= 1"):
            TD3Config(env_name="Test-v1", policy_delay=0)

    def test_config_validation_noise_type(self) -> None:
        """Тест валидации типа шума."""
        # Валидные типы
        for noise_type in ["normal", "ornstein_uhlenbeck", "none"]:
            config = TD3Config(env_name="Test-v1", action_noise_type=noise_type)
            assert config.action_noise_type == noise_type

        # Невалидный тип
        with pytest.raises(ValueError, match="action_noise_type должен быть одним из"):
            TD3Config(env_name="Test-v1", action_noise_type="invalid")

    def test_config_validation_buffer_size(self) -> None:
        """Тест валидации размера буфера."""
        # Валидные значения
        config = TD3Config(env_name="Test-v1", buffer_size=1000)
        assert config.buffer_size == 1000

        # Невалидные значения
        with pytest.raises(ValueError, match="buffer_size должен быть > 0"):
            TD3Config(env_name="Test-v1", buffer_size=0)

        with pytest.raises(ValueError, match="buffer_size должен быть > 0"):
            TD3Config(env_name="Test-v1", buffer_size=-1000)

    def test_policy_kwargs_setup(self) -> None:
        """Тест настройки policy_kwargs."""
        config = TD3Config(
            env_name="Test-v1", net_arch=[256, 256], activation_fn="tanh"
        )

        assert config.policy_kwargs["net_arch"] == [256, 256]
        assert config.policy_kwargs["activation_fn"] == torch.nn.Tanh

    def test_invalid_activation_function(self) -> None:
        """Тест обработки невалидной функции активации."""
        with pytest.raises(ValueError, match="Неподдерживаемая функция активации"):
            TD3Config(env_name="Test-v1", activation_fn="invalid")


class TestTD3MetricsCallback:
    """Тесты для колбэка метрик TD3."""

    def test_callback_initialization(self) -> None:
        """Тест инициализации колбэка метрик."""
        metrics_tracker = Mock()
        callback = TD3MetricsCallback(
            metrics_tracker=metrics_tracker, log_freq=1000, verbose=1
        )

        assert callback.metrics_tracker == metrics_tracker
        assert callback.log_freq == 1000
        assert callback.verbose == 1
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []

    def test_on_step_with_episodes(self) -> None:
        """Тест обработки шага с информацией об эпизодах."""
        metrics_tracker = Mock()
        callback = TD3MetricsCallback(metrics_tracker, log_freq=1)

        # Мок модели с информацией об эпизодах
        callback.model = Mock()
        callback.model.ep_info_buffer = [
            {"r": 100.0, "l": 50},
            {"r": 150.0, "l": 75},
        ]
        callback.model.logger = Mock()
        callback.model.logger.name_to_value = {
            "train/actor_loss": 0.1,
            "train/critic_loss": 0.2,
        }
        callback.num_timesteps = 1000
        callback.n_calls = 1

        result = callback._on_step()

        assert result is True
        assert callback.episode_rewards == [100.0, 150.0]
        assert callback.episode_lengths == [50, 75]

        # Проверяем, что метрики были записаны
        assert metrics_tracker.add_metric.called

    def test_log_metrics(self) -> None:
        """Тест логирования метрик."""
        metrics_tracker = Mock()
        callback = TD3MetricsCallback(metrics_tracker, log_freq=1)

        # Добавляем атрибут model к callback
        callback.model = Mock()
        callback.model.logger = Mock()
        callback.model.logger.name_to_value = {
            "train/actor_loss": 0.1,
            "train/critic_loss": 0.2,
        }

        callback.num_timesteps = 1000

        # Добавляем данные об эпизодах
        callback.episode_rewards = [100.0, 150.0, 120.0]
        callback.episode_lengths = [50, 75, 60]

        callback._log_metrics()

        # Проверяем, что add_metric был вызван для каждой метрики
        assert metrics_tracker.add_metric.call_count >= 6  # Минимум 6 основных метрик


class TestTD3EarlyStoppingCallback:
    """Тесты для колбэка ранней остановки TD3."""

    def test_callback_initialization(self) -> None:
        """Тест инициализации колбэка ранней остановки."""
        callback = TD3EarlyStoppingCallback(
            target_reward=200.0,
            patience_episodes=50,
            min_improvement=5.0,
            check_freq=1000,
        )

        assert callback.target_reward == 200.0
        assert callback.patience_episodes == 50
        assert callback.min_improvement == 5.0
        assert callback.check_freq == 1000
        assert callback.best_mean_reward == float("-inf")
        assert callback.episodes_without_improvement == 0

    def test_target_reward_reached(self) -> None:
        """Тест остановки при достижении целевой награды."""
        callback = TD3EarlyStoppingCallback(target_reward=200.0, check_freq=1)
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 210.0}] * 20
        callback.n_calls = 1

        result = callback._on_step()

        assert result is False  # Обучение должно остановиться

    def test_patience_exceeded(self) -> None:
        """Тест остановки при превышении терпения."""
        callback = TD3EarlyStoppingCallback(
            target_reward=300.0,  # Недостижимая награда
            patience_episodes=10,
            min_improvement=5.0,
            check_freq=1,
        )
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 100.0}] * 20
        callback.n_calls = 1
        callback.episodes_without_improvement = 15  # Превышено терпение
        callback.best_mean_reward = 100.0  # Установим лучшую награду выше текущих
        callback.last_check_episode = 0

        result = callback._on_step()

        assert result is False  # Обучение должно остановиться

    def test_improvement_detected(self) -> None:
        """Тест обнаружения улучшения."""
        callback = TD3EarlyStoppingCallback(
            target_reward=300.0, patience_episodes=10, min_improvement=5.0, check_freq=1
        )
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 150.0}] * 20
        callback.n_calls = 1
        callback.best_mean_reward = 100.0

        result = callback._on_step()

        assert result is True  # Обучение продолжается
        assert callback.best_mean_reward == 150.0
        assert callback.episodes_without_improvement == 0


class TestTD3Agent:
    """Тесты для TD3 агента."""

    @pytest.fixture
    def mock_env(self) -> Mock:
        """Создать мок среды с непрерывным пространством действий."""
        env = Mock(spec=gym.Env)
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        env.reset.return_value = (np.zeros(4), {})
        env.step.return_value = (np.zeros(4), 0.0, False, False, {})
        return env

    @pytest.fixture
    def td3_config(self) -> TD3Config:
        """Создать базовую конфигурацию TD3."""
        return TD3Config(
            env_name="LunarLander-v3",
            total_timesteps=1000,
            learning_starts=100,
            batch_size=32,
            buffer_size=10000,
            eval_freq=500,
            save_freq=1000,
        )

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_agent_initialization(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест инициализации TD3 агента."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        assert agent.config == td3_config
        assert agent.env == mock_env
        assert isinstance(agent.config, TD3Config)
        assert agent.is_trained is False

    def test_discrete_action_space_validation(self, td3_config: TD3Config) -> None:
        """Тест валидации дискретного пространства действий."""
        discrete_env = Mock(spec=gym.Env)
        discrete_env.action_space = gym.spaces.Discrete(4)
        discrete_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,)
        )

        with (
            patch("src.utils.set_seed"),
            patch("src.utils.get_experiment_logger"),
            patch("src.utils.get_metrics_tracker"),
        ):
            with pytest.raises(
                ValueError, match="Алгоритм TD3 не поддерживает дискретные действия"
            ):
                TD3Agent(config=td3_config, env=discrete_env)

    def test_invalid_config_type(self, mock_env: Mock) -> None:
        """Тест обработки невалидного типа конфигурации."""
        from src.agents.base import AgentConfig

        invalid_config = AgentConfig(algorithm="TD3", env_name="Test-v1")

        with pytest.raises(ValueError, match="Ожидается TD3Config"):
            TD3Agent(config=invalid_config, env=mock_env)

    @patch("src.agents.td3_agent.NormalActionNoise")
    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_normal_action_noise_creation(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        mock_noise_class: Mock,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест создания Gaussian шума для действий."""
        td3_config.action_noise_type = "normal"
        td3_config.action_noise_std = 0.2

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Проверяем, что NormalActionNoise был создан с правильными параметрами
        mock_noise_class.assert_called_once()
        call_args = mock_noise_class.call_args

        # Проверяем mean и sigma
        np.testing.assert_array_equal(call_args[1]["mean"], np.zeros(2))
        np.testing.assert_array_equal(call_args[1]["sigma"], np.full(2, 0.2))

    @patch("src.agents.td3_agent.OrnsteinUhlenbeckActionNoise")
    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_ou_action_noise_creation(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        mock_noise_class: Mock,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест создания Ornstein-Uhlenbeck шума для действий."""
        td3_config.action_noise_type = "ornstein_uhlenbeck"
        td3_config.ou_sigma = 0.3
        td3_config.ou_theta = 0.2

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Проверяем, что OrnsteinUhlenbeckActionNoise был создан с правильными параметрами
        mock_noise_class.assert_called_once()
        call_args = mock_noise_class.call_args

        assert call_args[1]["theta"] == 0.2
        # Проверяем sigma с помощью numpy.testing
        np.testing.assert_allclose(call_args[1]["sigma"], 0.3)

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_no_action_noise_creation(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест создания агента без шума действий."""
        td3_config.action_noise_type = "none"

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        assert agent.action_noise is None

    @patch("stable_baselines3.common.utils.LinearSchedule")
    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_learning_rate_schedule_creation(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        mock_schedule: Mock,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест создания расписания learning rate."""
        td3_config.use_lr_schedule = True
        td3_config.lr_schedule_type = "linear"
        td3_config.learning_rate = 1e-3
        td3_config.lr_final_ratio = 0.1

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Проверяем, что get_linear_fn был вызван с правильными параметрами
        mock_schedule.assert_called_once_with(
            start=1e-3,
            end=1e-4,  # 1e-3 * 0.1
            end_fraction=1.0,
        )

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_exponential_learning_rate_schedule(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест экспоненциального расписания learning rate."""
        td3_config.use_lr_schedule = True
        td3_config.lr_schedule_type = "exponential"
        td3_config.learning_rate = 1e-3
        td3_config.lr_final_ratio = 0.1

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Получаем функцию расписания
        lr_schedule = agent._create_learning_rate_schedule()

        # Проверяем, что это функция
        assert callable(lr_schedule)

        # Проверяем значения в начале и конце
        assert lr_schedule(1.0) == 1e-3  # В начале
        assert abs(lr_schedule(0.0) - 1e-4) < 1e-6  # В конце

    @patch("stable_baselines3.common.utils.LinearSchedule")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_model_creation(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_make_vec_env,
        mock_td3_class: Mock,
        mock_get_linear_fn,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест создания модели TD3."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем функцию расписания
        mock_get_linear_fn.return_value = td3_config.learning_rate

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Проверяем, что TD3 был создан с правильными параметрами
        mock_td3_class.assert_called_once()
        call_kwargs = mock_td3_class.call_args[1]

        assert call_kwargs["buffer_size"] == td3_config.buffer_size
        assert call_kwargs["batch_size"] == td3_config.batch_size
        assert call_kwargs["tau"] == td3_config.tau
        assert call_kwargs["gamma"] == td3_config.gamma
        assert call_kwargs["policy_delay"] == td3_config.policy_delay

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_train_method(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест метода обучения."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        # Настройка мока модели
        mock_model = Mock()
        mock_model.learn.return_value = mock_model
        mock_td3_class.return_value = mock_model

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Мок для evaluate
        with patch.object(agent, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = {"mean_reward": 150.0, "std_reward": 25.0}

            result = agent.train(total_timesteps=1000)

            assert isinstance(result, TrainingResult)
            assert result.total_timesteps == 1000
            assert result.final_mean_reward == 150.0
            assert result.final_std_reward == 25.0
            assert result.success is True
            assert agent.is_trained is True

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_predict_method(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест метода предсказания."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        # Настройка мока модели
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0.5, -0.3]), None)
        mock_td3_class.return_value = mock_model

        agent = TD3Agent(config=td3_config, env=mock_env)
        agent.is_trained = True

        observation = np.array([1.0, 2.0, 3.0, 4.0])
        action, state = agent.predict(observation, deterministic=True)

        np.testing.assert_array_equal(action, np.array([0.5, -0.3]))
        assert state is None
        # Проверяем вызов с учетом того, что numpy массивы могут быть разных типов
        assert mock_model.predict.call_count == 1
        call_args = mock_model.predict.call_args
        # Проверяем, что первый аргумент - это массив с теми же значениями
        np.testing.assert_array_equal(call_args[0][0], observation)
        assert call_args[1]["deterministic"] is True

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_predict_untrained_model(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест предсказания с необученной моделью."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        observation = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(RuntimeError, match="Модель не обучена"):
            agent.predict(observation)

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_save_and_load_model(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест сохранения и загрузки модели."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_td3_model.zip"

            # Создание и сохранение агента
            mock_vec_env = MagicMock(spec=DummyVecEnv)
            mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
            mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
            mock_vec_env.num_envs = 1
            mock_make_vec_env.return_value = mock_vec_env

            mock_model = Mock()
            mock_td3_class.return_value = mock_model

            agent = TD3Agent(config=td3_config, env=mock_env)

            with patch.object(agent, "_save_config"):
                agent.save(str(model_path))

            mock_model.save.assert_called_once_with(str(model_path))

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_get_model_info(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест получения информации о модели."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        info = agent.get_model_info()

        assert info["algorithm"] == "TD3"
        assert info["env_name"] == "LunarLander-v3"
        assert info["buffer_size"] == td3_config.buffer_size
        assert info["batch_size"] == td3_config.batch_size
        assert info["tau"] == td3_config.tau
        assert info["policy_delay"] == td3_config.policy_delay
        assert info["action_noise_type"] == td3_config.action_noise_type

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_reset_model(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест сброса модели."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)
        agent.is_trained = True

        with patch.object(mock_vec_env, "close"):
            agent.reset_model()

        assert agent.is_trained is False
        assert agent.training_result is None

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_update_noise_schedule_normal(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест обновления расписания шума для Normal шума."""
        td3_config.action_noise_type = "normal"
        td3_config.use_noise_schedule = True
        td3_config.noise_schedule_type = "linear"
        td3_config.action_noise_std = 0.2
        td3_config.noise_final_ratio = 0.1

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Мок для action_noise
        agent.action_noise = Mock(spec=NormalActionNoise)
        agent.action_noise.sigma = np.array([0.2, 0.2])

        # Обновляем шум на 50% прогресса
        agent.update_noise_schedule(0.5)

        # Ожидаемый фактор: 1.0 - 0.5 * (1.0 - 0.1) = 0.55
        expected_sigma = 0.2 * 0.55
        np.testing.assert_array_almost_equal(
            agent.action_noise.sigma, np.array([expected_sigma, expected_sigma])
        )

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_update_noise_schedule_ou(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест обновления расписания шума для OU шума."""
        td3_config.action_noise_type = "ornstein_uhlenbeck"
        td3_config.use_noise_schedule = True
        td3_config.noise_schedule_type = "exponential"
        td3_config.ou_sigma = 0.3
        td3_config.noise_final_ratio = 0.1

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=td3_config, env=mock_env)

        # Мок для action_noise
        agent.action_noise = Mock(spec=OrnsteinUhlenbeckActionNoise)
        agent.action_noise.sigma = 0.3

        # Обновляем шум на 50% прогресса
        agent.update_noise_schedule(0.5)

        # Ожидаемый фактор: 0.1 ** 0.5 ≈ 0.316
        expected_sigma = 0.3 * (0.1**0.5)
        assert abs(agent.action_noise.sigma - expected_sigma) < 1e-6

    @patch("src.agents.td3_agent.make_vec_env")
    @patch("src.agents.td3_agent.TD3")
    @patch("src.utils.set_seed")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_callbacks_setup(
        self,
        mock_get_metrics_tracker,
        mock_get_logger,
        mock_set_seed,
        mock_td3_class: Mock,
        mock_make_vec_env,
        td3_config: TD3Config,
        mock_env: Mock,
    ) -> None:
        """Тест настройки колбэков."""
        td3_config.early_stopping = True
        td3_config.eval_freq = 1000
        td3_config.save_freq = 2000
        td3_config.model_save_path = "/tmp/model.zip"

        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем CheckpointManager, чтобы избежать ошибки с missing experiment_id
        with patch("src.utils.CheckpointManager") as mock_checkpoint_manager:
            # Настройка мока CheckpointManager с правильной сигнатурой
            mock_checkpoint_manager_instance = MagicMock()
            mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance

            agent = TD3Agent(config=td3_config, env=mock_env)

        # Проверяем, что колбэки были созданы
        assert len(agent.callbacks) > 0

        # Проверяем типы колбэков
        callback_types = [type(cb).__name__ for cb in agent.callbacks]
        assert "TD3MetricsCallback" in callback_types
        assert "TD3EarlyStoppingCallback" in callback_types
        assert "EvalCallback" in callback_types
        assert "CheckpointCallback" in callback_types


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_reproducibility(seed: int) -> None:
    """Тест воспроизводимости результатов с разными seed."""
    config = TD3Config(
        env_name="Pendulum-v1",
        total_timesteps=100,
        seed=seed,
        learning_starts=50,
        batch_size=16,
    )

    mock_env = Mock(spec=gym.Env)
    mock_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    mock_env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

    with (
        patch("src.utils.CheckpointManager"),
        patch("src.agents.td3_agent.make_vec_env") as mock_make_vec_env,
        patch("src.agents.td3_agent.TD3"),
        patch("src.utils.set_seed") as mock_set_seed,
        patch("src.utils.get_experiment_logger"),
        patch("src.utils.get_metrics_tracker"),
    ):
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(3,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        agent = TD3Agent(config=config, env=mock_env)

        # Проверяем, что set_seed был вызван с правильным значением
        mock_set_seed.assert_called_with(seed)


def test_training_error_handling() -> None:
    """Тест обработки ошибок во время обучения."""
    config = TD3Config(env_name="Test-v1", total_timesteps=100)

    mock_env = Mock(spec=gym.Env)
    mock_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    mock_env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    with (
        patch("src.agents.td3_agent.make_vec_env") as mock_make_vec_env,
        patch("src.agents.td3_agent.TD3") as mock_td3_class,
        patch("src.utils.set_seed"),
        patch("src.utils.get_experiment_logger"),
        patch("src.utils.get_metrics_tracker"),
    ):
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env

        # Настройка мока модели для генерации ошибки
        mock_model = Mock()
        mock_model.learn.side_effect = RuntimeError("Ошибка обучения")
        mock_td3_class.return_value = mock_model

        agent = TD3Agent(config=config, env=mock_env)

        with pytest.raises(RuntimeError, match="Ошибка обучения TD3 агента"):
            agent.train()

        # Проверяем, что результат обучения содержит информацию об ошибке
        assert agent.training_result is not None
        assert agent.training_result.success is False
        assert "Ошибка обучения" in agent.training_result.error_message
