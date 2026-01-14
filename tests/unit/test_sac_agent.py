"""Тесты для SAC агента.

Модуль содержит комплексные тесты для проверки функциональности SAC агента,
включая инициализацию, конфигурацию, обучение, предсказание, сохранение и загрузку.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from src.agents.sac_agent import SACAgent, SACConfig, SACMetricsCallback, SACEarlyStoppingCallback
from src.utils import MetricsTracker


class TestSACConfig:
    """Тесты для конфигурации SAC агента."""

    def test_default_config(self) -> None:
        """Тест создания конфигурации с параметрами по умолчанию."""
        config = SACConfig(env_name="LunarLander-v3")
        
        assert config.algorithm == "SAC"
        assert config.env_name == "LunarLander-v3"
        assert config.learning_rate == 3e-4
        assert config.buffer_size == 1_000_000
        assert config.batch_size == 256
        assert config.tau == 0.005
        assert config.gamma == 0.99
        assert config.ent_coef == "auto"
        assert config.target_entropy == "auto"
        assert config.net_arch == [256, 256]
        assert config.activation_fn == "relu"
        assert config.normalize_env is False
        assert config.early_stopping is True
        assert config.target_reward == 200.0

    def test_custom_config(self) -> None:
        """Тест создания конфигурации с кастомными параметрами."""
        config = SACConfig(
            env_name="Pendulum-v1",
            learning_rate=1e-3,
            buffer_size=500_000,
            batch_size=128,
            tau=0.01,
            gamma=0.95,
            ent_coef=0.1,
            target_entropy=-2.0,
            net_arch=[128, 128],
            activation_fn="tanh",
            normalize_env=True,
            early_stopping=False,
            target_reward=100.0,
        )
        
        assert config.env_name == "Pendulum-v1"
        assert config.learning_rate == 1e-3
        assert config.buffer_size == 500_000
        assert config.batch_size == 128
        assert config.tau == 0.01
        assert config.gamma == 0.95
        assert config.ent_coef == 0.1
        assert config.target_entropy == -2.0
        assert config.net_arch == [128, 128]
        assert config.activation_fn == "tanh"
        assert config.normalize_env is True
        assert config.early_stopping is False
        assert config.target_reward == 100.0

    def test_policy_kwargs_setup(self) -> None:
        """Тест настройки policy_kwargs."""
        config = SACConfig(
            env_name="LunarLander-v3",
            net_arch=[64, 64],
            activation_fn="elu",
        )
        
        assert "net_arch" in config.policy_kwargs
        assert "activation_fn" in config.policy_kwargs
        assert config.policy_kwargs["net_arch"] == [64, 64]
        assert config.policy_kwargs["activation_fn"] == torch.nn.ELU

    def test_invalid_buffer_size(self) -> None:
        """Тест валидации размера буфера."""
        with pytest.raises(ValueError, match="buffer_size должен быть > 0"):
            SACConfig(env_name="LunarLander-v3", buffer_size=0)

    def test_invalid_tau(self) -> None:
        """Тест валидации параметра tau."""
        with pytest.raises(ValueError, match="tau должен быть в \\(0, 1\\]"):
            SACConfig(env_name="LunarLander-v3", tau=0.0)
        
        with pytest.raises(ValueError, match="tau должен быть в \\(0, 1\\]"):
            SACConfig(env_name="LunarLander-v3", tau=1.5)

    def test_invalid_learning_starts(self) -> None:
        """Тест валидации параметра learning_starts."""
        with pytest.raises(ValueError, match="learning_starts должен быть >= 0"):
            SACConfig(env_name="LunarLander-v3", learning_starts=-1)

    def test_invalid_train_freq(self) -> None:
        """Тест валидации параметра train_freq."""
        with pytest.raises(ValueError, match="train_freq должен быть > 0"):
            SACConfig(env_name="LunarLander-v3", train_freq=0)

    def test_invalid_gradient_steps(self) -> None:
        """Тест валидации параметра gradient_steps."""
        with pytest.raises(ValueError, match="gradient_steps должен быть > 0"):
            SACConfig(env_name="LunarLander-v3", gradient_steps=0)

    def test_invalid_ent_coef(self) -> None:
        """Тест валидации коэффициента энтропии."""
        with pytest.raises(ValueError, match="ent_coef должен быть 'auto' или числом"):
            SACConfig(env_name="LunarLander-v3", ent_coef="invalid")

    def test_invalid_target_entropy(self) -> None:
        """Тест валидации целевой энтропии."""
        with pytest.raises(ValueError, match="target_entropy должен быть 'auto' или числом"):
            SACConfig(env_name="LunarLander-v3", target_entropy="invalid")

    def test_invalid_activation_fn(self) -> None:
        """Тест валидации функции активации."""
        with pytest.raises(ValueError, match="Неподдерживаемая функция активации"):
            SACConfig(env_name="LunarLander-v3", activation_fn="invalid")

    def test_invalid_action_noise_type(self) -> None:
        """Тест валидации типа шума."""
        with pytest.raises(ValueError, match="Неподдерживаемый тип шума"):
            SACConfig(env_name="LunarLander-v3", action_noise_type="invalid")

    def test_invalid_action_noise_std(self) -> None:
        """Тест валидации стандартного отклонения шума."""
        with pytest.raises(ValueError, match="action_noise_std должен быть > 0"):
            SACConfig(
                env_name="LunarLander-v3",
                action_noise_type="normal",
                action_noise_std=0.0,
            )


class TestSACMetricsCallback:
    """Тесты для колбэка метрик SAC."""

    def test_init(self) -> None:
        """Тест инициализации колбэка."""
        metrics_tracker = Mock(spec=MetricsTracker)
        callback = SACMetricsCallback(
            metrics_tracker=metrics_tracker,
            log_freq=1000,
            verbose=1,
        )
        
        assert callback.metrics_tracker is metrics_tracker
        assert callback.log_freq == 1000
        assert callback.verbose == 1
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []

    def test_on_step_no_episodes(self) -> None:
        """Тест _on_step без эпизодов."""
        metrics_tracker = Mock(spec=MetricsTracker)
        callback = SACMetricsCallback(metrics_tracker, log_freq=1000)
        
        # Мок модели без эпизодов
        callback.model = Mock()
        callback.model.ep_info_buffer = []
        callback.n_calls = 1000
        callback.num_timesteps = 1000
        
        result = callback._on_step()
        assert result is True
        assert len(callback.episode_rewards) == 0

    def test_on_step_with_episodes(self) -> None:
        """Тест _on_step с эпизодами."""
        metrics_tracker = Mock(spec=MetricsTracker)
        callback = SACMetricsCallback(metrics_tracker, log_freq=1000)
        
        # Мок модели с эпизодами
        callback.model = Mock()
        callback.model.ep_info_buffer = [
            {"r": 100.0, "l": 200},
            {"r": 150.0, "l": 250},
        ]
        callback.model.logger = Mock()
        callback.model.logger.name_to_value = {
            "train/actor_loss": 0.1,
            "train/critic_loss": 0.2,
        }
        callback.n_calls = 1000
        callback.num_timesteps = 1000
        
        result = callback._on_step()
        assert result is True
        assert len(callback.episode_rewards) == 2
        assert callback.episode_rewards == [100.0, 150.0]
        assert callback.episode_lengths == [200, 250]
        
        # Проверка вызова add_metric
        assert metrics_tracker.add_metric.called


class TestSACEarlyStoppingCallback:
    """Тесты для колбэка ранней остановки SAC."""

    def test_init(self) -> None:
        """Тест инициализации колбэка."""
        callback = SACEarlyStoppingCallback(
            target_reward=200.0,
            patience_episodes=100,
            min_improvement=5.0,
            check_freq=10000,
            verbose=1,
        )
        
        assert callback.target_reward == 200.0
        assert callback.patience_episodes == 100
        assert callback.min_improvement == 5.0
        assert callback.check_freq == 10000
        assert callback.verbose == 1
        assert callback.best_mean_reward == float("-inf")
        assert callback.episodes_without_improvement == 0

    def test_on_step_not_check_time(self) -> None:
        """Тест _on_step когда не время проверки."""
        callback = SACEarlyStoppingCallback(target_reward=200.0, check_freq=10000)
        callback.n_calls = 5000
        
        result = callback._on_step()
        assert result is True

    def test_on_step_no_episodes(self) -> None:
        """Тест _on_step без эпизодов."""
        callback = SACEarlyStoppingCallback(target_reward=200.0, check_freq=10000)
        callback.n_calls = 10000
        callback.model = Mock()
        callback.model.ep_info_buffer = []
        
        result = callback._on_step()
        assert result is True

    def test_on_step_insufficient_data(self) -> None:
        """Тест _on_step с недостаточными данными."""
        callback = SACEarlyStoppingCallback(target_reward=200.0, check_freq=10000)
        callback.n_calls = 10000
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 100.0, "l": 200}]  # Только 1 эпизод
        
        result = callback._on_step()
        assert result is True

    def test_on_step_target_reached(self) -> None:
        """Тест _on_step при достижении целевой награды."""
        callback = SACEarlyStoppingCallback(target_reward=200.0, check_freq=10000, verbose=1)
        callback.n_calls = 10000
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 250.0, "l": 200} for _ in range(20)]
        
        result = callback._on_step()
        assert result is False  # Остановить обучение

    def test_on_step_improvement(self) -> None:
        """Тест _on_step с улучшением."""
        callback = SACEarlyStoppingCallback(
            target_reward=300.0,
            min_improvement=5.0,
            check_freq=10000,
        )
        callback.n_calls = 10000
        callback.best_mean_reward = 100.0
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 110.0, "l": 200} for _ in range(20)]
        
        result = callback._on_step()
        assert result is True
        assert callback.best_mean_reward == 110.0
        assert callback.episodes_without_improvement == 0

    def test_on_step_no_improvement_patience_not_exceeded(self) -> None:
        """Тест _on_step без улучшения, терпение не исчерпано."""
        callback = SACEarlyStoppingCallback(
            target_reward=300.0,
            patience_episodes=100,
            min_improvement=5.0,
            check_freq=10000,
        )
        callback.n_calls = 10000
        callback.best_mean_reward = 150.0
        callback.episodes_without_improvement = 50
        callback.last_check_episode = 0
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 140.0, "l": 200} for _ in range(20)]
        
        result = callback._on_step()
        assert result is True
        assert callback.episodes_without_improvement == 70  # 50 + 20

    def test_on_step_patience_exceeded(self) -> None:
        """Тест _on_step при исчерпании терпения."""
        callback = SACEarlyStoppingCallback(
            target_reward=300.0,
            patience_episodes=50,
            min_improvement=5.0,
            check_freq=10000,
            verbose=1,
        )
        callback.n_calls = 10000
        callback.best_mean_reward = 150.0
        callback.episodes_without_improvement = 40
        callback.last_check_episode = 0
        callback.model = Mock()
        callback.model.ep_info_buffer = [{"r": 140.0, "l": 200} for _ in range(20)]
        
        result = callback._on_step()
        assert result is False  # Остановить обучение


class TestSACAgent:
    """Тесты для SAC агента."""

    @pytest.fixture
    def mock_env(self) -> Mock:
        """Создать мок среды с непрерывным пространством действий."""
        env = Mock()
        env.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        env.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        env.reset.return_value = (np.zeros(8), {})
        env.step.return_value = (np.zeros(8), 0.0, False, False, {})
        return env

    @pytest.fixture
    def sac_config(self) -> SACConfig:
        """Создать конфигурацию SAC для тестов."""
        return SACConfig(
            env_name="LunarLander-v3",
            total_timesteps=1000,
            learning_rate=3e-4,
            buffer_size=10000,  # Уменьшено для тестов
            batch_size=32,
            learning_starts=100,
            eval_freq=0,  # Отключено для тестов
            save_freq=0,  # Отключено для тестов
            verbose=0,
        )

    def test_init_with_valid_config(self, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест инициализации с корректной конфигурацией."""
        with patch("src.agents.sac_agent.make_vec_env"), \
             patch("src.agents.sac_agent.SAC"):
            agent = SACAgent(config=sac_config, env=mock_env)
            
            assert isinstance(agent.config, SACConfig)
            assert agent.env is mock_env
            assert agent.is_trained is False
            assert agent.model is not None

    def test_init_with_invalid_config_type(self, mock_env: Mock) -> None:
        """Тест инициализации с некорректным типом конфигурации."""
        from src.agents.base import AgentConfig
        
        invalid_config = AgentConfig(algorithm="SAC", env_name="LunarLander-v3")
        
        with pytest.raises(ValueError, match="Ожидается SACConfig"):
            SACAgent(config=invalid_config, env=mock_env)

    def test_init_with_discrete_action_space(self, sac_config: SACConfig) -> None:
        """Тест инициализации с дискретным пространством действий."""
        discrete_env = Mock()
        discrete_env.action_space = Discrete(4)
        discrete_env.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        
        with pytest.raises(ValueError, match="Алгоритм SAC не поддерживает дискретные действия"):
            SACAgent(config=sac_config, env=discrete_env)

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_create_action_noise_normal(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест создания нормального шума для действий."""
        sac_config.action_noise_type = "normal"
        sac_config.action_noise_std = 0.1
        
        agent = SACAgent(config=sac_config, env=mock_env)
        
        assert agent.action_noise is not None
        assert hasattr(agent.action_noise, "_sigma")

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_create_action_noise_none(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест отсутствия шума для действий."""
        sac_config.action_noise_type = None
        
        agent = SACAgent(config=sac_config, env=mock_env)
        
        assert agent.action_noise is None

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_learning_rate_schedule_linear(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест создания линейного расписания learning rate."""
        sac_config.use_lr_schedule = True
        sac_config.lr_schedule_type = "linear"
        
        agent = SACAgent(config=sac_config, env=mock_env)
        lr_schedule = agent._create_learning_rate_schedule()
        
        assert callable(lr_schedule)

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_learning_rate_schedule_constant(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест константного learning rate."""
        sac_config.use_lr_schedule = False
        
        agent = SACAgent(config=sac_config, env=mock_env)
        lr_schedule = agent._create_learning_rate_schedule()
        
        assert lr_schedule == sac_config.learning_rate

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_predict_not_trained(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест предсказания на необученной модели."""
        agent = SACAgent(config=sac_config, env=mock_env)
        agent.is_trained = False
        
        observation = np.zeros(8)
        
        with pytest.raises(RuntimeError, match="Модель не обучена"):
            agent.predict(observation)

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_predict_trained(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест предсказания на обученной модели."""
        agent = SACAgent(config=sac_config, env=mock_env)
        agent.is_trained = True
        
        # Мок предсказания модели
        expected_action = np.array([0.5, -0.3])
        agent.model.predict.return_value = (expected_action, None)
        
        observation = np.zeros(8)
        action, state = agent.predict(observation, deterministic=True)
        
        assert np.array_equal(action, expected_action)
        assert state is None
        agent.model.predict.assert_called_once_with(observation, deterministic=True)

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_train_success(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест успешного обучения."""
        agent = SACAgent(config=sac_config, env=mock_env)
        
        # Мок обучения
        agent.model.learn.return_value = agent.model
        
        # Мок оценки
        with patch.object(agent, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = {
                "mean_reward": 150.0,
                "std_reward": 25.0,
            }
            
            result = agent.train(total_timesteps=1000)
            
            assert result.success is True
            assert result.total_timesteps == 1000
            assert result.final_mean_reward == 150.0
            assert result.final_std_reward == 25.0
            assert agent.is_trained is True
            
            agent.model.learn.assert_called_once()
            mock_evaluate.assert_called_once()

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_train_failure(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест неудачного обучения."""
        agent = SACAgent(config=sac_config, env=mock_env)
        
        # Мок ошибки обучения
        agent.model.learn.side_effect = RuntimeError("Training failed")
        
        with pytest.raises(RuntimeError, match="Ошибка обучения SAC агента"):
            agent.train(total_timesteps=1000)
        
        assert agent.training_result is not None
        assert agent.training_result.success is False
        assert "Training failed" in agent.training_result.error_message

    @patch("src.agents.sac_agent.make_vec_env")
    @patch("src.agents.sac_agent.SAC")
    def test_save_and_load(self, mock_sac, mock_make_vec_env, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест сохранения и загрузки модели."""
        agent = SACAgent(config=sac_config, env=mock_env)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_sac_model.zip"
            
            # Мок сохранения
            agent.model.save = Mock()
            
            with patch("builtins.open", create=True), \
                 patch("yaml.dump"):
                agent.save(str(model_path))
                agent.model.save.assert_called_once()

    def test_get_model_info(self, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест получения информации о модели."""
        with patch("src.agents.sac_agent.make_vec_env"), \
             patch("src.agents.sac_agent.SAC"):
            agent = SACAgent(config=sac_config, env=mock_env)
            
            info = agent.get_model_info()
            
            assert "algorithm" in info
            assert "buffer_size" in info
            assert "batch_size" in info
            assert "tau" in info
            assert "ent_coef" in info
            assert info["algorithm"] == "SAC"
            assert info["buffer_size"] == sac_config.buffer_size

    def test_reset_model(self, sac_config: SACConfig, mock_env: Mock) -> None:
        """Тест сброса модели."""
        with patch("src.agents.sac_agent.make_vec_env") as mock_make_vec_env, \
             patch("src.agents.sac_agent.SAC") as mock_sac:
            
            # Мок векторизованной среды с методом close
            mock_vec_env = Mock()
            mock_make_vec_env.return_value = mock_vec_env
            
            agent = SACAgent(config=sac_config, env=mock_env)
            agent.is_trained = True
            
            agent.reset_model()
            
            assert agent.is_trained is False
            assert agent.model is not None
            mock_vec_env.close.assert_called_once()


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_sac_agent_reproducibility(seed: int) -> None:
    """Тест воспроизводимости SAC агента с разными seeds."""
    config = SACConfig(
        env_name="LunarLander-v3",
        total_timesteps=100,
        seed=seed,
        verbose=0,
        eval_freq=0,  # Отключено для тестов
    )
    
    mock_env = Mock()
    mock_env.action_space = Box(low=-1.0, high=1.0, shape=(2,))
    mock_env.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,))
    
    with patch("src.agents.sac_agent.make_vec_env"), \
         patch("src.agents.sac_agent.SAC"):
        agent = SACAgent(config=config, env=mock_env)
        
        assert agent.config.seed == seed


def test_sac_config_inheritance() -> None:
    """Тест наследования SACConfig от AgentConfig."""
    from src.agents.base import AgentConfig
    
    config = SACConfig(env_name="LunarLander-v3")
    
    assert isinstance(config, AgentConfig)
    assert hasattr(config, "algorithm")
    assert hasattr(config, "env_name")
    assert hasattr(config, "total_timesteps")
    assert hasattr(config, "buffer_size")  # SAC-специфичный параметр