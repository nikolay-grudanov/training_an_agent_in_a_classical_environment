"""Тесты для A2C агента.

Этот модуль содержит комплексные тесты для A2CAgent и A2CConfig,
включая проверку инициализации, обучения, предсказания, сохранения/загрузки,
валидации конфигурации и интеграции с системой метрик.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from src.agents.a2c_agent import A2CAgent, A2CConfig, A2CMetricsCallback, A2CEarlyStoppingCallback
from src.agents.base import TrainingResult


class TestA2CConfig:
    """Тесты для конфигурации A2C агента."""

    def test_default_config(self) -> None:
        """Тест создания конфигурации с параметрами по умолчанию."""
        config = A2CConfig(env_name="LunarLander-v3")

        assert config.algorithm == "A2C"
        assert config.env_name == "LunarLander-v3"
        assert config.learning_rate == 7e-4
        assert config.n_steps == 5
        assert config.gamma == 0.99
        assert config.gae_lambda == 1.0
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.25
        assert config.use_rms_prop is True
        assert config.rms_prop_eps == 1e-5
        assert config.normalize_advantage is False

    def test_custom_config(self) -> None:
        """Тест создания конфигурации с кастомными параметрами."""
        config = A2CConfig(
            env_name="CartPole-v1",
            learning_rate=1e-3,
            n_steps=10,
            gamma=0.95,
            ent_coef=0.02,
            vf_coef=0.5,
            use_rms_prop=False,
        )

        assert config.env_name == "CartPole-v1"
        assert config.learning_rate == 1e-3
        assert config.n_steps == 10
        assert config.gamma == 0.95
        assert config.ent_coef == 0.02
        assert config.vf_coef == 0.5
        assert config.use_rms_prop is False

    def test_net_arch_default(self) -> None:
        """Тест установки архитектуры сети по умолчанию."""
        config = A2CConfig(env_name="LunarLander-v3")

        assert config.net_arch is not None
        assert len(config.net_arch) == 1
        assert isinstance(config.net_arch[0], dict)
        assert "pi" in config.net_arch[0]
        assert "vf" in config.net_arch[0]
        assert config.net_arch[0]["pi"] == [64, 64]
        assert config.net_arch[0]["vf"] == [64, 64]

    def test_policy_kwargs_setup(self) -> None:
        """Тест настройки policy_kwargs."""
        config = A2CConfig(
            env_name="LunarLander-v3",
            activation_fn="relu",
            ortho_init=False,
        )

        assert "net_arch" in config.policy_kwargs
        assert "activation_fn" in config.policy_kwargs
        assert "ortho_init" in config.policy_kwargs
        assert config.policy_kwargs["activation_fn"] == torch.nn.ReLU
        assert config.policy_kwargs["ortho_init"] is False

    def test_invalid_n_steps(self) -> None:
        """Тест валидации n_steps."""
        with pytest.raises(ValueError, match="n_steps должен быть > 0"):
            A2CConfig(env_name="LunarLander-v3", n_steps=0)

    def test_invalid_rms_prop_eps(self) -> None:
        """Тест валидации rms_prop_eps."""
        with pytest.raises(ValueError, match="rms_prop_eps должен быть > 0"):
            A2CConfig(env_name="LunarLander-v3", rms_prop_eps=0)

    def test_invalid_activation_fn(self) -> None:
        """Тест валидации функции активации."""
        with pytest.raises(ValueError, match="Неподдерживаемая функция активации"):
            A2CConfig(env_name="LunarLander-v3", activation_fn="invalid")

    @pytest.mark.parametrize("activation_fn,expected_class", [
        ("tanh", torch.nn.Tanh),
        ("relu", torch.nn.ReLU),
        ("elu", torch.nn.ELU),
    ])
    def test_activation_functions(self, activation_fn: str, expected_class: type) -> None:
        """Тест различных функций активации."""
        config = A2CConfig(env_name="LunarLander-v3", activation_fn=activation_fn)
        assert config.policy_kwargs["activation_fn"] == expected_class


class TestA2CMetricsCallback:
    """Тесты для колбэка метрик A2C."""

    def test_init(self) -> None:
        """Тест инициализации колбэка метрик."""
        metrics_tracker = MagicMock()
        callback = A2CMetricsCallback(
            metrics_tracker=metrics_tracker,
            log_freq=1000,
            verbose=1,
        )

        assert callback.metrics_tracker == metrics_tracker
        assert callback.log_freq == 1000
        assert callback.verbose == 1
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []

    def test_on_step_no_episodes(self) -> None:
        """Тест _on_step без эпизодов."""
        metrics_tracker = MagicMock()
        callback = A2CMetricsCallback(metrics_tracker, log_freq=1)

        # Мок модели без эпизодов
        callback.model = MagicMock()
        callback.model.ep_info_buffer = []
        callback.n_calls = 1
        callback.num_timesteps = 1000

        result = callback._on_step()
        assert result is True
        assert len(callback.episode_rewards) == 0

    def test_on_step_with_episodes(self) -> None:
        """Тест _on_step с эпизодами."""
        metrics_tracker = MagicMock()
        callback = A2CMetricsCallback(metrics_tracker, log_freq=1)

        # Мок модели с эпизодами
        callback.model = MagicMock()
        callback.model.ep_info_buffer = [
            {"r": 100.0, "l": 200},
            {"r": 150.0, "l": 250},
        ]
        callback.n_calls = 1
        callback.num_timesteps = 1000

        result = callback._on_step()
        assert result is True
        assert len(callback.episode_rewards) == 2
        assert callback.episode_rewards == [100.0, 150.0]
        assert callback.episode_lengths == [200, 250]


class TestA2CEarlyStoppingCallback:
    """Тесты для колбэка ранней остановки A2C."""

    def test_init(self) -> None:
        """Тест инициализации колбэка ранней остановки."""
        callback = A2CEarlyStoppingCallback(
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

    def test_on_step_target_reached(self) -> None:
        """Тест остановки при достижении целевой награды."""
        callback = A2CEarlyStoppingCallback(target_reward=200.0, check_freq=1)

        # Мок модели с высокими наградами
        callback.model = MagicMock()
        callback.model.ep_info_buffer = [{"r": 250.0}] * 20
        callback.n_calls = 1

        result = callback._on_step()
        assert result is False  # Должна остановить обучение

    def test_on_step_patience_exceeded(self) -> None:
        """Тест остановки при превышении терпения."""
        callback = A2CEarlyStoppingCallback(
            target_reward=200.0,
            patience_episodes=10,
            min_improvement=5.0,
            check_freq=1,
        )

        # Мок модели с низкими наградами
        callback.model = MagicMock()
        callback.model.ep_info_buffer = [{"r": 50.0}] * 20
        callback.n_calls = 1
        callback.best_mean_reward = 100.0  # Установим лучшую награду выше текущих
        callback.episodes_without_improvement = 15  # Превышает терпение
        callback.last_check_episode = 0

        result = callback._on_step()
        assert result is False  # Должна остановить обучение


class TestA2CAgent:
    """Тесты для A2C агента."""

    @pytest.fixture
    def config(self) -> A2CConfig:
        """Фикстура конфигурации для тестов."""
        return A2CConfig(
            env_name="CartPole-v1",
            total_timesteps=1000,
            learning_rate=1e-3,
            n_steps=5,
            verbose=0,
        )

    @pytest.fixture
    def mock_env(self):
        """Фикстура мок-среды."""
        env = MagicMock()
        env.action_space = spaces.Discrete(2)
        env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
        env.step.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1]),
            1.0,
            False,
            False,
            {},
        )
        return env

    def test_init_invalid_config_type(self, mock_env) -> None:
        """Тест инициализации с неправильным типом конфигурации."""
        from src.agents.base import AgentConfig

        invalid_config = AgentConfig(algorithm="A2C", env_name="CartPole-v1")

        with pytest.raises(ValueError, match="Ожидается A2CConfig"):
            A2CAgent(config=invalid_config, env=mock_env)

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_init_success(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест успешной инициализации агента."""
        # Настройка моков
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Создаем мок-объект, который имитирует DummyVecEnv с необходимыми атрибутами
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1  # Необходимый атрибут
        mock_vec_env.reset.return_value = (np.array([[0.0, 0.0, 0.0, 0.0]]), {})  # Для reset
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)

            assert isinstance(agent.config, A2CConfig)
            assert agent.model == mock_model
            assert len(agent.callbacks) > 0

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_create_learning_rate_schedule(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест создания расписания learning rate."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            # Тест константного learning rate
            config.use_lr_schedule = False
            agent = A2CAgent(config=config, env=mock_env)
            lr_schedule = agent._create_learning_rate_schedule()
            assert lr_schedule == config.learning_rate

            # Тест линейного расписания
            config.use_lr_schedule = True
            config.lr_schedule_type = "linear"
            agent = A2CAgent(config=config, env=mock_env)
            lr_schedule = agent._create_learning_rate_schedule()
            assert callable(lr_schedule)

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_train_success(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест успешного обучения."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)

            # Мок evaluate для финальной оценки
            with patch.object(agent, "evaluate") as mock_evaluate:
                mock_evaluate.return_value = {
                    "mean_reward": 100.0,
                    "std_reward": 10.0,
                }

                result = agent.train(total_timesteps=500)

                assert isinstance(result, TrainingResult)
                assert result.success is True
                assert result.total_timesteps == 500
                assert result.final_mean_reward == 100.0
                assert result.final_std_reward == 10.0
                assert agent.is_trained is True

                # Проверка вызова learn
                mock_model.learn.assert_called_once()

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_train_failure(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест ошибки обучения."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_model.learn.side_effect = Exception("Training failed")
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)

            with pytest.raises(RuntimeError, match="Ошибка обучения A2C агента"):
                agent.train()

            assert agent.training_result is not None
            assert agent.training_result.success is False

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_predict_not_trained(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест предсказания без обучения."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)

            observation = np.array([0.1, 0.2, 0.3, 0.4])

            with pytest.raises(RuntimeError, match="Модель не обучена"):
                agent.predict(observation)

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_predict_success(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест успешного предсказания."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([1]), None)
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)
            agent.is_trained = True

            observation = np.array([0.1, 0.2, 0.3, 0.4])
            action, state = agent.predict(observation)

            assert action.shape == (1,)
            assert action[0] == 1
            assert state is None

    def test_save_load_integration(self, config) -> None:
        """Интеграционный тест сохранения и загрузки."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_a2c_model.zip"

            with patch("src.agents.a2c_agent.make_vec_env") as mock_make_vec_env, \
                 patch("src.agents.a2c_agent.A2C") as mock_a2c_class, \
                 patch("src.agents.base.set_seed"), \
                 patch("src.agents.a2c_agent.get_experiment_logger"), \
                 patch("src.agents.a2c_agent.get_metrics_tracker"):

                # Настройка моков
                mock_vec_env = MagicMock(spec=DummyVecEnv)
                mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
                mock_vec_env.action_space = spaces.Discrete(2)
                mock_vec_env.num_envs = 1
                mock_make_vec_env.return_value = mock_vec_env

                # Создание и сохранение агента
                mock_model = MagicMock()
                mock_a2c_class.return_value = mock_model

                agent = A2CAgent(config=config)
                agent.save(str(model_path))

                # Проверка вызова сохранения
                mock_model.save.assert_called_once_with(str(model_path))

                # Проверка создания файла конфигурации
                config_path = model_path.with_suffix(".yaml")
                assert config_path.exists()

    @patch("src.agents.a2c_agent.A2C")
    def test_load_success(self, mock_a2c_class, config) -> None:
        """Тест успешной загрузки агента."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.zip"
            model_path.touch()  # Создаем файл

            mock_model = MagicMock()
            mock_a2c_class.load.return_value = mock_model

            with patch("src.agents.a2c_agent.make_vec_env") as mock_make_vec_env, \
                 patch("src.agents.base.set_seed"), \
                 patch("src.agents.a2c_agent.get_experiment_logger"), \
                 patch("src.agents.a2c_agent.get_metrics_tracker"):

                # Настройка моков
                mock_vec_env = MagicMock(spec=DummyVecEnv)
                mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
                mock_vec_env.action_space = spaces.Discrete(2)
                mock_vec_env.num_envs = 1
                mock_make_vec_env.return_value = mock_vec_env

                agent = A2CAgent.load(str(model_path), config=config)

                assert agent.is_trained is True
                assert agent.model == mock_model

    def test_load_file_not_found(self) -> None:
        """Тест загрузки несуществующего файла."""
        with pytest.raises(FileNotFoundError, match="Файл модели не найден"):
            A2CAgent.load("/nonexistent/path/model.zip")

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_get_model_info(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест получения информации о модели."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)
            info = agent.get_model_info()

            assert "algorithm" in info
            assert "n_steps" in info
            assert "ent_coef" in info
            assert "vf_coef" in info
            assert "use_rms_prop" in info
            assert info["algorithm"] == "A2C"
            assert info["n_steps"] == config.n_steps

    @patch("src.agents.a2c_agent.make_vec_env")
    @patch("src.agents.a2c_agent.A2C")
    def test_reset_model(self, mock_a2c_class, mock_make_vec_env, config, mock_env) -> None:
        """Тест сброса модели."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        mock_make_vec_env.return_value = mock_vec_env
        mock_model = MagicMock()
        mock_a2c_class.return_value = mock_model

        with patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            agent = A2CAgent(config=config, env=mock_env)
            agent.is_trained = True

            agent.reset_model()

            assert agent.is_trained is False
            assert agent.training_result is None
            mock_vec_env.close.assert_called_once()

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_reproducibility(self, seed: int, config) -> None:
        """Тест воспроизводимости с разными seed."""
        config.seed = seed

        with patch("src.agents.a2c_agent.make_vec_env") as mock_make_vec_env, \
             patch("src.agents.a2c_agent.A2C"), \
             patch("src.agents.base.set_seed") as mock_set_seed, \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            # Настройка моков
            mock_vec_env = MagicMock(spec=DummyVecEnv)
            mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
            mock_vec_env.action_space = spaces.Discrete(2)
            mock_vec_env.num_envs = 1
            mock_make_vec_env.return_value = mock_vec_env

            A2CAgent(config=config)
            mock_set_seed.assert_called_once_with(seed)

    def test_cleanup_on_delete(self, config) -> None:
        """Тест очистки ресурсов при удалении объекта."""
        with patch("src.agents.a2c_agent.make_vec_env") as mock_make_vec_env, \
             patch("src.agents.a2c_agent.A2C"), \
             patch("src.agents.base.set_seed"), \
             patch("src.agents.a2c_agent.get_experiment_logger"), \
             patch("src.agents.a2c_agent.get_metrics_tracker"):

            mock_vec_env = MagicMock(spec=DummyVecEnv)
            mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
            mock_vec_env.action_space = spaces.Discrete(2)
            mock_vec_env.num_envs = 1
            mock_make_vec_env.return_value = mock_vec_env

            agent = A2CAgent(config=config)
            del agent

            mock_vec_env.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])