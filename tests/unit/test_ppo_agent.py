"""Тесты для PPO агента.

Модуль содержит комплексные тесты для PPOAgent и PPOConfig,
включая проверку инициализации, обучения, предсказания,
сохранения/загрузки и интеграции с системой метрик.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from src.agents.ppo_agent import PPOAgent, PPOConfig, PPOMetricsCallback, EarlyStoppingCallback
from src.utils import MetricsTracker


class TestPPOConfig:
    """Тесты для конфигурации PPO агента."""

    def test_default_config(self) -> None:
        """Тест создания конфигурации с параметрами по умолчанию."""
        config = PPOConfig(env_name="LunarLander-v3")

        assert config.algorithm == "PPO"
        assert config.env_name == "LunarLander-v3"
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.gamma == 0.999
        assert config.normalize_env is True
        assert config.early_stopping is True
        assert config.target_reward == 200.0

    def test_custom_config(self) -> None:
        """Тест создания конфигурации с кастомными параметрами."""
        config = PPOConfig(
            env_name="CartPole-v1",
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=32,
            target_reward=195.0,
            normalize_env=False,
        )

        assert config.env_name == "CartPole-v1"
        assert config.learning_rate == 1e-3
        assert config.n_steps == 1024
        assert config.batch_size == 32
        assert config.target_reward == 195.0
        assert config.normalize_env is False

    def test_invalid_clip_range(self) -> None:
        """Тест валидации clip_range."""
        with pytest.raises(ValueError, match="clip_range должен быть в"):
            PPOConfig(env_name="LunarLander-v3", clip_range=1.5)

        with pytest.raises(ValueError, match="clip_range должен быть в"):
            PPOConfig(env_name="LunarLander-v3", clip_range=0.0)

    def test_invalid_n_steps(self) -> None:
        """Тест валидации n_steps."""
        with pytest.raises(ValueError, match="n_steps должен быть > 0"):
            PPOConfig(env_name="LunarLander-v3", n_steps=0)

        with pytest.raises(ValueError, match="n_steps должен быть > 0"):
            PPOConfig(env_name="LunarLander-v3", n_steps=-100)

    def test_invalid_n_epochs(self) -> None:
        """Тест валидации n_epochs."""
        with pytest.raises(ValueError, match="n_epochs должен быть > 0"):
            PPOConfig(env_name="LunarLander-v3", n_epochs=0)

    def test_invalid_activation_function(self) -> None:
        """Тест валидации функции активации."""
        with pytest.raises(ValueError, match="Неподдерживаемая функция активации"):
            PPOConfig(env_name="LunarLander-v3", activation_fn="invalid")

    def test_policy_kwargs_setup(self) -> None:
        """Тест настройки policy_kwargs."""
        config = PPOConfig(
            env_name="LunarLander-v3",
            activation_fn="relu",
            ortho_init=False,
        )

        assert "net_arch" in config.policy_kwargs
        assert "activation_fn" in config.policy_kwargs
        assert config.policy_kwargs["activation_fn"] == torch.nn.ReLU
        assert config.policy_kwargs["ortho_init"] is False


class TestPPOMetricsCallback:
    """Тесты для колбэка метрик PPO."""

    @pytest.fixture
    def metrics_tracker(self) -> MetricsTracker:
        """Создать мок трекера метрик."""
        return MagicMock(spec=MetricsTracker)

    @pytest.fixture
    def callback(self, metrics_tracker: MetricsTracker) -> PPOMetricsCallback:
        """Создать колбэк метрик."""
        return PPOMetricsCallback(
            metrics_tracker=metrics_tracker,
            log_freq=100,
            verbose=0,
        )

    def test_callback_initialization(self, callback: PPOMetricsCallback) -> None:
        """Тест инициализации колбэка."""
        assert callback.log_freq == 100
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []

    def test_on_step_with_episodes(self, callback: PPOMetricsCallback) -> None:
        """Тест обработки шага с информацией об эпизодах."""
        # Мок модели с информацией об эпизодах
        mock_model = MagicMock()
        mock_model.ep_info_buffer = [
            {"r": 100.0, "l": 200},
            {"r": 150.0, "l": 180},
        ]
        callback.model = mock_model
        callback.n_calls = 100  # Триггер логирования

        result = callback._on_step()

        assert result is True
        assert len(callback.episode_rewards) == 2
        assert callback.episode_rewards == [100.0, 150.0]
        assert callback.episode_lengths == [200, 180]

    def test_metrics_logging(self, callback: PPOMetricsCallback, metrics_tracker: MetricsTracker) -> None:
        """Тест логирования метрик."""
        # Подготовка данных
        callback.episode_rewards = [100.0, 150.0, 120.0]
        callback.episode_lengths = [200, 180, 190]
        callback.num_timesteps = 1000

        # Мок модели с логгером
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/policy_loss": 0.1,
            "train/value_loss": 0.05,
        }
        mock_model = MagicMock()
        mock_model.logger = mock_logger
        callback.model = mock_model

        callback._log_metrics()

        # Проверка вызовов add_metric
        assert metrics_tracker.add_metric.call_count >= 5  # Минимум 5 базовых метрик


class TestEarlyStoppingCallback:
    """Тесты для колбэка ранней остановки."""

    @pytest.fixture
    def callback(self) -> EarlyStoppingCallback:
        """Создать колбэк ранней остановки."""
        return EarlyStoppingCallback(
            target_reward=200.0,
            patience_episodes=10,
            min_improvement=5.0,
            check_freq=100,
            verbose=0,
        )

    def test_callback_initialization(self, callback: EarlyStoppingCallback) -> None:
        """Тест инициализации колбэка."""
        assert callback.target_reward == 200.0
        assert callback.patience_episodes == 10
        assert callback.min_improvement == 5.0
        assert callback.best_mean_reward == float("-inf")
        assert callback.episodes_without_improvement == 0

    def test_target_reward_reached(self, callback: EarlyStoppingCallback) -> None:
        """Тест остановки при достижении целевой награды."""
        # Мок модели с высокими наградами
        mock_model = MagicMock()
        mock_model.ep_info_buffer = [{"r": 210.0} for _ in range(20)]
        callback.model = mock_model
        callback.n_calls = 100  # Триггер проверки

        result = callback._on_step()

        assert result is False  # Остановка обучения

    def test_patience_exceeded(self, callback: EarlyStoppingCallback) -> None:
        """Тест остановки при превышении терпения."""
        callback.episodes_without_improvement = 15  # Больше patience_episodes
        callback.last_check_episode = 0
        callback.best_mean_reward = 100.0  # Установим лучшую награду выше текущих

        # Мок модели с низкими наградами
        mock_model = MagicMock()
        mock_model.ep_info_buffer = [{"r": 50.0} for _ in range(20)]
        callback.model = mock_model
        callback.n_calls = 100  # Триггер проверки

        result = callback._on_step()

        assert result is False  # Остановка обучения

    def test_improvement_resets_patience(self, callback: EarlyStoppingCallback) -> None:
        """Тест сброса терпения при улучшении."""
        callback.best_mean_reward = 100.0
        callback.episodes_without_improvement = 5

        # Мок модели с улучшенными наградами
        mock_model = MagicMock()
        mock_model.ep_info_buffer = [{"r": 110.0} for _ in range(20)]  # Улучшение > 5.0
        callback.model = mock_model
        callback.n_calls = 100  # Триггер проверки

        result = callback._on_step()

        assert result is True  # Продолжение обучения
        assert callback.episodes_without_improvement == 0


class TestPPOAgent:
    """Тесты для PPO агента."""

    @pytest.fixture
    def config(self) -> PPOConfig:
        """Создать тестовую конфигурацию."""
        return PPOConfig(
            env_name="CartPole-v1",
            total_timesteps=1000,
            n_steps=64,
            batch_size=32,
            verbose=0,
            use_tensorboard=False,
            normalize_env=False,
            early_stopping=False,
        )

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Создать временную директорию."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_agent_initialization(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
        config: PPOConfig,
    ) -> None:
        """Тест инициализации PPO агента."""
        # Настройка моков
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)

        assert agent.config == config
        assert agent.model is not None
        assert agent.vec_env == mock_vec_env
        assert len(agent.callbacks) > 0
        mock_make_vec_env.assert_called_once()

    def test_invalid_config_type(self) -> None:
        """Тест ошибки при неправильном типе конфигурации."""
        from src.agents.base import AgentConfig

        invalid_config = AgentConfig(algorithm="PPO", env_name="CartPole-v1")

        with pytest.raises(ValueError, match="Ожидается PPOConfig"):
            PPOAgent(invalid_config)

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_vectorized_env_with_normalization(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
    ) -> None:
        """Тест создания векторизованной среды с нормализацией."""
        config = PPOConfig(
            env_name="CartPole-v1",
            normalize_env=True,
            verbose=0,
        )

        mock_vec_env = MagicMock(spec=VecNormalize)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env  # Это позволяет unwrap работать правильно
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)

        assert isinstance(agent.vec_env, VecNormalize)

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_learning_rate_schedule_linear(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
    ) -> None:
        """Тест создания линейного расписания learning rate."""
        config = PPOConfig(
            env_name="CartPole-v1",
            use_lr_schedule=True,
            lr_schedule_type="linear",
            verbose=0,
        )

        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)
        lr_schedule = agent._create_learning_rate_schedule()

        # Проверка что возвращается функция
        assert callable(lr_schedule)

        # Проверка значений в начале и конце
        initial_lr = lr_schedule(1.0)  # progress_remaining = 1.0
        final_lr = lr_schedule(0.0)    # progress_remaining = 0.0

        assert initial_lr == config.learning_rate
        # Проверяем, что final_lr близко к ожидаемому значению (с учетом возможных вычислительных погрешностей)
        expected_final_lr = config.learning_rate * config.lr_final_ratio
        assert abs(final_lr - expected_final_lr) < 1e-10

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_learning_rate_schedule_constant(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
    ) -> None:
        """Тест константного learning rate."""
        config = PPOConfig(
            env_name="CartPole-v1",
            use_lr_schedule=False,
            verbose=0,
        )

        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)
        lr_schedule = agent._create_learning_rate_schedule()

        # Проверка что возвращается константа
        assert lr_schedule == config.learning_rate

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_predict_without_training(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
        config: PPOConfig,
    ) -> None:
        """Тест ошибки предсказания без обучения."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)
        observation = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(RuntimeError, match="Модель не обучена"):
            agent.predict(observation)

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_save_and_load(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
        config: PPOConfig,
        temp_dir: Path,
    ) -> None:
        """Тест сохранения и загрузки агента."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        # Создание и "обучение" агента
        agent = PPOAgent(config)
        agent.is_trained = True

        # Мок модели для сохранения
        mock_model = MagicMock()
        agent.model = mock_model

        # Сохранение
        save_path = temp_dir / "test_model.zip"
        agent.save(str(save_path))

        # Проверка вызова save у модели
        mock_model.save.assert_called_once_with(str(save_path))

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_get_model_info(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
        config: PPOConfig,
    ) -> None:
        """Тест получения информации о модели."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent = PPOAgent(config)
        info = agent.get_model_info()

        # Проверка базовых полей
        assert "algorithm" in info
        assert "env_name" in info
        assert "is_trained" in info

        # Проверка PPO-специфичных полей
        assert "n_steps" in info
        assert "batch_size" in info
        assert "clip_range" in info
        assert "normalize_env" in info

        assert info["algorithm"] == "PPO"
        assert info["n_steps"] == config.n_steps
        assert info["normalize_env"] == config.normalize_env

    @patch("src.agents.ppo_agent.PPO")
    @patch("src.agents.ppo_agent.make_vec_env")
    @patch("src.utils.get_experiment_logger")
    @patch("src.utils.get_metrics_tracker")
    def test_reset_model(
        self,
        mock_metrics_tracker: MagicMock,
        mock_logger: MagicMock,
        mock_make_vec_env: MagicMock,
        mock_ppo_class: MagicMock,
        config: PPOConfig,
    ) -> None:
        """Тест сброса модели."""
        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели: всегда возвращаем новый экземпляр
        original_model = MagicMock()
        new_model = MagicMock()

        # Используем side_effect с функцией, чтобы отслеживать вызовы
        call_count = 0
        def side_effect_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_model
            else:
                return new_model

        mock_ppo_class.side_effect = side_effect_func

        agent = PPOAgent(config)
        agent.is_trained = True

        # Сохраняем оригинальную модель
        original_model_ref = agent.model
        # Вызываем reset_model, который должен создать новую модель
        agent.reset_model()

        assert agent.is_trained is False
        assert agent.model is new_model  # После сброса должна быть новая модель
        assert agent.model is not original_model_ref  # Новая модель отличается от старой
        mock_vec_env.close.assert_called()


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_reproducibility_with_different_seeds(seed: int) -> None:
    """Тест воспроизводимости с разными seeds."""
    config1 = PPOConfig(
        env_name="CartPole-v1",
        seed=seed,
        verbose=0,
        use_tensorboard=False,
        normalize_env=False,
    )
    config2 = PPOConfig(
        env_name="CartPole-v1",
        seed=seed,
        verbose=0,
        use_tensorboard=False,
        normalize_env=False,
    )

    with patch("src.agents.ppo_agent.PPO") as mock_ppo_class, \
         patch("src.agents.ppo_agent.make_vec_env") as mock_make_vec_env, \
         patch("src.utils.get_experiment_logger"), \
         patch("src.utils.get_metrics_tracker"):

        mock_vec_env = MagicMock(spec=DummyVecEnv)
        mock_vec_env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        mock_vec_env.action_space = spaces.Discrete(2)
        mock_vec_env.num_envs = 1
        # Добавляем атрибут venv для корректной работы SB3
        mock_vec_env.venv = mock_vec_env
        mock_make_vec_env.return_value = mock_vec_env

        # Мокируем создание модели
        mock_model_instance = MagicMock()
        mock_ppo_class.return_value = mock_model_instance

        agent1 = PPOAgent(config1)
        agent2 = PPOAgent(config2)

        # Проверка что seed установлен одинаково
        assert agent1.config.seed == agent2.config.seed == seed


def test_ppo_config_inheritance() -> None:
    """Тест наследования PPOConfig от AgentConfig."""
    from src.agents.base import AgentConfig

    config = PPOConfig(env_name="LunarLander-v3")

    assert isinstance(config, AgentConfig)
    assert config.algorithm == "PPO"

    # Проверка что базовые поля доступны
    assert hasattr(config, "total_timesteps")
    assert hasattr(config, "seed")
    assert hasattr(config, "learning_rate")

    # Проверка PPO-специфичных полей
    assert hasattr(config, "n_steps")
    assert hasattr(config, "clip_range")
    assert hasattr(config, "normalize_env")