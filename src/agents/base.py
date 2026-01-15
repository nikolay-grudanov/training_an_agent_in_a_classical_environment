"""Базовый класс для RL агентов с интеграцией Stable-Baselines3.

Этот модуль предоставляет абстрактный базовый класс Agent, который определяет
единый интерфейс для всех RL агентов в проекте. Включает поддержку конфигурации,
логирования, метрик, воспроизводимости и интеграции с SB3.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv

from src.utils import (
    CheckpointManager,
    set_seed,
    get_experiment_logger,
    get_metrics_tracker,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Конфигурация для RL агента.

    Содержит все параметры, необходимые для инициализации и обучения агента,
    включая гиперпараметры алгоритма, настройки среды и параметры эксперимента.
    """

    # Основные параметры
    algorithm: str = "Unknown"  # PPO, A2C, SAC, TD3, etc.
    env_name: str = "CartPole-v1"  # Название среды Gymnasium
    total_timesteps: int = 100_000
    seed: int = 42

    # Гиперпараметры алгоритма
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # Для on-policy алгоритмов
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Настройки модели
    policy: str = "MlpPolicy"
    policy_kwargs: Optional[Dict[str, Any]] = None
    device: str = "auto"
    verbose: int = 1

    # Настройки обучения
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    save_freq: int = 50_000
    log_interval: int = 1

    # Пути для сохранения
    model_save_path: Optional[str] = None
    tensorboard_log: Optional[str] = None

    # Дополнительные параметры
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None

    def __post_init__(self) -> None:
        """Валидация и нормализация параметров после инициализации."""
        if self.total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps должен быть > 0, получен: {self.total_timesteps}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate должен быть > 0, получен: {self.learning_rate}"
            )

        if self.gamma < 0 or self.gamma > 1:
            raise ValueError(f"gamma должен быть в [0, 1], получен: {self.gamma}")

        # Установка значений по умолчанию для policy_kwargs
        if self.policy_kwargs is None:
            self.policy_kwargs = {}


@dataclass
class TrainingResult:
    """Результат обучения агента.

    Содержит метрики производительности, информацию о процессе обучения
    и пути к сохраненным артефактам.
    """

    # Основные метрики
    total_timesteps: int
    training_time: float
    final_mean_reward: float
    final_std_reward: float

    # История обучения
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    timesteps_history: List[int] = field(default_factory=list)

    # Метрики оценки
    eval_mean_rewards: List[float] = field(default_factory=list)
    eval_std_rewards: List[float] = field(default_factory=list)
    eval_timesteps: List[int] = field(default_factory=list)

    # Пути к артефактам
    model_path: Optional[str] = None
    logs_path: Optional[str] = None

    # Дополнительные метрики
    best_mean_reward: float = float("-inf")
    convergence_timestep: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class ModelProtocol(Protocol):
    """Протокол для SB3 моделей."""

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> "ModelProtocol":
        """Обучить модель."""
        ...

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказать действие."""
        ...

    def save(self, path: str) -> None:
        """Сохранить модель."""
        ...

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None) -> "ModelProtocol":
        """Загрузить модель."""
        ...


class Agent(ABC):
    """Абстрактный базовый класс для всех RL агентов.

    Определяет единый интерфейс для обучения, предсказания, сохранения и загрузки
    моделей. Обеспечивает интеграцию с системой логирования, метрик и чекпоинтов.

    Поддерживаемые алгоритмы:
    - PPO (Proximal Policy Optimization)
    - A2C (Advantage Actor-Critic)
    - SAC (Soft Actor-Critic)
    - TD3 (Twin Delayed Deep Deterministic Policy Gradient)
    - DQN (Deep Q-Network)
    """

    def __init__(
        self,
        config: AgentConfig,
        env: Optional[gym.Env] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Инициализировать агента.

        Args:
            config: Конфигурация агента
            env: Среда Gymnasium (создается автоматически если None)
            experiment_name: Имя эксперимента для логирования

        Raises:
            ValueError: При некорректной конфигурации
            RuntimeError: При ошибке инициализации среды или модели
        """
        self.config = config
        self.experiment_name = (
            experiment_name or f"{config.algorithm}_{config.env_name}"
        )

        # Установка seed для воспроизводимости
        set_seed(config.seed)

        # Инициализация логирования
        self.logger = get_experiment_logger(
            experiment_id=self.experiment_name,
        )

        # Инициализация трекера метрик
        self.metrics_tracker = get_metrics_tracker(experiment_id=self.experiment_name)

        # Инициализация среды
        self.env = env
        if self.env is None:
            self.env = self._create_environment()

        # Валидация совместимости среды и алгоритма
        self._validate_env_algorithm_compatibility()

        # Инициализация модели (должна быть реализована в подклассах)
        self.model: Optional[ModelProtocol] = None

        # Менеджер чекпоинтов
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.model_save_path:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(config.model_save_path).parent,
                experiment_id=experiment_name or f"{config.algorithm}_{config.env_name}",
                max_checkpoints=5,
            )

        # Состояние обучения
        self.is_trained = False
        self.training_result: Optional[TrainingResult] = None

        self.logger.info(
            f"Инициализирован агент {self.__class__.__name__}",
            extra={
                "algorithm": config.algorithm,
                "env_name": config.env_name,
                "seed": config.seed,
                "total_timesteps": config.total_timesteps,
            },
        )

    def _create_environment(self) -> gym.Env:
        """Создать среду Gymnasium.

        Returns:
            Инициализированная среда

        Raises:
            RuntimeError: При ошибке создания среды
        """
        try:
            env = gym.make(self.config.env_name)
            self.logger.info(f"Создана среда: {self.config.env_name}")
            return env
        except Exception as e:
            error_msg = f"Ошибка создания среды {self.config.env_name}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_env_algorithm_compatibility(self) -> None:
        """Проверить совместимость среды и алгоритма.

        Raises:
            ValueError: При несовместимости среды и алгоритма
        """
        if self.env is None:
            raise ValueError("Среда не инициализирована")

        action_space = self.env.action_space
        algorithm = self.config.algorithm.upper()

        # Проверка дискретных действий
        action_space_type = action_space.__class__.__name__
        
        if action_space_type == "Discrete":
            if algorithm in ["SAC", "TD3", "DDPG"]:
                raise ValueError(
                    f"Алгоритм {algorithm} не поддерживает дискретные действия. "
                    f"Используйте PPO, A2C или DQN."
                )

        # Проверка непрерывных действий
        elif action_space_type == "Box":
            if algorithm in ["DQN"]:
                raise ValueError(
                    f"Алгоритм {algorithm} не поддерживает непрерывные действия. "
                    f"Используйте PPO, A2C, SAC или TD3."
                )

        self.logger.debug(
            "Совместимость среды и алгоритма проверена",
            extra={
                "action_space": str(action_space),
                "algorithm": algorithm,
            },
        )

    @abstractmethod
    def _create_model(self) -> ModelProtocol:
        """Создать модель SB3.

        Должен быть реализован в подклассах для конкретных алгоритмов.

        Returns:
            Инициализированная модель SB3

        Raises:
            NotImplementedError: Если не реализован в подклассе
        """
        raise NotImplementedError(
            "Метод _create_model должен быть реализован в подклассе"
        )

    @abstractmethod
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Обучить агента.

        Args:
            total_timesteps: Количество шагов обучения (по умолчанию из config)
            callback: Callback для мониторинга обучения
            **kwargs: Дополнительные параметры для алгоритма

        Returns:
            Результат обучения с метриками и путями к артефактам

        Raises:
            RuntimeError: При ошибке обучения
        """
        raise NotImplementedError("Метод train должен быть реализован в подклассе")

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказать действие для наблюдения.

        Args:
            observation: Наблюдение из среды
            deterministic: Использовать детерминистическую политику
            **kwargs: Дополнительные параметры

        Returns:
            Кортеж (действие, состояние) где состояние может быть None

        Raises:
            RuntimeError: Если модель не обучена
        """
        raise NotImplementedError("Метод predict должен быть реализован в подклассе")

    def save(self, path: str) -> None:
        """Сохранить модель агента.

        Args:
            path: Путь для сохранения модели

        Raises:
            RuntimeError: Если модель не инициализирована
            OSError: При ошибке записи файла
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Сначала создайте модель.")

        try:
            # Создание директории если не существует
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Сохранение модели
            self.model.save(str(save_path))

            # Сохранение конфигурации
            config_path = save_path.with_suffix(".yaml")
            self._save_config(config_path)

            self.logger.info(f"Модель сохранена: {path}")

        except Exception as e:
            error_msg = f"Ошибка сохранения модели в {path}: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from e

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> "Agent":
        """Загрузить агента из файла.

        Args:
            path: Путь к сохраненной модели
            env: Среда (создается автоматически если None)
            **kwargs: Дополнительные параметры

        Returns:
            Загруженный агент

        Raises:
            FileNotFoundError: Если файл модели не найден
            RuntimeError: При ошибке загрузки
        """
        raise NotImplementedError("Метод load должен быть реализован в подклассе")

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """Оценить производительность агента.

        Args:
            n_episodes: Количество эпизодов для оценки
            deterministic: Использовать детерминистическую политику
            render: Отображать среду во время оценки

        Returns:
            Словарь с метриками оценки

        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите train().")

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
        }

        self.logger.info(f"Оценка завершена за {n_episodes} эпизодов", extra=metrics)

        return metrics

    def _save_config(self, path: Path) -> None:
        """Сохранить конфигурацию агента.

        Args:
            path: Путь для сохранения конфигурации
        """
        import yaml
        from dataclasses import asdict

        try:
            config_dict = asdict(self.config)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            self.logger.debug(f"Конфигурация сохранена: {path}")

        except Exception as e:
            self.logger.warning(f"Ошибка сохранения конфигурации: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели.

        Returns:
            Словарь с информацией о модели и конфигурации
        """
        info = {
            "algorithm": self.config.algorithm,
            "env_name": self.config.env_name,
            "is_trained": self.is_trained,
            "total_timesteps": self.config.total_timesteps,
            "seed": self.config.seed,
        }

        if self.env is not None:
            info.update(
                {
                    "observation_space": str(self.env.observation_space),
                    "action_space": str(self.env.action_space),
                }
            )

        if self.training_result is not None:
            info.update(
                {
                    "final_mean_reward": self.training_result.final_mean_reward,
                    "training_time": self.training_result.training_time,
                    "best_mean_reward": self.training_result.best_mean_reward,
                }
            )

        return info

    def reset_model(self) -> None:
        """Сбросить модель для переобучения."""
        self.model = None
        self.is_trained = False
        self.training_result = None

        # Пересоздание модели
        self.model = self._create_model()

        self.logger.info("Модель сброшена для переобучения")

    def __repr__(self) -> str:
        """Строковое представление агента."""
        return (
            f"{self.__class__.__name__}("
            f"algorithm={self.config.algorithm}, "
            f"env={self.config.env_name}, "
            f"trained={self.is_trained})"
        )
