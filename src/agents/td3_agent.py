"""TD3 агент с оптимизированными гиперпараметрами для непрерывных сред.

Этот модуль реализует агента на основе алгоритма Twin Delayed Deep Deterministic
Policy Gradient (TD3) с использованием Stable-Baselines3. Предназначен для
непрерывных пространств действий с поддержкой шумовых параметров для исследования,
кастомных колбэков для мониторинга, интеграции с системой логирования и метрик.

Особенности:
- Оптимизированные гиперпараметры для непрерывных сред (LunarLander-v3 continuous)
- Поддержка различных типов шума для исследования (Gaussian, OrnsteinUhlenbeck)
- Адаптивное расписание learning rate и шума
- Кастомные колбэки для мониторинга и ранней остановки
- Поддержка TensorBoard логирования
- Автоматическое сохранение чекпоинтов
- Комплексная обработка ошибок и валидация
- Интеграция с системой метрик проекта
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    ActionNoise,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.base import Agent, AgentConfig, TrainingResult
from src.utils import (
    MetricsTracker,
)

logger = logging.getLogger(__name__)


@dataclass
class TD3Config(AgentConfig):
    """Конфигурация для TD3 агента с оптимизированными параметрами.

    Расширяет базовую конфигурацию агента специфичными для TD3 параметрами,
    оптимизированными для непрерывных сред управления.
    """

    # Переопределение базовых параметров для TD3
    algorithm: str = "TD3"

    # Оптимизированные гиперпараметры для непрерывных сред
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005  # Soft update coefficient
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = (1, "step")
    gradient_steps: int = 1

    # TD3-специфичные параметры
    policy_delay: int = 2  # Задержка обновления политики
    target_policy_noise: float = 0.2  # Шум для целевой политики
    target_noise_clip: float = 0.5  # Ограничение шума целевой политики

    # Параметры шума для исследования
    action_noise_type: str = "normal"  # "normal", "ornstein_uhlenbeck", "none"
    action_noise_std: float = 0.1  # Стандартное отклонение шума
    action_noise_mean: float = 0.0  # Среднее значение шума

    # Параметры для OrnsteinUhlenbeck шума
    ou_theta: float = 0.15  # Скорость возврата к среднему
    ou_dt: float = 1e-2  # Временной шаг
    ou_sigma: float = 0.2  # Волатильность

    # Расписание learning rate и шума
    use_lr_schedule: bool = True
    lr_schedule_type: str = "linear"  # "linear", "constant", "exponential"
    lr_final_ratio: float = 0.1

    use_noise_schedule: bool = True
    noise_schedule_type: str = "linear"  # "linear", "exponential"
    noise_final_ratio: float = 0.01

    # Параметры сети
    net_arch: Optional[List[int]] = None
    activation_fn: str = "relu"  # "relu", "tanh", "elu"

    # Нормализация
    normalize_env: bool = True
    norm_obs: bool = True
    norm_reward: bool = False  # Обычно не используется для TD3
    clip_obs: float = 10.0

    # Ранняя остановка
    early_stopping: bool = True
    target_reward: float = 200.0  # Для LunarLander-v3 continuous
    patience_episodes: int = 100
    min_improvement: float = 10.0

    # Мониторинг и логирование
    use_tensorboard: bool = True

    # Дополнительные параметры для оптимизации
    stats_window_size: int = 100
    policy_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Валидация и установка значений по умолчанию."""
        super().__post_init__()

        # Валидация TD3-специфичных параметров
        if self.tau <= 0 or self.tau > 1:
            raise ValueError(f"tau должен быть в (0, 1], получен: {self.tau}")

        if self.policy_delay < 1:
            raise ValueError(
                f"policy_delay должен быть >= 1, получен: {self.policy_delay}"
            )

        if self.target_policy_noise < 0:
            raise ValueError(
                f"target_policy_noise должен быть >= 0, получен: {self.target_policy_noise}"
            )

        if self.buffer_size <= 0:
            raise ValueError(
                f"buffer_size должен быть > 0, получен: {self.buffer_size}"
            )

        if self.learning_starts < 0:
            raise ValueError(
                f"learning_starts должен быть >= 0, получен: {self.learning_starts}"
            )

        # Установка архитектуры сети по умолчанию
        if self.net_arch is None:
            self.net_arch = [400, 300]

        # Валидация типа шума
        valid_noise_types = ["normal", "ornstein_uhlenbeck", "none"]
        if self.action_noise_type not in valid_noise_types:
            raise ValueError(
                f"action_noise_type должен быть одним из {valid_noise_types}, "
                f"получен: {self.action_noise_type}"
            )

        # Установка policy_kwargs
        activation_functions = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "elu": torch.nn.ELU,
        }

        if self.activation_fn not in activation_functions:
            raise ValueError(
                f"Неподдерживаемая функция активации: {self.activation_fn}. "
                f"Доступные: {list(activation_functions.keys())}"
            )

        self.policy_kwargs = self.policy_kwargs or {}
        self.policy_kwargs.update(
            {
                "net_arch": self.net_arch,
                "activation_fn": activation_functions[self.activation_fn],
            }
        )


class TD3MetricsCallback(BaseCallback):
    """Колбэк для отслеживания метрик обучения TD3 агента.

    Интегрируется с системой метрик проекта для записи детальной
    информации о процессе обучения, включая специфичные для TD3 метрики.
    """

    def __init__(
        self,
        metrics_tracker: MetricsTracker,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """Инициализировать колбэк метрик.

        Args:
            metrics_tracker: Трекер метрик проекта
            log_freq: Частота логирования метрик
            verbose: Уровень детализации логов
        """
        super().__init__(verbose)
        self.metrics_tracker = metrics_tracker
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        """Вызывается на каждом шаге обучения."""
        # Сбор информации об эпизодах
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info["r"])
                self.episode_lengths.append(info["l"])

        # Логирование метрик с заданной частотой
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Записать текущие метрики."""
        if len(self.episode_rewards) == 0:
            return

        # Основные метрики
        recent_rewards = self.episode_rewards[-100:]  # Последние 100 эпизодов
        recent_lengths = self.episode_lengths[-100:]

        metrics = {
            "timesteps": self.num_timesteps,
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "mean_length": np.mean(recent_lengths),
            "max_reward": np.max(recent_rewards),
            "min_reward": np.min(recent_rewards),
        }

        # Добавление TD3-специфичных метрик если доступны
        if hasattr(self.model, "logger") and self.model.logger is not None:
            td3_metrics = [
                "train/actor_loss",
                "train/critic_loss",
                "train/n_updates",
            ]
            for key in td3_metrics:
                if key in self.model.logger.name_to_value:
                    metrics[key.replace("train/", "")] = (
                        self.model.logger.name_to_value[key]
                    )

        # Запись метрик
        for name, value in metrics.items():
            self.metrics_tracker.add_metric(
                name=name,
                value=float(value),
                timestep=self.num_timesteps,
                metadata={"phase": "training", "algorithm": "TD3"},
            )

        if self.verbose > 0:
            logger.info(
                f"Шаг {self.num_timesteps}: средняя награда = {metrics['mean_reward']:.2f} "
                f"± {metrics['std_reward']:.2f}"
            )


class TD3EarlyStoppingCallback(BaseCallback):
    """Колбэк для ранней остановки обучения TD3 при достижении целевой производительности."""

    def __init__(
        self,
        target_reward: float,
        patience_episodes: int = 100,
        min_improvement: float = 10.0,
        check_freq: int = 10000,
        verbose: int = 0,
    ):
        """Инициализировать колбэк ранней остановки.

        Args:
            target_reward: Целевая средняя награда для остановки
            patience_episodes: Количество эпизодов без улучшения для остановки
            min_improvement: Минимальное улучшение для сброса счетчика терпения
            check_freq: Частота проверки условий остановки
            verbose: Уровень детализации логов
        """
        super().__init__(verbose)
        self.target_reward = target_reward
        self.patience_episodes = patience_episodes
        self.min_improvement = min_improvement
        self.check_freq = check_freq

        self.best_mean_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.last_check_episode = 0

    def _on_step(self) -> bool:
        """Проверить условия ранней остановки."""
        if self.n_calls % self.check_freq != 0:
            return True

        if len(self.model.ep_info_buffer) == 0:
            return True

        # Вычисление средней награды за последние эпизоды
        recent_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer[-100:]]
        if len(recent_rewards) < 10:  # Недостаточно данных
            return True

        current_mean_reward = np.mean(recent_rewards)
        current_episode = len(self.model.ep_info_buffer)

        # Проверка достижения целевой награды
        if current_mean_reward >= self.target_reward:
            if self.verbose > 0:
                logger.info(
                    f"Достигнута целевая награда {self.target_reward:.2f}! "
                    f"Текущая средняя награда: {current_mean_reward:.2f}"
                )
            return False  # Остановить обучение

        # Проверка улучшения
        if current_mean_reward > self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = current_mean_reward
            self.episodes_without_improvement = 0
            if self.verbose > 1:
                logger.debug(
                    f"Новая лучшая средняя награда: {self.best_mean_reward:.2f}"
                )
        else:
            episodes_passed = current_episode - self.last_check_episode
            self.episodes_without_improvement += episodes_passed

        self.last_check_episode = current_episode

        # Проверка терпения
        if self.episodes_without_improvement >= self.patience_episodes:
            if self.verbose > 0:
                logger.info(
                    f"Ранняя остановка: {self.episodes_without_improvement} эпизодов "
                    f"без улучшения (лучшая награда: {self.best_mean_reward:.2f})"
                )
            return False  # Остановить обучение

        return True


class TD3Agent(Agent):
    """TD3 агент с оптимизированными гиперпараметрами для непрерывных сред.

    Реализует алгоритм Twin Delayed Deep Deterministic Policy Gradient с поддержкой:
    - Непрерывных пространств действий
    - Различных типов шума для исследования
    - Адаптивного расписания learning rate и шума
    - Нормализации наблюдений
    - Ранней остановки обучения
    - Детального мониторинга и логирования
    - Автоматического сохранения чекпоинтов
    """

    def __init__(
        self,
        config: TD3Config,
        env: Optional[gym.Env] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Инициализировать TD3 агента.

        Args:
            config: Конфигурация TD3 агента
            env: Среда Gymnasium (создается автоматически если None)
            experiment_name: Имя эксперимента для логирования

        Raises:
            ValueError: При некорректной конфигурации или дискретном пространстве действий
            RuntimeError: При ошибке инициализации
        """
        # Валидация типа конфигурации
        if not isinstance(config, TD3Config):
            raise ValueError(f"Ожидается TD3Config, получен {type(config).__name__}")

        super().__init__(config, env, experiment_name)

        # Приведение типа для удобства
        self.config: TD3Config = config

        # Дополнительная валидация для TD3 (только непрерывные действия)
        self._validate_continuous_action_space()

        # Создание векторизованной среды с нормализацией
        self.vec_env = self._create_vectorized_env()

        # Создание шума для действий
        self.action_noise = self._create_action_noise()

        # Создание модели TD3
        self.model = self._create_model()

        # Инициализация колбэков
        self.callbacks: List[BaseCallback] = []
        self._setup_callbacks()

        logger.info(
            "TD3 агент инициализирован",
            extra={
                "learning_rate": config.learning_rate,
                "buffer_size": config.buffer_size,
                "batch_size": config.batch_size,
                "action_noise_type": config.action_noise_type,
                "normalize_env": config.normalize_env,
            },
        )

    def _validate_continuous_action_space(self) -> None:
        """Проверить, что пространство действий непрерывное.

        Raises:
            ValueError: Если пространство действий не является Box (непрерывным)
        """
        if self.env is None:
            raise ValueError("Среда не инициализирована")

        action_space = self.env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError(
                f"TD3 поддерживает только непрерывные пространства действий (Box). "
                f"Получено: {type(action_space).__name__}. "
                f"Для дискретных действий используйте PPO, A2C или DQN."
            )

        logger.debug(
            "Валидация пространства действий пройдена",
            extra={
                "action_space": str(action_space),
                "action_dim": action_space.shape[0],
                "action_low": action_space.low.tolist(),
                "action_high": action_space.high.tolist(),
            },
        )

    def _create_vectorized_env(self) -> Union[DummyVecEnv, VecNormalize]:
        """Создать векторизованную среду с нормализацией.

        Returns:
            Векторизованная среда с опциональной нормализацией
        """
        # Создание векторизованной среды
        vec_env = make_vec_env(
            self.config.env_name,
            n_envs=1,
            seed=self.config.seed,
            wrapper_class=Monitor,
        )

        # Добавление нормализации если требуется
        if self.config.normalize_env:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=self.config.norm_obs,
                norm_reward=self.config.norm_reward,
                clip_obs=self.config.clip_obs,
                gamma=self.config.gamma,
            )

            logger.debug(
                "Добавлена нормализация среды",
                extra={
                    "norm_obs": self.config.norm_obs,
                    "norm_reward": self.config.norm_reward,
                    "clip_obs": self.config.clip_obs,
                },
            )

        return vec_env

    def _create_action_noise(self) -> Optional[ActionNoise]:
        """Создать шум для действий.

        Returns:
            Объект шума для действий или None если шум не используется
        """
        if self.config.action_noise_type == "none":
            return None

        action_dim = self.env.action_space.shape[0]

        if self.config.action_noise_type == "normal":
            noise = NormalActionNoise(
                mean=np.full(action_dim, self.config.action_noise_mean),
                sigma=np.full(action_dim, self.config.action_noise_std),
            )
            logger.debug(
                "Создан Gaussian шум для действий",
                extra={
                    "mean": self.config.action_noise_mean,
                    "std": self.config.action_noise_std,
                    "action_dim": action_dim,
                },
            )

        elif self.config.action_noise_type == "ornstein_uhlenbeck":
            noise = OrnsteinUhlenbeckActionNoise(
                mean=np.full(action_dim, self.config.action_noise_mean),
                sigma=np.full(action_dim, self.config.ou_sigma),
                theta=self.config.ou_theta,
                dt=self.config.ou_dt,
            )
            logger.debug(
                "Создан Ornstein-Uhlenbeck шум для действий",
                extra={
                    "mean": self.config.action_noise_mean,
                    "sigma": self.config.ou_sigma,
                    "theta": self.config.ou_theta,
                    "dt": self.config.ou_dt,
                    "action_dim": action_dim,
                },
            )

        else:
            raise ValueError(
                f"Неподдерживаемый тип шума: {self.config.action_noise_type}"
            )

        return noise

    def _create_learning_rate_schedule(self) -> Union[float, Callable[[float], float]]:
        """Создать расписание learning rate.

        Returns:
            Функция расписания или константное значение
        """
        if not self.config.use_lr_schedule:
            return self.config.learning_rate

        if self.config.lr_schedule_type == "linear":
            return LinearSchedule(
                start=self.config.learning_rate,
                end=self.config.learning_rate * self.config.lr_final_ratio,
                end_fraction=1.0,
            )
        elif self.config.lr_schedule_type == "exponential":

            def exponential_schedule(progress_remaining: float) -> float:
                """Экспоненциальное убывание learning rate."""
                return self.config.learning_rate * (
                    self.config.lr_final_ratio ** (1 - progress_remaining)
                )

            return exponential_schedule
        else:
            logger.warning(
                f"Неизвестный тип расписания: {self.config.lr_schedule_type}. "
                f"Используется константный learning rate."
            )
            return self.config.learning_rate

    def _create_model(self) -> TD3:
        """Создать модель TD3.

        Returns:
            Инициализированная модель TD3

        Raises:
            RuntimeError: При ошибке создания модели
        """
        try:
            # Подготовка learning rate
            learning_rate = self._create_learning_rate_schedule()

            # Создание модели TD3
            model = TD3(
                policy=self.config.policy,
                env=self.vec_env,
                learning_rate=learning_rate,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                batch_size=self.config.batch_size,
                tau=self.config.tau,
                gamma=self.config.gamma,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                action_noise=self.action_noise,
                policy_delay=self.config.policy_delay,
                target_policy_noise=self.config.target_policy_noise,
                target_noise_clip=self.config.target_noise_clip,
                tensorboard_log=self.config.tensorboard_log
                if self.config.use_tensorboard
                else None,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
            )

            logger.info("Модель TD3 создана успешно")
            return model

        except Exception as e:
            error_msg = f"Ошибка создания модели TD3: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _setup_callbacks(self) -> None:
        """Настроить колбэки для обучения."""
        self.callbacks = []

        # Колбэк для метрик
        metrics_callback = TD3MetricsCallback(
            metrics_tracker=self.metrics_tracker,
            log_freq=self.config.log_interval * 1000,
            verbose=self.config.verbose,
        )
        self.callbacks.append(metrics_callback)

        # Колбэк для ранней остановки
        if self.config.early_stopping:
            early_stopping_callback = TD3EarlyStoppingCallback(
                target_reward=self.config.target_reward,
                patience_episodes=self.config.patience_episodes,
                min_improvement=self.config.min_improvement,
                check_freq=self.config.eval_freq,
                verbose=self.config.verbose,
            )
            self.callbacks.append(early_stopping_callback)

        # Колбэк для оценки
        if self.config.eval_freq > 0:
            eval_callback = EvalCallback(
                eval_env=self.vec_env,
                best_model_save_path=str(
                    Path(self.config.model_save_path).parent / "best_model"
                )
                if self.config.model_save_path
                else None,
                log_path=str(Path(self.config.model_save_path).parent / "evaluations")
                if self.config.model_save_path
                else None,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.config.verbose,
            )
            self.callbacks.append(eval_callback)

        # Колбэк для сохранения чекпоинтов
        if self.config.save_freq > 0 and self.config.model_save_path:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.save_freq,
                save_path=str(Path(self.config.model_save_path).parent / "checkpoints"),
                name_prefix="td3_checkpoint",
                save_replay_buffer=True,  # TD3 использует replay buffer
                save_vecnormalize=self.config.normalize_env,
                verbose=self.config.verbose,
            )
            self.callbacks.append(checkpoint_callback)

        # Колбэк для остановки при достижении целевой награды
        # Добавляется только если есть EvalCallback
        if self.config.target_reward > float("-inf") and self.config.eval_freq > 0:
            reward_threshold_callback = StopTrainingOnRewardThreshold(
                reward_threshold=self.config.target_reward,
                verbose=self.config.verbose,
            )
            # Добавляем как дочерний к EvalCallback
            for callback in self.callbacks:
                if isinstance(callback, EvalCallback):
                    callback.callback_on_new_best = reward_threshold_callback
                    break

        logger.debug(f"Настроено {len(self.callbacks)} колбэков для обучения")

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Обучить TD3 агента.

        Args:
            total_timesteps: Количество шагов обучения (по умолчанию из config)
            callback: Дополнительный колбэк для обучения
            **kwargs: Дополнительные параметры для TD3.learn()

        Returns:
            Результат обучения с метриками и путями к артефактам

        Raises:
            RuntimeError: При ошибке обучения
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")

        timesteps = total_timesteps or self.config.total_timesteps
        start_time = time.time()

        try:
            logger.info(f"Начало обучения TD3 агента на {timesteps} шагов")

            # Подготовка колбэков
            all_callbacks = self.callbacks.copy()
            if callback is not None:
                all_callbacks.append(callback)

            callback_list = CallbackList(all_callbacks) if all_callbacks else None

            # Обучение модели
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback_list,
                **kwargs,
            )

            training_time = time.time() - start_time
            self.is_trained = True

            # Оценка финальной производительности
            final_metrics = self.evaluate(
                n_episodes=self.config.n_eval_episodes,
                deterministic=True,
            )

            # Создание результата обучения
            self.training_result = TrainingResult(
                total_timesteps=timesteps,
                training_time=training_time,
                final_mean_reward=final_metrics["mean_reward"],
                final_std_reward=final_metrics["std_reward"],
                success=True,
            )

            # Сохранение модели если указан путь
            if self.config.model_save_path:
                self.save(self.config.model_save_path)

            logger.info(
                f"Обучение завершено за {training_time:.2f} сек",
                extra={
                    "final_mean_reward": final_metrics["mean_reward"],
                    "final_std_reward": final_metrics["std_reward"],
                    "total_timesteps": timesteps,
                },
            )

            return self.training_result

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Ошибка обучения TD3 агента: {e}"
            logger.error(error_msg)

            self.training_result = TrainingResult(
                total_timesteps=timesteps,
                training_time=training_time,
                final_mean_reward=float("-inf"),
                final_std_reward=0.0,
                success=False,
                error_message=error_msg,
            )

            raise RuntimeError(error_msg) from e

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
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите train().")

        try:
            # Нормализация наблюдения если используется VecNormalize
            if isinstance(self.vec_env, VecNormalize):
                observation = self.vec_env.normalize_obs(observation.reshape(1, -1))
                observation = observation.flatten()

            action, state = self.model.predict(
                observation,
                deterministic=deterministic,
                **kwargs,
            )

            return action, state

        except Exception as e:
            error_msg = f"Ошибка предсказания действия: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def save(self, path: str) -> None:
        """Сохранить TD3 агента.

        Args:
            path: Путь для сохранения модели

        Raises:
            RuntimeError: Если модель не инициализирована
            OSError: При ошибке записи файла
        """
        super().save(path)

        # Дополнительное сохранение VecNormalize если используется
        if isinstance(self.vec_env, VecNormalize):
            vec_normalize_path = Path(path).with_suffix(".pkl")
            try:
                self.vec_env.save(str(vec_normalize_path))
                logger.info(f"VecNormalize сохранен: {vec_normalize_path}")
            except Exception as e:
                logger.warning(f"Ошибка сохранения VecNormalize: {e}")

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[gym.Env] = None,
        config: Optional[TD3Config] = None,
        **kwargs: Any,
    ) -> "TD3Agent":
        """Загрузить TD3 агента из файла.

        Args:
            path: Путь к сохраненной модели
            env: Среда (создается автоматически если None)
            config: Конфигурация агента (загружается из файла если None)
            **kwargs: Дополнительные параметры

        Returns:
            Загруженный TD3 агент

        Raises:
            FileNotFoundError: Если файл модели не найден
            RuntimeError: При ошибке загрузки
        """
        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        try:
            # Загрузка конфигурации если не предоставлена
            if config is None:
                config_path = model_path.with_suffix(".yaml")
                if config_path.exists():
                    import yaml

                    with open(config_path, "r", encoding="utf-8") as f:
                        config_dict = yaml.safe_load(f)
                    config = TD3Config(**config_dict)
                else:
                    logger.warning(
                        f"Файл конфигурации не найден: {config_path}. "
                        f"Используется конфигурация по умолчанию."
                    )
                    config = TD3Config()

            # Создание агента
            agent = cls(config=config, env=env, **kwargs)

            # Загрузка модели
            agent.model = TD3.load(str(model_path), env=agent.vec_env)
            agent.is_trained = True

            # Загрузка VecNormalize если существует
            vec_normalize_path = model_path.with_suffix(".pkl")
            if vec_normalize_path.exists() and isinstance(agent.vec_env, VecNormalize):
                try:
                    agent.vec_env = VecNormalize.load(
                        str(vec_normalize_path), agent.vec_env
                    )
                    logger.info(f"VecNormalize загружен: {vec_normalize_path}")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки VecNormalize: {e}")

            logger.info(f"TD3 агент загружен: {path}")
            return agent

        except Exception as e:
            error_msg = f"Ошибка загрузки TD3 агента из {path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о TD3 модели.

        Returns:
            Словарь с информацией о модели и конфигурации
        """
        info = super().get_model_info()

        # Добавление TD3-специфичной информации
        td3_info = {
            "buffer_size": self.config.buffer_size,
            "learning_starts": self.config.learning_starts,
            "batch_size": self.config.batch_size,
            "tau": self.config.tau,
            "policy_delay": self.config.policy_delay,
            "target_policy_noise": self.config.target_policy_noise,
            "target_noise_clip": self.config.target_noise_clip,
            "action_noise_type": self.config.action_noise_type,
            "normalize_env": self.config.normalize_env,
            "use_lr_schedule": self.config.use_lr_schedule,
            "early_stopping": self.config.early_stopping,
        }

        info.update(td3_info)

        # Информация о векторизованной среде
        if isinstance(self.vec_env, VecNormalize):
            info["vec_normalize_stats"] = {
                "obs_mean": self.vec_env.obs_rms.mean.tolist()
                if self.vec_env.obs_rms
                else None,
                "obs_var": self.vec_env.obs_rms.var.tolist()
                if self.vec_env.obs_rms
                else None,
            }

        # Информация о шуме действий
        if self.action_noise is not None:
            info["action_noise_info"] = {
                "type": type(self.action_noise).__name__,
                "sigma": getattr(self.action_noise, "sigma", None),
                "mean": getattr(self.action_noise, "mean", None),
            }

        return info

    def reset_model(self) -> None:
        """Сбросить TD3 модель для переобучения."""
        super().reset_model()

        # Пересоздание векторизованной среды
        if hasattr(self, "vec_env"):
            self.vec_env.close()
        self.vec_env = self._create_vectorized_env()

        # Пересоздание шума действий
        self.action_noise = self._create_action_noise()

        # Пересоздание модели
        self.model = self._create_model()

        # Переустановка колбэков
        self._setup_callbacks()

        logger.info("TD3 модель сброшена для переобучения")

    def update_noise_schedule(self, progress: float) -> None:
        """Обновить параметры шума согласно расписанию.

        Args:
            progress: Прогресс обучения (0.0 - 1.0)
        """
        if not self.config.use_noise_schedule or self.action_noise is None:
            return

        # Вычисление нового значения шума
        if self.config.noise_schedule_type == "linear":
            noise_factor = 1.0 - progress * (1.0 - self.config.noise_final_ratio)
        elif self.config.noise_schedule_type == "exponential":
            noise_factor = self.config.noise_final_ratio**progress
        else:
            return

        # Обновление параметров шума
        if isinstance(self.action_noise, NormalActionNoise):
            original_sigma = self.config.action_noise_std
            self.action_noise.sigma = np.full_like(
                self.action_noise.sigma, original_sigma * noise_factor
            )
        elif isinstance(self.action_noise, OrnsteinUhlenbeckActionNoise):
            original_sigma = self.config.ou_sigma
            self.action_noise.sigma = original_sigma * noise_factor

        logger.debug(f"Обновлен шум действий: фактор = {noise_factor:.3f}")

    def __del__(self) -> None:
        """Очистка ресурсов при удалении объекта."""
        try:
            if hasattr(self, "vec_env") and self.vec_env is not None:
                self.vec_env.close()
        except Exception as e:
            logger.warning(f"Ошибка при закрытии среды: {e}")
