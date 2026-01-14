"""A2C агент с оптимизированными гиперпараметрами для LunarLander-v3.

Этот модуль реализует агента на основе алгоритма Advantage Actor-Critic (A2C)
с использованием Stable-Baselines3. Включает поддержку дискретных и непрерывных
пространств действий, кастомные колбэки для мониторинга, раннюю остановку,
интеграцию с системой логирования и метрик проекта.

Особенности:
- Оптимизированные гиперпараметры для LunarLander-v3
- Адаптивное расписание learning rate
- Кастомные колбэки для мониторинга и ранней остановки
- Поддержка TensorBoard логирования
- Автоматическое сохранение чекпоинтов
- Комплексная обработка ошибок и валидация
- Интеграция с системой метрик проекта
- Оптимизированная архитектура сети для A2C
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.base import Agent, AgentConfig, TrainingResult
from src.utils import (
    MetricsTracker,
    set_seed,
    get_experiment_logger,
    get_metrics_tracker,
)

logger = logging.getLogger(__name__)


@dataclass
class A2CConfig(AgentConfig):
    """Конфигурация для A2C агента с оптимизированными параметрами.

    Расширяет базовую конфигурацию агента специфичными для A2C параметрами,
    оптимизированными для среды LunarLander-v3.
    """

    # Переопределение базовых параметров для A2C
    algorithm: str = "A2C"

    # Оптимизированные гиперпараметры для LunarLander-v3
    learning_rate: float = 7e-4
    n_steps: int = 5  # A2C использует меньше шагов чем PPO
    gamma: float = 0.99
    gae_lambda: float = 1.0  # A2C обычно использует 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.25  # Меньший коэффициент для value function
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5  # Специфично для A2C
    use_rms_prop: bool = True  # A2C обычно использует RMSprop

    # Расписание learning rate
    use_lr_schedule: bool = True
    lr_schedule_type: str = "linear"  # "linear", "constant", "exponential"
    lr_final_ratio: float = 0.1

    # Параметры сети (оптимизированы для A2C)
    net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None
    activation_fn: str = "tanh"  # "tanh", "relu", "elu"
    ortho_init: bool = True

    # Нормализация
    normalize_env: bool = True
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Ранняя остановка
    early_stopping: bool = True
    target_reward: float = 200.0  # Для LunarLander-v3
    patience_episodes: int = 100  # Больше терпения для A2C
    min_improvement: float = 2.0  # Меньшее улучшение для A2C

    # Мониторинг и логирование
    use_tensorboard: bool = True
    log_std_init: float = 0.0
    use_sde: bool = False
    sde_sample_freq: int = -1

    # Дополнительные параметры для оптимизации A2C
    stats_window_size: int = 100
    normalize_advantage: bool = False  # A2C обычно не нормализует advantage

    def __post_init__(self) -> None:
        """Валидация и установка значений по умолчанию."""
        super().__post_init__()

        # Установка архитектуры сети по умолчанию для A2C
        if self.net_arch is None:
            # A2C лучше работает с более простой архитектурой
            self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        # Валидация параметров A2C
        if self.n_steps <= 0:
            raise ValueError(f"n_steps должен быть > 0, получен: {self.n_steps}")

        if self.rms_prop_eps <= 0:
            raise ValueError(
                f"rms_prop_eps должен быть > 0, получен: {self.rms_prop_eps}"
            )

        # Установка policy_kwargs
        activation_functions = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
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
                "ortho_init": self.ortho_init,
                "log_std_init": self.log_std_init,
            }
        )


class A2CMetricsCallback(BaseCallback):
    """Колбэк для отслеживания метрик обучения A2C агента.

    Интегрируется с системой метрик проекта для записи детальной
    информации о процессе обучения.
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

        # Добавление метрик обучения если доступны
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key in ["train/policy_loss", "train/value_loss", "train/entropy_loss"]:
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
                metadata={"phase": "training", "algorithm": "A2C"},
            )

        if self.verbose > 0:
            logger.info(
                f"Шаг {self.num_timesteps}: средняя награда = {metrics['mean_reward']:.2f} "
                f"± {metrics['std_reward']:.2f}"
            )


class A2CEarlyStoppingCallback(BaseCallback):
    """Колбэк для ранней остановки обучения A2C при достижении целевой производительности."""

    def __init__(
        self,
        target_reward: float,
        patience_episodes: int = 100,
        min_improvement: float = 2.0,
        check_freq: int = 5000,  # Чаще проверяем для A2C
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


class A2CAgent(Agent):
    """A2C агент с оптимизированными гиперпараметрами для LunarLander-v3.

    Реализует алгоритм Advantage Actor-Critic с поддержкой:
    - Дискретных и непрерывных пространств действий
    - Адаптивного расписания learning rate
    - Нормализации наблюдений и наград
    - Ранней остановки обучения
    - Детального мониторинга и логирования
    - Автоматического сохранения чекпоинтов
    - Оптимизированной архитектуры сети для A2C
    """

    def __init__(
        self,
        config: A2CConfig,
        env: Optional[gym.Env] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Инициализировать A2C агента.

        Args:
            config: Конфигурация A2C агента
            env: Среда Gymnasium (создается автоматически если None)
            experiment_name: Имя эксперимента для логирования

        Raises:
            ValueError: При некорректной конфигурации
            RuntimeError: При ошибке инициализации
        """
        # Валидация типа конфигурации
        if not isinstance(config, A2CConfig):
            raise ValueError(f"Ожидается A2CConfig, получен {type(config).__name__}")

        super().__init__(config, env, experiment_name)

        # Приведение типа для удобства
        self.config: A2CConfig = config

        # Создание векторизованной среды с нормализацией
        self.vec_env = self._create_vectorized_env()

        # Создание модели A2C
        self.model = self._create_model()

        # Инициализация колбэков
        self.callbacks: List[BaseCallback] = []
        self._setup_callbacks()

        logger.info(
            "A2C агент инициализирован",
            extra={
                "learning_rate": config.learning_rate,
                "n_steps": config.n_steps,
                "use_rms_prop": config.use_rms_prop,
                "use_lr_schedule": config.use_lr_schedule,
                "normalize_env": config.normalize_env,
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
                clip_reward=self.config.clip_reward,
                gamma=self.config.gamma,
            )

            logger.debug(
                "Добавлена нормализация среды",
                extra={
                    "norm_obs": self.config.norm_obs,
                    "norm_reward": self.config.norm_reward,
                    "clip_obs": self.config.clip_obs,
                    "clip_reward": self.config.clip_reward,
                },
            )

        return vec_env

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

    def _create_model(self) -> A2C:
        """Создать модель A2C.

        Returns:
            Инициализированная модель A2C

        Raises:
            RuntimeError: При ошибке создания модели
        """
        try:
            # Подготовка learning rate
            learning_rate = self._create_learning_rate_schedule()

            # Создание модели A2C
            model = A2C(
                policy=self.config.policy,
                env=self.vec_env,
                learning_rate=learning_rate,
                n_steps=self.config.n_steps,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                rms_prop_eps=self.config.rms_prop_eps,
                use_rms_prop=self.config.use_rms_prop,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                normalize_advantage=self.config.normalize_advantage,
                tensorboard_log=self.config.tensorboard_log
                if self.config.use_tensorboard
                else None,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
            )

            logger.info("Модель A2C создана успешно")
            return model

        except Exception as e:
            error_msg = f"Ошибка создания модели A2C: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _setup_callbacks(self) -> None:
        """Настроить колбэки для обучения."""
        self.callbacks = []

        # Колбэк для метрик
        metrics_callback = A2CMetricsCallback(
            metrics_tracker=self.metrics_tracker,
            log_freq=self.config.log_interval * 1000,
            verbose=self.config.verbose,
        )
        self.callbacks.append(metrics_callback)

        # Колбэк для ранней остановки
        if self.config.early_stopping:
            early_stopping_callback = A2CEarlyStoppingCallback(
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
                name_prefix="a2c_checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=self.config.normalize_env,
                verbose=self.config.verbose,
            )
            self.callbacks.append(checkpoint_callback)

        # Колбэк для остановки при достижении целевой награды (только если есть EvalCallback)
        if self.config.target_reward > float("-inf") and self.config.eval_freq > 0:
            reward_threshold_callback = StopTrainingOnRewardThreshold(
                reward_threshold=self.config.target_reward,
                verbose=self.config.verbose,
            )
            self.callbacks.append(reward_threshold_callback)

        logger.debug(f"Настроено {len(self.callbacks)} колбэков для обучения")

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Обучить A2C агента.

        Args:
            total_timesteps: Количество шагов обучения (по умолчанию из config)
            callback: Дополнительный колбэк для обучения
            **kwargs: Дополнительные параметры для A2C.learn()

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
            logger.info(f"Начало обучения A2C агента на {timesteps} шагов")

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
            error_msg = f"Ошибка обучения A2C агента: {e}"
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
        """Сохранить A2C агента.

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
        config: Optional[A2CConfig] = None,
        **kwargs: Any,
    ) -> "A2CAgent":
        """Загрузить A2C агента из файла.

        Args:
            path: Путь к сохраненной модели
            env: Среда (создается автоматически если None)
            config: Конфигурация агента (загружается из файла если None)
            **kwargs: Дополнительные параметры

        Returns:
            Загруженный A2C агент

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
                    config = A2CConfig(**config_dict)
                else:
                    logger.warning(
                        f"Файл конфигурации не найден: {config_path}. "
                        f"Используется конфигурация по умолчанию."
                    )
                    config = A2CConfig()

            # Создание агента
            agent = cls(config=config, env=env, **kwargs)

            # Загрузка модели
            agent.model = A2C.load(str(model_path), env=agent.vec_env)
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

            logger.info(f"A2C агент загружен: {path}")
            return agent

        except Exception as e:
            error_msg = f"Ошибка загрузки A2C агента из {path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о A2C модели.

        Returns:
            Словарь с информацией о модели и конфигурации
        """
        info = super().get_model_info()

        # Добавление A2C-специфичной информации
        a2c_info = {
            "n_steps": self.config.n_steps,
            "ent_coef": self.config.ent_coef,
            "vf_coef": self.config.vf_coef,
            "rms_prop_eps": self.config.rms_prop_eps,
            "use_rms_prop": self.config.use_rms_prop,
            "normalize_env": self.config.normalize_env,
            "use_lr_schedule": self.config.use_lr_schedule,
            "early_stopping": self.config.early_stopping,
            "normalize_advantage": self.config.normalize_advantage,
        }

        info.update(a2c_info)

        # Информация о векторизованной среде
        if isinstance(self.vec_env, VecNormalize):
            info["vec_normalize_stats"] = {
                "obs_mean": self.vec_env.obs_rms.mean.tolist()
                if self.vec_env.obs_rms
                else None,
                "obs_var": self.vec_env.obs_rms.var.tolist()
                if self.vec_env.obs_rms
                else None,
                "ret_mean": float(self.vec_env.ret_rms.mean)
                if self.vec_env.ret_rms
                else None,
                "ret_var": float(self.vec_env.ret_rms.var)
                if self.vec_env.ret_rms
                else None,
            }

        return info

    def reset_model(self) -> None:
        """Сбросить A2C модель для переобучения."""
        super().reset_model()

        # Пересоздание векторизованной среды
        if hasattr(self, "vec_env"):
            self.vec_env.close()
        self.vec_env = self._create_vectorized_env()

        # Пересоздание модели
        self.model = self._create_model()

        # Переустановка колбэков
        self._setup_callbacks()

        logger.info("A2C модель сброшена для переобучения")

    def __del__(self) -> None:
        """Очистка ресурсов при удалении объекта."""
        try:
            if hasattr(self, "vec_env") and self.vec_env is not None:
                self.vec_env.close()
        except Exception as e:
            logger.warning(f"Ошибка при закрытии среды: {e}")