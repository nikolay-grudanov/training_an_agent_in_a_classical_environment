"""Комплексная система обучения RL агентов.

Этот модуль предоставляет высокоуровневый оркестратор обучения, который
координирует все компоненты системы: агентов, среды, конфигурацию,
логирование, метрики и чекпоинты.

Поддерживаемые алгоритмы:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic) 
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)

Основные возможности:
- Конфигурационное обучение через YAML/Hydra
- Восстановление прерванных сессий
- Комплексная система мониторинга
- Интеграция с экспериментальным трекингом
- Автоматическое управление чекпоинтами
- Поддержка различных режимов обучения
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.agents import (
    Agent,
    AgentConfig,
    TrainingResult as AgentTrainingResult,
    PPOAgent,
    A2CAgent,
    SACAgent,
    TD3Agent,
)
from src.environments import EnvironmentWrapper, LunarLanderEnvironment
from src.utils import (
    CheckpointManager,
    MetricsTracker,
    get_experiment_logger,
    get_metrics_tracker,
    set_seed,
    RLConfig,
    load_config,
)

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Режимы обучения."""
    
    TRAIN = "train"           # Обучение с нуля
    RESUME = "resume"         # Продолжение обучения
    EVALUATE = "evaluate"     # Только оценка
    FINETUNE = "finetune"     # Дообучение


@dataclass
class TrainerConfig:
    """Конфигурация системы обучения.
    
    Содержит все параметры, необходимые для координации обучения,
    включая настройки агента, среды, мониторинга и сохранения.
    """
    
    # Основные параметры
    experiment_name: str = "default_experiment"
    algorithm: str = "PPO"
    environment_name: str = "LunarLander-v3"
    mode: TrainingMode = TrainingMode.TRAIN
    
    # Параметры обучения
    total_timesteps: int = 100_000
    seed: int = 42
    n_envs: int = 1  # Количество параллельных сред
    
    # Конфигурация агента
    agent_config: Optional[AgentConfig] = None
    
    # Настройки среды
    env_config: Optional[Dict[str, Any]] = None
    use_lunar_lander_wrapper: bool = True
    render_mode: Optional[str] = None
    
    # Мониторинг и оценка
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Сохранение и чекпоинты
    save_freq: int = 50_000
    checkpoint_freq: int = 25_000
    max_checkpoints: int = 5
    
    # Пути
    output_dir: str = "results"
    model_save_path: Optional[str] = None
    logs_dir: Optional[str] = None
    tensorboard_log: Optional[str] = None
    
    # Восстановление
    resume_from_checkpoint: Optional[str] = None
    resume_timestep: Optional[int] = None
    
    # Раннее остановка
    early_stopping: bool = False
    patience: int = 5
    min_improvement: float = 0.01
    
    # Дополнительные настройки
    verbose: int = 1
    log_interval: int = 1000
    progress_bar: bool = True
    
    # Интеграция с экспериментами
    track_experiment: bool = True
    experiment_tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Валидация и нормализация конфигурации."""
        # Валидация алгоритма
        supported_algorithms = ["PPO", "A2C", "SAC", "TD3"]
        if self.algorithm.upper() not in supported_algorithms:
            raise ValueError(
                f"Неподдерживаемый алгоритм: {self.algorithm}. "
                f"Поддерживаемые: {supported_algorithms}"
            )
        
        # Нормализация алгоритма
        self.algorithm = self.algorithm.upper()
        
        # Валидация параметров
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps должен быть > 0: {self.total_timesteps}")
        
        if self.eval_freq <= 0:
            raise ValueError(f"eval_freq должен быть > 0: {self.eval_freq}")
        
        if self.n_eval_episodes <= 0:
            raise ValueError(f"n_eval_episodes должен быть > 0: {self.n_eval_episodes}")
        
        # Создание путей
        self._setup_paths()
        
        # Создание конфигурации агента если не задана
        if self.agent_config is None:
            self.agent_config = self._create_default_agent_config()
    
    def _setup_paths(self) -> None:
        """Настроить пути для сохранения."""
        output_path = Path(self.output_dir)
        experiment_path = output_path / self.experiment_name
        
        # Создание директорий
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Настройка путей
        if self.model_save_path is None:
            self.model_save_path = str(experiment_path / "models" / f"{self.algorithm.lower()}_model")
        
        if self.logs_dir is None:
            self.logs_dir = str(experiment_path / "logs")
        
        if self.tensorboard_log is None:
            self.tensorboard_log = str(experiment_path / "tensorboard")
        
        # Создание директорий
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)
    
    def _create_default_agent_config(self) -> AgentConfig:
        """Создать конфигурацию агента по умолчанию."""
        return AgentConfig(
            algorithm=self.algorithm,
            env_name=self.environment_name,
            total_timesteps=self.total_timesteps,
            seed=self.seed,
            model_save_path=self.model_save_path,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
        )
    
    @classmethod
    def from_rl_config(cls, rl_config: RLConfig) -> "TrainerConfig":
        """Создать TrainerConfig из RLConfig.
        
        Args:
            rl_config: Конфигурация RL системы
            
        Returns:
            Конфигурация тренера
        """
        # Создание конфигурации агента
        agent_config = AgentConfig(
            algorithm=rl_config.algorithm.name,
            env_name=rl_config.environment.name,
            total_timesteps=rl_config.training.total_timesteps,
            seed=rl_config.seed,
            learning_rate=rl_config.algorithm.learning_rate,
            batch_size=rl_config.algorithm.batch_size,
            n_steps=rl_config.algorithm.n_steps,
            n_epochs=rl_config.algorithm.n_epochs,
            gamma=rl_config.algorithm.gamma,
            gae_lambda=rl_config.algorithm.gae_lambda,
            clip_range=rl_config.algorithm.clip_range,
            ent_coef=rl_config.algorithm.ent_coef,
            vf_coef=rl_config.algorithm.vf_coef,
            max_grad_norm=rl_config.algorithm.max_grad_norm,
            use_sde=rl_config.algorithm.use_sde,
            sde_sample_freq=rl_config.algorithm.sde_sample_freq,
            target_kl=rl_config.algorithm.target_kl,
            device=rl_config.algorithm.device,
            verbose=rl_config.algorithm.verbose,
            policy_kwargs=rl_config.algorithm.policy_kwargs,
            tensorboard_log=rl_config.algorithm.tensorboard_log,
        )
        
        return cls(
            experiment_name=rl_config.experiment_name,
            algorithm=rl_config.algorithm.name,
            environment_name=rl_config.environment.name,
            total_timesteps=rl_config.training.total_timesteps,
            seed=rl_config.seed,
            agent_config=agent_config,
            eval_freq=rl_config.training.eval_freq,
            n_eval_episodes=rl_config.training.n_eval_episodes,
            save_freq=rl_config.training.save_freq,
            output_dir=rl_config.output_dir,
            early_stopping=rl_config.training.early_stopping,
            patience=rl_config.training.patience,
            min_improvement=rl_config.training.min_delta,
            log_interval=rl_config.training.log_interval,
        )


@dataclass
class TrainingResult:
    """Результат обучения с расширенной информацией."""
    
    # Основные метрики
    success: bool
    total_timesteps: int
    training_time: float
    final_mean_reward: float
    final_std_reward: float
    
    # Результат агента
    agent_result: Optional[AgentTrainingResult] = None
    
    # История обучения
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    evaluation_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Информация о модели
    model_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    
    # Метаданные
    experiment_name: str = ""
    algorithm: str = ""
    environment_name: str = ""
    seed: int = 42
    
    # Ошибки
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Дополнительные метрики
    best_mean_reward: float = float("-inf")
    convergence_timestep: Optional[int] = None
    early_stopped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать результат в словарь."""
        result_dict = {
            "success": self.success,
            "total_timesteps": self.total_timesteps,
            "training_time": self.training_time,
            "final_mean_reward": self.final_mean_reward,
            "final_std_reward": self.final_std_reward,
            "best_mean_reward": self.best_mean_reward,
            "convergence_timestep": self.convergence_timestep,
            "early_stopped": self.early_stopped,
            "experiment_name": self.experiment_name,
            "algorithm": self.algorithm,
            "environment_name": self.environment_name,
            "seed": self.seed,
            "model_path": self.model_path,
            "checkpoint_paths": self.checkpoint_paths,
            "config_path": self.config_path,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
        }
        
        if self.agent_result:
            result_dict["agent_result"] = {
                "total_timesteps": self.agent_result.total_timesteps,
                "training_time": self.agent_result.training_time,
                "final_mean_reward": self.agent_result.final_mean_reward,
                "final_std_reward": self.agent_result.final_std_reward,
                "episode_rewards": self.agent_result.episode_rewards,
                "episode_lengths": self.agent_result.episode_lengths,
                "best_mean_reward": self.agent_result.best_mean_reward,
                "success": self.agent_result.success,
            }
        
        return result_dict
    
    def save(self, path: Union[str, Path]) -> None:
        """Сохранить результат в файл.
        
        Args:
            path: Путь для сохранения
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


class Trainer:
    """Высокоуровневый оркестратор обучения RL агентов.
    
    Координирует все компоненты системы обучения:
    - Создание и настройка агентов
    - Управление средами
    - Мониторинг прогресса
    - Сохранение чекпоинтов
    - Восстановление сессий
    - Оценка производительности
    
    Поддерживает различные режимы работы и интеграцию
    с системами экспериментального трекинга.
    """
    
    # Маппинг алгоритмов на классы агентов
    AGENT_CLASSES: Dict[str, Type[Agent]] = {
        "PPO": PPOAgent,
        "A2C": A2CAgent,
        "SAC": SACAgent,
        "TD3": TD3Agent,
    }
    
    def __init__(self, config: TrainerConfig) -> None:
        """Инициализировать тренер.
        
        Args:
            config: Конфигурация обучения
            
        Raises:
            ValueError: При некорректной конфигурации
            RuntimeError: При ошибке инициализации компонентов
        """
        self.config = config
        self.experiment_name = config.experiment_name
        
        # Установка seed для воспроизводимости
        set_seed(config.seed)
        
        # Инициализация логирования
        self.logger = get_experiment_logger(
            experiment_id=self.experiment_name,
            log_level=logging.INFO if config.verbose > 0 else logging.WARNING,
        )
        
        # Инициализация трекера метрик
        self.metrics_tracker = get_metrics_tracker(
            experiment_id=self.experiment_name
        ) if config.track_experiment else None
        
        # Инициализация менеджера чекпоинтов
        checkpoint_dir = Path(config.output_dir) / config.experiment_name / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
        )
        
        # Состояние обучения
        self.agent: Optional[Agent] = None
        self.env: Optional[gym.Env] = None
        self.training_result: Optional[TrainingResult] = None
        self.current_timestep = 0
        self.best_mean_reward = float("-inf")
        self.patience_counter = 0
        
        # История обучения
        self.training_history: Dict[str, List[float]] = {
            "timesteps": [],
            "mean_rewards": [],
            "std_rewards": [],
            "episode_lengths": [],
        }
        
        self.evaluation_history: Dict[str, List[float]] = {
            "timesteps": [],
            "mean_rewards": [],
            "std_rewards": [],
        }
        
        self.logger.info(
            f"Инициализирован Trainer для эксперимента '{self.experiment_name}'",
            extra={
                "algorithm": config.algorithm,
                "environment": config.environment_name,
                "mode": config.mode.value,
                "total_timesteps": config.total_timesteps,
                "seed": config.seed,
            }
        )
    
    def setup(self) -> None:
        """Настроить все компоненты для обучения.
        
        Raises:
            RuntimeError: При ошибке настройки компонентов
        """
        try:
            # Создание среды
            self._setup_environment()
            
            # Создание агента
            self._setup_agent()
            
            # Восстановление состояния если необходимо
            if self.config.mode == TrainingMode.RESUME:
                self._resume_training()
            
            self.logger.info("Настройка компонентов завершена успешно")
            
        except Exception as e:
            error_msg = f"Ошибка настройки компонентов: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def train(self) -> TrainingResult:
        """Выполнить обучение агента.
        
        Returns:
            Результат обучения с метриками и путями к артефактам
            
        Raises:
            RuntimeError: При ошибке обучения
        """
        if self.config.mode == TrainingMode.EVALUATE:
            return self._evaluate_only()
        
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Начало обучения в режиме {self.config.mode.value}",
                extra={
                    "total_timesteps": self.config.total_timesteps,
                    "current_timestep": self.current_timestep,
                }
            )
            
            # Создание callbacks
            callbacks = self._create_callbacks()
            
            # Обучение агента
            remaining_timesteps = self.config.total_timesteps - self.current_timestep
            
            if remaining_timesteps > 0:
                agent_result = self.agent.train(
                    total_timesteps=remaining_timesteps,
                    callback=callbacks,
                )
            else:
                # Если обучение уже завершено, создаем результат
                agent_result = AgentTrainingResult(
                    total_timesteps=self.current_timestep,
                    training_time=0.0,
                    final_mean_reward=self.best_mean_reward,
                    final_std_reward=0.0,
                    success=True,
                )
            
            training_time = time.time() - start_time
            
            # Финальная оценка
            final_eval = self._evaluate_agent()
            
            # Сохранение финальной модели
            final_model_path = self._save_final_model()
            
            # Создание результата
            self.training_result = TrainingResult(
                success=True,
                total_timesteps=self.config.total_timesteps,
                training_time=training_time,
                final_mean_reward=final_eval["mean_reward"],
                final_std_reward=final_eval["std_reward"],
                agent_result=agent_result,
                training_history=self.training_history,
                evaluation_history=self.evaluation_history,
                model_path=final_model_path,
                checkpoint_paths=self.checkpoint_manager.get_checkpoint_paths(),
                experiment_name=self.experiment_name,
                algorithm=self.config.algorithm,
                environment_name=self.config.environment_name,
                seed=self.config.seed,
                best_mean_reward=self.best_mean_reward,
                early_stopped=self.patience_counter >= self.config.patience,
            )
            
            # Сохранение результата
            self._save_training_result()
            
            self.logger.info(
                "Обучение завершено успешно",
                extra={
                    "training_time": training_time,
                    "final_mean_reward": final_eval["mean_reward"],
                    "best_mean_reward": self.best_mean_reward,
                    "total_timesteps": self.config.total_timesteps,
                }
            )
            
            return self.training_result
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Ошибка во время обучения: {e}"
            self.logger.error(error_msg)
            
            # Создание результата с ошибкой
            self.training_result = TrainingResult(
                success=False,
                total_timesteps=self.current_timestep,
                training_time=training_time,
                final_mean_reward=self.best_mean_reward,
                final_std_reward=0.0,
                experiment_name=self.experiment_name,
                algorithm=self.config.algorithm,
                environment_name=self.config.environment_name,
                seed=self.config.seed,
                error_message=error_msg,
            )
            
            return self.training_result
    
    def evaluate(
        self,
        n_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        render: bool = False,
    ) -> Dict[str, float]:
        """Оценить производительность агента.
        
        Args:
            n_episodes: Количество эпизодов (по умолчанию из config)
            deterministic: Детерминистическая политика (по умолчанию из config)
            render: Отображать среду
            
        Returns:
            Словарь с метриками оценки
            
        Raises:
            RuntimeError: Если агент не обучен
        """
        if self.agent is None:
            raise RuntimeError("Агент не инициализирован. Вызовите setup().")
        
        n_episodes = n_episodes or self.config.n_eval_episodes
        deterministic = deterministic if deterministic is not None else self.config.eval_deterministic
        
        self.logger.info(
            f"Начало оценки агента на {n_episodes} эпизодах",
            extra={"deterministic": deterministic, "render": render}
        )
        
        return self.agent.evaluate(
            n_episodes=n_episodes,
            deterministic=deterministic,
            render=render,
        )
    
    def save_checkpoint(self, timestep: int) -> str:
        """Сохранить чекпоинт.
        
        Args:
            timestep: Текущий шаг обучения
            
        Returns:
            Путь к сохраненному чекпоинту
        """
        checkpoint_data = {
            "timestep": timestep,
            "config": self.config,
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
            "best_mean_reward": self.best_mean_reward,
            "patience_counter": self.patience_counter,
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            timestep=timestep,
        )
        
        # Сохранение модели агента
        if self.agent:
            model_path = Path(checkpoint_path).parent / f"model_{timestep}.zip"
            self.agent.save(str(model_path))
        
        self.logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Загрузить чекпоинт.
        
        Args:
            checkpoint_path: Путь к чекпоинту
            
        Raises:
            FileNotFoundError: Если чекпоинт не найден
            RuntimeError: При ошибке загрузки
        """
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            self.current_timestep = checkpoint_data["timestep"]
            self.training_history = checkpoint_data.get("training_history", {})
            self.evaluation_history = checkpoint_data.get("evaluation_history", {})
            self.best_mean_reward = checkpoint_data.get("best_mean_reward", float("-inf"))
            self.patience_counter = checkpoint_data.get("patience_counter", 0)
            
            # Загрузка модели агента
            model_path = Path(checkpoint_path).parent / f"model_{self.current_timestep}.zip"
            if model_path.exists() and self.agent:
                agent_class = self.AGENT_CLASSES[self.config.algorithm]
                self.agent = agent_class.load(str(model_path), env=self.env)
            
            self.logger.info(
                f"Чекпоинт загружен: {checkpoint_path}",
                extra={"timestep": self.current_timestep}
            )
            
        except Exception as e:
            error_msg = f"Ошибка загрузки чекпоинта {checkpoint_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def cleanup(self) -> None:
        """Очистить ресурсы."""
        if self.env:
            self.env.close()
        
        self.logger.info("Ресурсы очищены")
    
    def _setup_environment(self) -> None:
        """Настроить среду."""
        env_config = self.config.env_config or {}
        
        if self.config.use_lunar_lander_wrapper and "LunarLander" in self.config.environment_name:
            # Использование специализированного wrapper для LunarLander
            self.env = LunarLanderEnvironment(
                render_mode=self.config.render_mode,
                **env_config
            )
        else:
            # Использование универсального wrapper
            self.env = EnvironmentWrapper(
                env_id=self.config.environment_name,
                config=env_config,
                render_mode=self.config.render_mode,
            )
        
        # Установка seed для среды
        if hasattr(self.env, 'seed'):
            self.env.seed(self.config.seed)
        
        self.logger.info(
            f"Среда настроена: {self.config.environment_name}",
            extra={
                "action_space": str(self.env.action_space),
                "observation_space": str(self.env.observation_space),
            }
        )
    
    def _setup_agent(self) -> None:
        """Настроить агента."""
        agent_class = self.AGENT_CLASSES.get(self.config.algorithm)
        if agent_class is None:
            raise ValueError(f"Неподдерживаемый алгоритм: {self.config.algorithm}")
        
        self.agent = agent_class(
            config=self.config.agent_config,
            env=self.env,
            experiment_name=self.experiment_name,
        )
        
        self.logger.info(f"Агент {self.config.algorithm} настроен")
    
    def _resume_training(self) -> None:
        """Восстановить обучение из чекпоинта."""
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        elif self.config.resume_timestep:
            # Поиск ближайшего чекпоинта
            checkpoint_path = self.checkpoint_manager.find_checkpoint(self.config.resume_timestep)
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                self.logger.warning(
                    f"Чекпоинт для timestep {self.config.resume_timestep} не найден"
                )
    
    def _evaluate_only(self) -> TrainingResult:
        """Выполнить только оценку без обучения."""
        if self.agent is None or not self.agent.is_trained:
            raise RuntimeError("Агент не обучен. Невозможно выполнить оценку.")
        
        start_time = time.time()
        
        eval_result = self._evaluate_agent()
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            success=True,
            total_timesteps=0,
            training_time=training_time,
            final_mean_reward=eval_result["mean_reward"],
            final_std_reward=eval_result["std_reward"],
            experiment_name=self.experiment_name,
            algorithm=self.config.algorithm,
            environment_name=self.config.environment_name,
            seed=self.config.seed,
        )
    
    def _evaluate_agent(self) -> Dict[str, float]:
        """Оценить агента."""
        return self.agent.evaluate(
            n_episodes=self.config.n_eval_episodes,
            deterministic=self.config.eval_deterministic,
        )
    
    def _create_callbacks(self) -> CallbackList:
        """Создать callbacks для обучения."""
        callbacks = []
        
        # Callback для оценки
        if self.config.eval_freq > 0:
            eval_callback = self._create_eval_callback()
            callbacks.append(eval_callback)
        
        # Callback для сохранения
        if self.config.save_freq > 0:
            save_callback = self._create_save_callback()
            callbacks.append(save_callback)
        
        # Callback для чекпоинтов
        if self.config.checkpoint_freq > 0:
            checkpoint_callback = self._create_checkpoint_callback()
            callbacks.append(checkpoint_callback)
        
        return CallbackList(callbacks)
    
    def _create_eval_callback(self) -> BaseCallback:
        """Создать callback для оценки."""
        from stable_baselines3.common.callbacks import EvalCallback
        
        return EvalCallback(
            eval_env=self.env,
            n_eval_episodes=self.config.n_eval_episodes,
            eval_freq=self.config.eval_freq,
            deterministic=self.config.eval_deterministic,
            verbose=self.config.verbose,
        )
    
    def _create_save_callback(self) -> BaseCallback:
        """Создать callback для сохранения."""
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        return CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(Path(self.config.model_save_path).parent),
            name_prefix=f"{self.config.algorithm.lower()}_model",
            verbose=self.config.verbose,
        )
    
    def _create_checkpoint_callback(self) -> BaseCallback:
        """Создать callback для чекпоинтов."""
        class TrainerCheckpointCallback(BaseCallback):
            def __init__(self, trainer: "Trainer", freq: int):
                super().__init__()
                self.trainer = trainer
                self.freq = freq
            
            def _on_step(self) -> bool:
                if self.n_calls % self.freq == 0:
                    self.trainer.save_checkpoint(self.n_calls)
                return True
        
        return TrainerCheckpointCallback(self, self.config.checkpoint_freq)
    
    def _save_final_model(self) -> str:
        """Сохранить финальную модель."""
        final_path = f"{self.config.model_save_path}_final"
        self.agent.save(final_path)
        
        # Сохранение конфигурации
        config_path = f"{final_path}_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False, allow_unicode=True)
        
        return final_path
    
    def _save_training_result(self) -> None:
        """Сохранить результат обучения."""
        if self.training_result:
            result_path = Path(self.config.output_dir) / self.experiment_name / "training_result.yaml"
            self.training_result.config_path = str(result_path.parent / "config.yaml")
            self.training_result.save(result_path)
    
    def __enter__(self) -> "Trainer":
        """Контекстный менеджер."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Очистка ресурсов при выходе."""
        self.cleanup()


def create_trainer_from_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    **kwargs: Any,
) -> Trainer:
    """Создать тренер из конфигурационного файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        config_name: Имя конфигурации в директории configs/
        overrides: Список переопределений конфигурации
        **kwargs: Дополнительные параметры для TrainerConfig
        
    Returns:
        Настроенный тренер
        
    Raises:
        FileNotFoundError: Если конфигурация не найдена
        ValueError: При некорректной конфигурации
    """
    # Загрузка RLConfig
    rl_config = load_config(
        config_name=config_name,
        config_path=config_path,
        overrides=overrides,
    )
    
    # Преобразование в TrainerConfig
    trainer_config = TrainerConfig.from_rl_config(rl_config)
    
    # Применение дополнительных параметров
    for key, value in kwargs.items():
        if hasattr(trainer_config, key):
            setattr(trainer_config, key, value)
    
    return Trainer(trainer_config)


def main() -> None:
    """Главная функция для запуска из командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение RL агентов")
    parser.add_argument("--config", type=str, help="Путь к файлу конфигурации")
    parser.add_argument("--config-name", type=str, help="Имя конфигурации")
    parser.add_argument("--algorithm", type=str, choices=["PPO", "A2C", "SAC", "TD3"], 
                       help="Алгоритм обучения")
    parser.add_argument("--env", type=str, help="Название среды")
    parser.add_argument("--timesteps", type=int, help="Количество шагов обучения")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--mode", type=str, choices=["train", "resume", "evaluate"], 
                       default="train", help="Режим работы")
    parser.add_argument("--resume-from", type=str, help="Путь к чекпоинту для восстановления")
    parser.add_argument("--output-dir", type=str, default="results", help="Директория для результатов")
    parser.add_argument("--experiment-name", type=str, help="Имя эксперимента")
    
    args = parser.parse_args()
    
    # Подготовка переопределений
    overrides = []
    if args.algorithm:
        overrides.append(f"algorithm.name={args.algorithm}")
    if args.env:
        overrides.append(f"environment.name={args.env}")
    if args.timesteps:
        overrides.append(f"training.total_timesteps={args.timesteps}")
    if args.seed:
        overrides.append(f"seed={args.seed}")
    if args.experiment_name:
        overrides.append(f"experiment_name={args.experiment_name}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    
    # Создание тренера
    trainer = create_trainer_from_config(
        config_path=args.config,
        config_name=args.config_name,
        overrides=overrides,
        mode=TrainingMode(args.mode),
        resume_from_checkpoint=args.resume_from,
    )
    
    # Обучение
    with trainer:
        result = trainer.train()
        
        if result.success:
            print(f"✅ Обучение завершено успешно!")
            print(f"📊 Финальная награда: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
            print(f"⏱️  Время обучения: {result.training_time:.1f} сек")
            print(f"💾 Модель сохранена: {result.model_path}")
        else:
            print(f"❌ Обучение завершилось с ошибкой: {result.error_message}")


if __name__ == "__main__":
    main()