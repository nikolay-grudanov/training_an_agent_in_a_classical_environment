"""Детальная реализация тренировочного цикла для RL агентов.

Этот модуль предоставляет низкоуровневый тренировочный цикл с поддержкой
различных стратегий обучения, мониторинга прогресса в реальном времени,
управления ресурсами и интеграции с системами логирования.

Основные возможности:
- Эпизодическое и временное обучение
- Мониторинг прогресса в реальном времени
- Обработка прерываний и graceful shutdown
- Управление памятью и ресурсами
- Интеграция с TensorBoard и метриками
- Поддержка пользовательских callbacks
- Детальная статистика обучения
"""

import gc
import logging
import psutil
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import torch
from stable_baselines3.common.type_aliases import GymEnv
from torch.utils.tensorboard import SummaryWriter

from src.agents import Agent
from src.utils import (
    CheckpointManager,
    get_experiment_logger,
    get_metrics_tracker,
)

logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """Стратегии обучения."""

    EPISODIC = "episodic"  # Обучение по эпизодам
    TIMESTEP_BASED = "timestep"  # Обучение по временным шагам
    MIXED = "mixed"  # Смешанная стратегия
    ADAPTIVE = "adaptive"  # Адаптивная стратегия


class TrainingState(Enum):
    """Состояния тренировочного цикла."""

    IDLE = "idle"  # Ожидание
    INITIALIZING = "initializing"  # Инициализация
    TRAINING = "training"  # Обучение
    EVALUATING = "evaluating"  # Оценка
    SAVING = "saving"  # Сохранение
    PAUSED = "paused"  # Приостановлено
    INTERRUPTED = "interrupted"  # Прервано
    COMPLETED = "completed"  # Завершено
    ERROR = "error"  # Ошибка


@dataclass
class TrainingProgress:
    """Прогресс обучения с детальной информацией."""

    # Основные счетчики
    current_timestep: int = 0
    current_episode: int = 0
    total_timesteps: int = 0
    total_episodes: int = 0

    # Временные метрики
    start_time: float = 0.0
    elapsed_time: float = 0.0
    estimated_time_remaining: float = 0.0

    # Производительность
    steps_per_second: float = 0.0
    episodes_per_minute: float = 0.0

    # Награды и длины эпизодов
    current_episode_reward: float = 0.0
    current_episode_length: int = 0
    recent_episode_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))

    # Статистика
    mean_episode_reward: float = 0.0
    std_episode_reward: float = 0.0
    best_episode_reward: float = float("-inf")
    worst_episode_reward: float = float("inf")

    # Состояние
    state: TrainingState = TrainingState.IDLE
    last_checkpoint_timestep: int = 0
    last_evaluation_timestep: int = 0

    # Ресурсы
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0

    def update_episode_stats(self) -> None:
        """Обновить статистику эпизодов."""
        if self.recent_episode_rewards:
            rewards = list(self.recent_episode_rewards)
            self.mean_episode_reward = float(np.mean(rewards))
            self.std_episode_reward = float(np.std(rewards))
            self.best_episode_reward = max(self.best_episode_reward, max(rewards))
            self.worst_episode_reward = min(self.worst_episode_reward, min(rewards))

    def update_performance_stats(self, current_time: float) -> None:
        """Обновить статистику производительности."""
        self.elapsed_time = current_time - self.start_time

        if self.elapsed_time > 0:
            self.steps_per_second = self.current_timestep / self.elapsed_time
            self.episodes_per_minute = (self.current_episode * 60.0) / self.elapsed_time

            # Оценка оставшегося времени
            if self.current_timestep > 0 and self.total_timesteps > 0:
                progress_ratio = self.current_timestep / self.total_timesteps
                if progress_ratio > 0:
                    self.estimated_time_remaining = (
                        self.elapsed_time * (1 - progress_ratio) / progress_ratio
                    )

    def update_resource_usage(self) -> None:
        """Обновить информацию об использовании ресурсов."""
        try:
            # CPU и память
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.cpu_usage_percent = process.cpu_percent()

            # GPU память (если доступна)
            if torch.cuda.is_available():
                self.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception as e:
            logger.debug(f"Ошибка при получении информации о ресурсах: {e}")

    def get_progress_percentage(self) -> float:
        """Получить процент выполнения."""
        if self.total_timesteps > 0:
            return (self.current_timestep / self.total_timesteps) * 100.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "current_timestep": self.current_timestep,
            "current_episode": self.current_episode,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "elapsed_time": self.elapsed_time,
            "estimated_time_remaining": self.estimated_time_remaining,
            "steps_per_second": self.steps_per_second,
            "episodes_per_minute": self.episodes_per_minute,
            "mean_episode_reward": self.mean_episode_reward,
            "std_episode_reward": self.std_episode_reward,
            "best_episode_reward": self.best_episode_reward,
            "state": self.state.value,
            "progress_percentage": self.get_progress_percentage(),
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


@dataclass
class TrainingStatistics:
    """Детальная статистика обучения."""

    # Основные метрики
    total_training_time: float = 0.0
    total_timesteps_completed: int = 0
    total_episodes_completed: int = 0

    # Награды
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    timestep_rewards: List[float] = field(default_factory=list)

    # Статистика по эпизодам
    best_episode_reward: float = float("-inf")
    worst_episode_reward: float = float("inf")
    mean_episode_reward: float = 0.0
    std_episode_reward: float = 0.0
    median_episode_reward: float = 0.0

    # Статистика по длинам эпизодов
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0
    min_episode_length: int = float("inf")
    max_episode_length: int = 0

    # Производительность
    average_steps_per_second: float = 0.0
    average_episodes_per_minute: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_cpu_usage: float = 0.0

    # События
    num_checkpoints_saved: int = 0
    num_evaluations_performed: int = 0
    num_interruptions: int = 0

    # Сходимость
    convergence_timestep: Optional[int] = None
    convergence_episode: Optional[int] = None
    convergence_threshold: Optional[float] = None

    def update_from_progress(self, progress: TrainingProgress) -> None:
        """Обновить статистику из прогресса."""
        self.total_training_time = progress.elapsed_time
        self.total_timesteps_completed = progress.current_timestep
        self.total_episodes_completed = progress.current_episode

        if progress.recent_episode_rewards:
            self.episode_rewards = list(progress.recent_episode_rewards)
            self.mean_episode_reward = progress.mean_episode_reward
            self.std_episode_reward = progress.std_episode_reward
            self.best_episode_reward = progress.best_episode_reward
            self.worst_episode_reward = progress.worst_episode_reward

            if self.episode_rewards:
                self.median_episode_reward = float(np.median(self.episode_rewards))

        if progress.recent_episode_lengths:
            self.episode_lengths = list(progress.recent_episode_lengths)
            if self.episode_lengths:
                self.mean_episode_length = float(np.mean(self.episode_lengths))
                self.std_episode_length = float(np.std(self.episode_lengths))
                self.min_episode_length = min(self.episode_lengths)
                self.max_episode_length = max(self.episode_lengths)

        self.average_steps_per_second = progress.steps_per_second
        self.average_episodes_per_minute = progress.episodes_per_minute
        self.peak_memory_usage_mb = max(
            self.peak_memory_usage_mb, progress.memory_usage_mb
        )
        self.average_cpu_usage = progress.cpu_usage_percent

    def detect_convergence(
        self, threshold: float, window_size: int = 50, stability_episodes: int = 20
    ) -> bool:
        """Определить сходимость обучения."""
        if len(self.episode_rewards) < window_size:
            return False

        # Проверяем последние эпизоды
        recent_rewards = self.episode_rewards[-window_size:]
        mean_recent = np.mean(recent_rewards)

        if mean_recent >= threshold:
            # Проверяем стабильность
            last_rewards = self.episode_rewards[-stability_episodes:]
            if len(last_rewards) >= stability_episodes:
                stable = all(r >= threshold * 0.9 for r in last_rewards)
                if stable and self.convergence_timestep is None:
                    self.convergence_timestep = self.total_timesteps_completed
                    self.convergence_episode = self.total_episodes_completed
                    self.convergence_threshold = threshold
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "total_training_time": self.total_training_time,
            "total_timesteps_completed": self.total_timesteps_completed,
            "total_episodes_completed": self.total_episodes_completed,
            "best_episode_reward": self.best_episode_reward,
            "worst_episode_reward": self.worst_episode_reward,
            "mean_episode_reward": self.mean_episode_reward,
            "std_episode_reward": self.std_episode_reward,
            "median_episode_reward": self.median_episode_reward,
            "mean_episode_length": self.mean_episode_length,
            "std_episode_length": self.std_episode_length,
            "min_episode_length": self.min_episode_length,
            "max_episode_length": self.max_episode_length,
            "average_steps_per_second": self.average_steps_per_second,
            "average_episodes_per_minute": self.average_episodes_per_minute,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "average_cpu_usage": self.average_cpu_usage,
            "num_checkpoints_saved": self.num_checkpoints_saved,
            "num_evaluations_performed": self.num_evaluations_performed,
            "num_interruptions": self.num_interruptions,
            "convergence_timestep": self.convergence_timestep,
            "convergence_episode": self.convergence_episode,
            "convergence_threshold": self.convergence_threshold,
        }


class TrainingHook(Protocol):
    """Протокол для пользовательских хуков обучения."""

    def on_training_start(self, progress: TrainingProgress) -> None:
        """Вызывается в начале обучения."""
        ...

    def on_episode_start(self, progress: TrainingProgress) -> None:
        """Вызывается в начале эпизода."""
        ...

    def on_step(self, progress: TrainingProgress, step_info: Dict[str, Any]) -> None:
        """Вызывается на каждом шаге."""
        ...

    def on_episode_end(
        self, progress: TrainingProgress, episode_info: Dict[str, Any]
    ) -> None:
        """Вызывается в конце эпизода."""
        ...

    def on_training_end(
        self, progress: TrainingProgress, statistics: TrainingStatistics
    ) -> None:
        """Вызывается в конце обучения."""
        ...


class ProgressReporter:
    """Репортер прогресса обучения."""

    def __init__(
        self,
        update_interval: float = 5.0,
        tensorboard_log_dir: Optional[str] = None,
        enable_console_output: bool = True,
        enable_file_logging: bool = True,
    ):
        """Инициализация репортера.

        Args:
            update_interval: Интервал обновления в секундах
            tensorboard_log_dir: Директория для TensorBoard логов
            enable_console_output: Включить вывод в консоль
            enable_file_logging: Включить логирование в файл
        """
        self.update_interval = update_interval
        self.enable_console_output = enable_console_output
        self.enable_file_logging = enable_file_logging

        # TensorBoard writer
        self.tb_writer: Optional[SummaryWriter] = None
        if tensorboard_log_dir:
            self.tb_writer = SummaryWriter(tensorboard_log_dir)

        # Последнее время обновления
        self.last_update_time = 0.0

        # Метрики для TensorBoard
        self.tb_step = 0

    def should_update(self, current_time: float) -> bool:
        """Проверить, нужно ли обновлять прогресс."""
        return current_time - self.last_update_time >= self.update_interval

    def report_progress(self, progress: TrainingProgress) -> None:
        """Сообщить о прогрессе."""
        current_time = time.time()

        if not self.should_update(current_time):
            return

        self.last_update_time = current_time

        # Консольный вывод
        if self.enable_console_output:
            self._print_console_progress(progress)

        # Логирование в файл
        if self.enable_file_logging:
            self._log_progress(progress)

        # TensorBoard
        if self.tb_writer:
            self._log_tensorboard(progress)

    def _print_console_progress(self, progress: TrainingProgress) -> None:
        """Вывести прогресс в консоль."""
        progress_pct = progress.get_progress_percentage()

        # Создаем прогресс-бар
        bar_length = 30
        filled_length = int(bar_length * progress_pct / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # Форматируем время
        elapsed_str = self._format_time(progress.elapsed_time)
        remaining_str = self._format_time(progress.estimated_time_remaining)

        print(
            f"\r[{bar}] {progress_pct:.1f}% | "
            f"Episode: {progress.current_episode} | "
            f"Step: {progress.current_timestep}/{progress.total_timesteps} | "
            f"Reward: {progress.mean_episode_reward:.2f}±{progress.std_episode_reward:.2f} | "
            f"Speed: {progress.steps_per_second:.1f} steps/s | "
            f"Time: {elapsed_str}/{remaining_str} | "
            f"Mem: {progress.memory_usage_mb:.0f}MB",
            end="",
        )

    def _log_progress(self, progress: TrainingProgress) -> None:
        """Логировать прогресс в файл."""
        logger.info(
            "Training progress update",
            extra={
                "timestep": progress.current_timestep,
                "episode": progress.current_episode,
                "progress_pct": progress.get_progress_percentage(),
                "mean_reward": progress.mean_episode_reward,
                "std_reward": progress.std_episode_reward,
                "steps_per_second": progress.steps_per_second,
                "memory_mb": progress.memory_usage_mb,
                "state": progress.state.value,
            },
        )

    def _log_tensorboard(self, progress: TrainingProgress) -> None:
        """Логировать в TensorBoard."""
        step = progress.current_timestep

        # Основные метрики
        self.tb_writer.add_scalar("Training/Episode", progress.current_episode, step)
        self.tb_writer.add_scalar(
            "Training/MeanReward", progress.mean_episode_reward, step
        )
        self.tb_writer.add_scalar(
            "Training/StdReward", progress.std_episode_reward, step
        )
        self.tb_writer.add_scalar(
            "Training/BestReward", progress.best_episode_reward, step
        )

        # Производительность
        self.tb_writer.add_scalar(
            "Performance/StepsPerSecond", progress.steps_per_second, step
        )
        self.tb_writer.add_scalar(
            "Performance/EpisodesPerMinute", progress.episodes_per_minute, step
        )

        # Ресурсы
        self.tb_writer.add_scalar("Resources/MemoryMB", progress.memory_usage_mb, step)
        self.tb_writer.add_scalar(
            "Resources/CPUPercent", progress.cpu_usage_percent, step
        )
        if progress.gpu_memory_mb > 0:
            self.tb_writer.add_scalar(
                "Resources/GPUMemoryMB", progress.gpu_memory_mb, step
            )

        # Прогресс
        self.tb_writer.add_scalar(
            "Progress/Percentage", progress.get_progress_percentage(), step
        )

        self.tb_writer.flush()

    def _format_time(self, seconds: float) -> str:
        """Форматировать время в читаемый вид."""
        if seconds < 0 or not np.isfinite(seconds):
            return "??:??"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def close(self) -> None:
        """Закрыть репортер."""
        if self.tb_writer:
            self.tb_writer.close()


class TrainingLoop:
    """Основной класс тренировочного цикла."""

    def __init__(
        self,
        agent: Agent,
        env: GymEnv,
        strategy: TrainingStrategy = TrainingStrategy.TIMESTEP_BASED,
        total_timesteps: int = 100_000,
        max_episodes: Optional[int] = None,
        eval_freq: int = 10_000,
        checkpoint_freq: int = 25_000,
        save_freq: int = 50_000,
        progress_update_interval: float = 5.0,
        memory_limit_mb: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        early_stopping_patience: int = 10,
        tensorboard_log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Инициализация тренировочного цикла.

        Args:
            agent: RL агент для обучения
            env: Среда для обучения
            strategy: Стратегия обучения
            total_timesteps: Общее количество временных шагов
            max_episodes: Максимальное количество эпизодов
            eval_freq: Частота оценки
            checkpoint_freq: Частота сохранения чекпоинтов
            save_freq: Частота сохранения модели
            progress_update_interval: Интервал обновления прогресса
            memory_limit_mb: Лимит памяти в МБ
            convergence_threshold: Порог сходимости
            early_stopping_patience: Терпение для раннего останова
            tensorboard_log_dir: Директория для TensorBoard
            experiment_name: Имя эксперимента
        """
        self.agent = agent
        self.env = env
        self.strategy = strategy
        self.total_timesteps = total_timesteps
        self.max_episodes = max_episodes
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_freq = save_freq
        self.memory_limit_mb = memory_limit_mb
        self.convergence_threshold = convergence_threshold
        self.early_stopping_patience = early_stopping_patience

        # Эксперимент и логирование
        self.experiment_name = experiment_name or f"training_{int(time.time())}"
        self.logger = get_experiment_logger(experiment_id=self.experiment_name)
        self.metrics_tracker = get_metrics_tracker(experiment_id=self.experiment_name)

        # Прогресс и статистика
        self.progress = TrainingProgress(total_timesteps=total_timesteps)
        self.statistics = TrainingStatistics()

        # Репортер прогресса
        self.progress_reporter = ProgressReporter(
            update_interval=progress_update_interval,
            tensorboard_log_dir=tensorboard_log_dir,
        )

        # Менеджер чекпоинтов
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if hasattr(agent, "checkpoint_manager") and agent.checkpoint_manager:
            self.checkpoint_manager = agent.checkpoint_manager

        # Обработка сигналов для graceful shutdown
        self.interrupted = False
        self.pause_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Пользовательские хуки
        self.hooks: List[TrainingHook] = []

        # Блокировка для потокобезопасности
        self._lock = threading.Lock()

        self.logger.info(
            f"Инициализирован TrainingLoop для эксперимента '{self.experiment_name}'",
            extra={
                "strategy": strategy.value,
                "total_timesteps": total_timesteps,
                "max_episodes": max_episodes,
                "eval_freq": eval_freq,
            },
        )

    def add_hook(self, hook: TrainingHook) -> None:
        """Добавить пользовательский хук."""
        self.hooks.append(hook)
        self.logger.info(f"Добавлен хук: {hook.__class__.__name__}")

    def run(self) -> TrainingStatistics:
        """Запустить тренировочный цикл.

        Returns:
            Статистика обучения

        Raises:
            RuntimeError: При ошибке обучения
        """
        try:
            self.progress.state = TrainingState.INITIALIZING
            self.progress.start_time = time.time()

            self.logger.info("Начало тренировочного цикла")

            # Вызываем хуки начала обучения
            for hook in self.hooks:
                hook.on_training_start(self.progress)

            # Основной цикл обучения
            if self.strategy == TrainingStrategy.EPISODIC:
                self._run_episodic_training()
            elif self.strategy == TrainingStrategy.TIMESTEP_BASED:
                self._run_timestep_training()
            elif self.strategy == TrainingStrategy.MIXED:
                self._run_mixed_training()
            elif self.strategy == TrainingStrategy.ADAPTIVE:
                self._run_adaptive_training()
            else:
                raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")

            # Финализация
            self._finalize_training()

            return self.statistics

        except KeyboardInterrupt:
            self.logger.info("Обучение прервано пользователем")
            self.progress.state = TrainingState.INTERRUPTED
            self.statistics.num_interruptions += 1
            return self.statistics

        except Exception as e:
            self.logger.error(f"Ошибка во время обучения: {e}")
            self.progress.state = TrainingState.ERROR
            raise RuntimeError(f"Ошибка тренировочного цикла: {e}") from e

        finally:
            self._cleanup()

    def _run_timestep_training(self) -> None:
        """Запустить обучение на основе временных шагов."""
        self.progress.state = TrainingState.TRAINING

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for timestep in range(self.total_timesteps):
            if self.interrupted:
                break

            # Проверка паузы
            while self.pause_requested and not self.interrupted:
                time.sleep(0.1)

            # Обновление прогресса
            self.progress.current_timestep = timestep + 1
            self.progress.current_episode_reward = episode_reward
            self.progress.current_episode_length = episode_length

            # Действие агента
            action, _ = self.agent.predict(obs, deterministic=False)

            # Шаг в среде
            step_info = {
                "timestep": timestep,
                "action": action,
                "observation": obs,
            }

            # Вызываем хуки шага
            for hook in self.hooks:
                hook.on_step(self.progress, step_info)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Добавляем метрики
            self.metrics_tracker.add_metric("step_reward", reward, timestep)

            # Конец эпизода
            if done:
                self._handle_episode_end(episode_reward, episode_length, timestep)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Периодические операции
            self._handle_periodic_operations(timestep)

            # Проверка ресурсов
            if self._should_stop_due_to_resources():
                break

    def _run_episodic_training(self) -> None:
        """Запустить эпизодическое обучение."""
        self.progress.state = TrainingState.TRAINING

        episode = 0
        total_timesteps = 0

        while (
            self.max_episodes is None or episode < self.max_episodes
        ) and total_timesteps < self.total_timesteps:
            if self.interrupted:
                break

            # Начало эпизода
            self.progress.current_episode = episode + 1
            episode_reward = 0.0
            episode_length = 0

            # Вызываем хуки начала эпизода
            for hook in self.hooks:
                hook.on_episode_start(self.progress)

            obs, _ = self.env.reset()
            done = False

            while not done and total_timesteps < self.total_timesteps:
                if self.interrupted:
                    break

                # Проверка паузы
                while self.pause_requested and not self.interrupted:
                    time.sleep(0.1)

                # Обновление прогресса
                total_timesteps += 1
                self.progress.current_timestep = total_timesteps
                self.progress.current_episode_reward = episode_reward
                self.progress.current_episode_length = episode_length

                # Действие агента
                action, _ = self.agent.predict(obs, deterministic=False)

                # Шаг в среде
                step_info = {
                    "timestep": total_timesteps,
                    "episode": episode,
                    "action": action,
                    "observation": obs,
                }

                # Вызываем хуки шага
                for hook in self.hooks:
                    hook.on_step(self.progress, step_info)

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Добавляем метрики
                self.metrics_tracker.add_metric("step_reward", reward, total_timesteps)

                # Периодические операции
                self._handle_periodic_operations(total_timesteps)

            # Конец эпизода
            if not self.interrupted:
                self._handle_episode_end(
                    episode_reward, episode_length, total_timesteps
                )

            episode += 1

            # Проверка ресурсов
            if self._should_stop_due_to_resources():
                break

    def _run_mixed_training(self) -> None:
        """Запустить смешанное обучение."""
        # Комбинация эпизодического и временного обучения
        # Переключение между стратегиями в зависимости от прогресса

        if self.progress.current_timestep < self.total_timesteps // 2:
            self.logger.info("Переключение на эпизодическое обучение")
            self._run_episodic_training()
        else:
            self.logger.info("Переключение на временное обучение")
            self._run_timestep_training()

    def _run_adaptive_training(self) -> None:
        """Запустить адаптивное обучение."""
        # Адаптивная стратегия на основе производительности

        # Начинаем с эпизодического обучения
        current_strategy = TrainingStrategy.EPISODIC
        last_switch_timestep = 0
        switch_interval = 10_000

        while self.progress.current_timestep < self.total_timesteps:
            if self.interrupted:
                break

            # Проверяем, нужно ли переключить стратегию
            if (
                self.progress.current_timestep - last_switch_timestep
            ) >= switch_interval:
                # Анализируем производительность
                if self.progress.steps_per_second < 100:  # Медленно
                    current_strategy = TrainingStrategy.TIMESTEP_BASED
                else:
                    current_strategy = TrainingStrategy.EPISODIC

                last_switch_timestep = self.progress.current_timestep
                self.logger.info(f"Адаптивное переключение на {current_strategy.value}")

            # Выполняем обучение с текущей стратегией
            if current_strategy == TrainingStrategy.EPISODIC:
                self._run_episodic_training_batch(switch_interval)
            else:
                self._run_timestep_training_batch(switch_interval)

    def _run_episodic_training_batch(self, max_timesteps: int) -> None:
        """Запустить пакет эпизодического обучения."""
        start_timestep = self.progress.current_timestep

        while (
            self.progress.current_timestep - start_timestep
        ) < max_timesteps and self.progress.current_timestep < self.total_timesteps:
            if self.interrupted:
                break

            # Один эпизод
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and self.progress.current_timestep < self.total_timesteps:
                if self.interrupted:
                    break

                self.progress.current_timestep += 1
                action, _ = self.agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                self.metrics_tracker.add_metric(
                    "step_reward", reward, self.progress.current_timestep
                )

                # Периодические операции
                self._handle_periodic_operations(self.progress.current_timestep)

            if not self.interrupted:
                self._handle_episode_end(
                    episode_reward, episode_length, self.progress.current_timestep
                )

    def _run_timestep_training_batch(self, max_timesteps: int) -> None:
        """Запустить пакет временного обучения."""
        start_timestep = self.progress.current_timestep

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        while (
            self.progress.current_timestep - start_timestep
        ) < max_timesteps and self.progress.current_timestep < self.total_timesteps:
            if self.interrupted:
                break

            self.progress.current_timestep += 1
            action, _ = self.agent.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            self.metrics_tracker.add_metric(
                "step_reward", reward, self.progress.current_timestep
            )

            if done:
                self._handle_episode_end(
                    episode_reward, episode_length, self.progress.current_timestep
                )
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Периодические операции
            self._handle_periodic_operations(self.progress.current_timestep)

    def _handle_episode_end(
        self, episode_reward: float, episode_length: int, timestep: int
    ) -> None:
        """Обработать конец эпизода."""
        self.progress.current_episode += 1
        self.progress.recent_episode_rewards.append(episode_reward)
        self.progress.recent_episode_lengths.append(episode_length)
        self.progress.update_episode_stats()

        # Добавляем метрики эпизода
        self.metrics_tracker.add_episode_metrics(
            episode=self.progress.current_episode,
            timestep=timestep,
            reward=episode_reward,
            length=episode_length,
        )

        # Информация об эпизоде
        episode_info = {
            "episode": self.progress.current_episode,
            "reward": episode_reward,
            "length": episode_length,
            "timestep": timestep,
        }

        # Вызываем хуки конца эпизода
        for hook in self.hooks:
            hook.on_episode_end(self.progress, episode_info)

        # Проверка сходимости
        if self.convergence_threshold:
            converged = self.statistics.detect_convergence(
                self.convergence_threshold, window_size=50, stability_episodes=20
            )
            if converged:
                self.logger.info(
                    f"Достигнута сходимость на timestep {timestep}",
                    extra={"threshold": self.convergence_threshold},
                )

        self.logger.debug(
            f"Эпизод {self.progress.current_episode} завершен",
            extra={
                "reward": episode_reward,
                "length": episode_length,
                "timestep": timestep,
                "mean_reward": self.progress.mean_episode_reward,
            },
        )

    def _handle_periodic_operations(self, timestep: int) -> None:
        """Обработать периодические операции."""
        current_time = time.time()

        # Обновление прогресса
        self.progress.update_performance_stats(current_time)
        self.progress.update_resource_usage()

        # Обновление статистики
        self.statistics.update_from_progress(self.progress)

        # Репорт прогресса
        self.progress_reporter.report_progress(self.progress)

        # Оценка
        if self.eval_freq > 0 and timestep % self.eval_freq == 0:
            self._perform_evaluation(timestep)

        # Чекпоинт
        if self.checkpoint_freq > 0 and timestep % self.checkpoint_freq == 0:
            self._save_checkpoint(timestep)

        # Сохранение модели
        if self.save_freq > 0 and timestep % self.save_freq == 0:
            self._save_model(timestep)

        # Очистка памяти
        if timestep % 1000 == 0:
            self._cleanup_memory()

    def _perform_evaluation(self, timestep: int) -> None:
        """Выполнить оценку агента."""
        self.progress.state = TrainingState.EVALUATING
        self.progress.last_evaluation_timestep = timestep

        try:
            eval_results = self.agent.evaluate(n_episodes=10, deterministic=True)

            # Добавляем метрики оценки
            self.metrics_tracker.add_metric(
                "eval_mean_reward", eval_results["mean_reward"], timestep
            )
            self.metrics_tracker.add_metric(
                "eval_std_reward", eval_results["std_reward"], timestep
            )

            self.statistics.num_evaluations_performed += 1

            self.logger.info(
                f"Оценка на timestep {timestep}",
                extra={
                    "mean_reward": eval_results["mean_reward"],
                    "std_reward": eval_results["std_reward"],
                },
            )

        except Exception as e:
            self.logger.error(f"Ошибка при оценке: {e}")

        finally:
            self.progress.state = TrainingState.TRAINING

    def _save_checkpoint(self, timestep: int) -> None:
        """Сохранить чекпоинт."""
        if not self.checkpoint_manager:
            return

        self.progress.state = TrainingState.SAVING
        self.progress.last_checkpoint_timestep = timestep

        try:
            checkpoint_data = {
                "timestep": timestep,
                "progress": self.progress.to_dict(),
                "statistics": self.statistics.to_dict(),
                "agent_state": self.agent.get_model_info(),
            }

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint_data, timestep=timestep
            )

            # Сохраняем модель агента
            if hasattr(self.agent, "save"):
                model_path = Path(checkpoint_path).parent / f"model_{timestep}.zip"
                self.agent.save(str(model_path))

            self.statistics.num_checkpoints_saved += 1

            self.logger.info(f"Чекпоинт сохранен: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении чекпоинта: {e}")

        finally:
            self.progress.state = TrainingState.TRAINING

    def _save_model(self, timestep: int) -> None:
        """Сохранить модель."""
        try:
            if hasattr(self.agent, "save"):
                model_path = f"model_{self.experiment_name}_{timestep}.zip"
                self.agent.save(model_path)
                self.logger.info(f"Модель сохранена: {model_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {e}")

    def _should_stop_due_to_resources(self) -> bool:
        """Проверить, нужно ли остановиться из-за ресурсов."""
        if (
            self.memory_limit_mb
            and self.progress.memory_usage_mb > self.memory_limit_mb
        ):
            self.logger.warning(
                f"Превышен лимит памяти: {self.progress.memory_usage_mb:.1f}MB > {self.memory_limit_mb}MB"
            )
            return True

        return False

    def _cleanup_memory(self) -> None:
        """Очистить память."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.debug(f"Ошибка при очистке памяти: {e}")

    def _finalize_training(self) -> None:
        """Финализировать обучение."""
        self.progress.state = TrainingState.COMPLETED

        # Финальная статистика
        self.statistics.update_from_progress(self.progress)

        # Финальная оценка
        if hasattr(self.agent, "evaluate"):
            try:
                final_eval = self.agent.evaluate(n_episodes=20, deterministic=True)
                self.logger.info(
                    "Финальная оценка",
                    extra={
                        "mean_reward": final_eval["mean_reward"],
                        "std_reward": final_eval["std_reward"],
                    },
                )
            except Exception as e:
                self.logger.error(f"Ошибка финальной оценки: {e}")

        # Сохранение финальных метрик
        self.metrics_tracker.export_to_json()
        self.metrics_tracker.export_to_csv()

        # Вызываем хуки конца обучения
        for hook in self.hooks:
            hook.on_training_end(self.progress, self.statistics)

        self.logger.info(
            "Обучение завершено",
            extra={
                "total_timesteps": self.progress.current_timestep,
                "total_episodes": self.progress.current_episode,
                "training_time": self.progress.elapsed_time,
                "final_mean_reward": self.progress.mean_episode_reward,
            },
        )

    def _signal_handler(self, signum: int, frame) -> None:
        """Обработчик сигналов для graceful shutdown."""
        signal_names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
        signal_name = signal_names.get(signum, f"Signal {signum}")

        self.logger.info(
            f"Получен сигнал {signal_name}, инициируется graceful shutdown"
        )
        self.interrupted = True

    def _cleanup(self) -> None:
        """Очистить ресурсы."""
        try:
            # Закрываем репортер прогресса
            self.progress_reporter.close()

            # Очищаем память
            self._cleanup_memory()

            self.logger.info("Ресурсы очищены")

        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}")

    def pause(self) -> None:
        """Приостановить обучение."""
        self.pause_requested = True
        self.progress.state = TrainingState.PAUSED
        self.logger.info("Обучение приостановлено")

    def resume(self) -> None:
        """Возобновить обучение."""
        self.pause_requested = False
        self.progress.state = TrainingState.TRAINING
        self.logger.info("Обучение возобновлено")

    def stop(self) -> None:
        """Остановить обучение."""
        self.interrupted = True
        self.logger.info("Запрошена остановка обучения")


# Предопределенные хуки
class LoggingHook:
    """Хук для детального логирования."""

    def __init__(self, log_interval: int = 1000):
        self.log_interval = log_interval
        self.logger = logging.getLogger(f"{__name__}.LoggingHook")

    def on_training_start(self, progress: TrainingProgress) -> None:
        self.logger.info("Начало обучения", extra=progress.to_dict())

    def on_episode_start(self, progress: TrainingProgress) -> None:
        if progress.current_episode % 100 == 0:
            self.logger.debug(f"Начало эпизода {progress.current_episode}")

    def on_step(self, progress: TrainingProgress, step_info: Dict[str, Any]) -> None:
        if progress.current_timestep % self.log_interval == 0:
            self.logger.debug(
                f"Шаг {progress.current_timestep}", extra={"step_info": step_info}
            )

    def on_episode_end(
        self, progress: TrainingProgress, episode_info: Dict[str, Any]
    ) -> None:
        self.logger.debug(
            f"Конец эпизода {episode_info['episode']}", extra=episode_info
        )

    def on_training_end(
        self, progress: TrainingProgress, statistics: TrainingStatistics
    ) -> None:
        self.logger.info(
            "Обучение завершено",
            extra={
                "final_statistics": statistics.to_dict(),
                "final_progress": progress.to_dict(),
            },
        )


class EarlyStoppingHook:
    """Хук для раннего останова."""

    def __init__(
        self,
        patience: int = 10,
        min_improvement: float = 0.01,
        metric_name: str = "mean_episode_reward",
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric_name = metric_name
        self.best_value = float("-inf")
        self.patience_counter = 0
        self.logger = logging.getLogger(f"{__name__}.EarlyStoppingHook")

    def on_training_start(self, progress: TrainingProgress) -> None:
        self.best_value = float("-inf")
        self.patience_counter = 0

    def on_episode_start(self, progress: TrainingProgress) -> None:
        pass

    def on_step(self, progress: TrainingProgress, step_info: Dict[str, Any]) -> None:
        pass

    def on_episode_end(
        self, progress: TrainingProgress, episode_info: Dict[str, Any]
    ) -> None:
        current_value = progress.mean_episode_reward

        if current_value > self.best_value + self.min_improvement:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.logger.info(
                f"Ранний останов: нет улучшения за {self.patience} эпизодов",
                extra={
                    "best_value": self.best_value,
                    "current_value": current_value,
                    "patience": self.patience,
                },
            )
            # Здесь можно установить флаг для остановки обучения

    def on_training_end(
        self, progress: TrainingProgress, statistics: TrainingStatistics
    ) -> None:
        pass


def create_training_loop(
    agent: Agent,
    env: GymEnv,
    config: Dict[str, Any],
    experiment_name: Optional[str] = None,
) -> TrainingLoop:
    """Создать тренировочный цикл из конфигурации.

    Args:
        agent: RL агент
        env: Среда обучения
        config: Конфигурация обучения
        experiment_name: Имя эксперимента

    Returns:
        Настроенный тренировочный цикл
    """
    strategy_map = {
        "episodic": TrainingStrategy.EPISODIC,
        "timestep": TrainingStrategy.TIMESTEP_BASED,
        "mixed": TrainingStrategy.MIXED,
        "adaptive": TrainingStrategy.ADAPTIVE,
    }

    strategy = strategy_map.get(
        config.get("strategy", "timestep"), TrainingStrategy.TIMESTEP_BASED
    )

    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=strategy,
        total_timesteps=config.get("total_timesteps", 100_000),
        max_episodes=config.get("max_episodes"),
        eval_freq=config.get("eval_freq", 10_000),
        checkpoint_freq=config.get("checkpoint_freq", 25_000),
        save_freq=config.get("save_freq", 50_000),
        progress_update_interval=config.get("progress_update_interval", 5.0),
        memory_limit_mb=config.get("memory_limit_mb"),
        convergence_threshold=config.get("convergence_threshold"),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        tensorboard_log_dir=config.get("tensorboard_log_dir"),
        experiment_name=experiment_name,
    )

    # Добавляем стандартные хуки
    if config.get("enable_logging_hook", True):
        training_loop.add_hook(
            LoggingHook(log_interval=config.get("log_interval", 1000))
        )

    if config.get("enable_early_stopping", False):
        training_loop.add_hook(
            EarlyStoppingHook(
                patience=config.get("early_stopping_patience", 10),
                min_improvement=config.get("min_improvement", 0.01),
            )
        )

    return training_loop
