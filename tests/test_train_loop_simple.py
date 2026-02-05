"""Упрощенные тесты для модуля тренировочного цикла.

Проверяет основную функциональность без зависимостей от gymnasium и stable-baselines3.
"""

import pytest
from unittest.mock import Mock, patch
from collections import deque


# Импортируем только те части, которые не зависят от внешних библиотек
from src.training.train_loop import (
    TrainingProgress,
    TrainingStatistics,
    TrainingStrategy,
    TrainingState,
    ProgressReporter,
)


class TestTrainingProgress:
    """Тесты для класса TrainingProgress."""

    def test_init_default_values(self):
        """Тест инициализации с значениями по умолчанию."""
        progress = TrainingProgress()

        assert progress.current_timestep == 0
        assert progress.current_episode == 0
        assert progress.total_timesteps == 0
        assert progress.total_episodes == 0
        assert progress.state == TrainingState.IDLE
        assert progress.mean_episode_reward == 0.0
        assert isinstance(progress.recent_episode_rewards, deque)

    def test_update_episode_stats(self):
        """Тест обновления статистики эпизодов."""
        progress = TrainingProgress()

        # Добавляем награды
        rewards = [10.0, 20.0, 15.0, 25.0, 18.0]
        for reward in rewards:
            progress.recent_episode_rewards.append(reward)

        progress.update_episode_stats()

        assert progress.mean_episode_reward == pytest.approx(17.6)
        assert progress.std_episode_reward == pytest.approx(5.367, rel=1e-3)
        assert progress.best_episode_reward == 25.0
        assert progress.worst_episode_reward == 10.0

    def test_update_performance_stats(self):
        """Тест обновления статистики производительности."""
        progress = TrainingProgress()
        progress.start_time = 1000.0
        progress.current_timestep = 500
        progress.current_episode = 10
        progress.total_timesteps = 1000

        current_time = 1010.0  # 10 секунд прошло
        progress.update_performance_stats(current_time)

        assert progress.elapsed_time == 10.0
        assert progress.steps_per_second == 50.0  # 500 шагов за 10 секунд
        assert progress.episodes_per_minute == 60.0  # 10 эпизодов за 10 секунд
        assert progress.estimated_time_remaining == pytest.approx(10.0)  # 50% прогресс

    def test_get_progress_percentage(self):
        """Тест вычисления процента выполнения."""
        progress = TrainingProgress()
        progress.total_timesteps = 1000

        progress.current_timestep = 0
        assert progress.get_progress_percentage() == 0.0

        progress.current_timestep = 250
        assert progress.get_progress_percentage() == 25.0

        progress.current_timestep = 1000
        assert progress.get_progress_percentage() == 100.0

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        progress = TrainingProgress()
        progress.current_timestep = 100
        progress.current_episode = 5
        progress.total_timesteps = 1000
        progress.state = TrainingState.TRAINING

        result = progress.to_dict()

        assert isinstance(result, dict)
        assert result["current_timestep"] == 100
        assert result["current_episode"] == 5
        assert result["total_timesteps"] == 1000
        assert result["state"] == "training"
        assert result["progress_percentage"] == 10.0


class TestTrainingStatistics:
    """Тесты для класса TrainingStatistics."""

    def test_init_default_values(self):
        """Тест инициализации с значениями по умолчанию."""
        stats = TrainingStatistics()

        assert stats.total_training_time == 0.0
        assert stats.total_timesteps_completed == 0
        assert stats.total_episodes_completed == 0
        assert stats.episode_rewards == []
        assert stats.best_episode_reward == float("-inf")
        assert stats.worst_episode_reward == float("inf")
        assert stats.convergence_timestep is None

    def test_update_from_progress(self):
        """Тест обновления из прогресса."""
        progress = TrainingProgress()
        progress.elapsed_time = 120.0
        progress.current_timestep = 500
        progress.current_episode = 25
        progress.mean_episode_reward = 15.5
        progress.std_episode_reward = 3.2
        progress.best_episode_reward = 22.0
        progress.worst_episode_reward = 8.0
        progress.steps_per_second = 50.0
        progress.episodes_per_minute = 12.5
        progress.memory_usage_mb = 150.0
        progress.cpu_usage_percent = 80.0

        # Добавляем данные в deque
        rewards = [10.0, 15.0, 20.0, 12.0, 18.0]
        lengths = [100, 120, 95, 110, 105]
        for r, length in zip(rewards, lengths):
            progress.recent_episode_rewards.append(r)
            progress.recent_episode_lengths.append(length)

        stats = TrainingStatistics()
        stats.update_from_progress(progress)

        assert stats.total_training_time == 120.0
        assert stats.total_timesteps_completed == 500
        assert stats.total_episodes_completed == 25
        assert stats.mean_episode_reward == 15.5
        assert stats.std_episode_reward == 3.2
        assert stats.best_episode_reward == 22.0
        assert stats.worst_episode_reward == 8.0
        assert stats.average_steps_per_second == 50.0
        assert stats.average_episodes_per_minute == 12.5
        assert stats.peak_memory_usage_mb == 150.0
        assert stats.average_cpu_usage == 80.0
        assert stats.episode_rewards == rewards
        assert stats.episode_lengths == lengths
        assert stats.mean_episode_length == 106.0
        assert stats.min_episode_length == 95
        assert stats.max_episode_length == 120

    def test_detect_convergence(self):
        """Тест определения сходимости."""
        stats = TrainingStatistics()

        # Недостаточно данных
        stats.episode_rewards = [10.0, 15.0, 20.0]
        assert not stats.detect_convergence(threshold=18.0, window_size=50)

        # Создаем данные с сходимостью
        rewards = [10.0] * 30 + [20.0] * 50  # Улучшение после 30 эпизодов
        stats.episode_rewards = rewards
        stats.total_timesteps_completed = 1000
        stats.total_episodes_completed = 80

        # Проверяем сходимость
        converged = stats.detect_convergence(
            threshold=18.0, window_size=50, stability_episodes=20
        )

        assert converged
        assert stats.convergence_timestep == 1000
        assert stats.convergence_episode == 80
        assert stats.convergence_threshold == 18.0

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        stats = TrainingStatistics()
        stats.total_training_time = 300.0
        stats.total_timesteps_completed = 1000
        stats.best_episode_reward = 25.0
        stats.convergence_timestep = 800

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert result["total_training_time"] == 300.0
        assert result["total_timesteps_completed"] == 1000
        assert result["best_episode_reward"] == 25.0
        assert result["convergence_timestep"] == 800


class TestProgressReporter:
    """Тесты для класса ProgressReporter."""

    def test_init(self):
        """Тест инициализации."""
        reporter = ProgressReporter(
            update_interval=10.0,
            enable_console_output=False,
            enable_file_logging=True,
        )

        assert reporter.update_interval == 10.0
        assert not reporter.enable_console_output
        assert reporter.enable_file_logging
        assert reporter.last_update_time == 0.0

    def test_should_update(self):
        """Тест проверки необходимости обновления."""
        reporter = ProgressReporter(update_interval=5.0)

        # Первое обновление
        assert reporter.should_update(10.0)

        # Обновляем время
        reporter.last_update_time = 10.0

        # Слишком рано для обновления
        assert not reporter.should_update(12.0)

        # Время для обновления
        assert reporter.should_update(16.0)

    @patch("builtins.print")
    def test_print_console_progress(self, mock_print):
        """Тест вывода прогресса в консоль."""
        reporter = ProgressReporter(enable_console_output=True)

        progress = TrainingProgress()
        progress.current_episode = 10
        progress.current_timestep = 500
        progress.total_timesteps = 1000
        progress.mean_episode_reward = 15.5
        progress.std_episode_reward = 2.3
        progress.steps_per_second = 50.0
        progress.elapsed_time = 120.0
        progress.estimated_time_remaining = 120.0
        progress.memory_usage_mb = 100.0

        reporter._print_console_progress(progress)

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "50.0%" in call_args
        assert "Episode: 10" in call_args
        assert "15.5±2.3" in call_args
        assert "50.0 steps/s" in call_args

    def test_format_time(self):
        """Тест форматирования времени."""
        reporter = ProgressReporter()

        assert reporter._format_time(0) == "00:00"
        assert reporter._format_time(65) == "01:05"
        assert reporter._format_time(3665) == "01:01:05"
        assert reporter._format_time(-1) == "??:??"
        assert reporter._format_time(float("inf")) == "??:??"


class TestTrainingEnums:
    """Тесты для перечислений."""

    def test_training_strategy_enum(self):
        """Тест перечисления стратегий обучения."""
        assert TrainingStrategy.EPISODIC.value == "episodic"
        assert TrainingStrategy.TIMESTEP_BASED.value == "timestep"
        assert TrainingStrategy.MIXED.value == "mixed"
        assert TrainingStrategy.ADAPTIVE.value == "adaptive"

    def test_training_state_enum(self):
        """Тест перечисления состояний обучения."""
        assert TrainingState.IDLE.value == "idle"
        assert TrainingState.INITIALIZING.value == "initializing"
        assert TrainingState.TRAINING.value == "training"
        assert TrainingState.EVALUATING.value == "evaluating"
        assert TrainingState.SAVING.value == "saving"
        assert TrainingState.PAUSED.value == "paused"
        assert TrainingState.INTERRUPTED.value == "interrupted"
        assert TrainingState.COMPLETED.value == "completed"
        assert TrainingState.ERROR.value == "error"


class TestTrainingProgressResourceMonitoring:
    """Тесты для мониторинга ресурсов."""

    @patch("psutil.Process")
    def test_update_resource_usage_success(self, mock_process):
        """Тест успешного обновления информации о ресурсах."""
        # Настройка мока
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process_instance.cpu_percent.return_value = 75.5
        mock_process.return_value = mock_process_instance

        progress = TrainingProgress()

        with patch("torch.cuda.is_available", return_value=False):
            progress.update_resource_usage()

        assert progress.memory_usage_mb == 100.0
        assert progress.cpu_usage_percent == 75.5
        assert progress.gpu_memory_mb == 0.0

    @patch("psutil.Process")
    def test_update_resource_usage_with_gpu(self, mock_process):
        """Тест обновления с GPU информацией."""
        # Настройка мока для процесса
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 50  # 50MB
        mock_process_instance.cpu_percent.return_value = 60.0
        mock_process.return_value = mock_process_instance

        progress = TrainingProgress()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 25),
        ):  # 25MB
            progress.update_resource_usage()

        assert progress.memory_usage_mb == 50.0
        assert progress.cpu_usage_percent == 60.0
        assert progress.gpu_memory_mb == 25.0

    @patch("psutil.Process")
    def test_update_resource_usage_exception(self, mock_process):
        """Тест обработки исключений при мониторинге ресурсов."""
        # Настройка мока для генерации исключения
        mock_process.side_effect = Exception("Process error")

        progress = TrainingProgress()

        # Не должно вызывать исключение
        progress.update_resource_usage()

        # Значения должны остаться по умолчанию
        assert progress.memory_usage_mb == 0.0
        assert progress.cpu_usage_percent == 0.0
        assert progress.gpu_memory_mb == 0.0


class TestTrainingStatisticsAdvanced:
    """Расширенные тесты для статистики обучения."""

    def test_detect_convergence_insufficient_data(self):
        """Тест определения сходимости с недостаточными данными."""
        stats = TrainingStatistics()

        # Пустые данные
        assert not stats.detect_convergence(threshold=10.0)

        # Недостаточно данных для окна
        stats.episode_rewards = [5.0, 8.0, 12.0]
        assert not stats.detect_convergence(threshold=10.0, window_size=50)

    def test_detect_convergence_no_stability(self):
        """Тест определения сходимости без стабильности."""
        stats = TrainingStatistics()

        # Данные превышают порог, но нестабильны
        rewards = [5.0] * 30 + [15.0, 8.0, 16.0, 7.0, 14.0] * 10  # Нестабильные данные
        stats.episode_rewards = rewards
        stats.total_timesteps_completed = 1000
        stats.total_episodes_completed = len(rewards)

        converged = stats.detect_convergence(
            threshold=12.0, window_size=50, stability_episodes=10
        )

        # Не должно определить сходимость из-за нестабильности
        assert not converged
        assert stats.convergence_timestep is None

    def test_detect_convergence_multiple_calls(self):
        """Тест множественных вызовов определения сходимости."""
        stats = TrainingStatistics()

        # Первый вызов - нет сходимости
        rewards = [5.0] * 60
        stats.episode_rewards = rewards
        stats.total_timesteps_completed = 600
        stats.total_episodes_completed = 60

        assert not stats.detect_convergence(threshold=10.0)
        assert stats.convergence_timestep is None

        # Второй вызов - есть сходимость
        rewards.extend([15.0] * 30)
        stats.episode_rewards = rewards
        stats.total_timesteps_completed = 900
        stats.total_episodes_completed = 90

        converged = stats.detect_convergence(threshold=10.0, stability_episodes=20)
        assert converged
        assert stats.convergence_timestep == 900

        # Третий вызов - сходимость уже определена
        rewards.extend([20.0] * 10)
        stats.episode_rewards = rewards
        stats.total_timesteps_completed = 1000
        stats.total_episodes_completed = 100

        # Не должно изменить уже установленную сходимость
        stats.detect_convergence(threshold=18.0)
        assert stats.convergence_timestep == 900  # Остается прежним


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
