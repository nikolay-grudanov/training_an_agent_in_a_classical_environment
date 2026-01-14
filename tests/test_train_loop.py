"""Тесты для модуля тренировочного цикла.

Проверяет функциональность TrainingLoop, TrainingProgress, TrainingStatistics
и связанных компонентов с различными стратегиями обучения.
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from collections import deque

import numpy as np

from src.training.train_loop import (
    TrainingLoop,
    TrainingProgress,
    TrainingStatistics,
    TrainingStrategy,
    TrainingState,
    ProgressReporter,
    LoggingHook,
    EarlyStoppingHook,
    create_training_loop,
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
    
    @patch('psutil.Process')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_update_resource_usage(self, mock_memory, mock_cuda, mock_process):
        """Тест обновления информации о ресурсах."""
        # Настройка моков
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process_instance.cpu_percent.return_value = 75.5
        mock_process.return_value = mock_process_instance
        
        mock_cuda.return_value = True
        mock_memory.return_value = 1024 * 1024 * 50  # 50MB
        
        progress = TrainingProgress()
        progress.update_resource_usage()
        
        assert progress.memory_usage_mb == 100.0
        assert progress.cpu_usage_percent == 75.5
        assert progress.gpu_memory_mb == 50.0
    
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
        for r, l in zip(rewards, lengths):
            progress.recent_episode_rewards.append(r)
            progress.recent_episode_lengths.append(l)
        
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
            threshold=18.0, 
            window_size=50, 
            stability_episodes=20
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
    
    @patch('builtins.print')
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
        assert reporter._format_time(float('inf')) == "??:??"


class TestTrainingLoop:
    """Тесты для класса TrainingLoop."""
    
    @pytest.fixture
    def mock_agent(self):
        """Создать мок агента."""
        agent = Mock()
        agent.predict.return_value = (np.array([0]), None)
        agent.evaluate.return_value = {
            "mean_reward": 15.0,
            "std_reward": 2.0,
        }
        agent.save = Mock()
        agent.get_model_info.return_value = {"algorithm": "PPO"}
        agent.checkpoint_manager = None
        return agent
    
    @pytest.fixture
    def mock_env(self):
        """Создать мок среды."""
        env = Mock()
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
        env.step.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1]),  # observation
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}  # info
        )
        return env
    
    def test_init(self, mock_agent, mock_env):
        """Тест инициализации."""
        loop = TrainingLoop(
            agent=mock_agent,
            env=mock_env,
            strategy=TrainingStrategy.TIMESTEP_BASED,
            total_timesteps=1000,
            experiment_name="test_experiment",
        )
        
        assert loop.agent == mock_agent
        assert loop.env == mock_env
        assert loop.strategy == TrainingStrategy.TIMESTEP_BASED
        assert loop.total_timesteps == 1000
        assert loop.experiment_name == "test_experiment"
        assert loop.progress.total_timesteps == 1000
        assert not loop.interrupted
        assert not loop.pause_requested
    
    def test_add_hook(self, mock_agent, mock_env):
        """Тест добавления хука."""
        loop = TrainingLoop(mock_agent, mock_env)
        hook = LoggingHook()
        
        loop.add_hook(hook)
        
        assert hook in loop.hooks
    
    @patch('time.time')
    def test_run_timestep_training_short(self, mock_time, mock_agent, mock_env):
        """Тест короткого обучения по временным шагам."""
        # Настройка времени
        mock_time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0]
        
        # Настройка среды для завершения эпизода
        mock_env.step.side_effect = [
            (np.array([0.1]), 1.0, True, False, {}),  # Завершение эпизода
            (np.array([0.2]), 2.0, False, False, {}),
            (np.array([0.3]), 3.0, False, False, {}),
        ]
        
        loop = TrainingLoop(
            agent=mock_agent,
            env=mock_env,
            strategy=TrainingStrategy.TIMESTEP_BASED,
            total_timesteps=3,
            eval_freq=0,  # Отключаем оценку
            checkpoint_freq=0,  # Отключаем чекпоинты
            save_freq=0,  # Отключаем сохранение
            progress_update_interval=0.1,
        )
        
        statistics = loop.run()
        
        assert isinstance(statistics, TrainingStatistics)
        assert loop.progress.current_timestep == 3
        assert loop.progress.current_episode >= 1
        assert loop.progress.state == TrainingState.COMPLETED
    
    def test_pause_resume(self, mock_agent, mock_env):
        """Тест приостановки и возобновления."""
        loop = TrainingLoop(mock_agent, mock_env)
        
        # Проверяем начальное состояние
        assert not loop.pause_requested
        assert loop.progress.state == TrainingState.IDLE
        
        # Приостанавливаем
        loop.pause()
        assert loop.pause_requested
        assert loop.progress.state == TrainingState.PAUSED
        
        # Возобновляем
        loop.resume()
        assert not loop.pause_requested
        assert loop.progress.state == TrainingState.TRAINING
    
    def test_stop(self, mock_agent, mock_env):
        """Тест остановки."""
        loop = TrainingLoop(mock_agent, mock_env)
        
        assert not loop.interrupted
        
        loop.stop()
        assert loop.interrupted
    
    def test_signal_handler(self, mock_agent, mock_env):
        """Тест обработчика сигналов."""
        loop = TrainingLoop(mock_agent, mock_env)
        
        assert not loop.interrupted
        
        # Имитируем получение сигнала
        loop._signal_handler(2, None)  # SIGINT
        assert loop.interrupted
    
    @patch('gc.collect')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_cleanup_memory(self, mock_empty_cache, mock_cuda, mock_gc, mock_agent, mock_env):
        """Тест очистки памяти."""
        mock_cuda.return_value = True
        
        loop = TrainingLoop(mock_agent, mock_env)
        loop._cleanup_memory()
        
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()
    
    def test_should_stop_due_to_resources(self, mock_agent, mock_env):
        """Тест проверки остановки из-за ресурсов."""
        loop = TrainingLoop(
            mock_agent, 
            mock_env, 
            memory_limit_mb=100.0
        )
        
        # Нормальное использование памяти
        loop.progress.memory_usage_mb = 50.0
        assert not loop._should_stop_due_to_resources()
        
        # Превышение лимита
        loop.progress.memory_usage_mb = 150.0
        assert loop._should_stop_due_to_resources()


class TestHooks:
    """Тесты для хуков обучения."""
    
    def test_logging_hook(self):
        """Тест хука логирования."""
        hook = LoggingHook(log_interval=100)
        progress = TrainingProgress()
        
        # Тестируем методы хука
        hook.on_training_start(progress)
        hook.on_episode_start(progress)
        hook.on_step(progress, {"timestep": 1})
        hook.on_episode_end(progress, {"episode": 1, "reward": 10.0})
        hook.on_training_end(progress, TrainingStatistics())
        
        # Проверяем, что методы выполняются без ошибок
        assert True
    
    def test_early_stopping_hook(self):
        """Тест хука раннего останова."""
        hook = EarlyStoppingHook(patience=3, min_improvement=1.0)
        progress = TrainingProgress()
        
        # Начальное состояние
        hook.on_training_start(progress)
        assert hook.best_value == float("-inf")
        assert hook.patience_counter == 0
        
        # Улучшение
        progress.mean_episode_reward = 10.0
        hook.on_episode_end(progress, {"reward": 10.0})
        assert hook.best_value == 10.0
        assert hook.patience_counter == 0
        
        # Нет улучшения
        progress.mean_episode_reward = 9.5
        hook.on_episode_end(progress, {"reward": 9.5})
        assert hook.patience_counter == 1
        
        # Еще нет улучшения
        progress.mean_episode_reward = 9.0
        hook.on_episode_end(progress, {"reward": 9.0})
        assert hook.patience_counter == 2
        
        # Значительное улучшение
        progress.mean_episode_reward = 12.0
        hook.on_episode_end(progress, {"reward": 12.0})
        assert hook.best_value == 12.0
        assert hook.patience_counter == 0


class TestCreateTrainingLoop:
    """Тесты для функции создания тренировочного цикла."""
    
    def test_create_training_loop_basic(self):
        """Тест базового создания тренировочного цикла."""
        mock_agent = Mock()
        mock_env = Mock()
        
        config = {
            "strategy": "timestep",
            "total_timesteps": 5000,
            "eval_freq": 1000,
        }
        
        loop = create_training_loop(
            agent=mock_agent,
            env=mock_env,
            config=config,
            experiment_name="test_experiment",
        )
        
        assert isinstance(loop, TrainingLoop)
        assert loop.agent == mock_agent
        assert loop.env == mock_env
        assert loop.strategy == TrainingStrategy.TIMESTEP_BASED
        assert loop.total_timesteps == 5000
        assert loop.eval_freq == 1000
        assert loop.experiment_name == "test_experiment"
    
    def test_create_training_loop_with_hooks(self):
        """Тест создания с хуками."""
        mock_agent = Mock()
        mock_env = Mock()
        
        config = {
            "strategy": "episodic",
            "enable_logging_hook": True,
            "enable_early_stopping": True,
            "early_stopping_patience": 5,
            "log_interval": 500,
        }
        
        loop = create_training_loop(
            agent=mock_agent,
            env=mock_env,
            config=config,
        )
        
        assert len(loop.hooks) == 2  # LoggingHook и EarlyStoppingHook
        assert loop.strategy == TrainingStrategy.EPISODIC
    
    def test_create_training_loop_all_strategies(self):
        """Тест создания со всеми стратегиями."""
        mock_agent = Mock()
        mock_env = Mock()
        
        strategies = ["episodic", "timestep", "mixed", "adaptive"]
        expected = [
            TrainingStrategy.EPISODIC,
            TrainingStrategy.TIMESTEP_BASED,
            TrainingStrategy.MIXED,
            TrainingStrategy.ADAPTIVE,
        ]
        
        for strategy_name, expected_strategy in zip(strategies, expected):
            config = {"strategy": strategy_name}
            loop = create_training_loop(mock_agent, mock_env, config)
            assert loop.strategy == expected_strategy


@pytest.mark.integration
class TestTrainingLoopIntegration:
    """Интеграционные тесты тренировочного цикла."""
    
    @pytest.fixture
    def simple_agent(self):
        """Простой агент для тестирования."""
        class SimpleAgent:
            def predict(self, obs, deterministic=False):
                return np.array([0]), None
            
            def evaluate(self, n_episodes=10, deterministic=True):
                return {"mean_reward": 10.0, "std_reward": 1.0}
            
            def save(self, path):
                pass
            
            def get_model_info(self):
                return {"algorithm": "Simple"}
        
        return SimpleAgent()
    
    @pytest.fixture
    def simple_env(self):
        """Простая среда для тестирования."""
        class SimpleEnv:
            def __init__(self):
                self.step_count = 0
            
            def reset(self):
                self.step_count = 0
                return np.array([0.0]), {}
            
            def step(self, action):
                self.step_count += 1
                obs = np.array([0.1])
                reward = 1.0
                terminated = self.step_count >= 10  # Эпизод длится 10 шагов
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info
        
        return SimpleEnv()
    
    def test_full_training_cycle(self, simple_agent, simple_env):
        """Тест полного цикла обучения."""
        loop = TrainingLoop(
            agent=simple_agent,
            env=simple_env,
            strategy=TrainingStrategy.TIMESTEP_BASED,
            total_timesteps=50,
            eval_freq=0,  # Отключаем оценку
            checkpoint_freq=0,  # Отключаем чекпоинты
            save_freq=0,  # Отключаем сохранение
            progress_update_interval=1.0,
        )
        
        # Добавляем хук для тестирования
        hook_calls = []
        
        class TestHook:
            def on_training_start(self, progress):
                hook_calls.append("start")
            
            def on_episode_start(self, progress):
                hook_calls.append("episode_start")
            
            def on_step(self, progress, step_info):
                hook_calls.append("step")
            
            def on_episode_end(self, progress, episode_info):
                hook_calls.append("episode_end")
            
            def on_training_end(self, progress, statistics):
                hook_calls.append("end")
        
        loop.add_hook(TestHook())
        
        # Запускаем обучение
        statistics = loop.run()
        
        # Проверяем результаты
        assert isinstance(statistics, TrainingStatistics)
        assert loop.progress.current_timestep == 50
        assert loop.progress.current_episode >= 5  # 50 шагов / 10 шагов на эпизод
        assert loop.progress.state == TrainingState.COMPLETED
        
        # Проверяем вызовы хуков
        assert "start" in hook_calls
        assert "end" in hook_calls
        assert "step" in hook_calls
        assert "episode_end" in hook_calls
        assert hook_calls.count("step") == 50


if __name__ == "__main__":
    pytest.main([__file__])