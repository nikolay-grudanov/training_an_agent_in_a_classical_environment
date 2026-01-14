"""Примеры использования тренировочного цикла.

Демонстрирует различные способы использования TrainingLoop для обучения
RL агентов с различными стратегиями, мониторингом и настройками.
"""

import logging
import time
from pathlib import Path

import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Импорты проекта
from src.training.train_loop import (
    TrainingLoop,
    TrainingStrategy,
    TrainingProgress,
    TrainingStatistics,
    LoggingHook,
    EarlyStoppingHook,
    create_training_loop,
)
from src.agents import PPOAgent, AgentConfig
from src.environments import LunarLanderEnvironment


def create_simple_agent_and_env():
    """Создать простого агента и среду для демонстрации."""
    
    # Конфигурация агента
    config = AgentConfig(
        algorithm="PPO",
        env_name="LunarLander-v3",
        total_timesteps=10_000,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        verbose=1,
    )
    
    # Создание среды
    env = LunarLanderEnvironment()
    
    # Создание агента
    agent = PPOAgent(config=config, env=env, experiment_name="train_loop_demo")
    
    return agent, env


def example_basic_training():
    """Пример базового обучения с тренировочным циклом."""
    print("\n=== Пример 1: Базовое обучение ===")
    
    agent, env = create_simple_agent_and_env()
    
    # Создание тренировочного цикла
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.TIMESTEP_BASED,
        total_timesteps=5_000,
        eval_freq=1_000,
        checkpoint_freq=2_000,
        save_freq=2_500,
        progress_update_interval=2.0,
        experiment_name="basic_training_demo",
    )
    
    # Запуск обучения
    try:
        statistics = training_loop.run()
        
        print(f"✅ Обучение завершено!")
        print(f"📊 Общее время: {statistics.total_training_time:.1f} сек")
        print(f"🎯 Финальная награда: {statistics.mean_episode_reward:.2f}")
        print(f"📈 Лучшая награда: {statistics.best_episode_reward:.2f}")
        print(f"🏃 Скорость: {statistics.average_steps_per_second:.1f} шагов/сек")
        
    except KeyboardInterrupt:
        print("❌ Обучение прервано пользователем")
    
    finally:
        env.close()


def example_episodic_training():
    """Пример эпизодического обучения."""
    print("\n=== Пример 2: Эпизодическое обучение ===")
    
    agent, env = create_simple_agent_and_env()
    
    # Создание тренировочного цикла с эпизодической стратегией
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.EPISODIC,
        total_timesteps=8_000,
        max_episodes=50,  # Ограничение по эпизодам
        eval_freq=0,  # Отключаем оценку для скорости
        checkpoint_freq=0,
        save_freq=0,
        progress_update_interval=3.0,
        experiment_name="episodic_training_demo",
    )
    
    # Добавляем хук логирования
    logging_hook = LoggingHook(log_interval=500)
    training_loop.add_hook(logging_hook)
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Эпизодическое обучение завершено!")
        print(f"📊 Эпизодов: {statistics.total_episodes_completed}")
        print(f"⏱️  Время: {statistics.total_training_time:.1f} сек")
        print(f"📏 Средняя длина эпизода: {statistics.mean_episode_length:.1f}")
        print(f"🎯 Средняя награда: {statistics.mean_episode_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def example_training_with_early_stopping():
    """Пример обучения с ранним остановом."""
    print("\n=== Пример 3: Обучение с ранним остановом ===")
    
    agent, env = create_simple_agent_and_env()
    
    # Создание тренировочного цикла
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.TIMESTEP_BASED,
        total_timesteps=15_000,
        eval_freq=1_500,
        convergence_threshold=200.0,  # Порог сходимости
        early_stopping_patience=5,
        progress_update_interval=2.0,
        experiment_name="early_stopping_demo",
    )
    
    # Добавляем хук раннего останова
    early_stopping_hook = EarlyStoppingHook(
        patience=3,
        min_improvement=5.0,
        metric_name="mean_episode_reward"
    )
    training_loop.add_hook(early_stopping_hook)
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Обучение с ранним остановом завершено!")
        print(f"🎯 Финальная награда: {statistics.mean_episode_reward:.2f}")
        
        if statistics.convergence_timestep:
            print(f"🎉 Сходимость достигнута на шаге {statistics.convergence_timestep}")
        else:
            print("⚠️  Сходимость не достигнута")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def example_adaptive_training():
    """Пример адаптивного обучения."""
    print("\n=== Пример 4: Адаптивное обучение ===")
    
    agent, env = create_simple_agent_and_env()
    
    # Создание тренировочного цикла с адаптивной стратегией
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.ADAPTIVE,
        total_timesteps=10_000,
        eval_freq=2_000,
        memory_limit_mb=500.0,  # Лимит памяти
        progress_update_interval=1.5,
        experiment_name="adaptive_training_demo",
    )
    
    # Пользовательский хук для мониторинга адаптации
    class AdaptiveMonitoringHook:
        def __init__(self):
            self.strategy_switches = 0
            self.last_strategy = None
        
        def on_training_start(self, progress):
            print("🚀 Начало адаптивного обучения")
        
        def on_episode_start(self, progress):
            pass
        
        def on_step(self, progress, step_info):
            # Мониторинг производительности
            if progress.current_timestep % 1000 == 0:
                print(f"📊 Шаг {progress.current_timestep}: "
                      f"{progress.steps_per_second:.1f} шагов/сек, "
                      f"память: {progress.memory_usage_mb:.1f}MB")
        
        def on_episode_end(self, progress, episode_info):
            pass
        
        def on_training_end(self, progress, statistics):
            print(f"🏁 Адаптивное обучение завершено")
            print(f"🔄 Переключений стратегии: {self.strategy_switches}")
    
    adaptive_hook = AdaptiveMonitoringHook()
    training_loop.add_hook(adaptive_hook)
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Адаптивное обучение завершено!")
        print(f"💾 Пиковое использование памяти: {statistics.peak_memory_usage_mb:.1f}MB")
        print(f"⚡ Средняя скорость: {statistics.average_steps_per_second:.1f} шагов/сек")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def example_training_with_config():
    """Пример создания тренировочного цикла из конфигурации."""
    print("\n=== Пример 5: Обучение из конфигурации ===")
    
    agent, env = create_simple_agent_and_env()
    
    # Конфигурация обучения
    training_config = {
        "strategy": "mixed",
        "total_timesteps": 6_000,
        "max_episodes": 30,
        "eval_freq": 1_500,
        "checkpoint_freq": 3_000,
        "save_freq": 0,
        "progress_update_interval": 2.5,
        "memory_limit_mb": 300.0,
        "convergence_threshold": 150.0,
        "early_stopping_patience": 4,
        "tensorboard_log_dir": "results/tensorboard/config_demo",
        "enable_logging_hook": True,
        "enable_early_stopping": True,
        "log_interval": 750,
        "min_improvement": 3.0,
    }
    
    # Создание тренировочного цикла из конфигурации
    training_loop = create_training_loop(
        agent=agent,
        env=env,
        config=training_config,
        experiment_name="config_based_training",
    )
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Обучение из конфигурации завершено!")
        print(f"📊 Статистика:")
        print(f"  - Шагов: {statistics.total_timesteps_completed}")
        print(f"  - Эпизодов: {statistics.total_episodes_completed}")
        print(f"  - Время: {statistics.total_training_time:.1f} сек")
        print(f"  - Награда: {statistics.mean_episode_reward:.2f} ± {statistics.std_episode_reward:.2f}")
        print(f"  - Лучшая награда: {statistics.best_episode_reward:.2f}")
        print(f"  - Чекпоинтов: {statistics.num_checkpoints_saved}")
        print(f"  - Оценок: {statistics.num_evaluations_performed}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def example_pause_resume_training():
    """Пример приостановки и возобновления обучения."""
    print("\n=== Пример 6: Приостановка и возобновление ===")
    
    agent, env = create_simple_agent_and_env()
    
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.TIMESTEP_BASED,
        total_timesteps=8_000,
        eval_freq=0,
        checkpoint_freq=0,
        save_freq=0,
        progress_update_interval=1.0,
        experiment_name="pause_resume_demo",
    )
    
    # Пользовательский хук для демонстрации паузы
    class PauseResumeHook:
        def __init__(self, training_loop):
            self.training_loop = training_loop
            self.paused = False
        
        def on_training_start(self, progress):
            print("🚀 Начало обучения с возможностью паузы")
        
        def on_episode_start(self, progress):
            pass
        
        def on_step(self, progress, step_info):
            # Приостанавливаем на 2000 шагов для демонстрации
            if progress.current_timestep == 2000 and not self.paused:
                print("⏸️  Приостанавливаем обучение на 2 секунды...")
                self.training_loop.pause()
                self.paused = True
                
                # Имитируем паузу
                import threading
                def resume_after_delay():
                    time.sleep(2)
                    print("▶️  Возобновляем обучение")
                    self.training_loop.resume()
                
                threading.Thread(target=resume_after_delay).start()
        
        def on_episode_end(self, progress, episode_info):
            pass
        
        def on_training_end(self, progress, statistics):
            print("🏁 Обучение с паузой завершено")
    
    pause_hook = PauseResumeHook(training_loop)
    training_loop.add_hook(pause_hook)
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Обучение с паузой завершено!")
        print(f"⏱️  Общее время: {statistics.total_training_time:.1f} сек")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def example_resource_monitoring():
    """Пример мониторинга ресурсов."""
    print("\n=== Пример 7: Мониторинг ресурсов ===")
    
    agent, env = create_simple_agent_and_env()
    
    training_loop = TrainingLoop(
        agent=agent,
        env=env,
        strategy=TrainingStrategy.TIMESTEP_BASED,
        total_timesteps=5_000,
        eval_freq=0,
        checkpoint_freq=0,
        save_freq=0,
        memory_limit_mb=200.0,  # Низкий лимит для демонстрации
        progress_update_interval=1.0,
        experiment_name="resource_monitoring_demo",
    )
    
    # Хук для мониторинга ресурсов
    class ResourceMonitoringHook:
        def __init__(self):
            self.max_memory = 0.0
            self.max_cpu = 0.0
        
        def on_training_start(self, progress):
            print("📊 Начало мониторинга ресурсов")
        
        def on_episode_start(self, progress):
            pass
        
        def on_step(self, progress, step_info):
            self.max_memory = max(self.max_memory, progress.memory_usage_mb)
            self.max_cpu = max(self.max_cpu, progress.cpu_usage_percent)
            
            # Выводим информацию каждые 1000 шагов
            if progress.current_timestep % 1000 == 0:
                print(f"💾 Память: {progress.memory_usage_mb:.1f}MB "
                      f"(макс: {self.max_memory:.1f}MB)")
                print(f"🖥️  CPU: {progress.cpu_usage_percent:.1f}% "
                      f"(макс: {self.max_cpu:.1f}%)")
                if progress.gpu_memory_mb > 0:
                    print(f"🎮 GPU память: {progress.gpu_memory_mb:.1f}MB")
        
        def on_episode_end(self, progress, episode_info):
            pass
        
        def on_training_end(self, progress, statistics):
            print(f"📈 Пиковое использование:")
            print(f"  - Память: {self.max_memory:.1f}MB")
            print(f"  - CPU: {self.max_cpu:.1f}%")
    
    resource_hook = ResourceMonitoringHook()
    training_loop.add_hook(resource_hook)
    
    try:
        statistics = training_loop.run()
        
        print(f"✅ Мониторинг ресурсов завершен!")
        print(f"💾 Пиковая память: {statistics.peak_memory_usage_mb:.1f}MB")
        print(f"🖥️  Средний CPU: {statistics.average_cpu_usage:.1f}%")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    finally:
        env.close()


def main():
    """Главная функция для запуска всех примеров."""
    print("🎯 Демонстрация возможностей TrainingLoop")
    print("=" * 50)
    
    examples = [
        ("Базовое обучение", example_basic_training),
        ("Эпизодическое обучение", example_episodic_training),
        ("Ранний останов", example_training_with_early_stopping),
        ("Адаптивное обучение", example_adaptive_training),
        ("Конфигурационное обучение", example_training_with_config),
        ("Пауза и возобновление", example_pause_resume_training),
        ("Мониторинг ресурсов", example_resource_monitoring),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            print(f"\n🔄 Запуск примера {i}: {name}")
            example_func()
            print(f"✅ Пример {i} завершен успешно")
        except KeyboardInterrupt:
            print(f"\n⏹️  Пример {i} прерван пользователем")
            break
        except Exception as e:
            print(f"\n❌ Ошибка в примере {i}: {e}")
            continue
        
        # Пауза между примерами
        if i < len(examples):
            print("\n⏳ Пауза 2 секунды перед следующим примером...")
            time.sleep(2)
    
    print("\n🎉 Демонстрация завершена!")


if __name__ == "__main__":
    main()