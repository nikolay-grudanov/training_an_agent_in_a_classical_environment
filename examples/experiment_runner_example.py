"""Пример использования ExperimentRunner для проведения RL экспериментов.

Этот скрипт демонстрирует различные способы использования ExperimentRunner
для оркестрации контролируемых экспериментов с baseline и variant конфигурациями.
"""

import logging
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.config import Configuration
from src.experiments.experiment import Experiment
from src.experiments.runner import ExperimentRunner, ExecutionMode
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig


def create_sample_experiment() -> Experiment:
    """Создать пример эксперимента для демонстрации.
    
    Returns:
        Настроенный объект эксперимента
    """
    print("📋 Создание примера эксперимента...")
    
    # Baseline конфигурация - стандартный PPO
    baseline_algorithm = AlgorithmConfig(
        name="PPO",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    
    baseline_environment = EnvironmentConfig(
        name="LunarLander-v2",
        render_mode=None,
    )
    
    baseline_training = TrainingConfig(
        total_timesteps=50_000,  # Уменьшено для быстрого примера
        eval_freq=10_000,
        n_eval_episodes=5,
        save_freq=25_000,
    )
    
    baseline_config = RLConfig(
        algorithm=baseline_algorithm,
        environment=baseline_environment,
        training=baseline_training,
        seed=42,
        experiment_name="baseline_ppo",
        output_dir="results/examples",
    )
    
    # Variant конфигурация - PPO с измененным learning rate
    variant_algorithm = AlgorithmConfig(
        name="PPO",
        learning_rate=1e-3,  # Увеличенный learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    
    variant_training = TrainingConfig(
        total_timesteps=50_000,
        eval_freq=10_000,
        n_eval_episodes=5,
        save_freq=25_000,
    )
    
    variant_config = RLConfig(
        algorithm=variant_algorithm,
        environment=baseline_environment,  # Та же среда
        training=variant_training,
        seed=42,  # Тот же seed для честного сравнения
        experiment_name="variant_ppo_high_lr",
        output_dir="results/examples",
    )
    
    # Создание эксперимента
    experiment = Experiment(
        baseline_config=baseline_config,
        variant_config=variant_config,
        hypothesis="Увеличенный learning rate (1e-3) должен ускорить обучение по сравнению со стандартным (3e-4)",
        output_dir="results/examples",
    )
    
    print(f"✅ Эксперимент создан: {experiment.experiment_id}")
    print(f"📁 Директория: {experiment.experiment_dir}")
    
    return experiment


def example_sequential_execution():
    """Пример последовательного выполнения эксперимента."""
    print("\n🔄 Пример 1: Последовательное выполнение")
    print("=" * 50)
    
    # Создание эксперимента
    experiment = create_sample_experiment()
    
    # Создание runner'а для последовательного выполнения
    runner = ExperimentRunner(
        experiment=experiment,
        execution_mode=ExecutionMode.SEQUENTIAL,
        enable_monitoring=True,
        checkpoint_frequency=10_000,
    )
    
    print("🚀 Запуск последовательного выполнения...")
    
    try:
        # Выполнение эксперимента
        success = runner.run()
        
        if success:
            print("✅ Эксперимент выполнен успешно!")
            
            # Получение статуса
            status = runner.get_status()
            print(f"⏱️  Время выполнения: {status['execution_time']:.1f} сек")
            
            # Результаты
            if runner.baseline_result and runner.variant_result:
                baseline_reward = runner.baseline_result.final_mean_reward
                variant_reward = runner.variant_result.final_mean_reward
                improvement = variant_reward - baseline_reward
                
                print(f"📊 Результаты:")
                print(f"   Baseline награда: {baseline_reward:.2f}")
                print(f"   Variant награда: {variant_reward:.2f}")
                print(f"   Улучшение: {improvement:+.2f}")
                
                if improvement > 0:
                    print("🎉 Гипотеза подтверждена: variant показал лучшие результаты!")
                else:
                    print("🤔 Гипотеза не подтверждена: baseline показал лучшие результаты")
        else:
            print("❌ Эксперимент завершился с ошибкой")
            
    except KeyboardInterrupt:
        print("\n⚠️  Эксперимент прерван пользователем")
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
    finally:
        # Очистка ресурсов
        runner.cleanup()


def example_parallel_execution():
    """Пример параллельного выполнения эксперимента."""
    print("\n⚡ Пример 2: Параллельное выполнение")
    print("=" * 50)
    
    # Создание эксперимента
    experiment = create_sample_experiment()
    
    # Создание runner'а для параллельного выполнения
    runner = ExperimentRunner(
        experiment=experiment,
        execution_mode=ExecutionMode.PARALLEL,
        max_workers=2,  # Baseline и variant параллельно
        enable_monitoring=True,
        resource_limits={
            "memory_mb": 4096,  # 4GB лимит памяти
            "cpu_percent": 80.0,  # 80% CPU
        }
    )
    
    print("🚀 Запуск параллельного выполнения...")
    print("⚡ Baseline и variant будут обучаться одновременно")
    
    try:
        success = runner.run()
        
        if success:
            print("✅ Параллельное выполнение завершено успешно!")
            
            # Сравнение времени выполнения
            status = runner.get_status()
            print(f"⏱️  Общее время: {status['execution_time']:.1f} сек")
            print("💡 Параллельное выполнение должно быть быстрее последовательного")
        else:
            print("❌ Параллельное выполнение завершилось с ошибкой")
            
    except Exception as e:
        print(f"❌ Ошибка параллельного выполнения: {e}")
    finally:
        runner.cleanup()


def example_validation_mode():
    """Пример режима валидации (dry-run)."""
    print("\n🔍 Пример 3: Режим валидации")
    print("=" * 50)
    
    # Создание эксперимента
    experiment = create_sample_experiment()
    
    # Создание runner'а для валидации
    runner = ExperimentRunner(
        experiment=experiment,
        execution_mode=ExecutionMode.VALIDATION,
        enable_monitoring=False,  # Не нужен для валидации
    )
    
    print("🔍 Запуск валидации конфигураций...")
    print("💡 Обучение не будет выполняться, только проверка настроек")
    
    try:
        success = runner.run()
        
        if success:
            print("✅ Валидация прошла успешно!")
            print("🎯 Все конфигурации корректны и готовы к выполнению")
        else:
            print("❌ Валидация выявила ошибки в конфигурациях")
            
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
    finally:
        runner.cleanup()


def example_monitoring_and_progress():
    """Пример мониторинга прогресса выполнения."""
    print("\n📊 Пример 4: Мониторинг и прогресс")
    print("=" * 50)
    
    # Создание эксперимента
    experiment = create_sample_experiment()
    
    # Создание runner'а с мониторингом
    runner = ExperimentRunner(
        experiment=experiment,
        execution_mode=ExecutionMode.SEQUENTIAL,
        enable_monitoring=True,
        checkpoint_frequency=5_000,
    )
    
    print("📊 Демонстрация мониторинга прогресса...")
    
    try:
        # Запуск в отдельном потоке для демонстрации мониторинга
        import threading
        import time
        
        def run_experiment():
            runner.run()
        
        # Запуск эксперимента в фоне
        experiment_thread = threading.Thread(target=run_experiment)
        experiment_thread.start()
        
        # Мониторинг прогресса
        print("🔄 Мониторинг прогресса (первые 30 секунд):")
        
        for i in range(6):  # 6 итераций по 5 секунд
            time.sleep(5)
            
            # Получение прогресса
            progress = runner.monitor_progress()
            status = runner.get_status()
            
            print(f"   Шаг {i+1}: {progress.current_phase}, "
                  f"Baseline: {progress.baseline_progress:.1f}%, "
                  f"Variant: {progress.variant_progress:.1f}%, "
                  f"CPU: {status['resource_usage']['cpu_percent']:.1f}%, "
                  f"Memory: {status['resource_usage']['memory_mb']:.1f}MB")
            
            # Прерываем если эксперимент завершен
            if status['status'] in ['completed', 'failed']:
                break
        
        # Ожидание завершения
        experiment_thread.join(timeout=60)  # Максимум 1 минута
        
        print("📊 Мониторинг завершен")
        
    except Exception as e:
        print(f"❌ Ошибка мониторинга: {e}")
    finally:
        runner.cleanup()


def example_error_handling():
    """Пример обработки ошибок."""
    print("\n🛠️  Пример 5: Обработка ошибок")
    print("=" * 50)
    
    # Создание эксперимента с некорректными параметрами
    try:
        # Некорректная конфигурация
        invalid_algorithm = AlgorithmConfig(
            name="INVALID_ALGORITHM",  # Неподдерживаемый алгоритм
            learning_rate=3e-4,
        )
        
        invalid_environment = EnvironmentConfig(
            name="LunarLander-v2",
        )
        
        invalid_training = TrainingConfig(
            total_timesteps=1000,
        )
        
        invalid_config = RLConfig(
            algorithm=invalid_algorithm,
            environment=invalid_environment,
            training=invalid_training,
            seed=42,
            experiment_name="invalid_experiment",
        )
        
        # Попытка создания эксперимента с некорректной конфигурацией
        experiment = Experiment(
            baseline_config=invalid_config,
            variant_config=invalid_config,
            hypothesis="Тест обработки ошибок",
        )
        
        runner = ExperimentRunner(experiment)
        
        print("🧪 Тестирование обработки ошибок...")
        success = runner.run()
        
        if not success:
            print("✅ Ошибка корректно обработана")
            status = runner.get_status()
            print(f"📋 Статус: {status['status']}")
        
    except Exception as e:
        print(f"✅ Ошибка корректно перехвачена: {e}")


def main():
    """Главная функция с примерами использования."""
    print("🎯 Примеры использования ExperimentRunner")
    print("=" * 60)
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Создание директории для результатов
    results_dir = Path("results/examples")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Пример 1: Последовательное выполнение
        example_sequential_execution()
        
        # Пример 2: Параллельное выполнение
        # example_parallel_execution()  # Закомментировано для экономии времени
        
        # Пример 3: Режим валидации
        example_validation_mode()
        
        # Пример 4: Мониторинг прогресса
        # example_monitoring_and_progress()  # Закомментировано для экономии времени
        
        # Пример 5: Обработка ошибок
        example_error_handling()
        
        print("\n🎉 Все примеры выполнены!")
        print(f"📁 Результаты сохранены в: {results_dir.absolute()}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Примеры прерваны пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()