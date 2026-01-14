"""Пример базового использования утилит RL системы."""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.seeding import set_seed, SeedManager
from src.utils.config import RLConfig, ConfigLoader
from src.utils.logging import setup_logging, get_experiment_logger
from src.utils.metrics import MetricsTracker
from src.utils.checkpointing import CheckpointManager, create_checkpoint_metadata
from src.experiments.base import create_experiment


def demo_seeding():
    """Демонстрация работы с seed'ами."""
    print("=== Демонстрация модуля seeding ===")
    
    # Установка глобального seed
    set_seed(42)
    print("✓ Установлен глобальный seed: 42")
    
    # Использование SeedManager
    manager = SeedManager(42)
    seed1 = manager.set_experiment_seed("experiment_1")
    seed2 = manager.set_experiment_seed("experiment_2")
    
    print(f"✓ Seed для эксперимента 1: {seed1}")
    print(f"✓ Seed для эксперимента 2: {seed2}")
    print()


def demo_config():
    """Демонстрация работы с конфигурацией."""
    print("=== Демонстрация модуля config ===")
    
    # Создание конфигурации по умолчанию
    config = RLConfig()
    print(f"✓ Создана конфигурация: {config.experiment_name}")
    print(f"✓ Алгоритм: {config.algorithm.name}")
    print(f"✓ Среда: {config.environment.name}")
    print(f"✓ Seed: {config.seed}")
    
    # Загрузчик конфигураций
    loader = ConfigLoader()
    print("✓ Создан загрузчик конфигураций")
    print()


def demo_logging():
    """Демонстрация системы логирования."""
    print("=== Демонстрация модуля logging ===")
    
    # Настройка логирования
    logger = setup_logging(
        log_level="INFO",
        console_output=True,
        json_format=False
    )
    print("✓ Настроено базовое логирование")
    
    # Логгер для эксперимента
    exp_logger = get_experiment_logger("demo_experiment")
    exp_logger.info("Это сообщение от логгера эксперимента")
    
    # Логирование шага обучения
    exp_logger.log_training_step(
        timestep=100,
        episode=5,
        reward=15.5,
        loss=0.02
    )
    print("✓ Записан шаг обучения")
    print()


def demo_metrics():
    """Демонстрация трекера метрик."""
    print("=== Демонстрация модуля metrics ===")
    
    # Создание трекера
    tracker = MetricsTracker("demo_experiment")
    print("✓ Создан трекер метрик")
    
    # Добавление метрик
    for i in range(10):
        tracker.add_metric("reward", i * 2.5, timestep=i*100, episode=i)
        tracker.add_metric("loss", 1.0 / (i + 1), timestep=i*100)
    
    print("✓ Добавлены тестовые метрики")
    
    # Получение статистики
    summary = tracker.get_metric_summary("reward")
    if summary:
        print(f"✓ Статистика по reward: mean={summary.mean:.2f}, max={summary.max:.2f}")
    
    # Экспорт метрик
    json_path = tracker.export_to_json("demo_metrics.json")
    print(f"✓ Метрики экспортированы в: {json_path}")
    print()


def demo_checkpointing():
    """Демонстрация системы чекпоинтов."""
    print("=== Демонстрация модуля checkpointing ===")
    
    # Создание менеджера чекпоинтов
    manager = CheckpointManager(
        checkpoint_dir="demo_checkpoints",
        experiment_id="demo_experiment"
    )
    print("✓ Создан менеджер чекпоинтов")
    
    # Создание метаданных
    metadata = create_checkpoint_metadata(
        experiment_id="demo_experiment",
        timestep=1000,
        episode=50,
        reward=25.5,
        model_class="PPO",
        algorithm="PPO",
        environment="LunarLander-v3",
        seed=42,
        hyperparameters={"learning_rate": 3e-4}
    )
    print("✓ Созданы метаданные чекпоинта")
    
    # Простая модель для демонстрации
    import torch
    dummy_model = torch.nn.Linear(4, 2)
    
    try:
        # Сохранение чекпоинта
        checkpoint_path = manager.save_checkpoint(dummy_model, metadata)
        print(f"✓ Чекпоинт сохранен: {checkpoint_path}")
        
        # Статистика чекпоинтов
        stats = manager.get_checkpoint_stats()
        print(f"✓ Статистика: {stats['total_checkpoints']} чекпоинтов")
    except Exception as e:
        print(f"⚠ Ошибка при сохранении чекпоинта: {e}")
    
    print()


def demo_experiment():
    """Демонстрация системы экспериментов."""
    print("=== Демонстрация модуля experiments ===")
    
    # Создание конфигурации
    config = RLConfig(experiment_name="demo_experiment")
    
    # Функция выполнения эксперимента
    def execute_func(experiment):
        experiment.logger.info("Выполнение демо-эксперимента")
        
        # Симуляция обучения
        for step in range(5):
            reward = step * 2.0
            experiment.metrics_tracker.add_metric("reward", reward, timestep=step)
            experiment.logger.info(f"Шаг {step}: reward={reward}")
        
        experiment.logger.info("Демо-эксперимент завершен")
    
    # Создание и запуск эксперимента
    experiment = create_experiment(
        experiment_id="demo_experiment",
        config=config,
        execute_func=execute_func
    )
    
    print("✓ Создан эксперимент")
    
    # Запуск эксперимента
    result = experiment.run()
    print(f"✓ Эксперимент завершен со статусом: {result.status}")
    print(f"✓ Длительность: {result.duration_seconds:.2f} секунд")
    print()


def main():
    """Главная функция демонстрации."""
    print("🚀 Демонстрация утилит RL системы обучения агентов\n")
    
    try:
        demo_seeding()
        demo_config()
        demo_logging()
        demo_metrics()
        demo_checkpointing()
        demo_experiment()
        
        print("🎉 Все демонстрации успешно выполнены!")
        
    except Exception as e:
        print(f"❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()