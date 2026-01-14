#!/usr/bin/env python3
"""Пример использования системы проверки воспроизводимости RL экспериментов.

Этот скрипт демонстрирует основные возможности модуля reproducibility_checker:
- Создание и регистрация экспериментальных запусков
- Проверка воспроизводимости результатов
- Диагностика проблем с детерминированностью
- Генерация отчетов и рекомендаций
"""

import logging
import tempfile
from pathlib import Path

import numpy as np

from src.utils.reproducibility_checker import (
    ReproducibilityChecker,
    StrictnessLevel,
    create_simple_reproducibility_test,
    quick_reproducibility_check,
    validate_experiment_reproducibility,
)
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig
from src.utils.seeding import set_seed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_training_simulation(seed: int, algorithm: str = "PPO") -> dict:
    """Симуляция простого процесса обучения RL агента.
    
    Args:
        seed: Сид для воспроизводимости
        algorithm: Название алгоритма
        
    Returns:
        Результаты симуляции обучения
    """
    # Устанавливаем сид для детерминированности
    set_seed(seed)
    
    # Симулируем процесс обучения
    num_episodes = 100
    rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # Симулируем эпизод
        episode_reward = 0.0
        episode_length = 0
        
        # Базовая награда зависит от алгоритма
        base_reward = 100.0 if algorithm == "PPO" else 80.0
        
        # Добавляем случайность и прогресс обучения
        progress_factor = min(1.0, episode / 50.0)  # Улучшение за первые 50 эпизодов
        noise = np.random.normal(0, 10.0)
        
        episode_reward = base_reward * progress_factor + noise
        episode_length = int(50 + np.random.normal(0, 5))
        
        rewards.append(episode_reward)
        episode_lengths.append(max(1, episode_length))
    
    # Вычисляем финальные метрики
    final_reward = np.mean(rewards[-10:])  # Среднее за последние 10 эпизодов
    final_length = np.mean(episode_lengths[-10:])
    
    return {
        'final_reward': final_reward,
        'final_episode_length': final_length,
        'total_episodes': num_episodes,
        'algorithm': algorithm,
        'metrics': {
            'episode_rewards': rewards,
            'episode_lengths': episode_lengths,
            'reward_trend': np.convolve(rewards, np.ones(10)/10, mode='valid').tolist()  # Скользящее среднее
        }
    }


def demonstrate_basic_reproducibility_check():
    """Демонстрация базовой проверки воспроизводимости."""
    logger.info("=== Демонстрация базовой проверки воспроизводимости ===")
    
    # Создаем временную директорию для примера
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Инициализируем проверщик воспроизводимости
        checker = ReproducibilityChecker(
            project_root=project_root,
            strictness_level=StrictnessLevel.STANDARD
        )
        
        # Создаем конфигурацию эксперимента
        config = RLConfig(
            experiment_name="demo_experiment",
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=42),
            environment=EnvironmentConfig(name="CartPole-v1")
        )
        
        logger.info("Выполнение нескольких запусков с одинаковыми сидами...")
        
        # Выполняем несколько запусков с одинаковыми параметрами
        for run_idx in range(3):
            logger.info(f"Запуск {run_idx + 1}/3")
            
            # Симулируем обучение
            results = simple_training_simulation(seed=42, algorithm="PPO")
            
            # Регистрируем запуск
            run_id = checker.register_experiment_run(
                experiment_id="demo_experiment",
                config=config,
                results={
                    'final_reward': results['final_reward'],
                    'final_episode_length': results['final_episode_length'],
                    'total_episodes': results['total_episodes']
                },
                metrics=results['metrics'],
                metadata={'run_index': run_idx, 'algorithm': results['algorithm']}
            )
            
            logger.info(f"Зарегистрирован запуск: {run_id}")
        
        logger.info("Проверка воспроизводимости...")
        
        # Проверяем воспроизводимость
        report = checker.check_reproducibility("demo_experiment")
        
        # Выводим результаты
        logger.info(f"Результат проверки: {'✓ Воспроизводимо' if report.is_reproducible else '✗ Не воспроизводимо'}")
        logger.info(f"Уверенность: {report.confidence_score:.2f}")
        logger.info(f"Количество запусков: {len(report.runs)}")
        logger.info(f"Обнаружено проблем: {len(report.issues)}")
        
        if report.issues:
            logger.info("Найденные проблемы:")
            for issue in report.issues[:3]:  # Показываем первые 3 проблемы
                logger.info(f"  - {issue.severity.upper()}: {issue.description}")
        
        if report.recommendations:
            logger.info("Рекомендации:")
            for rec in report.recommendations[:3]:  # Показываем первые 3 рекомендации
                logger.info(f"  - {rec}")
        
        return report


def demonstrate_determinism_validation():
    """Демонстрация валидации детерминизма функций."""
    logger.info("\n=== Демонстрация валидации детерминизма ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        checker = ReproducibilityChecker(project_root=project_root)
        
        # Тест детерминистической функции
        def deterministic_function():
            np.random.seed(42)
            return {
                'value': np.random.random(),
                'array': np.random.random(5).tolist()
            }
        
        logger.info("Тестирование детерминистической функции...")
        result = checker.validate_determinism(
            test_function=deterministic_function,
            seed=42,
            num_runs=5
        )
        
        logger.info(f"Детерминирована: {'✓' if result['is_deterministic'] else '✗'}")
        logger.info(f"Уникальных результатов: {result['unique_results']}")
        logger.info(f"Успешность: {result['success_rate']:.1%}")
        
        # Тест недетерминистической функции
        import time
        def nondeterministic_function():
            return {'timestamp': time.time()}
        
        logger.info("\nТестирование недетерминистической функции...")
        result = checker.validate_determinism(
            test_function=nondeterministic_function,
            seed=42,
            num_runs=3
        )
        
        logger.info(f"Детерминирована: {'✓' if result['is_deterministic'] else '✗'}")
        logger.info(f"Уникальных результатов: {result['unique_results']}")


def demonstrate_different_seeds_comparison():
    """Демонстрация сравнения экспериментов с разными сидами."""
    logger.info("\n=== Демонстрация сравнения разных сидов ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        checker = ReproducibilityChecker(
            project_root=project_root,
            strictness_level=StrictnessLevel.STRICT
        )
        
        # Тестируем с разными сидами
        seeds = [42, 123, 456]
        
        for seed in seeds:
            config = RLConfig(
                experiment_name="different_seeds_demo",
                seed=seed,
                algorithm=AlgorithmConfig(name="PPO", seed=seed),
                environment=EnvironmentConfig(name="CartPole-v1")
            )
            
            logger.info(f"Запуск с сидом {seed}...")
            
            results = simple_training_simulation(seed=seed, algorithm="PPO")
            
            checker.register_experiment_run(
                experiment_id="different_seeds_demo",
                config=config,
                results={
                    'final_reward': results['final_reward'],
                    'final_episode_length': results['final_episode_length']
                },
                metrics=results['metrics'],
                metadata={'seed_used': seed}
            )
        
        # Проверяем воспроизводимость
        report = checker.check_reproducibility("different_seeds_demo")
        
        logger.info(f"Результат (разные сиды): {'✓' if report.is_reproducible else '✗'}")
        logger.info(f"Уверенность: {report.confidence_score:.2f}")
        logger.info(f"Проблем найдено: {len(report.issues)}")
        
        # Показываем проблемы с сидами
        seed_issues = [issue for issue in report.issues if "сид" in issue.description.lower()]
        if seed_issues:
            logger.info("Проблемы с сидами:")
            for issue in seed_issues:
                logger.info(f"  - {issue.description}")


def demonstrate_automatic_testing():
    """Демонстрация автоматического тестирования воспроизводимости."""
    logger.info("\n=== Демонстрация автоматического тестирования ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        checker = ReproducibilityChecker(project_root=project_root)
        
        # Создаем тестовую функцию
        def test_function(seed):
            return simple_training_simulation(seed=seed, algorithm="A2C")
        
        # Создаем конфигурацию
        config = RLConfig(
            experiment_name="auto_test_demo",
            algorithm=AlgorithmConfig(name="A2C"),
            environment=EnvironmentConfig(name="CartPole-v1")
        )
        
        logger.info("Запуск автоматического теста воспроизводимости...")
        
        # Запускаем автоматический тест
        report = checker.run_reproducibility_test(
            test_function=test_function,
            experiment_id="auto_test_demo",
            seeds=[42, 42, 42],  # Одинаковые сиды для проверки воспроизводимости
            config=config
        )
        
        logger.info(f"Автоматический тест: {'✓' if report.is_reproducible else '✗'}")
        logger.info(f"Уверенность: {report.confidence_score:.2f}")
        
        if report.statistics:
            logger.info("Статистический анализ:")
            for metric, stats in report.statistics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    logger.info(f"  {metric}: среднее={stats['mean']:.2f}, std={stats['std']:.2f}")


def demonstrate_quick_check():
    """Демонстрация быстрой проверки воспроизводимости."""
    logger.info("\n=== Демонстрация быстрой проверки ===")
    
    # Быстрая проверка системы
    logger.info("Выполнение быстрой проверки воспроизводимости системы...")
    
    is_reproducible = quick_reproducibility_check(
        experiment_id="quick_demo",
        num_runs=3,
        seed=42
    )
    
    logger.info(f"Быстрая проверка: {'✓ Система воспроизводима' if is_reproducible else '✗ Проблемы с воспроизводимостью'}")


def demonstrate_config_validation():
    """Демонстрация валидации конфигурации эксперимента."""
    logger.info("\n=== Демонстрация валидации конфигурации ===")
    
    # Создаем конфигурацию с потенциальными проблемами
    config = RLConfig(
        experiment_name="config_validation_demo",
        seed=42,
        algorithm=AlgorithmConfig(
            name="PPO",
            seed=123,  # Разный сид!
            use_sde=True  # Стохастические дифференциальные уравнения
        ),
        environment=EnvironmentConfig(name="CartPole-v1")
    )
    
    logger.info("Валидация конфигурации с потенциальными проблемами...")
    
    is_valid = validate_experiment_reproducibility(
        config=config,
        num_validation_runs=2
    )
    
    logger.info(f"Конфигурация валидна: {'✓' if is_valid else '✗'}")
    
    # Исправляем конфигурацию
    config.algorithm.seed = 42  # Синхронизируем сиды
    config.algorithm.use_sde = False  # Отключаем SDE
    config.enforce_seed_consistency()  # Принудительная синхронизация
    
    logger.info("Валидация исправленной конфигурации...")
    
    is_valid_fixed = validate_experiment_reproducibility(
        config=config,
        num_validation_runs=2
    )
    
    logger.info(f"Исправленная конфигурация валидна: {'✓' if is_valid_fixed else '✗'}")


def demonstrate_reproducibility_guide():
    """Демонстрация генерации руководства по воспроизводимости."""
    logger.info("\n=== Демонстрация генерации руководства ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        checker = ReproducibilityChecker(project_root=project_root)
        
        logger.info("Генерация руководства по воспроизводимости...")
        
        guide = checker.generate_reproducibility_guide()
        
        # Сохраняем руководство
        guide_path = project_root / "reproducibility_guide.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        logger.info(f"Руководство сохранено: {guide_path}")
        logger.info(f"Размер руководства: {len(guide)} символов")
        
        # Показываем первые несколько строк
        lines = guide.split('\n')
        logger.info("Начало руководства:")
        for line in lines[:5]:
            if line.strip():
                logger.info(f"  {line}")


def main():
    """Главная функция с демонстрацией всех возможностей."""
    logger.info("🚀 Демонстрация системы проверки воспроизводимости RL экспериментов")
    logger.info("=" * 80)
    
    try:
        # Базовая проверка воспроизводимости
        demonstrate_basic_reproducibility_check()
        
        # Валидация детерминизма
        demonstrate_determinism_validation()
        
        # Сравнение разных сидов
        demonstrate_different_seeds_comparison()
        
        # Автоматическое тестирование
        demonstrate_automatic_testing()
        
        # Быстрая проверка
        demonstrate_quick_check()
        
        # Валидация конфигурации
        demonstrate_config_validation()
        
        # Генерация руководства
        demonstrate_reproducibility_guide()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Демонстрация завершена успешно!")
        logger.info("\nОсновные возможности системы:")
        logger.info("• Проверка воспроизводимости экспериментов")
        logger.info("• Валидация детерминизма функций")
        logger.info("• Статистический анализ результатов")
        logger.info("• Диагностика проблем с сидами и зависимостями")
        logger.info("• Генерация отчетов и рекомендаций")
        logger.info("• Автоматическое тестирование")
        logger.info("• Валидация конфигураций")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()