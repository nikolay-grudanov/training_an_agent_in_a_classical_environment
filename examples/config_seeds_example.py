#!/usr/bin/env python3
"""Пример использования улучшенной функциональности управления сидами в конфигурации.

Этот скрипт демонстрирует:
1. Загрузку конфигурации с автоматической синхронизацией сидов
2. Валидацию настроек воспроизводимости
3. Применение сидов к системе
4. Генерацию отчетов о воспроизводимости
5. Проверку консистентности между несколькими конфигурациями
"""

import logging
from pathlib import Path
from typing import List

from src.utils.config import (
    RLConfig, AlgorithmConfig, EnvironmentConfig, ReproducibilityConfig,
    load_config_with_seeds, create_reproducibility_report,
    validate_configs_seed_consistency, enforce_global_seed_consistency
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_seed_consistency():
    """Демонстрация принудительной синхронизации сидов."""
    logger.info("=== Демонстрация синхронизации сидов ===")
    
    # Создаем конфигурацию с конфликтующими сидами
    config = RLConfig(
        experiment_name="seed_consistency_demo",
        seed=42,
        algorithm=AlgorithmConfig(name="PPO", seed=123),  # Конфликтующий сид
        reproducibility=ReproducibilityConfig(
            seed=456,  # Еще один конфликтующий сид
            enforce_seed_consistency=True,
            auto_propagate_seeds=True
        )
    )
    
    logger.info(f"До синхронизации:")
    logger.info(f"  Основной сид: {config.seed}")
    logger.info(f"  Сид алгоритма: {config.algorithm.seed}")
    logger.info(f"  Сид воспроизводимости: {config.reproducibility.seed}")
    
    # Принудительная синхронизация (вызывается автоматически при auto_propagate_seeds=True)
    config.enforce_seed_consistency()
    
    logger.info(f"После синхронизации:")
    logger.info(f"  Основной сид: {config.seed}")
    logger.info(f"  Сид алгоритма: {config.algorithm.seed}")
    logger.info(f"  Сид воспроизводимости: {config.reproducibility.seed}")
    
    return config


def demonstrate_reproducibility_validation():
    """Демонстрация валидации настроек воспроизводимости."""
    logger.info("\n=== Демонстрация валидации воспроизводимости ===")
    
    # Создаем конфигурацию с потенциальными проблемами
    config = RLConfig(
        experiment_name="reproducibility_validation_demo",
        seed=42,
        algorithm=AlgorithmConfig(
            name="PPO",
            seed=42,
            use_sde=True,  # Может снизить воспроизводимость
            device="auto"  # Может привести к различному поведению
        ),
        reproducibility=ReproducibilityConfig(
            seed=42,
            deterministic=True,
            benchmark=True,  # Конфликт с deterministic
            validate_determinism=True,
            warn_on_seed_conflicts=True
        )
    )
    
    # Валидируем настройки
    is_valid, warnings = config.validate_reproducibility()
    
    logger.info(f"Результат валидации: {'✓ Валидна' if is_valid else '✗ Есть проблемы'}")
    logger.info(f"Количество предупреждений: {len(warnings)}")
    
    for i, warning in enumerate(warnings, 1):
        logger.warning(f"  {i}. {warning}")
    
    return config


def demonstrate_reproducibility_report():
    """Демонстрация генерации отчета о воспроизводимости."""
    logger.info("\n=== Демонстрация отчета о воспроизводимости ===")
    
    config = RLConfig(
        experiment_name="reproducibility_report_demo",
        seed=42,
        algorithm=AlgorithmConfig(name="PPO", seed=42),
        reproducibility=ReproducibilityConfig(seed=42, deterministic=True)
    )
    
    # Генерируем отчет
    report = config.get_reproducibility_report()
    
    logger.info("Отчет о воспроизводимости:")
    logger.info(f"  Эксперимент: {report['experiment_name']}")
    logger.info(f"  Статус: {'✓ Валидна' if report['is_valid'] else '✗ Есть проблемы'}")
    logger.info(f"  Основной сид: {report['seeds']['main_seed']}")
    logger.info(f"  Сиды консистентны: {report['seeds']['seeds_consistent']}")
    logger.info(f"  Детерминистический режим: {report['determinism']['deterministic_mode']}")
    logger.info(f"  Алгоритм: {report['algorithm']['algorithm_name']}")
    
    if report['warnings']:
        logger.info(f"  Предупреждения ({len(report['warnings'])}):")
        for warning in report['warnings']:
            logger.warning(f"    - {warning}")
    
    if report['recommendations']:
        logger.info(f"  Рекомендации:")
        for rec in report['recommendations']:
            logger.info(f"    - {rec}")
    
    # Сохраняем отчет в файл
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "reproducibility_report.json"
    
    create_reproducibility_report(config, report_file)
    logger.info(f"Отчет сохранен в: {report_file}")
    
    return report


def demonstrate_apply_seeds():
    """Демонстрация применения сидов к системе."""
    logger.info("\n=== Демонстрация применения сидов ===")
    
    config = RLConfig(
        experiment_name="apply_seeds_demo",
        seed=42,
        reproducibility=ReproducibilityConfig(
            seed=42,
            deterministic=True,
            validate_determinism=True
        )
    )
    
    try:
        # Применяем сиды к системе
        logger.info("Применение сидов к системе...")
        config.apply_seeds()
        logger.info("✓ Сиды успешно применены")
        
        # Проверяем, что сиды работают
        import numpy as np
        import random
        
        # Генерируем несколько случайных чисел для демонстрации
        logger.info("Демонстрация детерминистического поведения:")
        logger.info(f"  NumPy random: {np.random.random():.6f}")
        logger.info(f"  Python random: {random.random():.6f}")
        
        # Сбрасываем сид и проверяем воспроизводимость
        config.apply_seeds()
        logger.info("После повторного применения сидов:")
        logger.info(f"  NumPy random: {np.random.random():.6f}")
        logger.info(f"  Python random: {random.random():.6f}")
        
    except Exception as e:
        logger.error(f"Ошибка при применении сидов: {e}")


def demonstrate_multi_config_consistency():
    """Демонстрация проверки консистентности между конфигурациями."""
    logger.info("\n=== Демонстрация консистентности между конфигурациями ===")
    
    # Создаем несколько конфигураций
    configs = [
        RLConfig(
            experiment_name="ppo_experiment",
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=42)
        ),
        RLConfig(
            experiment_name="a2c_experiment", 
            seed=42,
            algorithm=AlgorithmConfig(name="A2C", seed=42)
        ),
        RLConfig(
            experiment_name="sac_experiment",
            seed=123,  # Другой сид!
            algorithm=AlgorithmConfig(name="SAC", seed=123)
        )
    ]
    
    logger.info(f"Создано {len(configs)} конфигураций:")
    for config in configs:
        logger.info(f"  {config.experiment_name}: seed={config.seed}")
    
    # Принудительная синхронизация сидов
    logger.info("\nПринудительная синхронизация сидов...")
    synced_configs = enforce_global_seed_consistency(configs, master_seed=777)
    
    logger.info("После синхронизации:")
    for config in synced_configs:
        logger.info(f"  {config.experiment_name}: seed={config.seed}")


def demonstrate_config_loading_with_seeds():
    """Демонстрация загрузки конфигурации с автоматическим применением сидов."""
    logger.info("\n=== Демонстрация загрузки конфигурации с сидами ===")
    
    # Создаем временную конфигурацию
    config_dir = Path("temp_configs")
    config_dir.mkdir(exist_ok=True)
    
    config_data = {
        'experiment_name': 'seed_loading_demo',
        'seed': 42,
        'algorithm': {
            'name': 'PPO',
            'learning_rate': 3e-4,
            'seed': 42
        },
        'environment': {
            'name': 'LunarLander-v3'
        },
        'reproducibility': {
            'seed': 42,
            'deterministic': True,
            'validate_determinism': True,
            'auto_propagate_seeds': True
        }
    }
    
    config_file = config_dir / "demo_config.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    try:
        # Загружаем конфигурацию с автоматическим применением сидов
        logger.info(f"Загрузка конфигурации из {config_file}")
        config = load_config_with_seeds(
            config_path=config_file,
            apply_seeds=True,
            validate_reproducibility=True
        )
        
        logger.info("✓ Конфигурация загружена и сиды применены")
        logger.info(f"  Эксперимент: {config.experiment_name}")
        logger.info(f"  Сид: {config.seed}")
        logger.info(f"  Алгоритм: {config.algorithm.name}")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
    
    finally:
        # Очищаем временные файлы
        import shutil
        if config_dir.exists():
            shutil.rmtree(config_dir)


def main():
    """Главная функция с демонстрацией всех возможностей."""
    logger.info("Демонстрация улучшенной функциональности управления сидами")
    logger.info("=" * 70)
    
    try:
        # 1. Синхронизация сидов
        config1 = demonstrate_seed_consistency()
        
        # 2. Валидация воспроизводимости
        config2 = demonstrate_reproducibility_validation()
        
        # 3. Отчет о воспроизводимости
        report = demonstrate_reproducibility_report()
        
        # 4. Применение сидов
        demonstrate_apply_seeds()
        
        # 5. Консистентность между конфигурациями
        demonstrate_multi_config_consistency()
        
        # 6. Загрузка конфигурации с сидами
        demonstrate_config_loading_with_seeds()
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Демонстрация завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка во время демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()