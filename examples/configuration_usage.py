"""Примеры использования класса Configuration для управления конфигурацией RL экспериментов.

Этот файл демонстрирует различные способы создания, валидации, сравнения
и сериализации конфигураций экспериментов.
"""

import tempfile
from pathlib import Path

from src.experiments.config import (
    Configuration,
    compare_configs,
    create_a2c_config,
    create_ppo_config,
    create_sac_config,
    create_td3_config,
)


def example_basic_usage() -> None:
    """Пример базового использования Configuration."""
    print("=== Базовое использование Configuration ===")
    
    # Создание конфигурации с минимальными параметрами
    config = Configuration(
        algorithm="PPO",
        environment="LunarLander-v2"
    )
    
    print(f"Создана конфигурация: {config.experiment_name}")
    print(f"Алгоритм: {config.algorithm}")
    print(f"Среда: {config.environment}")
    print(f"Шагов обучения: {config.training_steps:,}")
    print(f"Количество гиперпараметров: {len(config.hyperparameters)}")
    print()


def example_custom_configuration() -> None:
    """Пример создания кастомной конфигурации."""
    print("=== Кастомная конфигурация ===")
    
    # Создание конфигурации с пользовательскими параметрами
    config = Configuration(
        algorithm="A2C",
        environment="Pendulum-v1",
        hyperparameters={
            "learning_rate": 1e-3,
            "gamma": 0.95,
            "n_steps": 10  # Переопределяем значение по умолчанию
        },
        seed=123,
        training_steps=200_000,
        evaluation_frequency=20_000,
        experiment_name="custom_a2c_experiment",
        description="Эксперимент с кастомными параметрами A2C"
    )
    
    print(str(config))
    print(f"Learning rate: {config.hyperparameters['learning_rate']}")
    print(f"N steps: {config.hyperparameters['n_steps']}")
    print()


def example_factory_functions() -> None:
    """Пример использования фабричных функций."""
    print("=== Фабричные функции ===")
    
    # Создание конфигураций с помощью фабричных функций
    ppo_config = create_ppo_config(
        environment="LunarLander-v2",
        experiment_name="ppo_baseline",
        training_steps=150_000
    )
    
    a2c_config = create_a2c_config(
        environment="LunarLander-v2",
        experiment_name="a2c_baseline",
        training_steps=150_000
    )
    
    sac_config = create_sac_config(
        environment="Pendulum-v1",
        experiment_name="sac_continuous",
        training_steps=100_000
    )
    
    td3_config = create_td3_config(
        environment="Pendulum-v1",
        experiment_name="td3_continuous",
        training_steps=100_000
    )
    
    configs = [ppo_config, a2c_config, sac_config, td3_config]
    
    for config in configs:
        print(f"{config.algorithm}: {config.experiment_name}")
        print(f"  Среда: {config.environment}")
        print(f"  Learning rate: {config.hyperparameters['learning_rate']}")
        print()


def example_configuration_comparison() -> None:
    """Пример сравнения конфигураций."""
    print("=== Сравнение конфигураций ===")
    
    # Создаем две похожие конфигурации
    baseline_config = create_ppo_config(
        environment="LunarLander-v2",
        experiment_name="ppo_baseline",
        seed=42
    )
    
    variant_config = create_ppo_config(
        environment="LunarLander-v2",
        experiment_name="ppo_variant",
        seed=42
    )
    
    # Изменяем learning rate в варианте
    variant_config.hyperparameters["learning_rate"] = 1e-3
    variant_config.training_steps = 200_000
    
    # Сравниваем конфигурации
    differences = baseline_config.get_differences(variant_config)
    
    print("Различия между baseline и variant:")
    for field, diff in differences.items():
        print(f"  {field}:")
        if isinstance(diff, dict) and 'self' in diff:
            print(f"    baseline: {diff['self']}")
            print(f"    variant: {diff['other']}")
        else:
            print(f"    различия в гиперпараметрах: {len(diff) if isinstance(diff, dict) else 'N/A'}")
    
    # Используем функцию сравнения
    comparison = compare_configs(baseline_config, variant_config)
    print(f"\nИдентичные конфигурации: {comparison['identical']}")
    print(f"Количество различий: {comparison['differences_count']}")
    print()


def example_configuration_merge() -> None:
    """Пример объединения конфигураций."""
    print("=== Объединение конфигураций ===")
    
    # Базовая конфигурация
    base_config = Configuration(
        algorithm="PPO",
        environment="LunarLander-v2",
        experiment_name="base_experiment"
    )
    
    # Конфигурация с изменениями
    override_config = Configuration(
        algorithm="PPO",
        environment="LunarLander-v2",
        experiment_name="override_experiment",
        training_steps=300_000,
        seed=999
    )
    
    # Изменяем некоторые гиперпараметры
    override_config.hyperparameters["learning_rate"] = 5e-4
    override_config.hyperparameters["custom_param"] = "test_value"
    
    # Объединяем конфигурации
    merged_config = base_config.merge(override_config)
    
    print("Объединенная конфигурация:")
    print(f"  Название: {merged_config.experiment_name}")
    print(f"  Шагов обучения: {merged_config.training_steps:,}")
    print(f"  Seed: {merged_config.seed}")
    print(f"  Learning rate: {merged_config.hyperparameters['learning_rate']}")
    print(f"  Custom param: {merged_config.hyperparameters.get('custom_param', 'N/A')}")
    print()


def example_serialization() -> None:
    """Пример сериализации и десериализации конфигураций."""
    print("=== Сериализация конфигураций ===")
    
    # Создаем конфигурацию
    original_config = Configuration(
        algorithm="SAC",
        environment="Pendulum-v1",
        experiment_name="sac_serialization_test",
        description="Тест сериализации конфигурации SAC",
        training_steps=75_000,
        seed=456
    )
    
    # Сохраняем в YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = Path(f.name)
    
    original_config.save(yaml_path, format_type="yaml")
    print(f"Конфигурация сохранена в YAML: {yaml_path}")
    
    # Загружаем из YAML
    loaded_yaml_config = Configuration.load(yaml_path)
    print(f"Конфигурация загружена из YAML: {loaded_yaml_config.experiment_name}")
    
    # Сохраняем в JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    
    original_config.save(json_path, format_type="json")
    print(f"Конфигурация сохранена в JSON: {json_path}")
    
    # Загружаем из JSON
    loaded_json_config = Configuration.load(json_path)
    print(f"Конфигурация загружена из JSON: {loaded_json_config.experiment_name}")
    
    # Проверяем, что все конфигурации идентичны
    assert original_config == loaded_yaml_config == loaded_json_config
    print("✓ Все конфигурации идентичны после сериализации")
    
    # Очищаем временные файлы
    yaml_path.unlink()
    json_path.unlink()
    print()


def example_configuration_copy() -> None:
    """Пример создания копий конфигурации."""
    print("=== Копирование конфигураций ===")
    
    # Создаем оригинальную конфигурацию
    original = create_td3_config(
        environment="Pendulum-v1",
        experiment_name="td3_original"
    )
    
    # Создаем копию
    copy_config = original.copy()
    
    print(f"Оригинал: {original.experiment_name}")
    print(f"Копия: {copy_config.experiment_name}")
    print(f"Идентичны: {original == copy_config}")
    print(f"Разные объекты: {original is not copy_config}")
    
    # Изменяем копию
    copy_config.experiment_name = "td3_modified_copy"
    copy_config.hyperparameters["learning_rate"] = 1e-3
    copy_config.training_steps = 50_000
    
    print(f"\nПосле изменения копии:")
    print(f"Оригинал: {original.experiment_name}, LR: {original.hyperparameters['learning_rate']}")
    print(f"Копия: {copy_config.experiment_name}, LR: {copy_config.hyperparameters['learning_rate']}")
    print(f"Идентичны: {original == copy_config}")
    print()


def example_algorithm_defaults() -> None:
    """Пример работы с настройками по умолчанию для алгоритмов."""
    print("=== Настройки алгоритмов по умолчанию ===")
    
    algorithms = ["PPO", "A2C", "SAC", "TD3"]
    
    for algorithm in algorithms:
        defaults = Configuration.get_algorithm_defaults(algorithm)
        print(f"{algorithm} (параметров: {len(defaults)}):")
        
        # Показываем основные параметры
        key_params = ["learning_rate", "gamma", "batch_size", "n_steps", "buffer_size"]
        for param in key_params:
            if param in defaults:
                print(f"  {param}: {defaults[param]}")
        print()


def example_validation_errors() -> None:
    """Пример обработки ошибок валидации."""
    print("=== Обработка ошибок валидации ===")
    
    try:
        # Попытка создать конфигурацию с невалидными параметрами
        invalid_config = Configuration(
            algorithm="PPO",
            environment="LunarLander-v2",
            seed=-1,  # Невалидный seed
            training_steps=0,  # Невалидное количество шагов
            evaluation_frequency=200_000,  # Больше чем training_steps
            experiment_name=""  # Пустое название
        )
    except Exception as e:
        print(f"Поймана ошибка валидации: {type(e).__name__}")
        print(f"Сообщение: {e}")
    
    try:
        # Попытка создать конфигурацию с неподдерживаемым алгоритмом
        invalid_algorithm_config = Configuration(
            algorithm="INVALID_ALGO",
            environment="LunarLander-v2"
        )
    except Exception as e:
        print(f"\nПоймана ошибка алгоритма: {type(e).__name__}")
        print(f"Сообщение: {e}")
    
    print()


def main() -> None:
    """Запуск всех примеров."""
    print("🚀 Примеры использования Configuration для RL экспериментов\n")
    
    example_basic_usage()
    example_custom_configuration()
    example_factory_functions()
    example_configuration_comparison()
    example_configuration_merge()
    example_serialization()
    example_configuration_copy()
    example_algorithm_defaults()
    example_validation_errors()
    
    print("✅ Все примеры выполнены успешно!")


if __name__ == "__main__":
    main()