#!/usr/bin/env python3
"""Базовый тест функциональности тренера без внешних зависимостей."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Мокаем все внешние зависимости
mock_modules = {
    'gymnasium': MagicMock(),
    'stable_baselines3': MagicMock(),
    'stable_baselines3.common': MagicMock(),
    'stable_baselines3.common.callbacks': MagicMock(),
    'stable_baselines3.common.type_aliases': MagicMock(),
    'typer': MagicMock(),
    'rich': MagicMock(),
    'rich.console': MagicMock(),
    'rich.table': MagicMock(),
    'rich.progress': MagicMock(),
    'rich.panel': MagicMock(),
}

with patch.dict('sys.modules', mock_modules):
    # Теперь можем импортировать наши модули
    from training.trainer import TrainerConfig, TrainingMode, TrainingResult


def test_trainer_config_basic():
    """Тест базовой функциональности TrainerConfig."""
    print("🧪 Тестирование TrainerConfig...")
    
    # Тест создания конфигурации по умолчанию
    config = TrainerConfig()
    assert config.experiment_name == "default_experiment"
    assert config.algorithm == "PPO"
    assert config.environment_name == "LunarLander-v3"
    assert config.mode == TrainingMode.TRAIN
    assert config.total_timesteps == 100_000
    assert config.seed == 42
    print("✅ Конфигурация по умолчанию создана корректно")
    
    # Тест пользовательской конфигурации
    custom_config = TrainerConfig(
        experiment_name="test_experiment",
        algorithm="A2C",
        environment_name="CartPole-v1",
        total_timesteps=50_000,
        seed=123,
    )
    assert custom_config.experiment_name == "test_experiment"
    assert custom_config.algorithm == "A2C"
    assert custom_config.environment_name == "CartPole-v1"
    assert custom_config.total_timesteps == 50_000
    assert custom_config.seed == 123
    print("✅ Пользовательская конфигурация создана корректно")
    
    # Тест валидации
    try:
        TrainerConfig(algorithm="INVALID")
        assert False, "Должна была быть ошибка валидации"
    except ValueError as e:
        assert "Неподдерживаемый алгоритм" in str(e)
        print("✅ Валидация алгоритма работает")
    
    try:
        TrainerConfig(total_timesteps=0)
        assert False, "Должна была быть ошибка валидации"
    except ValueError as e:
        assert "total_timesteps должен быть > 0" in str(e)
        print("✅ Валидация timesteps работает")
    
    # Тест настройки путей
    with tempfile.TemporaryDirectory() as temp_dir:
        path_config = TrainerConfig(
            experiment_name="test_paths",
            output_dir=temp_dir,
        )
        assert path_config.model_save_path is not None
        assert path_config.logs_dir is not None
        assert path_config.tensorboard_log is not None
        
        # Проверяем, что директории созданы
        assert Path(path_config.logs_dir).exists()
        assert Path(path_config.tensorboard_log).exists()
        print("✅ Настройка путей работает")


def test_training_mode():
    """Тест enum TrainingMode."""
    print("🧪 Тестирование TrainingMode...")
    
    assert TrainingMode.TRAIN.value == "train"
    assert TrainingMode.RESUME.value == "resume"
    assert TrainingMode.EVALUATE.value == "evaluate"
    assert TrainingMode.FINETUNE.value == "finetune"
    print("✅ Значения TrainingMode корректны")
    
    # Тест создания из строки
    assert TrainingMode("train") == TrainingMode.TRAIN
    assert TrainingMode("resume") == TrainingMode.RESUME
    print("✅ Создание TrainingMode из строки работает")


def test_training_result():
    """Тест TrainingResult."""
    print("🧪 Тестирование TrainingResult...")
    
    result = TrainingResult(
        success=True,
        total_timesteps=100_000,
        training_time=300.5,
        final_mean_reward=250.0,
        final_std_reward=50.0,
        experiment_name="test_exp",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        seed=42,
    )
    
    assert result.success is True
    assert result.total_timesteps == 100_000
    assert result.training_time == 300.5
    assert result.final_mean_reward == 250.0
    assert result.experiment_name == "test_exp"
    print("✅ TrainingResult создан корректно")
    
    # Тест преобразования в словарь
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["total_timesteps"] == 100_000
    assert result_dict["experiment_name"] == "test_exp"
    print("✅ Преобразование в словарь работает")
    
    # Тест сохранения
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = Path(temp_dir) / "test_result.yaml"
        result.save(result_path)
        assert result_path.exists()
        print("✅ Сохранение результата работает")


def test_supported_algorithms():
    """Тест поддерживаемых алгоритмов."""
    print("🧪 Тестирование поддерживаемых алгоритмов...")
    
    supported = ["PPO", "A2C", "SAC", "TD3"]
    
    for algorithm in supported:
        config = TrainerConfig(algorithm=algorithm)
        assert config.algorithm == algorithm
        print(f"✅ Алгоритм {algorithm} поддерживается")
    
    # Тест нормализации регистра
    config = TrainerConfig(algorithm="ppo")
    assert config.algorithm == "PPO"
    print("✅ Нормализация регистра работает")


def main():
    """Главная функция тестирования."""
    print("🎮 Запуск базовых тестов системы обучения")
    print("=" * 50)
    
    try:
        test_trainer_config_basic()
        test_training_mode()
        test_training_result()
        test_supported_algorithms()
        
        print("\n🎉 Все тесты пройдены успешно!")
        print("✅ Основная функциональность работает корректно")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()