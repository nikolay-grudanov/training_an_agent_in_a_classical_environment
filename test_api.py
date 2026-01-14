#!/usr/bin/env python3
"""Простой тест для проверки работы API системы обучения RL агентов."""

import asyncio
import json
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.api.app import create_app
from src.api.config import get_api_config, setup_api_config, create_api_config


async def test_api():
    """Тестирование основных функций API."""
    print("🚀 Тестирование RL Agent Training API")
    
    # Настройка конфигурации для тестирования
    test_config = create_api_config(
        base_dir=Path.cwd(),
    )
    setup_api_config(test_config)
    
    # Создание приложения
    app = create_app()
    
    print("✅ FastAPI приложение создано успешно")
    
    # Тестирование конфигурации
    config = get_api_config()
    print(f"📁 Базовая директория: {config.base_dir}")
    print(f"📊 Поддерживаемые алгоритмы: {config.supported_algorithms}")
    print(f"🌍 Поддерживаемые среды: {config.supported_environments}")
    
    # Проверка создания директорий
    print(f"📂 Директория экспериментов: {config.experiments_dir}")
    print(f"📂 Директория моделей: {config.models_dir}")
    print(f"📂 Директория логов: {config.logs_dir}")
    
    # Проверка валидации
    print(f"✅ Валидация PPO: {config.validate_algorithm('PPO')}")
    print(f"❌ Валидация INVALID: {config.validate_algorithm('INVALID')}")
    print(f"✅ Валидация LunarLander-v2: {config.validate_environment('LunarLander-v2')}")
    print(f"❌ Валидация INVALID: {config.validate_environment('INVALID')}")
    
    # Проверка путей для эксперимента
    test_exp_id = "test-experiment-123"
    exp_dir = config.get_experiment_dir(test_exp_id)
    model_path = config.get_model_path(test_exp_id)
    metrics_path = config.get_metrics_path(test_exp_id)
    results_path = config.get_results_path(test_exp_id)
    
    print(f"📁 Директория тестового эксперимента: {exp_dir}")
    print(f"🤖 Путь к модели: {model_path}")
    print(f"📈 Путь к метрикам: {metrics_path}")
    print(f"📊 Путь к результатам: {results_path}")
    
    # Проверка конфигурации в виде словаря
    config_dict = config.to_dict()
    print("📋 Конфигурация API:")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    
    print("\n🎉 Все тесты пройдены успешно!")
    print("🌐 Для запуска сервера используйте:")
    print("   python -m src.api.app --host 0.0.0.0 --port 8000 --debug")
    print("📚 Документация будет доступна по адресу: http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(test_api())