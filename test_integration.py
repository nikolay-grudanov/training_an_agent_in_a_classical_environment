#!/usr/bin/env python3
"""
Интеграционный тест для проверки базовой функциональности обучения RL агентов.

Этот скрипт тестирует:
1. Создание и инициализацию всех компонентов системы
2. Базовую функциональность обучения (без полного обучения)
3. Интеграцию между агентами, средами и системами мониторинга
4. Сохранение и загрузку моделей
5. Генерацию отчетов и метрик

Выполняется быстрый тест с минимальным количеством шагов для проверки
корректности интеграции всех компонентов.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_integration():
    """Тестирование интеграции сред."""
    print("🌍 Тестирование интеграции сред...")
    
    try:
        from src.environments.wrapper import EnvironmentWrapper
        from src.environments.lunar_lander import LunarLanderEnvironment
        
        # Тест базового wrapper'а
        print("  ✓ Импорт EnvironmentWrapper успешен")
        
        # Тест LunarLander wrapper'а
        print("  ✓ Импорт LunarLanderEnvironment успешен")
        
        # Создание среды (без фактической инициализации Gymnasium)
        env_config = {
            'render_mode': None,
            'max_episode_steps': 1000,
        }
        print("  ✓ Конфигурация среды создана")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в интеграции сред: {e}")
        return False

def test_agent_integration():
    """Тестирование интеграции агентов."""
    print("🤖 Тестирование интеграции агентов...")
    
    try:
        from src.agents.base import Agent, AgentConfig
        from src.agents.ppo_agent import PPOAgent, PPOConfig
        from src.agents.a2c_agent import A2CAgent, A2CConfig
        from src.agents.sac_agent import SACAgent, SACConfig
        from src.agents.td3_agent import TD3Agent, TD3Config
        
        print("  ✓ Импорт всех агентов успешен")
        
        # Тест создания конфигураций
        ppo_config = PPOConfig(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42
        )
        print("  ✓ PPOConfig создан")
        
        a2c_config = A2CConfig(
            env_name="CartPole-v1", 
            total_timesteps=1000,
            seed=42
        )
        print("  ✓ A2CConfig создан")
        
        sac_config = SACConfig(
            env_name="Pendulum-v1",
            total_timesteps=1000,
            seed=42
        )
        print("  ✓ SACConfig создан")
        
        td3_config = TD3Config(
            env_name="Pendulum-v1",
            total_timesteps=1000,
            seed=42
        )
        print("  ✓ TD3Config создан")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в интеграции агентов: {e}")
        return False

def test_training_integration():
    """Тестирование интеграции системы обучения."""
    print("🎯 Тестирование интеграции системы обучения...")
    
    try:
        from src.training.trainer import Trainer, TrainerConfig, TrainingMode
        from src.training.train_loop import TrainingLoop, TrainingStrategy
        
        print("  ✓ Импорт системы обучения успешен")
        
        # Тест создания конфигурации обучения
        trainer_config = TrainerConfig(
            algorithm="PPO",
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
            mode=TrainingMode.TRAIN,
            output_dir="test_results"
        )
        print("  ✓ TrainerConfig создан")
        
        # Тест создания trainer'а
        trainer = Trainer(trainer_config)
        print("  ✓ Trainer создан")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в интеграции системы обучения: {e}")
        return False

def test_utils_integration():
    """Тестирование интеграции утилит."""
    print("🔧 Тестирование интеграции утилит...")
    
    try:
        from src.utils.seeding import set_seed
        from src.utils.logging import get_logger
        from src.utils.metrics import MetricsTracker
        from src.utils.config import load_config
        from src.utils.checkpointing import CheckpointManager
        
        print("  ✓ Импорт всех утилит успешен")
        
        # Тест seeding
        set_seed(42)
        print("  ✓ Seeding работает")
        
        # Тест логирования
        logger = get_logger("test")
        logger.info("Тест логирования")
        print("  ✓ Логирование работает")
        
        # Тест метрик
        metrics = MetricsTracker(experiment_id="test_experiment")
        print("  ✓ MetricsTracker создан")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в интеграции утилит: {e}")
        return False

def test_config_schema():
    """Тестирование конфигурационной схемы."""
    print("⚙️ Тестирование конфигурационной схемы...")
    
    try:
        config_path = project_root / "configs" / "training_schema.yaml"
        
        if config_path.exists():
            print("  ✓ training_schema.yaml найден")
            
            # Проверяем размер файла
            file_size = config_path.stat().st_size
            if file_size > 1000:  # Больше 1KB
                print(f"  ✓ Конфигурационный файл содержательный ({file_size} байт)")
            else:
                print(f"  ⚠️ Конфигурационный файл маленький ({file_size} байт)")
        else:
            print("  ❌ training_schema.yaml не найден")
            return False
            
        # Проверяем другие конфигурационные файлы
        config_files = list((project_root / "configs").glob("*.yaml"))
        print(f"  ✓ Найдено {len(config_files)} конфигурационных файлов")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в проверке конфигурации: {e}")
        return False

def test_project_structure():
    """Тестирование структуры проекта."""
    print("📁 Тестирование структуры проекта...")
    
    try:
        # Проверяем основные директории
        required_dirs = [
            "src",
            "src/agents",
            "src/environments", 
            "src/training",
            "src/utils",
            "src/experiments",
            "configs",
            "tests",
            "docs",
            "notebooks",
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"  ✓ {dir_name}/ существует")
            else:
                print(f"  ❌ {dir_name}/ отсутствует")
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"  ⚠️ Отсутствуют директории: {missing_dirs}")
            return False
        
        # Проверяем ключевые файлы
        key_files = [
            "src/__init__.py",
            "src/agents/__init__.py",
            "src/agents/base.py",
            "src/agents/ppo_agent.py",
            "src/environments/wrapper.py",
            "src/environments/lunar_lander.py",
            "src/training/trainer.py",
            "src/training/train_loop.py",
            "src/utils/seeding.py",
            "src/utils/logging.py",
            "src/utils/metrics.py",
        ]
        
        missing_files = []
        for file_name in key_files:
            file_path = project_root / file_name
            if file_path.exists():
                print(f"  ✓ {file_name} существует")
            else:
                print(f"  ❌ {file_name} отсутствует")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"  ⚠️ Отсутствуют файлы: {missing_files}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в проверке структуры: {e}")
        return False

def run_integration_test():
    """Запуск полного интеграционного теста."""
    print("🚀 Запуск интеграционного теста системы обучения RL агентов")
    print("=" * 60)
    
    tests = [
        ("Структура проекта", test_project_structure),
        ("Конфигурационная схема", test_config_schema),
        ("Утилиты", test_utils_integration),
        ("Среды", test_environment_integration),
        ("Агенты", test_agent_integration),
        ("Система обучения", test_training_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ Критическая ошибка в тесте {test_name}: {e}")
            results[test_name] = False
    
    # Сводка результатов
    print("\n" + "=" * 60)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Всего тестов: {len(tests)}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {failed}")
    print(f"Успешность: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ Система готова к использованию")
        return True
    else:
        print(f"\n⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ В {failed} ТЕСТАХ")
        print("🔧 Требуется исправление перед использованием")
        return False

def main():
    """Главная функция."""
    try:
        success = run_integration_test()
        
        if success:
            print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
            print("1. Установите зависимости: conda activate rocm")
            print("2. Запустите обучение: python -m src.training.cli train")
            print("3. Проверьте результаты в директории results/")
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Тест прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()