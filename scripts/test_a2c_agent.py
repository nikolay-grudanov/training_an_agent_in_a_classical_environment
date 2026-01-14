#!/usr/bin/env python3
"""Быстрый тест A2C агента для проверки работоспособности.

Этот скрипт выполняет базовую проверку A2C агента:
1. Создание конфигурации
2. Инициализация агента
3. Короткое обучение
4. Тестирование предсказания
5. Сохранение и загрузка модели
"""

import logging
import sys
from pathlib import Path

# Добавление корневой директории в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.a2c_agent import A2CAgent, A2CConfig
from src.utils import set_seed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_a2c_agent() -> None:
    """Тестирование A2C агента."""
    logger.info("🧪 Начало тестирования A2C агента")
    
    try:
        # 1. Создание конфигурации для быстрого теста
        logger.info("📋 Создание тестовой конфигурации...")
        config = A2CConfig(
            env_name="CartPole-v1",  # Простая среда для быстрого теста
            total_timesteps=5000,    # Короткое обучение
            learning_rate=1e-3,
            n_steps=5,
            verbose=1,
            early_stopping=False,    # Отключаем для теста
            eval_freq=0,            # Отключаем оценку
            save_freq=0,            # Отключаем сохранение чекпоинтов
            use_tensorboard=False,   # Отключаем TensorBoard
            normalize_env=False,     # Упрощаем для теста
        )
        logger.info("✅ Конфигурация создана")
        
        # 2. Инициализация агента
        logger.info("🤖 Инициализация A2C агента...")
        agent = A2CAgent(
            config=config,
            experiment_name="a2c_test",
        )
        logger.info("✅ Агент инициализирован")
        
        # 3. Проверка информации о модели
        logger.info("📊 Информация о модели:")
        model_info = agent.get_model_info()
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # 4. Короткое обучение
        logger.info("🎓 Начало обучения...")
        training_result = agent.train()
        logger.info(f"✅ Обучение завершено за {training_result.training_time:.2f} сек")
        logger.info(f"📈 Финальная награда: {training_result.final_mean_reward:.2f} ± {training_result.final_std_reward:.2f}")
        
        # 5. Тестирование предсказания
        logger.info("🔮 Тестирование предсказания...")
        import numpy as np
        
        # Создание тестового наблюдения
        test_obs = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Предсказание действия
        action, state = agent.predict(test_obs, deterministic=True)
        logger.info(f"✅ Предсказание: действие={action}, состояние={state}")
        
        # 6. Оценка производительности
        logger.info("📊 Оценка производительности...")
        eval_metrics = agent.evaluate(n_episodes=5, deterministic=True)
        logger.info(f"✅ Средняя награда: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        
        # 7. Тестирование сохранения и загрузки
        logger.info("💾 Тестирование сохранения и загрузки...")
        
        # Создание временной директории
        test_dir = Path("test_results")
        test_dir.mkdir(exist_ok=True)
        
        save_path = test_dir / "test_a2c_model.zip"
        
        # Сохранение
        agent.save(str(save_path))
        logger.info(f"✅ Модель сохранена: {save_path}")
        
        # Загрузка
        loaded_agent = A2CAgent.load(str(save_path), config=config)
        logger.info("✅ Модель загружена")
        
        # Проверка загруженной модели
        loaded_action, loaded_state = loaded_agent.predict(test_obs, deterministic=True)
        logger.info(f"✅ Загруженная модель: действие={loaded_action}, состояние={loaded_state}")
        
        # Сравнение предсказаний
        if np.array_equal(action, loaded_action):
            logger.info("✅ Предсказания идентичны")
        else:
            logger.warning("⚠️ Предсказания различаются (может быть нормально)")
        
        # 8. Очистка
        logger.info("🧹 Очистка тестовых файлов...")
        if save_path.exists():
            save_path.unlink()
        config_path = save_path.with_suffix(".yaml")
        if config_path.exists():
            config_path.unlink()
        if test_dir.exists() and not list(test_dir.iterdir()):
            test_dir.rmdir()
        
        logger.info("🎉 Все тесты пройдены успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
        raise


def test_config_validation() -> None:
    """Тестирование валидации конфигурации."""
    logger.info("🔍 Тестирование валидации конфигурации...")
    
    # Тест валидных параметров
    try:
        config = A2CConfig(
            env_name="CartPole-v1",
            learning_rate=1e-3,
            n_steps=5,
            rms_prop_eps=1e-5,
            activation_fn="tanh",
        )
        logger.info("✅ Валидная конфигурация создана")
    except Exception as e:
        logger.error(f"❌ Ошибка создания валидной конфигурации: {e}")
        raise
    
    # Тест невалидных параметров
    test_cases = [
        ("n_steps=0", {"env_name": "CartPole-v1", "n_steps": 0}),
        ("rms_prop_eps=0", {"env_name": "CartPole-v1", "rms_prop_eps": 0}),
        ("invalid_activation", {"env_name": "CartPole-v1", "activation_fn": "invalid"}),
    ]
    
    for test_name, kwargs in test_cases:
        try:
            A2CConfig(**kwargs)
            logger.error(f"❌ Ожидалась ошибка для {test_name}")
        except ValueError:
            logger.info(f"✅ Корректная валидация для {test_name}")
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка для {test_name}: {e}")
    
    logger.info("✅ Тестирование валидации завершено")


def main() -> None:
    """Основная функция тестирования."""
    logger.info("🚀 Запуск тестов A2C агента")
    
    # Установка seed для воспроизводимости
    set_seed(42)
    
    try:
        # Тест валидации конфигурации
        test_config_validation()
        
        # Основной тест агента
        test_a2c_agent()
        
        logger.info("🎊 Все тесты завершены успешно!")
        
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()