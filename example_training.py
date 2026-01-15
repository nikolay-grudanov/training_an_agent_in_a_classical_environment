#!/usr/bin/env python3
"""
Пример запуска обучения PPO агента на CartPole-v1.

Этот скрипт демонстрирует базовое использование системы обучения RL агентов.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from src.training import Trainer, TrainerConfig
from src.agents import PPOConfig


def main():
    print("🚀 Запуск примера обучения PPO агента на CartPole-v1")
    print("=" * 60)

    # Создание конфигурации для обучения
    config = TrainerConfig(
        experiment_name="ppo_cartpole_example",
        algorithm="PPO",
        environment_name="CartPole-v1",
        total_timesteps=10000,  # Уменьшено для быстрого примера
        seed=42,
        
        # Настройки оценки
        eval_freq=2000,
        n_eval_episodes=5,
        
        # Настройки сохранения
        save_freq=5000,
        
        # Пути
        output_dir="results/example",
        
        # Мониторинг
        verbose=1,
        progress_bar=True,
    )

    print(f"📋 Конфигурация создана:")
    print(f"   • Эксперимент: {config.experiment_name}")
    print(f"   • Алгоритм: {config.algorithm}")
    print(f"   • Среда: {config.environment_name}")
    print(f"   • Шаги обучения: {config.total_timesteps:,}")
    print(f"   • Seed: {config.seed}")
    print()

    # Настройка базового логирования
    from src.utils.logging import setup_logging
    setup_logging(log_level="INFO", console_output=True)

    # Создание и запуск тренера
    print("🤖 Создание тренера...")
    with Trainer(config) as trainer:
        print("🎯 Запуск обучения...")
        result = trainer.train()

        if result.success:
            print("\n✅ Обучение завершено успешно!")
            print(f"📊 Финальная награда: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
            print(f"🏆 Лучшая награда: {result.best_mean_reward:.2f}")
            print(f"⏱️  Время обучения: {result.training_time:.1f} сек")
            print(f"💾 Модель сохранена: {result.model_path}")

            # Дополнительная оценка
            print("\n🔍 Дополнительная оценка...")
            eval_result = trainer.evaluate(n_episodes=10, render=False)
            print(f"📈 Средняя награда: {eval_result['mean_reward']:.2f}")
            print(f"📏 Средняя длина эпизода: {eval_result['mean_length']:.1f}")
        else:
            print(f"\n❌ Обучение завершилось с ошибкой: {result.error_message}")

    print("\n🎉 Пример завершен!")


if __name__ == "__main__":
    main()