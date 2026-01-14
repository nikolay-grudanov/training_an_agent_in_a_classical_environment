"""Пример базового использования системы обучения.

Этот пример демонстрирует основные возможности Trainer:
- Создание и настройка конфигурации
- Обучение различных алгоритмов
- Мониторинг прогресса
- Сохранение и загрузка моделей
- Оценка производительности
"""

import logging
from pathlib import Path

from src.training import (
    Trainer,
    TrainerConfig,
    TrainingMode,
    create_trainer_from_config,
)
from src.agents.base import AgentConfig
from src.utils.logging import setup_logging


def basic_ppo_training():
    """Базовое обучение PPO агента."""
    print("🚀 Запуск базового обучения PPO...")
    
    # Настройка логирования
    setup_logging(level=logging.INFO)
    
    # Создание конфигурации
    config = TrainerConfig(
        experiment_name="basic_ppo_lunarlander",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        total_timesteps=100_000,
        seed=42,
        
        # Настройки оценки
        eval_freq=10_000,
        n_eval_episodes=5,
        
        # Настройки сохранения
        save_freq=25_000,
        checkpoint_freq=20_000,
        
        # Пути
        output_dir="results/examples",
        
        # Мониторинг
        verbose=1,
        progress_bar=True,
    )
    
    # Создание и запуск тренера
    with Trainer(config) as trainer:
        result = trainer.train()
        
        if result.success:
            print(f"✅ Обучение завершено успешно!")
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
            print(f"❌ Обучение завершилось с ошибкой: {result.error_message}")
    
    return result


def compare_algorithms():
    """Сравнение различных алгоритмов."""
    print("\n🔬 Сравнение алгоритмов...")
    
    algorithms = ["PPO", "A2C"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🎯 Обучение {algorithm}...")
        
        config = TrainerConfig(
            experiment_name=f"comparison_{algorithm.lower()}",
            algorithm=algorithm,
            environment_name="LunarLander-v3",
            total_timesteps=50_000,  # Меньше шагов для быстрого сравнения
            seed=42,
            eval_freq=10_000,
            output_dir="results/comparison",
            verbose=0,  # Меньше вывода
        )
        
        with Trainer(config) as trainer:
            result = trainer.train()
            results[algorithm] = result
            
            if result.success:
                print(f"✅ {algorithm}: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
            else:
                print(f"❌ {algorithm}: Ошибка - {result.error_message}")
    
    # Сравнение результатов
    print("\n📊 Сравнение результатов:")
    print("-" * 50)
    for algorithm, result in results.items():
        if result.success:
            print(f"{algorithm:>8}: {result.final_mean_reward:>8.2f} ± {result.final_std_reward:>6.2f}")
        else:
            print(f"{algorithm:>8}: {'ОШИБКА':>15}")
    
    return results


def advanced_training_with_config():
    """Продвинутое обучение с детальной конфигурацией."""
    print("\n⚙️  Продвинутое обучение с настройкой...")
    
    # Детальная конфигурация агента
    agent_config = AgentConfig(
        algorithm="PPO",
        env_name="LunarLander-v3",
        total_timesteps=150_000,
        seed=42,
        
        # Гиперпараметры PPO
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Больше исследования
        vf_coef=0.5,
        
        # Настройки модели
        policy_kwargs={
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": "tanh",
        },
        
        verbose=1,
    )
    
    # Конфигурация тренера
    trainer_config = TrainerConfig(
        experiment_name="advanced_ppo_training",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        total_timesteps=150_000,
        seed=42,
        
        # Используем детальную конфигурацию агента
        agent_config=agent_config,
        
        # Частая оценка для мониторинга
        eval_freq=5_000,
        n_eval_episodes=10,
        eval_deterministic=True,
        
        # Настройки сохранения
        save_freq=15_000,
        checkpoint_freq=10_000,
        max_checkpoints=10,
        
        # Раннее остановка
        early_stopping=True,
        patience=3,
        min_improvement=5.0,
        
        # Пути
        output_dir="results/advanced",
        
        # Мониторинг
        verbose=1,
        log_interval=1000,
        track_experiment=True,
        experiment_tags=["advanced", "ppo", "lunarlander"],
    )
    
    with Trainer(trainer_config) as trainer:
        result = trainer.train()
        
        if result.success:
            print(f"✅ Продвинутое обучение завершено!")
            print(f"📊 Результат: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
            print(f"🛑 Раннее остановка: {result.early_stopped}")
            print(f"📈 История оценки: {len(result.evaluation_history.get('mean_rewards', []))} точек")
            
            # Анализ истории обучения
            if result.evaluation_history.get('mean_rewards'):
                rewards = result.evaluation_history['mean_rewards']
                print(f"📈 Прогресс: {rewards[0]:.1f} → {rewards[-1]:.1f}")
                print(f"📊 Максимум: {max(rewards):.1f}")
        
        return result


def resume_training_example():
    """Пример восстановления обучения."""
    print("\n🔄 Пример восстановления обучения...")
    
    # Сначала запускаем частичное обучение
    print("1️⃣ Запуск частичного обучения...")
    
    config = TrainerConfig(
        experiment_name="resume_example",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        total_timesteps=50_000,
        seed=42,
        checkpoint_freq=10_000,
        output_dir="results/resume_example",
        verbose=0,
    )
    
    # Первая часть обучения
    with Trainer(config) as trainer:
        # Имитируем прерывание после 20000 шагов
        config.total_timesteps = 20_000
        result1 = trainer.train()
        
        if result1.success:
            print(f"✅ Первая часть: {result1.final_mean_reward:.2f}")
            checkpoint_paths = result1.checkpoint_paths
        else:
            print("❌ Ошибка в первой части")
            return
    
    # Восстановление и продолжение
    print("2️⃣ Восстановление и продолжение...")
    
    resume_config = TrainerConfig(
        experiment_name="resume_example_continued",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        total_timesteps=50_000,  # Общее количество
        seed=42,
        mode=TrainingMode.RESUME,
        resume_from_checkpoint=checkpoint_paths[-1] if checkpoint_paths else None,
        output_dir="results/resume_example",
        verbose=0,
    )
    
    with Trainer(resume_config) as trainer:
        result2 = trainer.train()
        
        if result2.success:
            print(f"✅ Продолжение: {result2.final_mean_reward:.2f}")
            print(f"📈 Улучшение: {result2.final_mean_reward - result1.final_mean_reward:.2f}")
        else:
            print("❌ Ошибка при восстановлении")
    
    return result1, result2


def config_file_training():
    """Обучение с использованием конфигурационного файла."""
    print("\n📄 Обучение из конфигурационного файла...")
    
    # Создаем временный конфигурационный файл
    config_content = """
experiment_name: "config_file_example"
output_dir: "results/config_example"
seed: 42

algorithm:
  name: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99

environment:
  name: "LunarLander-v3"

training:
  total_timesteps: 75000
  eval_freq: 15000
  n_eval_episodes: 5
  save_freq: 25000

logging:
  level: "INFO"
  log_to_file: true

reproducibility:
  seed: 42
  deterministic: true
"""
    
    # Сохраняем конфигурацию
    config_dir = Path("configs/examples")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "example_config.yaml"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"📝 Конфигурация сохранена: {config_file}")
    
    try:
        # Создание тренера из конфигурации
        trainer = create_trainer_from_config(
            config_path=str(config_file),
            # Переопределения через командную строку
            overrides=[
                "training.total_timesteps=30000",  # Уменьшаем для примера
                "algorithm.learning_rate=0.001",
            ]
        )
        
        with trainer:
            result = trainer.train()
            
            if result.success:
                print(f"✅ Обучение из файла завершено!")
                print(f"📊 Результат: {result.final_mean_reward:.2f}")
                print(f"⚙️  Конфигурация: {config_file}")
            else:
                print(f"❌ Ошибка: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return None


def main():
    """Главная функция с примерами."""
    print("🎮 Примеры использования системы обучения RL агентов")
    print("=" * 60)
    
    try:
        # 1. Базовое обучение
        result1 = basic_ppo_training()
        
        # 2. Сравнение алгоритмов
        results2 = compare_algorithms()
        
        # 3. Продвинутое обучение
        result3 = advanced_training_with_config()
        
        # 4. Восстановление обучения
        results4 = resume_training_example()
        
        # 5. Обучение из конфигурационного файла
        result5 = config_file_training()
        
        print("\n🎉 Все примеры выполнены!")
        print("\n📋 Сводка результатов:")
        print("-" * 40)
        
        if result1 and result1.success:
            print(f"Базовое PPO:     {result1.final_mean_reward:>8.2f}")
        
        if results2:
            for alg, res in results2.items():
                if res.success:
                    print(f"Сравнение {alg}:   {res.final_mean_reward:>8.2f}")
        
        if result3 and result3.success:
            print(f"Продвинутое:     {result3.final_mean_reward:>8.2f}")
        
        if result5 and result5.success:
            print(f"Из файла:        {result5.final_mean_reward:>8.2f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Выполнение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()