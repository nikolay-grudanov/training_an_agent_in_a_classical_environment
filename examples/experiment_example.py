"""Пример использования класса Experiment для проведения RL экспериментов.

Этот пример демонстрирует полный жизненный цикл эксперимента:
1. Создание baseline и variant конфигураций
2. Инициализация эксперимента с гипотезой
3. Запуск эксперимента
4. Добавление результатов обучения
5. Автоматическое сравнение результатов
6. Сохранение и загрузка эксперимента
"""

import logging
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment import Experiment
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig
from src.utils.logging import setup_logging


def create_baseline_config() -> RLConfig:
    """Создать базовую конфигурацию для эксперимента."""
    return RLConfig(
        experiment_name="baseline_ppo_lunarlander",
        seed=42,
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        ),
        environment=EnvironmentConfig(
            name="LunarLander-v3",
            max_episode_steps=1000
        ),
        training=TrainingConfig(
            total_timesteps=200000,
            eval_freq=10000,
            n_eval_episodes=10
        )
    )


def create_variant_config() -> RLConfig:
    """Создать вариантную конфигурацию с увеличенным learning rate."""
    return RLConfig(
        experiment_name="variant_ppo_lunarlander",
        seed=42,
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=1e-3,  # Увеличенный learning rate
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        ),
        environment=EnvironmentConfig(
            name="LunarLander-v3",
            max_episode_steps=1000
        ),
        training=TrainingConfig(
            total_timesteps=200000,
            eval_freq=10000,
            n_eval_episodes=10
        )
    )


def simulate_training_results(config_type: str) -> dict:
    """Симулировать результаты обучения для демонстрации.
    
    В реальном использовании эти данные будут получены
    из фактического процесса обучения агента.
    """
    if config_type == "baseline":
        return {
            'mean_reward': 150.5,
            'final_reward': 180.2,
            'episode_length': 245,
            'convergence_timesteps': 120000,
            'training_time': 3600,
            'best_reward': 220.1,
            'worst_reward': 85.3,
            'reward_std': 45.2,
            'success_rate': 0.75
        }
    else:  # variant
        return {
            'mean_reward': 175.8,  # Лучше чем baseline
            'final_reward': 195.4,
            'episode_length': 230,  # Короче эпизоды
            'convergence_timesteps': 100000,  # Быстрее сходимость
            'training_time': 3400,
            'best_reward': 240.5,
            'worst_reward': 95.7,
            'reward_std': 38.9,
            'success_rate': 0.82
        }


def simulate_training_metrics(config_type: str) -> list:
    """Симулировать метрики по шагам обучения."""
    import random
    
    metrics = []
    base_reward = 150.5 if config_type == "baseline" else 175.8
    
    for timestep in range(0, 200000, 10000):
        # Симулируем прогресс обучения
        progress = timestep / 200000
        noise = random.uniform(-20, 20)
        reward = base_reward * progress + noise
        
        metrics.append({
            'timestep': timestep,
            'mean_reward': max(reward, -200),  # Ограничиваем снизу
            'episode_length': int(250 - progress * 20),
            'loss_policy': random.uniform(0.01, 0.1),
            'loss_value': random.uniform(0.05, 0.2)
        })
    
    return metrics


def main():
    """Основная функция демонстрации."""
    # Настройка логирования
    logger = setup_logging(
        log_level=logging.INFO,
        console_output=True,
        json_format=False
    )
    
    # Создание директории для результатов
    output_dir = Path("results/experiments/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Начинаем демонстрацию класса Experiment")
    
    # 1. Создание конфигураций
    logger.info("Создание конфигураций...")
    baseline_config = create_baseline_config()
    variant_config = create_variant_config()
    
    # 2. Создание эксперимента
    hypothesis = (
        "Увеличение learning rate с 3e-4 до 1e-3 улучшит производительность "
        "агента PPO в среде LunarLander-v3, что приведет к более высокой "
        "средней награде и более быстрой сходимости."
    )
    
    logger.info("Создание эксперимента...")
    experiment = Experiment(
        baseline_config=baseline_config,
        variant_config=variant_config,
        hypothesis=hypothesis,
        output_dir=output_dir
    )
    
    logger.info(f"Создан эксперимент: {experiment.experiment_id}")
    logger.info(f"Гипотеза: {experiment.hypothesis}")
    
    # 3. Запуск эксперимента
    logger.info("Запуск эксперимента...")
    experiment.start()
    
    # Проверка статуса
    status = experiment.get_status()
    logger.info(f"Статус эксперимента: {status['status']}")
    
    # 4. Симуляция обучения baseline конфигурации
    logger.info("Симуляция обучения baseline конфигурации...")
    baseline_results = simulate_training_results("baseline")
    baseline_metrics = simulate_training_metrics("baseline")
    
    experiment.add_result("baseline", baseline_results, baseline_metrics)
    logger.info("Результаты baseline добавлены")
    
    # 5. Симуляция обучения variant конфигурации
    logger.info("Симуляция обучения variant конфигурации...")
    variant_results = simulate_training_results("variant")
    variant_metrics = simulate_training_metrics("variant")
    
    experiment.add_result("variant", variant_results, variant_metrics)
    logger.info("Результаты variant добавлены")
    
    # 6. Анализ результатов
    logger.info("Анализ результатов...")
    comparison = experiment.compare_results()
    
    logger.info("Результаты сравнения:")
    for metric, data in comparison['performance_metrics'].items():
        improvement = data['improvement_percent']
        better = data['better']
        logger.info(
            f"  {metric}: {better} лучше на {improvement:.1f}% "
            f"(baseline: {data['baseline']:.2f}, variant: {data['variant']:.2f})"
        )
    
    # Общий вывод
    overall_better = comparison['summary']['overall_better']
    reward_improvement = comparison['summary']['reward_improvement']
    logger.info(f"Общий результат: {overall_better} лучше на {reward_improvement:.2f} награды")
    
    # 7. Завершение эксперимента
    logger.info("Завершение эксперимента...")
    experiment.stop()
    
    # 8. Сохранение результатов
    logger.info("Сохранение эксперимента...")
    json_path = experiment.save(format_type='json')
    pickle_path = experiment.save(format_type='pickle')
    
    logger.info(f"Эксперимент сохранен в:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Pickle: {pickle_path}")
    
    # 9. Демонстрация загрузки
    logger.info("Демонстрация загрузки эксперимента...")
    loaded_experiment = Experiment.load(json_path)
    
    logger.info(f"Загружен эксперимент: {loaded_experiment.experiment_id}")
    logger.info(f"Статус: {loaded_experiment.status.value}")
    
    # 10. Получение сводки
    logger.info("Получение сводки эксперимента...")
    summary = loaded_experiment.get_summary()
    
    logger.info("Сводка эксперимента:")
    logger.info(f"  ID: {summary['experiment_id']}")
    logger.info(f"  Статус: {summary['status']}")
    logger.info(f"  Baseline алгоритм: {summary['configurations']['baseline']['algorithm']}")
    logger.info(f"  Variant алгоритм: {summary['configurations']['variant']['algorithm']}")
    logger.info(f"  Среда: {summary['configurations']['baseline']['environment']}")
    
    if 'results' in summary:
        overall_result = summary['results']['summary']['overall_better']
        logger.info(f"  Лучший результат: {overall_result}")
    
    logger.info("Демонстрация завершена успешно!")
    
    # Возвращаем эксперимент для дальнейшего использования
    return loaded_experiment


if __name__ == "__main__":
    experiment = main()
    print(f"\nЭксперимент завершен: {experiment}")
    print(f"Результаты сохранены в: {experiment.experiment_dir}")