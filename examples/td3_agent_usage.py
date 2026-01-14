"""Пример использования TD3 агента для непрерывных сред.

Этот скрипт демонстрирует полный цикл работы с TD3 агентом:
- Создание и настройка конфигурации
- Инициализация агента
- Обучение модели
- Оценка производительности
- Сохранение и загрузка модели
- Визуализация результатов

Примеры для различных непрерывных сред:
- LunarLander-v3 (continuous)
- Pendulum-v1
- BipedalWalker-v3
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.agents.td3_agent import TD3Agent, TD3Config
from src.utils import set_seed, setup_logging

# Настройка логирования
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_td3_config(
    env_name: str = "LunarLander-v3",
    total_timesteps: int = 100_000,
    seed: int = 42,
    **kwargs,
) -> TD3Config:
    """Создать оптимизированную конфигурацию TD3 для заданной среды.
    
    Args:
        env_name: Название среды Gymnasium
        total_timesteps: Количество шагов обучения
        seed: Seed для воспроизводимости
        **kwargs: Дополнительные параметры конфигурации
    
    Returns:
        Настроенная конфигурация TD3
    """
    # Базовые параметры
    base_config = {
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "verbose": 1,
    }
    
    # Оптимизированные параметры для разных сред
    env_configs = {
        "LunarLander-v3": {
            "learning_rate": 1e-3,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "action_noise_type": "normal",
            "action_noise_std": 0.1,
            "target_reward": 200.0,
            "early_stopping": True,
            "patience_episodes": 100,
        },
        "Pendulum-v1": {
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "learning_starts": 1_000,
            "batch_size": 128,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "action_noise_type": "normal",
            "action_noise_std": 0.1,
            "target_reward": -200.0,  # Для Pendulum цель - минимизация
            "early_stopping": True,
            "patience_episodes": 50,
        },
        "BipedalWalker-v3": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "action_noise_type": "ornstein_uhlenbeck",
            "ou_sigma": 0.2,
            "ou_theta": 0.15,
            "target_reward": 300.0,
            "early_stopping": True,
            "patience_episodes": 100,
        },
    }
    
    # Объединение конфигураций
    config_dict = base_config.copy()
    if env_name in env_configs:
        config_dict.update(env_configs[env_name])
    config_dict.update(kwargs)
    
    return TD3Config(**config_dict)


def train_td3_agent(
    config: TD3Config,
    save_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> TD3Agent:
    """Обучить TD3 агента с заданной конфигурацией.
    
    Args:
        config: Конфигурация TD3 агента
        save_path: Путь для сохранения модели
        experiment_name: Имя эксперимента
    
    Returns:
        Обученный TD3 агент
    """
    logger.info(f"Начало обучения TD3 агента для среды {config.env_name}")
    
    # Создание агента
    agent = TD3Agent(
        config=config,
        experiment_name=experiment_name or f"td3_{config.env_name}_{int(time.time())}",
    )
    
    # Логирование информации о модели
    model_info = agent.get_model_info()
    logger.info(f"Информация о модели: {model_info}")
    
    # Обучение
    start_time = time.time()
    training_result = agent.train()
    training_time = time.time() - start_time
    
    logger.info(
        f"Обучение завершено за {training_time:.2f} сек. "
        f"Финальная средняя награда: {training_result.final_mean_reward:.2f} ± "
        f"{training_result.final_std_reward:.2f}"
    )
    
    # Сохранение модели
    if save_path:
        agent.save(save_path)
        logger.info(f"Модель сохранена: {save_path}")
    
    return agent


def evaluate_agent(
    agent: TD3Agent,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Оценить производительность агента.
    
    Args:
        agent: Обученный TD3 агент
        n_episodes: Количество эпизодов для оценки
        render: Отображать среду во время оценки
        deterministic: Использовать детерминистическую политику
    
    Returns:
        Словарь с метриками производительности
    """
    logger.info(f"Оценка агента на {n_episodes} эпизодах")
    
    metrics = agent.evaluate(
        n_episodes=n_episodes,
        deterministic=deterministic,
        render=render,
    )
    
    logger.info(f"Результаты оценки: {metrics}")
    return metrics


def demonstrate_agent(
    agent: TD3Agent,
    n_episodes: int = 3,
    max_steps: int = 1000,
) -> List[Dict[str, float]]:
    """Демонстрация работы агента с детальным логированием.
    
    Args:
        agent: Обученный TD3 агент
        n_episodes: Количество эпизодов для демонстрации
        max_steps: Максимальное количество шагов в эпизоде
    
    Returns:
        Список результатов эпизодов
    """
    logger.info(f"Демонстрация агента на {n_episodes} эпизодах")
    
    results = []
    
    for episode in range(n_episodes):
        obs, _ = agent.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        logger.info(f"Эпизод {episode + 1} начат")
        
        while not done and episode_length < max_steps:
            # Предсказание действия
            action, _ = agent.predict(obs, deterministic=True)
            
            # Выполнение действия
            obs, reward, terminated, truncated, info = agent.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Логирование каждые 100 шагов
            if episode_length % 100 == 0:
                logger.debug(
                    f"Эпизод {episode + 1}, шаг {episode_length}: "
                    f"действие={action}, награда={reward:.3f}, "
                    f"общая награда={episode_reward:.3f}"
                )
        
        episode_result = {
            "episode": episode + 1,
            "reward": episode_reward,
            "length": episode_length,
            "success": episode_reward > 0,  # Простой критерий успеха
        }
        
        results.append(episode_result)
        
        logger.info(
            f"Эпизод {episode + 1} завершен: "
            f"награда={episode_reward:.2f}, длина={episode_length}"
        )
    
    return results


def plot_training_progress(
    agent: TD3Agent,
    save_path: Optional[str] = None,
) -> None:
    """Построить графики прогресса обучения.
    
    Args:
        agent: Обученный TD3 агент
        save_path: Путь для сохранения графика
    """
    if agent.training_result is None:
        logger.warning("Нет данных об обучении для построения графика")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Прогресс обучения TD3 - {agent.config.env_name}", fontsize=16)
    
    # График наград эпизодов
    if agent.training_result.episode_rewards:
        axes[0, 0].plot(agent.training_result.episode_rewards)
        axes[0, 0].set_title("Награды эпизодов")
        axes[0, 0].set_xlabel("Эпизод")
        axes[0, 0].set_ylabel("Награда")
        axes[0, 0].grid(True)
    
    # График длин эпизодов
    if agent.training_result.episode_lengths:
        axes[0, 1].plot(agent.training_result.episode_lengths)
        axes[0, 1].set_title("Длины эпизодов")
        axes[0, 1].set_xlabel("Эпизод")
        axes[0, 1].set_ylabel("Длина")
        axes[0, 1].grid(True)
    
    # График средних наград при оценке
    if agent.training_result.eval_mean_rewards:
        axes[1, 0].plot(
            agent.training_result.eval_timesteps,
            agent.training_result.eval_mean_rewards,
            marker='o'
        )
        axes[1, 0].set_title("Средние награды при оценке")
        axes[1, 0].set_xlabel("Временные шаги")
        axes[1, 0].set_ylabel("Средняя награда")
        axes[1, 0].grid(True)
    
    # Гистограмма финальных наград
    if agent.training_result.episode_rewards:
        final_rewards = agent.training_result.episode_rewards[-100:]  # Последние 100
        axes[1, 1].hist(final_rewards, bins=20, alpha=0.7)
        axes[1, 1].set_title("Распределение финальных наград")
        axes[1, 1].set_xlabel("Награда")
        axes[1, 1].set_ylabel("Частота")
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен: {save_path}")
    
    plt.show()


def compare_noise_types(
    env_name: str = "Pendulum-v1",
    total_timesteps: int = 50_000,
    n_runs: int = 3,
) -> Dict[str, List[float]]:
    """Сравнить различные типы шума для действий.
    
    Args:
        env_name: Название среды
        total_timesteps: Количество шагов обучения
        n_runs: Количество запусков для каждого типа шума
    
    Returns:
        Словарь с результатами для каждого типа шума
    """
    logger.info(f"Сравнение типов шума для среды {env_name}")
    
    noise_types = ["normal", "ornstein_uhlenbeck", "none"]
    results = {noise_type: [] for noise_type in noise_types}
    
    for noise_type in noise_types:
        logger.info(f"Тестирование шума типа: {noise_type}")
        
        for run in range(n_runs):
            logger.info(f"Запуск {run + 1}/{n_runs}")
            
            # Создание конфигурации
            config = create_td3_config(
                env_name=env_name,
                total_timesteps=total_timesteps,
                action_noise_type=noise_type,
                seed=42 + run,  # Разные seed для каждого запуска
                early_stopping=False,  # Отключаем для честного сравнения
            )
            
            # Обучение агента
            agent = train_td3_agent(config)
            
            # Оценка
            metrics = evaluate_agent(agent, n_episodes=10)
            results[noise_type].append(metrics["mean_reward"])
            
            logger.info(
                f"Шум {noise_type}, запуск {run + 1}: "
                f"средняя награда = {metrics['mean_reward']:.2f}"
            )
    
    # Вывод сводных результатов
    logger.info("Сводные результаты сравнения шума:")
    for noise_type, rewards in results.items():
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logger.info(
            f"{noise_type}: {mean_reward:.2f} ± {std_reward:.2f} "
            f"(мин: {np.min(rewards):.2f}, макс: {np.max(rewards):.2f})"
        )
    
    return results


def hyperparameter_tuning_example(
    env_name: str = "LunarLander-v3",
    total_timesteps: int = 50_000,
) -> Dict[str, float]:
    """Пример настройки гиперпараметров TD3.
    
    Args:
        env_name: Название среды
        total_timesteps: Количество шагов обучения
    
    Returns:
        Словарь с лучшими гиперпараметрами и результатом
    """
    logger.info(f"Настройка гиперпараметров для среды {env_name}")
    
    # Сетка гиперпараметров для поиска
    param_grid = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [128, 256, 512],
        "tau": [0.001, 0.005, 0.01],
        "policy_delay": [1, 2, 3],
    }
    
    best_reward = float("-inf")
    best_params = {}
    
    # Простой grid search (в реальности лучше использовать более эффективные методы)
    for lr in param_grid["learning_rate"]:
        for batch_size in param_grid["batch_size"]:
            for tau in param_grid["tau"]:
                for policy_delay in param_grid["policy_delay"]:
                    
                    params = {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "tau": tau,
                        "policy_delay": policy_delay,
                    }
                    
                    logger.info(f"Тестирование параметров: {params}")
                    
                    # Создание конфигурации
                    config = create_td3_config(
                        env_name=env_name,
                        total_timesteps=total_timesteps,
                        early_stopping=False,
                        **params
                    )
                    
                    # Обучение и оценка
                    agent = train_td3_agent(config)
                    metrics = evaluate_agent(agent, n_episodes=5)
                    
                    reward = metrics["mean_reward"]
                    logger.info(f"Результат: {reward:.2f}")
                    
                    # Обновление лучших параметров
                    if reward > best_reward:
                        best_reward = reward
                        best_params = params.copy()
                        logger.info(f"Новые лучшие параметры: {best_params}, награда: {best_reward:.2f}")
    
    logger.info(f"Лучшие параметры: {best_params}")
    logger.info(f"Лучшая награда: {best_reward:.2f}")
    
    return {"params": best_params, "reward": best_reward}


def main() -> None:
    """Главная функция с примерами использования TD3 агента."""
    # Установка seed для воспроизводимости
    set_seed(42)
    
    # Создание директории для результатов
    results_dir = Path("results/td3_examples")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Примеры использования TD3 агента ===")
    
    # Пример 1: Базовое обучение для LunarLander-v3
    logger.info("\n1. Базовое обучение TD3 для LunarLander-v3")
    
    config = create_td3_config(
        env_name="LunarLander-v3",
        total_timesteps=100_000,
        model_save_path=str(results_dir / "td3_lunarlander.zip"),
        tensorboard_log=str(results_dir / "tensorboard"),
    )
    
    agent = train_td3_agent(
        config=config,
        save_path=str(results_dir / "td3_lunarlander.zip"),
        experiment_name="td3_lunarlander_example",
    )
    
    # Оценка агента
    metrics = evaluate_agent(agent, n_episodes=10)
    
    # Демонстрация
    demo_results = demonstrate_agent(agent, n_episodes=3)
    
    # Построение графиков
    plot_training_progress(
        agent,
        save_path=str(results_dir / "td3_lunarlander_progress.png")
    )
    
    # Пример 2: Обучение для Pendulum-v1
    logger.info("\n2. Обучение TD3 для Pendulum-v1")
    
    pendulum_config = create_td3_config(
        env_name="Pendulum-v1",
        total_timesteps=50_000,
        action_noise_type="ornstein_uhlenbeck",
    )
    
    pendulum_agent = train_td3_agent(
        config=pendulum_config,
        experiment_name="td3_pendulum_example",
    )
    
    pendulum_metrics = evaluate_agent(pendulum_agent, n_episodes=10)
    
    # Пример 3: Сравнение типов шума
    logger.info("\n3. Сравнение типов шума")
    
    noise_comparison = compare_noise_types(
        env_name="Pendulum-v1",
        total_timesteps=30_000,
        n_runs=2,  # Уменьшено для примера
    )
    
    # Пример 4: Загрузка и тестирование сохраненной модели
    logger.info("\n4. Загрузка сохраненной модели")
    
    model_path = results_dir / "td3_lunarlander.zip"
    if model_path.exists():
        loaded_agent = TD3Agent.load(str(model_path))
        loaded_metrics = evaluate_agent(loaded_agent, n_episodes=5)
        logger.info(f"Загруженная модель: {loaded_metrics}")
    
    # Пример 5: Настройка гиперпараметров (упрощенная версия)
    logger.info("\n5. Пример настройки гиперпараметров")
    
    # Для демонстрации используем ограниченную сетку
    tuning_results = hyperparameter_tuning_example(
        env_name="Pendulum-v1",
        total_timesteps=20_000,
    )
    
    logger.info("=== Примеры завершены ===")
    
    # Сводка результатов
    logger.info("\nСводка результатов:")
    logger.info(f"LunarLander-v3: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"Pendulum-v1: {pendulum_metrics['mean_reward']:.2f} ± {pendulum_metrics['std_reward']:.2f}")
    logger.info(f"Лучшие параметры: {tuning_results['params']}")
    logger.info(f"Результаты сохранены в: {results_dir}")


if __name__ == "__main__":
    main()