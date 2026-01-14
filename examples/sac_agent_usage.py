"""Пример использования SAC агента для непрерывных сред.

Этот скрипт демонстрирует полный цикл работы с SAC агентом:
- Создание и настройка конфигурации
- Инициализация агента
- Обучение с мониторингом
- Оценка производительности
- Сохранение и загрузка модели
- Визуализация результатов

Примеры для различных непрерывных сред:
- LunarLander-v3 (непрерывная версия)
- Pendulum-v1
- BipedalWalker-v3
- MountainCarContinuous-v0
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from src.agents import SACAgent, SACConfig
from src.utils import set_seed, setup_logging

# Настройка логирования
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """Колбэк для отображения прогресса обучения."""
    
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer[-10:]]
                mean_reward = np.mean(recent_rewards)
                logger.info(
                    f"Шаг {self.num_timesteps:,} | "
                    f"Время: {elapsed_time:.1f}с | "
                    f"Средняя награда: {mean_reward:.2f}"
                )
        return True


def create_sac_config_for_environment(env_name: str) -> SACConfig:
    """Создать оптимизированную конфигурацию SAC для конкретной среды.
    
    Args:
        env_name: Название среды Gymnasium
        
    Returns:
        Настроенная конфигурация SAC
    """
    base_config = {
        "env_name": env_name,
        "total_timesteps": 100_000,
        "seed": 42,
        "verbose": 1,
        "use_tensorboard": True,
        "tensorboard_log": f"./logs/sac_{env_name.lower().replace('-', '_')}",
        "model_save_path": f"./models/sac_{env_name.lower().replace('-', '_')}.zip",
        "eval_freq": 10_000,
        "n_eval_episodes": 10,
        "save_freq": 25_000,
    }
    
    # Специфичные настройки для разных сред
    if "LunarLander" in env_name:
        # Оптимизировано для LunarLander-v3 (непрерывная версия)
        config = SACConfig(
            **base_config,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            net_arch=[256, 256],
            activation_fn="relu",
            target_reward=200.0,
            early_stopping=True,
            patience_episodes=100,
            action_noise_type=None,  # SAC имеет встроенное исследование
        )
    
    elif "Pendulum" in env_name:
        # Оптимизировано для Pendulum-v1
        config = SACConfig(
            **base_config,
            total_timesteps=50_000,
            learning_rate=1e-3,
            buffer_size=200_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.02,
            gamma=0.98,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            net_arch=[128, 128],
            activation_fn="relu",
            target_reward=-200.0,  # Для Pendulum цель - минимизировать отрицательную награду
            early_stopping=True,
            patience_episodes=50,
            action_noise_type="normal",
            action_noise_std=0.1,
        )
    
    elif "BipedalWalker" in env_name:
        # Оптимизировано для BipedalWalker-v3
        config = SACConfig(
            **base_config,
            total_timesteps=1_000_000,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            net_arch=[400, 300],
            activation_fn="relu",
            target_reward=300.0,
            early_stopping=True,
            patience_episodes=200,
            action_noise_type=None,
        )
    
    elif "MountainCarContinuous" in env_name:
        # Оптимизировано для MountainCarContinuous-v0
        config = SACConfig(
            **base_config,
            total_timesteps=100_000,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128,
            tau=0.01,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            net_arch=[128, 128],
            activation_fn="tanh",
            target_reward=90.0,
            early_stopping=True,
            patience_episodes=50,
            action_noise_type="normal",
            action_noise_std=0.2,
        )
    
    else:
        # Универсальная конфигурация
        config = SACConfig(**base_config)
    
    return config


def train_sac_agent(env_name: str) -> SACAgent:
    """Обучить SAC агента для указанной среды.
    
    Args:
        env_name: Название среды Gymnasium
        
    Returns:
        Обученный SAC агент
    """
    logger.info(f"Начало обучения SAC агента для среды: {env_name}")
    
    # Создание конфигурации
    config = create_sac_config_for_environment(env_name)
    
    # Создание директорий
    Path(config.tensorboard_log).parent.mkdir(parents=True, exist_ok=True)
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Инициализация агента
    agent = SACAgent(config=config, experiment_name=f"sac_{env_name}")
    
    # Создание колбэка для прогресса
    progress_callback = ProgressCallback(check_freq=10_000, verbose=1)
    
    # Обучение
    start_time = time.time()
    training_result = agent.train(callback=progress_callback)
    training_time = time.time() - start_time
    
    # Логирование результатов
    logger.info(f"Обучение завершено за {training_time:.2f} секунд")
    logger.info(f"Финальная средняя награда: {training_result.final_mean_reward:.2f} ± {training_result.final_std_reward:.2f}")
    logger.info(f"Модель сохранена: {config.model_save_path}")
    
    return agent


def evaluate_agent(agent: SACAgent, n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
    """Оценить производительность агента.
    
    Args:
        agent: Обученный SAC агент
        n_episodes: Количество эпизодов для оценки
        render: Отображать среду во время оценки
        
    Returns:
        Словарь с метриками оценки
    """
    logger.info(f"Оценка агента на {n_episodes} эпизодах")
    
    metrics = agent.evaluate(n_episodes=n_episodes, render=render)
    
    logger.info("Результаты оценки:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    return metrics


def demonstrate_agent(agent: SACAgent, n_episodes: int = 3) -> None:
    """Демонстрация работы обученного агента.
    
    Args:
        agent: Обученный SAC агент
        n_episodes: Количество эпизодов для демонстрации
    """
    logger.info(f"Демонстрация агента на {n_episodes} эпизодах")
    
    env = agent.env
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False
        
        logger.info(f"Эпизод {episode + 1}:")
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if step_count % 50 == 0:
                logger.info(f"  Шаг {step_count}: награда = {episode_reward:.2f}")
        
        logger.info(f"  Финальная награда: {episode_reward:.2f} за {step_count} шагов")


def plot_training_progress(agent: SACAgent, save_path: str = "./plots/sac_training_progress.png") -> None:
    """Построить график прогресса обучения.
    
    Args:
        agent: Обученный SAC агент
        save_path: Путь для сохранения графика
    """
    if agent.training_result is None:
        logger.warning("Нет данных об обучении для построения графика")
        return
    
    # Создание директории для графиков
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Прогресс обучения SAC агента - {agent.config.env_name}", fontsize=14)
    
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
    
    # График средних наград оценки
    if agent.training_result.eval_mean_rewards:
        axes[1, 0].plot(agent.training_result.eval_timesteps, agent.training_result.eval_mean_rewards, 'o-')
        axes[1, 0].set_title("Средние награды оценки")
        axes[1, 0].set_xlabel("Временные шаги")
        axes[1, 0].set_ylabel("Средняя награда")
        axes[1, 0].grid(True)
    
    # Информация о модели
    info_text = f"""
    Алгоритм: {agent.config.algorithm}
    Среда: {agent.config.env_name}
    Общие шаги: {agent.config.total_timesteps:,}
    Время обучения: {agent.training_result.training_time:.1f}с
    Финальная награда: {agent.training_result.final_mean_reward:.2f}
    """
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"График сохранен: {save_path}")


def compare_sac_configurations() -> None:
    """Сравнить различные конфигурации SAC на одной среде."""
    env_name = "Pendulum-v1"
    logger.info(f"Сравнение конфигураций SAC на среде: {env_name}")
    
    # Различные конфигурации для сравнения
    configs = {
        "baseline": SACConfig(
            env_name=env_name,
            total_timesteps=20_000,
            learning_rate=3e-4,
            batch_size=256,
            ent_coef="auto",
            verbose=0,
        ),
        "high_lr": SACConfig(
            env_name=env_name,
            total_timesteps=20_000,
            learning_rate=1e-3,
            batch_size=256,
            ent_coef="auto",
            verbose=0,
        ),
        "large_batch": SACConfig(
            env_name=env_name,
            total_timesteps=20_000,
            learning_rate=3e-4,
            batch_size=512,
            ent_coef="auto",
            verbose=0,
        ),
        "fixed_entropy": SACConfig(
            env_name=env_name,
            total_timesteps=20_000,
            learning_rate=3e-4,
            batch_size=256,
            ent_coef=0.1,
            verbose=0,
        ),
    }
    
    results = {}
    
    for config_name, config in configs.items():
        logger.info(f"Обучение конфигурации: {config_name}")
        
        # Установка уникального seed для каждой конфигурации
        config.seed = 42 + hash(config_name) % 1000
        set_seed(config.seed)
        
        # Обучение агента
        agent = SACAgent(config=config, experiment_name=f"sac_comparison_{config_name}")
        training_result = agent.train()
        
        # Оценка
        eval_metrics = agent.evaluate(n_episodes=5)
        
        results[config_name] = {
            "final_reward": training_result.final_mean_reward,
            "training_time": training_result.training_time,
            "eval_reward": eval_metrics["mean_reward"],
            "eval_std": eval_metrics["std_reward"],
        }
        
        logger.info(f"  Результат {config_name}: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    
    # Вывод сравнения
    logger.info("\nСравнение результатов:")
    logger.info("Конфигурация | Финальная награда | Время обучения | Оценка")
    logger.info("-" * 65)
    
    for config_name, result in results.items():
        logger.info(
            f"{config_name:12} | {result['final_reward']:15.2f} | "
            f"{result['training_time']:12.1f}с | "
            f"{result['eval_reward']:6.2f} ± {result['eval_std']:4.2f}"
        )


def main() -> None:
    """Основная функция демонстрации SAC агента."""
    logger.info("=== Демонстрация SAC агента ===")
    
    # Установка seed для воспроизводимости
    set_seed(42)
    
    # Выбор среды для демонстрации
    env_name = "Pendulum-v1"  # Быстрая среда для демонстрации
    
    try:
        # 1. Обучение агента
        logger.info("\n1. Обучение SAC агента")
        agent = train_sac_agent(env_name)
        
        # 2. Оценка производительности
        logger.info("\n2. Оценка производительности")
        eval_metrics = evaluate_agent(agent, n_episodes=10)
        
        # 3. Демонстрация работы
        logger.info("\n3. Демонстрация работы агента")
        demonstrate_agent(agent, n_episodes=2)
        
        # 4. Построение графиков
        logger.info("\n4. Построение графиков")
        plot_training_progress(agent)
        
        # 5. Информация о модели
        logger.info("\n5. Информация о модели")
        model_info = agent.get_model_info()
        for key, value in model_info.items():
            if not key.endswith("_stats"):  # Пропускаем сложные статистики
                logger.info(f"  {key}: {value}")
        
        # 6. Тест сохранения и загрузки
        logger.info("\n6. Тест сохранения и загрузки")
        model_path = f"./models/test_sac_{env_name.lower().replace('-', '_')}.zip"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        agent.save(model_path)
        loaded_agent = SACAgent.load(model_path)
        
        # Сравнение предсказаний
        test_obs = np.array([0.0, 0.0, 1.0])  # Для Pendulum-v1
        original_action, _ = agent.predict(test_obs, deterministic=True)
        loaded_action, _ = loaded_agent.predict(test_obs, deterministic=True)
        
        logger.info(f"  Оригинальное действие: {original_action}")
        logger.info(f"  Загруженное действие: {loaded_action}")
        logger.info(f"  Разность: {np.abs(original_action - loaded_action).max():.6f}")
        
        # 7. Сравнение конфигураций (опционально)
        logger.info("\n7. Сравнение конфигураций")
        compare_sac_configurations()
        
        logger.info("\n=== Демонстрация завершена успешно ===")
        
    except Exception as e:
        logger.error(f"Ошибка во время демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()