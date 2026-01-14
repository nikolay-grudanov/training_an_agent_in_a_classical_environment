"""Пример использования A2C агента для обучения в среде LunarLander-v3.

Этот скрипт демонстрирует:
1. Создание и настройку A2C агента
2. Обучение с мониторингом метрик
3. Оценку производительности
4. Сохранение и загрузку модели
5. Визуализацию результатов
6. Сравнение с PPO агентом
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.agents.a2c_agent import A2CAgent, A2CConfig
from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.utils import set_seed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_a2c_config() -> A2CConfig:
    """Создать оптимизированную конфигурацию A2C для LunarLander-v3.
    
    Returns:
        Настроенная конфигурация A2C агента
    """
    return A2CConfig(
        # Основные параметры
        env_name="LunarLander-v3",
        total_timesteps=200_000,
        seed=42,
        
        # Гиперпараметры A2C
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-5,
        
        # Расписание learning rate
        use_lr_schedule=True,
        lr_schedule_type="linear",
        lr_final_ratio=0.1,
        
        # Архитектура сети
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn="tanh",
        ortho_init=True,
        
        # Нормализация
        normalize_env=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        
        # Ранняя остановка
        early_stopping=True,
        target_reward=200.0,
        patience_episodes=100,
        min_improvement=2.0,
        
        # Мониторинг
        eval_freq=10_000,
        n_eval_episodes=10,
        save_freq=50_000,
        log_interval=1,
        use_tensorboard=True,
        
        # Пути сохранения
        model_save_path="results/models/a2c_lunar_lander.zip",
        tensorboard_log="results/logs/a2c_tensorboard/",
        
        # Дополнительные параметры
        verbose=1,
        device="cpu",
    )


def train_a2c_agent(config: A2CConfig) -> A2CAgent:
    """Обучить A2C агента.
    
    Args:
        config: Конфигурация агента
        
    Returns:
        Обученный A2C агент
    """
    logger.info("Создание A2C агента...")
    
    # Установка seed для воспроизводимости
    set_seed(config.seed)
    
    # Создание агента
    agent = A2CAgent(
        config=config,
        experiment_name="a2c_lunar_lander_experiment",
    )
    
    logger.info("Начало обучения A2C агента...")
    start_time = time.time()
    
    # Обучение
    training_result = agent.train()
    
    training_time = time.time() - start_time
    
    logger.info(
        f"Обучение завершено за {training_time:.2f} сек",
        extra={
            "final_mean_reward": training_result.final_mean_reward,
            "final_std_reward": training_result.final_std_reward,
            "total_timesteps": training_result.total_timesteps,
            "success": training_result.success,
        },
    )
    
    return agent


def evaluate_agent(agent: A2CAgent, n_episodes: int = 20) -> Dict[str, float]:
    """Оценить производительность агента.
    
    Args:
        agent: Обученный агент
        n_episodes: Количество эпизодов для оценки
        
    Returns:
        Словарь с метриками оценки
    """
    logger.info(f"Оценка агента на {n_episodes} эпизодах...")
    
    metrics = agent.evaluate(
        n_episodes=n_episodes,
        deterministic=True,
        render=False,
    )
    
    logger.info(
        "Результаты оценки:",
        extra=metrics,
    )
    
    return metrics


def compare_with_ppo(a2c_config: A2CConfig) -> Dict[str, Dict[str, float]]:
    """Сравнить A2C с PPO агентом.
    
    Args:
        a2c_config: Конфигурация A2C для создания аналогичной PPO конфигурации
        
    Returns:
        Словарь с результатами сравнения
    """
    logger.info("Сравнение A2C с PPO...")
    
    # Создание PPO конфигурации на основе A2C
    ppo_config = PPOConfig(
        env_name=a2c_config.env_name,
        total_timesteps=a2c_config.total_timesteps,
        seed=a2c_config.seed,
        learning_rate=a2c_config.learning_rate,
        gamma=a2c_config.gamma,
        ent_coef=a2c_config.ent_coef,
        vf_coef=a2c_config.vf_coef,
        max_grad_norm=a2c_config.max_grad_norm,
        normalize_env=a2c_config.normalize_env,
        early_stopping=a2c_config.early_stopping,
        target_reward=a2c_config.target_reward,
        model_save_path="results/models/ppo_lunar_lander_comparison.zip",
        tensorboard_log="results/logs/ppo_comparison_tensorboard/",
        verbose=0,  # Меньше логов для сравнения
    )
    
    # Обучение A2C
    logger.info("Обучение A2C для сравнения...")
    a2c_agent = A2CAgent(config=a2c_config, experiment_name="a2c_comparison")
    a2c_result = a2c_agent.train()
    a2c_metrics = a2c_agent.evaluate(n_episodes=20, deterministic=True)
    
    # Обучение PPO
    logger.info("Обучение PPO для сравнения...")
    ppo_agent = PPOAgent(config=ppo_config, experiment_name="ppo_comparison")
    ppo_result = ppo_agent.train()
    ppo_metrics = ppo_agent.evaluate(n_episodes=20, deterministic=True)
    
    # Сравнение результатов
    comparison = {
        "A2C": {
            "mean_reward": a2c_metrics["mean_reward"],
            "std_reward": a2c_metrics["std_reward"],
            "training_time": a2c_result.training_time,
            "total_timesteps": a2c_result.total_timesteps,
        },
        "PPO": {
            "mean_reward": ppo_metrics["mean_reward"],
            "std_reward": ppo_metrics["std_reward"],
            "training_time": ppo_result.training_time,
            "total_timesteps": ppo_result.total_timesteps,
        },
    }
    
    logger.info("Результаты сравнения:")
    for algorithm, metrics in comparison.items():
        logger.info(f"{algorithm}: {metrics}")
    
    return comparison


def visualize_results(
    agent: A2CAgent,
    comparison_results: Dict[str, Dict[str, float]],
    save_path: str = "results/plots/",
) -> None:
    """Создать визуализацию результатов.
    
    Args:
        agent: Обученный агент
        comparison_results: Результаты сравнения с PPO
        save_path: Путь для сохранения графиков
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # График 1: Сравнение алгоритмов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    algorithms = list(comparison_results.keys())
    mean_rewards = [comparison_results[alg]["mean_reward"] for alg in algorithms]
    std_rewards = [comparison_results[alg]["std_reward"] for alg in algorithms]
    training_times = [comparison_results[alg]["training_time"] for alg in algorithms]
    
    # Средние награды
    bars1 = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5)
    ax1.set_title("Сравнение средних наград", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Средняя награда")
    ax1.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, reward in zip(bars1, mean_rewards):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{reward:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    # Время обучения
    bars2 = ax2.bar(algorithms, training_times)
    ax2.set_title("Сравнение времени обучения", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Время обучения (сек)")
    ax2.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    plt.tight_layout()
    plt.savefig(save_dir / "a2c_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # График 2: Информация о модели A2C
    model_info = agent.get_model_info()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Подготовка данных для отображения
    info_text = f"""
    A2C Agent Information
    =====================
    
    Environment: {model_info.get('env_name', 'N/A')}
    Algorithm: {model_info.get('algorithm', 'N/A')}
    Total Timesteps: {model_info.get('total_timesteps', 'N/A'):,}
    
    Hyperparameters:
    ----------------
    Learning Rate: {agent.config.learning_rate}
    N Steps: {model_info.get('n_steps', 'N/A')}
    Entropy Coefficient: {model_info.get('ent_coef', 'N/A')}
    Value Function Coefficient: {model_info.get('vf_coef', 'N/A')}
    RMSProp Epsilon: {agent.config.rms_prop_eps}
    Use RMSProp: {model_info.get('use_rms_prop', 'N/A')}
    
    Training Results:
    ----------------
    Final Mean Reward: {model_info.get('final_mean_reward', 'N/A'):.2f}
    Training Time: {model_info.get('training_time', 'N/A'):.2f} sec
    Best Mean Reward: {model_info.get('best_mean_reward', 'N/A'):.2f}
    
    Environment Normalization: {model_info.get('normalize_env', 'N/A')}
    Early Stopping: {model_info.get('early_stopping', 'N/A')}
    Learning Rate Schedule: {model_info.get('use_lr_schedule', 'N/A')}
    """
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A2C Agent Model Information", fontsize=16, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(save_dir / "a2c_model_info.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    logger.info(f"Графики сохранены в {save_dir}")


def demonstrate_save_load(agent: A2CAgent, config: A2CConfig) -> None:
    """Демонстрация сохранения и загрузки модели.
    
    Args:
        agent: Обученный агент
        config: Конфигурация агента
    """
    logger.info("Демонстрация сохранения и загрузки модели...")
    
    # Сохранение модели
    save_path = "results/models/a2c_demo_save.zip"
    agent.save(save_path)
    logger.info(f"Модель сохранена: {save_path}")
    
    # Загрузка модели
    loaded_agent = A2CAgent.load(save_path, config=config)
    logger.info("Модель загружена успешно")
    
    # Сравнение производительности
    original_metrics = agent.evaluate(n_episodes=5, deterministic=True)
    loaded_metrics = loaded_agent.evaluate(n_episodes=5, deterministic=True)
    
    logger.info("Сравнение оригинальной и загруженной модели:")
    logger.info(f"Оригинальная: {original_metrics['mean_reward']:.2f} ± {original_metrics['std_reward']:.2f}")
    logger.info(f"Загруженная: {loaded_metrics['mean_reward']:.2f} ± {loaded_metrics['std_reward']:.2f}")
    
    # Проверка идентичности
    diff = abs(original_metrics["mean_reward"] - loaded_metrics["mean_reward"])
    if diff < 1.0:  # Небольшая разница допустима из-за стохастичности среды
        logger.info("✅ Модель загружена корректно")
    else:
        logger.warning(f"⚠️ Большая разница в производительности: {diff:.2f}")


def main() -> None:
    """Основная функция демонстрации A2C агента."""
    logger.info("🚀 Демонстрация A2C агента для LunarLander-v3")
    
    # Создание директорий для результатов
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Создание конфигурации
        logger.info("📋 Создание конфигурации A2C...")
        config = create_a2c_config()
        
        # 2. Обучение агента
        logger.info("🎓 Обучение A2C агента...")
        agent = train_a2c_agent(config)
        
        # 3. Оценка производительности
        logger.info("📊 Оценка производительности...")
        evaluation_metrics = evaluate_agent(agent, n_episodes=20)
        
        # 4. Сравнение с PPO (опционально, занимает время)
        logger.info("⚖️ Сравнение с PPO агентом...")
        comparison_config = A2CConfig(
            env_name="LunarLander-v3",
            total_timesteps=50_000,  # Меньше для быстрого сравнения
            seed=42,
            verbose=0,
            model_save_path="results/models/a2c_comparison.zip",
            tensorboard_log="results/logs/a2c_comparison_tensorboard/",
        )
        comparison_results = compare_with_ppo(comparison_config)
        
        # 5. Визуализация результатов
        logger.info("📈 Создание визуализации...")
        visualize_results(agent, comparison_results)
        
        # 6. Демонстрация сохранения/загрузки
        logger.info("💾 Демонстрация сохранения и загрузки...")
        demonstrate_save_load(agent, config)
        
        # 7. Финальная информация
        logger.info("✅ Демонстрация завершена успешно!")
        logger.info(f"📁 Результаты сохранены в директории 'results/'")
        logger.info(f"🎯 Финальная производительность: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        
        # Рекомендации
        if evaluation_metrics["mean_reward"] >= 200:
            logger.info("🏆 Отличный результат! Агент успешно решил задачу.")
        elif evaluation_metrics["mean_reward"] >= 100:
            logger.info("👍 Хороший результат! Агент показывает стабильную производительность.")
        else:
            logger.info("📚 Агент требует дополнительного обучения или настройки гиперпараметров.")
            
    except Exception as e:
        logger.error(f"❌ Ошибка во время демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()