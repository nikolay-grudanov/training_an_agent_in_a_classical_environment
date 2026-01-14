"""Пример использования PPO агента для обучения в среде LunarLander-v3.

Этот скрипт демонстрирует полный цикл работы с PPO агентом:
- Создание конфигурации с оптимизированными гиперпараметрами
- Инициализация агента с кастомными колбэками
- Обучение с мониторингом метрик
- Оценка производительности
- Сохранение и загрузка модели
- Визуализация результатов

Использование:
    python examples/ppo_agent_usage.py
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.utils import configure_default_logging, set_seed


# Настройка логирования
configure_default_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """Колбэк для отображения прогресса обучения."""
    
    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            elapsed_time = time.time() - self.start_time
            progress = self.n_calls / self.locals.get("total_timesteps", 1)
            
            # Получение последних наград
            recent_rewards = []
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer[-10:]]
            
            if recent_rewards:
                mean_reward = np.mean(recent_rewards)
                logger.info(
                    f"Прогресс: {progress:.1%} | "
                    f"Шаги: {self.n_calls:,} | "
                    f"Время: {elapsed_time:.1f}с | "
                    f"Средняя награда (10 эп.): {mean_reward:.2f}"
                )
            else:
                logger.info(
                    f"Прогресс: {progress:.1%} | "
                    f"Шаги: {self.n_calls:,} | "
                    f"Время: {elapsed_time:.1f}с"
                )
        
        return True


def create_optimized_config() -> PPOConfig:
    """Создать оптимизированную конфигурацию для LunarLander-v3.
    
    Returns:
        Конфигурация PPO с оптимизированными гиперпараметрами
    """
    config = PPOConfig(
        # Основные параметры
        env_name="LunarLander-v3",
        total_timesteps=500_000,
        seed=42,
        
        # Оптимизированные гиперпараметры для LunarLander
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # Расписание learning rate
        use_lr_schedule=True,
        lr_schedule_type="linear",
        lr_final_ratio=0.1,
        
        # Архитектура сети
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn="tanh",
        ortho_init=True,
        
        # Нормализация среды
        normalize_env=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        
        # Ранняя остановка
        early_stopping=True,
        target_reward=200.0,
        patience_episodes=100,
        min_improvement=5.0,
        
        # Мониторинг
        eval_freq=25_000,
        n_eval_episodes=10,
        save_freq=100_000,
        log_interval=1,
        verbose=1,
        
        # Пути для сохранения
        model_save_path="results/models/ppo_lunar_lander.zip",
        tensorboard_log="results/logs/ppo_tensorboard/",
        use_tensorboard=True,
    )
    
    return config


def train_ppo_agent() -> PPOAgent:
    """Обучить PPO агента в среде LunarLander-v3.
    
    Returns:
        Обученный PPO агент
    """
    logger.info("🚀 Начало обучения PPO агента в LunarLander-v3")
    
    # Создание конфигурации
    config = create_optimized_config()
    logger.info(f"Конфигурация создана: {config.total_timesteps:,} шагов")
    
    # Создание директорий для сохранения
    model_path = Path(config.model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config.tensorboard_log:
        tb_path = Path(config.tensorboard_log)
        tb_path.mkdir(parents=True, exist_ok=True)
    
    # Инициализация агента
    agent = PPOAgent(
        config=config,
        experiment_name="ppo_lunar_lander_v3",
    )
    
    logger.info(f"Агент инициализирован: {agent}")
    
    # Создание кастомного колбэка для прогресса
    progress_callback = ProgressCallback(log_freq=25_000, verbose=1)
    
    # Обучение агента
    try:
        training_result = agent.train(callback=progress_callback)
        
        logger.info("✅ Обучение завершено успешно!")
        logger.info(f"Время обучения: {training_result.training_time:.2f} секунд")
        logger.info(f"Финальная средняя награда: {training_result.final_mean_reward:.2f} ± {training_result.final_std_reward:.2f}")
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Ошибка обучения: {e}")
        raise


def evaluate_agent(agent: PPOAgent, n_episodes: int = 20) -> Dict[str, float]:
    """Оценить производительность обученного агента.
    
    Args:
        agent: Обученный PPO агент
        n_episodes: Количество эпизодов для оценки
        
    Returns:
        Словарь с метриками оценки
    """
    logger.info(f"🧪 Оценка агента на {n_episodes} эпизодах")
    
    metrics = agent.evaluate(
        n_episodes=n_episodes,
        deterministic=True,
        render=False,
    )
    
    logger.info("📊 Результаты оценки:")
    logger.info(f"  Средняя награда: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"  Диапазон наград: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    logger.info(f"  Средняя длина эпизода: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    
    # Анализ успешности
    success_rate = sum(1 for _ in range(n_episodes) if metrics['mean_reward'] >= 200) / n_episodes * 100
    logger.info(f"  Процент успешных посадок: {success_rate:.1f}%")
    
    return metrics


def demonstrate_agent_usage(agent: PPOAgent, n_episodes: int = 3) -> List[float]:
    """Продемонстрировать работу агента с визуализацией.
    
    Args:
        agent: Обученный PPO агент
        n_episodes: Количество эпизодов для демонстрации
        
    Returns:
        Список наград за эпизоды
    """
    logger.info(f"🎮 Демонстрация работы агента ({n_episodes} эпизодов)")
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = agent.env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False
        
        logger.info(f"Эпизод {episode + 1}:")
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = agent.env.step(action)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Логирование ключевых моментов
            if step_count % 100 == 0:
                logger.debug(f"  Шаг {step_count}: награда = {reward:.3f}, общая = {episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        
        # Анализ результата эпизода
        if episode_reward >= 200:
            result = "🎯 Успешная посадка!"
        elif episode_reward >= 0:
            result = "🛬 Посадка с повреждениями"
        else:
            result = "💥 Крушение"
        
        logger.info(f"  Результат: {result}")
        logger.info(f"  Награда: {episode_reward:.2f}, Шагов: {step_count}")
    
    mean_reward = np.mean(episode_rewards)
    logger.info(f"Средняя награда за демонстрацию: {mean_reward:.2f}")
    
    return episode_rewards


def save_and_load_demo(agent: PPOAgent, save_path: str) -> PPOAgent:
    """Продемонстрировать сохранение и загрузку агента.
    
    Args:
        agent: Обученный агент для сохранения
        save_path: Путь для сохранения
        
    Returns:
        Загруженный агент
    """
    logger.info(f"💾 Сохранение агента в {save_path}")
    
    # Сохранение
    agent.save(save_path)
    logger.info("✅ Агент сохранен")
    
    # Загрузка
    logger.info(f"📂 Загрузка агента из {save_path}")
    loaded_agent = PPOAgent.load(save_path)
    logger.info("✅ Агент загружен")
    
    # Проверка что загруженный агент работает
    test_metrics = loaded_agent.evaluate(n_episodes=3, deterministic=True)
    logger.info(f"Тест загруженного агента: средняя награда = {test_metrics['mean_reward']:.2f}")
    
    return loaded_agent


def plot_training_progress(agent: PPOAgent) -> None:
    """Построить графики прогресса обучения.
    
    Args:
        agent: Обученный агент с результатами обучения
    """
    if agent.training_result is None:
        logger.warning("Нет данных для построения графиков")
        return
    
    logger.info("📈 Построение графиков прогресса обучения")
    
    # Получение данных из трекера метрик
    metrics_data = agent.metrics_tracker.get_summary()
    
    if not metrics_data.metrics:
        logger.warning("Нет метрик для построения графиков")
        return
    
    # Создание графиков
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Прогресс обучения PPO агента в LunarLander-v3", fontsize=14)
    
    # График наград
    reward_metrics = [m for m in metrics_data.metrics if m.name == "mean_reward"]
    if reward_metrics:
        steps = [m.step for m in reward_metrics]
        rewards = [m.value for m in reward_metrics]
        
        axes[0, 0].plot(steps, rewards, 'b-', alpha=0.7)
        axes[0, 0].set_title("Средняя награда")
        axes[0, 0].set_xlabel("Шаги обучения")
        axes[0, 0].set_ylabel("Награда")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=200, color='r', linestyle='--', alpha=0.7, label='Цель (200)')
        axes[0, 0].legend()
    
    # График длины эпизодов
    length_metrics = [m for m in metrics_data.metrics if m.name == "mean_length"]
    if length_metrics:
        steps = [m.step for m in length_metrics]
        lengths = [m.value for m in length_metrics]
        
        axes[0, 1].plot(steps, lengths, 'g-', alpha=0.7)
        axes[0, 1].set_title("Средняя длина эпизода")
        axes[0, 1].set_xlabel("Шаги обучения")
        axes[0, 1].set_ylabel("Шаги")
        axes[0, 1].grid(True, alpha=0.3)
    
    # График потерь политики
    policy_loss_metrics = [m for m in metrics_data.metrics if m.name == "policy_loss"]
    if policy_loss_metrics:
        steps = [m.step for m in policy_loss_metrics]
        losses = [m.value for m in policy_loss_metrics]
        
        axes[1, 0].plot(steps, losses, 'r-', alpha=0.7)
        axes[1, 0].set_title("Потери политики")
        axes[1, 0].set_xlabel("Шаги обучения")
        axes[1, 0].set_ylabel("Потери")
        axes[1, 0].grid(True, alpha=0.3)
    
    # График потерь функции ценности
    value_loss_metrics = [m for m in metrics_data.metrics if m.name == "value_loss"]
    if value_loss_metrics:
        steps = [m.step for m in value_loss_metrics]
        losses = [m.value for m in value_loss_metrics]
        
        axes[1, 1].plot(steps, losses, 'orange', alpha=0.7)
        axes[1, 1].set_title("Потери функции ценности")
        axes[1, 1].set_xlabel("Шаги обучения")
        axes[1, 1].set_ylabel("Потери")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение графика
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "ppo_training_progress.png"
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 График сохранен: {plot_path}")
    
    plt.show()


def main() -> None:
    """Главная функция демонстрации PPO агента."""
    logger.info("🎯 Демонстрация PPO агента для LunarLander-v3")
    logger.info("=" * 60)
    
    # Установка seed для воспроизводимости
    set_seed(42)
    
    try:
        # 1. Обучение агента
        agent = train_ppo_agent()
        
        # 2. Оценка производительности
        evaluation_metrics = evaluate_agent(agent, n_episodes=20)
        
        # 3. Демонстрация работы
        demo_rewards = demonstrate_agent_usage(agent, n_episodes=3)
        
        # 4. Сохранение и загрузка
        save_path = "results/models/ppo_demo_model.zip"
        save_and_load_demo(agent, save_path)
        
        # 5. Построение графиков
        plot_training_progress(agent)
        
        # 6. Итоговый отчет
        logger.info("=" * 60)
        logger.info("📋 ИТОГОВЫЙ ОТЧЕТ")
        logger.info("=" * 60)
        logger.info(f"Общее время обучения: {agent.training_result.training_time:.2f} сек")
        logger.info(f"Финальная производительность: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        logger.info(f"Лучшая награда: {evaluation_metrics['max_reward']:.2f}")
        logger.info(f"Демонстрационные награды: {[f'{r:.1f}' for r in demo_rewards]}")
        
        # Определение успешности обучения
        if evaluation_metrics['mean_reward'] >= 200:
            logger.info("🎉 ОБУЧЕНИЕ УСПЕШНО! Агент научился успешно приземляться.")
        elif evaluation_metrics['mean_reward'] >= 0:
            logger.info("⚠️  Частичный успех. Агент приземляется, но не оптимально.")
        else:
            logger.info("❌ Требуется дополнительное обучение.")
        
        logger.info("✅ Демонстрация завершена!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в демонстрации: {e}")
        raise


if __name__ == "__main__":
    main()