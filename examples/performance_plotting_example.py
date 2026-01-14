#!/usr/bin/env python3
"""Пример использования модуля performance_plots для визуализации обучения RL агентов.

Этот скрипт демонстрирует различные возможности модуля performance_plots:
- Создание статических и интерактивных графиков
- Сравнение нескольких агентов
- Создание дашбордов
- Экспорт в различные форматы
- Работа с различными источниками данных
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logging import setup_logging
from src.utils.metrics import MetricsTracker
from src.visualization.performance_plots import (
    PerformancePlotter,
    InteractivePlotter,
    DataLoader,
    create_performance_report,
    quick_reward_plot,
    quick_comparison_plot,
    export_plots_to_formats
)

# Настройка логирования
logger = setup_logging(log_level=logging.INFO, console_output=True)


def generate_sample_training_data(
    n_episodes: int = 1000,
    agent_name: str = "PPO",
    base_reward: float = 0.0,
    improvement_rate: float = 0.001,
    noise_level: float = 5.0
) -> pd.DataFrame:
    """Генерирует синтетические данные обучения RL агента.
    
    Args:
        n_episodes: Количество эпизодов
        agent_name: Название агента
        base_reward: Базовое вознаграждение
        improvement_rate: Скорость улучшения
        noise_level: Уровень шума
        
    Returns:
        DataFrame с данными обучения
    """
    np.random.seed(42)  # Для воспроизводимости
    
    episodes = range(1, n_episodes + 1)
    timesteps = np.cumsum(np.random.randint(50, 200, n_episodes))
    
    # Моделируем улучшение производительности с шумом
    trend = base_reward + improvement_rate * np.array(episodes)
    noise = np.random.normal(0, noise_level, n_episodes)
    rewards = trend + noise
    
    # Длина эпизодов (обратно коррелирует с производительностью)
    episode_lengths = np.maximum(
        50, 
        200 - 0.1 * np.array(episodes) + np.random.normal(0, 20, n_episodes)
    ).astype(int)
    
    # Функция потерь (убывает со временем)
    losses = np.maximum(
        0.01,
        1.0 * np.exp(-0.002 * np.array(episodes)) + np.random.exponential(0.1, n_episodes)
    )
    
    return pd.DataFrame({
        'episode': episodes,
        'timestep': timesteps,
        'episode_reward': rewards,
        'episode_length': episode_lengths,
        'training_loss': losses,
        'agent': agent_name
    })


def example_basic_plotting():
    """Пример базового построения графиков."""
    logger.info("=== Пример базового построения графиков ===")
    
    # Генерируем тестовые данные
    data = generate_sample_training_data(n_episodes=500, agent_name="PPO")
    
    # Создаем плоттер
    plotter = PerformancePlotter(style='seaborn-v0_8')
    
    # Создаем директорию для результатов
    output_dir = Path("results/examples/basic_plotting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # График вознаграждения
    reward_path = plotter.plot_reward_curve(
        data,
        x_col='timestep',
        y_col='episode_reward',
        smooth_window=50,
        save_path=output_dir / "reward_curve.png",
        title="Кривая обучения PPO агента"
    )
    logger.info(f"График вознаграждения сохранен: {reward_path}")
    
    # График длины эпизодов
    length_path = plotter.plot_episode_lengths(
        data,
        x_col='episode',
        y_col='episode_length',
        smooth_window=30,
        save_path=output_dir / "episode_lengths.png",
        title="Длина эпизодов PPO агента"
    )
    logger.info(f"График длины эпизодов сохранен: {length_path}")
    
    # График функции потерь
    loss_path = plotter.plot_loss_curves(
        {'training_loss': data[['timestep', 'training_loss']].rename(columns={'training_loss': 'value'})},
        loss_columns=['training_loss'],
        save_path=output_dir / "loss_curves.png",
        title="Функция потерь PPO агента"
    )
    logger.info(f"График функции потерь сохранен: {loss_path}")


def example_interactive_plotting():
    """Пример создания интерактивных графиков."""
    logger.info("=== Пример интерактивных графиков ===")
    
    # Генерируем тестовые данные
    data = generate_sample_training_data(n_episodes=800, agent_name="SAC")
    
    # Создаем интерактивный плоттер
    plotter = InteractivePlotter(theme='plotly_white')
    
    # Создаем директорию для результатов
    output_dir = Path("results/examples/interactive_plotting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Интерактивный график вознаграждения
    reward_path = plotter.plot_interactive_reward_curve(
        {'episode_reward': data[['timestep', 'episode_reward']].rename(columns={'episode_reward': 'value'})},
        save_path=output_dir / "interactive_reward.html",
        title="Интерактивная кривая обучения SAC агента"
    )
    logger.info(f"Интерактивный график вознаграждения сохранен: {reward_path}")
    
    # Интерактивный дашборд
    dashboard_data = {
        'episode_reward': data[['timestep', 'episode_reward']].rename(columns={'episode_reward': 'value'}),
        'episode_length': data[['timestep', 'episode_length']].rename(columns={'episode_length': 'value'}),
        'training_loss': data[['timestep', 'training_loss']].rename(columns={'training_loss': 'value'})
    }
    
    dashboard_path = plotter.create_interactive_dashboard(
        dashboard_data,
        save_path=output_dir / "interactive_dashboard.html",
        title="Интерактивный дашборд SAC агента"
    )
    logger.info(f"Интерактивный дашборд сохранен: {dashboard_path}")


def example_agent_comparison():
    """Пример сравнения нескольких агентов."""
    logger.info("=== Пример сравнения агентов ===")
    
    # Генерируем данные для разных агентов
    agents_data = {
        "PPO": generate_sample_training_data(
            n_episodes=600, agent_name="PPO", 
            base_reward=0, improvement_rate=0.015, noise_level=4.0
        ),
        "SAC": generate_sample_training_data(
            n_episodes=600, agent_name="SAC", 
            base_reward=2, improvement_rate=0.012, noise_level=3.5
        ),
        "A2C": generate_sample_training_data(
            n_episodes=600, agent_name="A2C", 
            base_reward=-1, improvement_rate=0.010, noise_level=5.5
        )
    }
    
    # Преобразуем данные в нужный формат
    formatted_data = {}
    for agent_name, agent_data in agents_data.items():
        formatted_data[agent_name] = {
            'episode_reward': agent_data[['timestep', 'episode_reward']].rename(
                columns={'episode_reward': 'value'}
            )
        }
    
    # Создаем плоттеры
    static_plotter = PerformancePlotter()
    interactive_plotter = InteractivePlotter()
    
    # Создаем директорию для результатов
    output_dir = Path("results/examples/agent_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Статический сравнительный график
    static_path = static_plotter.plot_multiple_agents(
        formatted_data,
        metric='episode_reward',
        save_path=output_dir / "agents_comparison_static.png",
        title="Сравнение производительности RL агентов"
    )
    logger.info(f"Статический сравнительный график сохранен: {static_path}")
    
    # Интерактивный сравнительный график
    interactive_path = interactive_plotter.plot_interactive_comparison(
        formatted_data,
        metric='episode_reward',
        save_path=output_dir / "agents_comparison_interactive.html",
        title="Интерактивное сравнение RL агентов"
    )
    logger.info(f"Интерактивный сравнительный график сохранен: {interactive_path}")


def example_metrics_tracker_integration():
    """Пример интеграции с MetricsTracker."""
    logger.info("=== Пример интеграции с MetricsTracker ===")
    
    # Создаем трекер метрик
    tracker = MetricsTracker("example_experiment")
    
    # Симулируем процесс обучения
    np.random.seed(42)
    for episode in range(1, 301):
        timestep = episode * 100
        
        # Добавляем метрики эпизода
        reward = -200 + episode * 0.5 + np.random.normal(0, 20)
        length = max(50, 200 - episode * 0.3 + np.random.normal(0, 15))
        
        tracker.add_episode_metrics(
            episode=episode,
            timestep=timestep,
            reward=reward,
            length=int(length)
        )
        
        # Добавляем метрики обучения
        if episode % 10 == 0:  # Каждые 10 эпизодов
            loss = max(0.01, 1.0 * np.exp(-0.01 * episode) + np.random.exponential(0.05))
            learning_rate = 0.001 * (0.99 ** (episode // 50))
            
            tracker.add_training_metrics(
                timestep=timestep,
                loss=loss,
                learning_rate=learning_rate
            )
    
    # Создаем полный отчет о производительности
    output_dir = Path("results/examples/metrics_tracker_integration")
    report_path = create_performance_report(
        tracker,
        output_dir=output_dir,
        include_interactive=True,
        include_static=True
    )
    logger.info(f"Отчет о производительности создан: {report_path}")
    
    # Экспортируем метрики
    csv_path = tracker.export_to_csv()
    json_path = tracker.export_to_json()
    logger.info(f"Метрики экспортированы: CSV={csv_path}, JSON={json_path}")


def example_data_loading():
    """Пример загрузки данных из различных источников."""
    logger.info("=== Пример загрузки данных ===")
    
    # Создаем тестовые файлы данных
    output_dir = Path("results/examples/data_loading")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем CSV файл
    test_data = generate_sample_training_data(n_episodes=200)
    csv_path = output_dir / "test_data.csv"
    test_data.to_csv(csv_path, index=False)
    logger.info(f"Тестовый CSV файл создан: {csv_path}")
    
    # Создаем JSON файл
    json_data = {
        "experiment_id": "test_experiment",
        "metrics": {
            "episode_reward": [
                {"timestep": row['timestep'], "value": row['episode_reward']}
                for _, row in test_data.iterrows()
            ]
        }
    }
    json_path = output_dir / "test_data.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Тестовый JSON файл создан: {json_path}")
    
    # Загружаем данные из CSV
    csv_data = DataLoader.load_from_csv(csv_path)
    logger.info(f"Загружены данные из CSV: {len(csv_data)} записей")
    
    # Загружаем данные из JSON
    json_data_loaded = DataLoader.load_from_json(json_path)
    logger.info(f"Загружены данные из JSON: {json_data_loaded['experiment_id']}")
    
    # Используем быстрые функции для создания графиков
    quick_reward_path = quick_reward_plot(
        csv_path,
        save_path=output_dir / "quick_reward_plot.png"
    )
    logger.info(f"Быстрый график вознаграждения создан: {quick_reward_path}")


def example_export_formats():
    """Пример экспорта графиков в различные форматы."""
    logger.info("=== Пример экспорта в различные форматы ===")
    
    # Генерируем тестовые данные
    data = generate_sample_training_data(n_episodes=300)
    
    # Создаем плоттер
    plotter = PerformancePlotter()
    
    # Создаем директорию для результатов
    output_dir = Path("results/examples/export_formats")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Экспортируем график в несколько форматов
    saved_files = export_plots_to_formats(
        plotter.plot_reward_curve,
        output_dir,
        formats=['png', 'pdf', 'svg'],
        data=data,
        x_col='timestep',
        y_col='episode_reward',
        title='Reward Curve Export Example'
    )
    
    logger.info("График экспортирован в форматы:")
    for fmt, path in saved_files.items():
        logger.info(f"  {fmt.upper()}: {path}")


def main():
    """Главная функция с примерами использования."""
    logger.info("Запуск примеров использования performance_plots")
    
    try:
        # Базовые графики
        example_basic_plotting()
        
        # Интерактивные графики
        example_interactive_plotting()
        
        # Сравнение агентов
        example_agent_comparison()
        
        # Интеграция с MetricsTracker
        example_metrics_tracker_integration()
        
        # Загрузка данных
        example_data_loading()
        
        # Экспорт в различные форматы
        example_export_formats()
        
        logger.info("Все примеры успешно выполнены!")
        logger.info("Результаты сохранены в директории: results/examples/")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении примеров: {e}")
        raise


if __name__ == "__main__":
    main()