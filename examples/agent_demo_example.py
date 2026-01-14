"""Пример использования модуля демонстрации агентов.

Демонстрирует различные способы создания видео демонстраций обученных RL агентов
с использованием высокоуровневых функций из src.visualization.agent_demo.
"""

import logging
from pathlib import Path

from src.agents.ppo_agent import PPOAgent
from src.agents.base import AgentConfig
from src.visualization.agent_demo import (
    DemoConfig,
    create_best_episode_demo,
    create_average_behavior_demo,
    create_before_after_demo,
    create_training_progress_demo,
    create_multi_agent_comparison,
    create_batch_demos,
    auto_demo_from_training_results,
    quick_demo,
    quick_comparison,
)
from src.visualization.video_generator import VideoConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_best_episode_demo():
    """Пример создания демо лучшего эпизода."""
    logger.info("=== Пример: Демо лучшего эпизода ===")
    
    # Создание и обучение агента
    config = AgentConfig(
        algorithm="PPO",
        env_name="LunarLander-v2",
        total_timesteps=50_000,
        seed=42,
    )
    
    agent = PPOAgent(config)
    
    # Обучение агента (в реальности это займет время)
    logger.info("Обучение агента...")
    training_result = agent.train()
    logger.info(f"Обучение завершено. Финальная награда: {training_result.final_mean_reward:.2f}")
    
    # Настройка демонстрации
    demo_config = DemoConfig(
        video_config=VideoConfig(
            fps=30,
            quality="high",
            show_metrics=True,
            show_episode_info=True,
        ),
        auto_naming=True,
        auto_compress=True,
        compression_level="medium",
    )
    
    # Создание демо лучшего эпизода
    output_path = Path("results/demos/best_episode_demo.mp4")
    
    result = create_best_episode_demo(
        agent=agent,
        env="LunarLander-v2",
        output_path=output_path,
        config=demo_config,
        num_candidates=20,  # Тестируем 20 эпизодов для поиска лучшего
    )
    
    logger.info(f"Демо создано: {result['output_path']}")
    logger.info(f"Лучшая награда: {result['best_reward']:.2f}")
    logger.info(f"Статистика наград: {result['reward_statistics']}")
    
    return result


def example_average_behavior_demo():
    """Пример создания демо среднего поведения."""
    logger.info("=== Пример: Демо среднего поведения ===")
    
    # Загрузка обученного агента
    agent = PPOAgent.load("results/models/trained_agent.zip")
    
    # Создание демо среднего поведения
    demo_config = DemoConfig(
        num_episodes=10,  # Анализируем 10 эпизодов
        auto_compress=True,
    )
    
    result = create_average_behavior_demo(
        agent=agent,
        env="LunarLander-v2",
        output_path=Path("results/demos/average_behavior_demo.mp4"),
        config=demo_config,
    )
    
    logger.info(f"Демо среднего поведения создано: {result['output_path']}")
    logger.info(f"Выбранная награда: {result['selected_reward']:.2f}")
    logger.info(f"Средняя награда: {result['mean_reward']:.2f}")
    
    return result


def example_before_after_demo():
    """Пример создания демо сравнения до/после обучения."""
    logger.info("=== Пример: Демо до/после обучения ===")
    
    # Создание необученного агента
    config = AgentConfig(
        algorithm="PPO",
        env_name="LunarLander-v2",
        seed=42,
    )
    untrained_agent = PPOAgent(config)
    
    # Загрузка обученного агента
    trained_agent = PPOAgent.load("results/models/trained_agent.zip")
    
    # Создание сравнительного демо
    demo_config = DemoConfig(
        video_config=VideoConfig(
            fps=30,
            quality="high",
            show_metrics=True,
        ),
        max_episode_length=500,  # Ограничиваем длину эпизода
    )
    
    result = create_before_after_demo(
        untrained_agent=untrained_agent,
        trained_agent=trained_agent,
        env="LunarLander-v2",
        output_path=Path("results/demos/before_after_demo.mp4"),
        config=demo_config,
    )
    
    logger.info(f"Сравнительное демо создано: {result['output_path']}")
    logger.info(f"Информация о сравнении: {result['comparison_info']}")
    
    return result


def example_training_progress_demo():
    """Пример создания демо прогресса обучения."""
    logger.info("=== Пример: Демо прогресса обучения ===")
    
    # Пути к чекпоинтам (предполагается, что они существуют)
    checkpoint_paths = [
        "results/checkpoints/checkpoint_10000.zip",
        "results/checkpoints/checkpoint_25000.zip",
        "results/checkpoints/checkpoint_50000.zip",
        "results/checkpoints/checkpoint_75000.zip",
        "results/checkpoints/checkpoint_100000.zip",
    ]
    
    # Названия для чекпоинтов
    checkpoint_names = [
        "10K шагов",
        "25K шагов", 
        "50K шагов",
        "75K шагов",
        "100K шагов (финал)",
    ]
    
    demo_config = DemoConfig(
        video_config=VideoConfig(
            fps=25,
            quality="presentation",
            show_metrics=True,
            show_agent_name=True,
        ),
        auto_compress=True,
        compression_level="low",  # Меньше сжатие для лучшего качества
    )
    
    result = create_training_progress_demo(
        checkpoint_paths=checkpoint_paths,
        agent_class=PPOAgent,
        env="LunarLander-v2",
        output_path=Path("results/demos/training_progress_demo.mp4"),
        config=demo_config,
        checkpoint_names=checkpoint_names,
    )
    
    logger.info(f"Демо прогресса создано: {result['output_path']}")
    logger.info(f"Загружено чекпоинтов: {result['checkpoints_loaded']}")
    
    return result


def example_multi_agent_comparison():
    """Пример создания сравнения нескольких агентов."""
    logger.info("=== Пример: Сравнение нескольких агентов ===")
    
    # Загрузка разных агентов для сравнения
    agents = [
        ("PPO (базовый)", PPOAgent.load("results/models/ppo_baseline.zip")),
        ("PPO (настроенный)", PPOAgent.load("results/models/ppo_tuned.zip")),
        ("A2C", PPOAgent.load("results/models/a2c_agent.zip")),  # Предположим, что есть A2C
    ]
    
    demo_config = DemoConfig(
        video_config=VideoConfig(
            fps=30,
            quality="high",
            show_metrics=True,
            show_agent_name=True,
        ),
        max_episode_length=300,
        auto_compress=True,
    )
    
    result = create_multi_agent_comparison(
        agents=agents,
        env="LunarLander-v2",
        output_path=Path("results/demos/multi_agent_comparison.mp4"),
        config=demo_config,
    )
    
    logger.info(f"Сравнительное демо создано: {result['output_path']}")
    logger.info(f"Сравнено агентов: {result['agents_compared']}")
    
    for agent_info in result['comparison_info']['agents']:
        logger.info(f"  {agent_info['name']}: награда {agent_info['total_reward']:.2f}")
    
    return result


def example_batch_demos():
    """Пример пакетного создания демонстраций."""
    logger.info("=== Пример: Пакетное создание демо ===")
    
    # Список агентов для демонстрации
    agents = [
        ("PPO Baseline", PPOAgent.load("results/models/ppo_baseline.zip")),
        ("PPO Tuned", PPOAgent.load("results/models/ppo_tuned.zip")),
        ("A2C Agent", PPOAgent.load("results/models/a2c_agent.zip")),
    ]
    
    # Типы демо для создания
    demo_types = ["best_episode", "average"]
    
    demo_config = DemoConfig(
        video_config=VideoConfig(fps=25, quality="medium"),
        auto_compress=True,
        continue_on_error=True,  # Продолжать при ошибках
    )
    
    result = create_batch_demos(
        agents=agents,
        env="LunarLander-v2",
        output_dir=Path("results/demos/batch"),
        demo_types=demo_types,
        config=demo_config,
    )
    
    logger.info(f"Пакетное создание завершено:")
    logger.info(f"  Обработано агентов: {result['agents_processed']}")
    logger.info(f"  Создано демо: {result['demos_created']}")
    logger.info(f"  Ошибок: {result['demos_failed']}")
    
    return result


def example_auto_demo_from_results():
    """Пример автоматического создания демо из результатов обучения."""
    logger.info("=== Пример: Автоматическое создание демо ===")
    
    demo_config = DemoConfig(
        video_config=VideoConfig(
            fps=30,
            quality="high",
            show_metrics=True,
        ),
        auto_compress=True,
        continue_on_error=True,
    )
    
    result = auto_demo_from_training_results(
        training_results_dir=Path("results"),
        agent_class=PPOAgent,
        env="LunarLander-v2",
        output_dir=Path("results/demos/auto"),
        config=demo_config,
    )
    
    logger.info(f"Автоматическое создание завершено:")
    logger.info(f"  Найдено моделей: {result['models_found']}")
    logger.info(f"  Создано демо: {result['demos_created']}")
    logger.info(f"  Сводный отчет: {result['summary_path']}")
    
    return result


def example_quick_functions():
    """Пример использования быстрых функций."""
    logger.info("=== Пример: Быстрые функции ===")
    
    # Быстрое создание демо
    agent = PPOAgent.load("results/models/trained_agent.zip")
    
    demo_path = quick_demo(
        agent=agent,
        env="LunarLander-v2",
        output_path=Path("results/demos/quick_demo.mp4"),
        demo_type="best_episode",
    )
    
    logger.info(f"Быстрое демо создано: {demo_path}")
    
    # Быстрое сравнение
    agents = [
        ("Агент 1", PPOAgent.load("results/models/agent1.zip")),
        ("Агент 2", PPOAgent.load("results/models/agent2.zip")),
    ]
    
    comparison_path = quick_comparison(
        agents=agents,
        env="LunarLander-v2",
        output_path=Path("results/demos/quick_comparison.mp4"),
    )
    
    logger.info(f"Быстрое сравнение создано: {comparison_path}")
    
    return demo_path, comparison_path


def example_custom_video_config():
    """Пример настройки параметров видео."""
    logger.info("=== Пример: Настройка параметров видео ===")
    
    # Кастомная конфигурация видео
    video_config = VideoConfig(
        fps=60,  # Высокий FPS
        quality="presentation",  # Максимальное качество
        format="mp4",
        width=1920,  # Full HD
        height=1080,
        show_metrics=True,
        show_episode_info=True,
        show_agent_name=True,
        font_size=20,  # Крупный шрифт
        text_color=(255, 255, 0),  # Желтый текст
        background_color=(0, 0, 0, 200),  # Полупрозрачный черный фон
    )
    
    demo_config = DemoConfig(
        video_config=video_config,
        auto_compress=False,  # Без сжатия для максимального качества
        title_prefix="HD Demo",
        include_timestamp=True,
    )
    
    agent = PPOAgent.load("results/models/trained_agent.zip")
    
    result = create_best_episode_demo(
        agent=agent,
        env="LunarLander-v2",
        output_path=Path("results/demos/hd_demo.mp4"),
        config=demo_config,
        num_candidates=50,  # Больше кандидатов для лучшего результата
    )
    
    logger.info(f"HD демо создано: {result['output_path']}")
    logger.info(f"Размер файла: {Path(result['output_path']).stat().st_size / 1024 / 1024:.1f} МБ")
    
    return result


def main():
    """Главная функция с примерами использования."""
    logger.info("Запуск примеров демонстрации агентов")
    
    # Создание директорий для результатов
    Path("results/demos").mkdir(parents=True, exist_ok=True)
    Path("results/demos/batch").mkdir(parents=True, exist_ok=True)
    Path("results/demos/auto").mkdir(parents=True, exist_ok=True)
    
    try:
        # Примеры различных типов демонстраций
        example_best_episode_demo()
        example_average_behavior_demo()
        example_before_after_demo()
        example_training_progress_demo()
        example_multi_agent_comparison()
        example_batch_demos()
        example_auto_demo_from_results()
        example_quick_functions()
        example_custom_video_config()
        
        logger.info("Все примеры выполнены успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения примеров: {e}")
        raise


if __name__ == "__main__":
    main()