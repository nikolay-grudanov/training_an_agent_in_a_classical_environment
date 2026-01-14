#!/usr/bin/env python3
"""Пример использования форматировщика результатов экспериментов.

Демонстрирует создание различных типов отчетов:
- Отчет по одному агенту
- Сравнительный отчет нескольких агентов
- Экспорт в различные форматы
"""

from pathlib import Path
from typing import Dict, List

from src.evaluation.evaluator import EvaluationMetrics
from src.reporting.results_formatter import ReportConfig, ResultsFormatter


class MockQuantitativeResults:
    """Мок-класс для демонстрации количественных результатов."""
    
    def __init__(self, rewards: List[float], episode_lengths: List[int]) -> None:
        self.rewards = rewards
        self.episode_lengths = episode_lengths


def create_sample_results() -> Dict[str, EvaluationMetrics]:
    """Создание примерных результатов оценки для демонстрации."""
    
    # Результаты для PPO агента
    ppo_results = EvaluationMetrics(
        mean_reward=185.7,
        std_reward=28.4,
        min_reward=150.0,
        max_reward=220.0,
        mean_episode_length=195.2,
        std_episode_length=32.1,
        min_episode_length=160,
        max_episode_length=230,
        reward_ci_lower=175.0,
        reward_ci_upper=196.4,
        success_rate=0.87,
        total_episodes=100,
        total_timesteps=19520,
        evaluation_time=150.0,
    )
    
    # Результаты для DQN агента
    dqn_results = EvaluationMetrics(
        mean_reward=142.3,
        std_reward=35.6,
        min_reward=90.0,
        max_reward=190.0,
        mean_episode_length=178.9,
        std_episode_length=41.2,
        min_episode_length=120,
        max_episode_length=220,
        reward_ci_lower=130.0,
        reward_ci_upper=154.6,
        success_rate=0.73,
        total_episodes=100,
        total_timesteps=17890,
        evaluation_time=140.0,
    )
    
    # Результаты для A2C агента
    a2c_results = EvaluationMetrics(
        mean_reward=156.8,
        std_reward=31.2,
        min_reward=110.0,
        max_reward=200.0,
        mean_episode_length=188.4,
        std_episode_length=29.7,
        min_episode_length=140,
        max_episode_length=230,
        reward_ci_lower=145.0,
        reward_ci_upper=168.6,
        success_rate=0.79,
        total_episodes=100,
        total_timesteps=18840,
        evaluation_time=145.0,
    )
    
    return {
        "PPO_Agent": ppo_results,
        "DQN_Agent": dqn_results,
        "A2C_Agent": a2c_results,
    }


def create_sample_quantitative_results() -> Dict[str, MockQuantitativeResults]:
    """Создание примерных количественных результатов."""
    
    # Количественные результаты для PPO
    ppo_quant = MockQuantitativeResults(
        rewards=[180.5, 195.2, 172.8, 201.3, 188.7, 176.9, 192.4, 184.1],
        episode_lengths=[190, 205, 185, 210, 195, 180, 200, 188],
    )
    
    # Количественные результаты для DQN
    dqn_quant = MockQuantitativeResults(
        rewards=[135.2, 158.7, 128.9, 149.6, 145.3, 139.8, 152.1, 141.4],
        episode_lengths=[170, 185, 165, 190, 175, 172, 188, 178],
    )
    
    # Количественные результаты для A2C
    a2c_quant = MockQuantitativeResults(
        rewards=[150.3, 168.9, 145.7, 162.4, 159.1, 148.6, 165.2, 154.8],
        episode_lengths=[185, 195, 180, 200, 190, 182, 198, 186],
    )
    
    return {
        "PPO_Agent": ppo_quant,
        "DQN_Agent": dqn_quant,
        "A2C_Agent": a2c_quant,
    }


def main() -> None:
    """Основная функция демонстрации."""
    
    print("🚀 Демонстрация форматировщика результатов экспериментов")
    print("=" * 60)
    
    # Создаем директории для примера
    output_dir = Path("examples/reports_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Конфигурация для русскоязычных отчетов
    config_ru = ReportConfig(
        language="ru",
        theme="default",
        include_plots=True,
        include_statistics=True,
        decimal_places=2,
    )
    
    # Конфигурация для англоязычных отчетов
    config_en = ReportConfig(
        language="en",
        theme="default",
        include_plots=True,
        include_statistics=True,
        decimal_places=3,
    )
    
    # Создаем форматировщики
    formatter_ru = ResultsFormatter(
        output_dir=output_dir / "ru",
        config=config_ru,
    )
    
    formatter_en = ResultsFormatter(
        output_dir=output_dir / "en",
        config=config_en,
    )
    
    # Получаем примерные данные
    agents_results = create_sample_results()
    quantitative_results = create_sample_quantitative_results()
    
    print("\n📊 Создание отчетов по отдельным агентам...")
    
    # Создаем отчеты по отдельным агентам
    for agent_name, results in agents_results.items():
        print(f"  • {agent_name}")
        
        # HTML отчет (русский)
        html_path = formatter_ru.format_single_agent_report(
            agent_name=agent_name,
            evaluation_results=results,
            quantitative_results=quantitative_results[agent_name],
            output_format="html",
            filename=f"{agent_name.lower()}_report_ru",
        )
        print(f"    HTML (RU): {html_path}")
        
        # Markdown отчет (английский)
        md_path = formatter_en.format_single_agent_report(
            agent_name=agent_name,
            evaluation_results=results,
            quantitative_results=quantitative_results[agent_name],
            output_format="markdown",
            filename=f"{agent_name.lower()}_report_en",
        )
        print(f"    Markdown (EN): {md_path}")
    
    print("\n🔄 Создание сравнительных отчетов...")
    
    # Сравнительный отчет HTML (русский)
    comparison_html = formatter_ru.format_comparison_report(
        agents_results=agents_results,
        quantitative_results=quantitative_results,
        output_format="html",
        filename="agents_comparison_ru",
    )
    print(f"  HTML сравнение (RU): {comparison_html}")
    
    # Сравнительный отчет HTML (английский)
    comparison_html_en = formatter_en.format_comparison_report(
        agents_results=agents_results,
        quantitative_results=quantitative_results,
        output_format="html",
        filename="agents_comparison_en",
    )
    print(f"  HTML сравнение (EN): {comparison_html_en}")
    
    print("\n📈 Создание отчета по эксперименту...")
    
    # Данные эксперимента
    experiment_data = {
        "name": "LunarLander_Comparison_Experiment",
        "description": "Сравнение алгоритмов PPO, DQN и A2C на задаче LunarLander-v2",
        "environment": "LunarLander-v2",
        "total_timesteps": 100_000,
        "agents": list(agents_results.keys()),
        "duration_hours": 2.5,
        "best_agent": "PPO_Agent",
        "best_reward": 185.7,
        "hyperparameters": {
            "PPO": {"learning_rate": 3e-4, "n_steps": 2048},
            "DQN": {"learning_rate": 1e-4, "buffer_size": 50000},
            "A2C": {"learning_rate": 7e-4, "n_steps": 5},
        },
        "results_summary": {
            "total_episodes": 300,
            "successful_episodes": 239,
            "average_training_time": "50 minutes per agent",
        },
    }
    
    experiment_report = formatter_ru.format_experiment_report(
        experiment_name="LunarLander_Comparison",
        experiment_data=experiment_data,
        output_format="html",
        filename="lunarlander_experiment",
    )
    print(f"  Отчет по эксперименту: {experiment_report}")
    
    print("\n📋 Создание сводного отчета...")
    
    # Данные для сводного отчета
    experiments_data = {
        "LunarLander_Experiment": {
            "description": "Сравнение алгоритмов на LunarLander-v2",
            "agents": ["PPO", "DQN", "A2C"],
            "best_agent": "PPO_Agent",
            "best_reward": 185.7,
            "duration": "2.5 hours",
            "status": "completed",
        },
        "CartPole_Experiment": {
            "description": "Тестирование на простой задаче CartPole-v1",
            "agents": ["PPO", "DQN"],
            "best_agent": "PPO_Agent",
            "best_reward": 500.0,
            "duration": "1.0 hour",
            "status": "completed",
        },
        "MountainCar_Experiment": {
            "description": "Сложная задача с разреженными наградами",
            "agents": ["PPO", "DQN", "A2C"],
            "best_agent": "A2C_Agent",
            "best_reward": -95.3,
            "duration": "3.2 hours",
            "status": "in_progress",
        },
    }
    
    summary_report = formatter_ru.format_summary_report(
        experiments_data=experiments_data,
        output_format="html",
        filename="experiments_summary",
    )
    print(f"  Сводный отчет: {summary_report}")
    
    print("\n💾 Экспорт в табличные форматы...")
    
    # Экспорт в CSV
    csv_path = formatter_ru.export_to_csv(
        data=agents_results,
        filename="agents_comparison_results",
    )
    print(f"  CSV файл: {csv_path}")
    
    # Экспорт в JSON
    json_data = {
        "experiment_info": experiment_data,
        "agents_results": {
            name: {
                "mean_reward": results.mean_reward,
                "std_reward": results.std_reward,
                "success_rate": results.success_rate,
                "total_episodes": results.total_episodes,
            }
            for name, results in agents_results.items()
        },
        "metadata": {
            "generated_by": "ResultsFormatter",
            "format_version": "1.0",
            "language": "ru",
        },
    }
    
    json_path = formatter_ru.export_to_json(
        data=json_data,
        filename="experiment_full_results",
    )
    print(f"  JSON файл: {json_path}")
    
    print("\n✅ Демонстрация завершена!")
    print(f"\nВсе отчеты сохранены в директории: {output_dir}")
    print("\nТипы созданных отчетов:")
    print("  • Отчеты по отдельным агентам (HTML, Markdown)")
    print("  • Сравнительные отчеты агентов (HTML)")
    print("  • Отчет по эксперименту (HTML)")
    print("  • Сводный отчет по экспериментам (HTML)")
    print("  • Экспорт данных (CSV, JSON)")
    print("\n🌐 Поддерживаемые языки: русский, английский")
    print("📊 Поддерживаемые форматы: HTML, Markdown, LaTeX, JSON, CSV")


if __name__ == "__main__":
    main()