#!/usr/bin/env python3
"""Пример использования модуля сравнения экспериментов.

Этот скрипт демонстрирует основные возможности модуля comparison.py
для анализа и сравнения результатов RL экспериментов.
"""

import logging
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.comparison import (
    ComparisonConfig,
    EffectSizeMethod,
    ExperimentComparator,
    MultipleComparisonMethod,
    StatisticalTest,
)
from src.experiments.experiment import Experiment
from src.utils.config import AlgorithmConfig, EnvironmentConfig, RLConfig, TrainingConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_experiments():
    """Создать образцы экспериментов для демонстрации."""
    logger.info("Создание образцов экспериментов...")
    
    # Базовая конфигурация
    base_config = RLConfig(
        experiment_name="baseline_experiment",
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64
        ),
        environment=EnvironmentConfig(name="LunarLander-v3"),
        training=TrainingConfig(total_timesteps=200000)
    )
    
    # Вариантные конфигурации
    configs = [
        # Эксперимент 1: Стандартный PPO
        (base_config, "Стандартная конфигурация PPO"),
        
        # Эксперимент 2: Увеличенная скорость обучения
        (RLConfig(
            experiment_name="high_lr_experiment",
            algorithm=AlgorithmConfig(
                name="PPO",
                learning_rate=1e-3,  # Увеличенная скорость обучения
                n_steps=2048,
                batch_size=64
            ),
            environment=EnvironmentConfig(name="LunarLander-v3"),
            training=TrainingConfig(total_timesteps=200000)
        ), "PPO с увеличенной скоростью обучения"),
        
        # Эксперимент 3: Больший batch size
        (RLConfig(
            experiment_name="large_batch_experiment",
            algorithm=AlgorithmConfig(
                name="PPO",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=128  # Увеличенный batch size
            ),
            environment=EnvironmentConfig(name="LunarLander-v3"),
            training=TrainingConfig(total_timesteps=200000)
        ), "PPO с увеличенным batch size"),
        
        # Эксперимент 4: Другой алгоритм (A2C)
        (RLConfig(
            experiment_name="a2c_experiment",
            algorithm=AlgorithmConfig(
                name="A2C",
                learning_rate=7e-4,
                n_steps=5
            ),
            environment=EnvironmentConfig(name="LunarLander-v3"),
            training=TrainingConfig(total_timesteps=200000)
        ), "Алгоритм A2C для сравнения")
    ]
    
    experiments = []
    
    for i, (config, hypothesis) in enumerate(configs):
        # Создаем эксперимент
        # Создаем слегка измененную variant конфигурацию
        variant_config = RLConfig(
            experiment_name=config.experiment_name + "_variant",
            algorithm=AlgorithmConfig(
                name=config.algorithm.name,
                learning_rate=config.algorithm.learning_rate * 1.1,  # Небольшое изменение
                n_steps=config.algorithm.n_steps,
                batch_size=config.algorithm.batch_size
            ),
            environment=config.environment,
            training=config.training
        )
        
        experiment = Experiment(
            baseline_config=config,
            variant_config=variant_config,
            hypothesis=hypothesis,
            experiment_id=f"exp_{i+1}"
        )
        
        # Симулируем результаты обучения
        # В реальности эти данные получаются из фактического обучения
        base_reward = 150 + i * 20  # Разные базовые награды
        noise_level = 0.1 + i * 0.05  # Разные уровни шума
        
        # Генерируем историю метрик
        metrics_history = []
        for episode in range(100):
            # Симулируем прогресс обучения с шумом
            progress = episode / 100.0
            reward = base_reward * (0.5 + 0.5 * progress) + \
                    (2 * (0.5 - __import__('random').random()) * noise_level * base_reward)
            
            metrics_history.append({
                "episode_reward": reward,
                "episode_length": 200 + int(50 * (0.5 - __import__('random').random())),
                "timestep": episode * 2000,
                "episode": episode
            })
        
        # Вычисляем агрегированные метрики
        episode_rewards = [m["episode_reward"] for m in metrics_history]
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        final_reward = sum(episode_rewards[-10:]) / 10  # Среднее за последние 10 эпизодов
        
        # Добавляем результаты
        experiment.add_result("baseline", {
            "mean_reward": mean_reward,
            "final_reward": final_reward,
            "max_reward": max(episode_rewards),
            "min_reward": min(episode_rewards),
            "std_reward": (__import__('statistics').stdev(episode_rewards) 
                          if len(episode_rewards) > 1 else 0),
            "training_time": 3600 + i * 600,  # Разное время обучения
            "convergence_timesteps": 50000 + i * 10000,
            "metrics_history": metrics_history
        })
        
        experiments.append(experiment)
        logger.info(f"Создан эксперимент {experiment.experiment_id}: {hypothesis}")
    
    return experiments


def demonstrate_basic_comparison():
    """Демонстрация базового сравнения экспериментов."""
    logger.info("=== Демонстрация базового сравнения экспериментов ===")
    
    # Создаем образцы экспериментов
    experiments = create_sample_experiments()
    
    # Создаем конфигурацию сравнения
    config = ComparisonConfig(
        significance_level=0.05,
        confidence_level=0.95,
        multiple_comparison_method=MultipleComparisonMethod.FDR_BH,
        effect_size_method=EffectSizeMethod.COHENS_D
    )
    
    # Создаем компаратор
    comparator = ExperimentComparator(config, "results/comparison_demo")
    
    # Выполняем сравнение
    metrics = ['mean_reward', 'stability_score', 'sample_efficiency']
    comparison_result = comparator.compare_experiments(experiments, metrics)
    
    # Выводим результаты
    logger.info(f"Сравнение {len(experiments)} экспериментов завершено")
    logger.info(f"Проанализированы метрики: {metrics}")
    
    # Показываем метрики производительности
    print("\n📊 Метрики производительности:")
    print("-" * 60)
    for exp_id, perf_metrics in comparison_result.performance_metrics.items():
        print(f"{exp_id}:")
        print(f"  Средняя награда: {perf_metrics.mean_reward:.2f}")
        print(f"  Стабильность: {perf_metrics.stability_score:.3f}")
        print(f"  Эффективность: {perf_metrics.sample_efficiency:.3f}")
        print()
    
    # Показываем рейтинги
    print("🏆 Рейтинги:")
    print("-" * 30)
    for metric, ranking in comparison_result.rankings.items():
        print(f"{metric}: {' > '.join(ranking)}")
    
    # Показываем рекомендации
    print("\n💡 Рекомендации:")
    print("-" * 40)
    for i, recommendation in enumerate(comparison_result.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    return comparison_result


def demonstrate_statistical_analysis():
    """Демонстрация статистического анализа."""
    logger.info("=== Демонстрация статистического анализа ===")
    
    # Создаем компаратор
    comparator = ExperimentComparator()
    
    # Генерируем образцы данных
    import random
    random.seed(42)
    
    data1 = [150 + random.gauss(0, 20) for _ in range(50)]  # Группа 1
    data2 = [170 + random.gauss(0, 25) for _ in range(50)]  # Группа 2 (лучше)
    
    # Проводим различные статистические тесты
    tests = [
        StatisticalTest.T_TEST,
        StatisticalTest.MANN_WHITNEY,
        StatisticalTest.BOOTSTRAP
    ]
    
    print("\n🔬 Результаты статистических тестов:")
    print("-" * 50)
    
    for test in tests:
        try:
            result = comparator.statistical_significance(data1, data2, test)
            
            print(f"\n{test.value.upper()}:")
            print(f"  Статистика: {result.statistic:.3f}")
            print(f"  p-value: {result.p_value:.4f}")
            print(f"  Значимо: {'Да' if result.significant else 'Нет'}")
            if result.effect_size:
                print(f"  Размер эффекта: {result.effect_size:.3f}")
        except Exception as e:
            print(f"{test.value}: Ошибка - {e}")
    
    # Доверительные интервалы
    ci_lower, ci_upper = comparator.confidence_intervals(data1)
    print(f"\n📊 95% доверительный интервал для группы 1: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    ci_lower, ci_upper = comparator.confidence_intervals(data2)
    print(f"📊 95% доверительный интервал для группы 2: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Размер эффекта
    effect_size = comparator.effect_size(data1, data2)
    print(f"\n📏 Размер эффекта (Cohen's d): {effect_size:.3f}")
    
    if effect_size < 0.2:
        effect_desc = "малый"
    elif effect_size < 0.5:
        effect_desc = "средний"
    elif effect_size < 0.8:
        effect_desc = "большой"
    else:
        effect_desc = "очень большой"
    
    print(f"   Интерпретация: {effect_desc} эффект")


def demonstrate_advanced_analysis():
    """Демонстрация продвинутого анализа."""
    logger.info("=== Демонстрация продвинутого анализа ===")
    
    experiments = create_sample_experiments()
    comparator = ExperimentComparator(output_dir="results/advanced_demo")
    
    # Анализ сходимости
    print("\n🎯 Анализ сходимости:")
    print("-" * 30)
    
    for exp in experiments[:2]:  # Анализируем первые 2 эксперимента
        try:
            convergence_info = comparator.convergence_analysis(exp, threshold=160.0)
            
            print(f"\n{exp.experiment_id}:")
            print(f"  Сошелся: {'Да' if convergence_info['converged'] else 'Нет'}")
            if convergence_info['convergence_timestep']:
                print(f"  Шаги до сходимости: {convergence_info['convergence_timestep']}")
            print(f"  Финальное значение: {convergence_info['final_value']:.2f}")
            print(f"  Максимальное значение: {convergence_info['max_value']:.2f}")
        except Exception as e:
            print(f"{exp.experiment_id}: Ошибка анализа сходимости - {e}")
    
    # Анализ эффективности обучения
    print("\n⚡ Анализ эффективности обучения:")
    print("-" * 40)
    
    efficiency_results = comparator.learning_efficiency(experiments, threshold=160.0)
    
    for exp_id, results in efficiency_results.items():
        print(f"\n{exp_id}:")
        print(f"  Достиг порога: {'Да' if results['achieved_threshold'] else 'Нет'}")
        if results['steps_to_threshold']:
            print(f"  Шаги до порога: {results['steps_to_threshold']}")
        print(f"  Эффективность выборки: {results['sample_efficiency']:.4f}")
    
    # Анализ стабильности
    print("\n🎢 Анализ стабильности:")
    print("-" * 25)
    
    stability_results = comparator.stability_analysis(experiments)
    
    for exp_id, results in stability_results.items():
        print(f"\n{exp_id}:")
        print(f"  Коэффициент вариации: {results['coefficient_of_variation']:.3f}")
        print(f"  Оценка стабильности: {results['stability_score']:.3f}")
        
        if results['stability_score'] > 0.8:
            stability_desc = "очень стабильный"
        elif results['stability_score'] > 0.6:
            stability_desc = "стабильный"
        elif results['stability_score'] > 0.4:
            stability_desc = "умеренно стабильный"
        else:
            stability_desc = "нестабильный"
        
        print(f"  Интерпретация: {stability_desc}")


def demonstrate_visualization():
    """Демонстрация создания визуализаций."""
    logger.info("=== Демонстрация визуализации ===")
    
    experiments = create_sample_experiments()
    comparator = ExperimentComparator(output_dir="results/visualization_demo")
    
    # Создаем сравнение
    comparison_result = comparator.compare_experiments(experiments)
    
    # Генерируем различные графики
    print("\n📈 Создание графиков...")
    
    try:
        # Кривые обучения
        learning_curves_path = comparator.learning_curves_comparison(
            experiments, metric='episode_reward'
        )
        if learning_curves_path:
            print(f"✅ Кривые обучения: {learning_curves_path}")
        
        # Распределения
        distribution_path = comparator.distribution_plots(
            experiments, metric='episode_reward'
        )
        if distribution_path:
            print(f"✅ Распределения: {distribution_path}")
        
        # Box plots
        box_plots_path = comparator.box_plots(
            experiments, metrics=['episode_reward']
        )
        if box_plots_path:
            print(f"✅ Box plots: {box_plots_path}")
        
        # Heatmap сравнения
        heatmap_path = comparator.heatmap_comparison(comparison_result)
        if heatmap_path:
            print(f"✅ Heatmap: {heatmap_path}")
        
        # Комплексные графики сравнения
        comparison_plots = comparator.generate_comparison_plots(comparison_result)
        for plot_type, plot_path in comparison_plots.items():
            print(f"✅ {plot_type}: {plot_path}")
            
    except Exception as e:
        print(f"❌ Ошибка создания графиков: {e}")
        logger.error(f"Visualization error: {e}")


def demonstrate_report_generation():
    """Демонстрация генерации отчетов."""
    logger.info("=== Демонстрация генерации отчетов ===")
    
    experiments = create_sample_experiments()
    comparator = ExperimentComparator(output_dir="results/reports_demo")
    
    # Создаем сравнение
    comparison_result = comparator.compare_experiments(experiments)
    
    print("\n📄 Генерация отчетов...")
    
    # HTML отчет
    try:
        html_report = comparator.generate_comparison_report(
            comparison_result, include_plots=True, output_format='html'
        )
        print(f"✅ HTML отчет: {html_report}")
    except Exception as e:
        print(f"❌ Ошибка HTML отчета: {e}")
    
    # Markdown отчет
    try:
        md_report = comparator.generate_comparison_report(
            comparison_result, include_plots=False, output_format='markdown'
        )
        print(f"✅ Markdown отчет: {md_report}")
    except Exception as e:
        print(f"❌ Ошибка Markdown отчета: {e}")
    
    # JSON отчет
    try:
        json_report = comparator.generate_comparison_report(
            comparison_result, include_plots=False, output_format='json'
        )
        print(f"✅ JSON отчет: {json_report}")
    except Exception as e:
        print(f"❌ Ошибка JSON отчета: {e}")
    
    # Экспорт в различные форматы
    try:
        exported_files = comparator.export_results(
            comparison_result, formats=['csv', 'json']
        )
        for format_type, file_path in exported_files.items():
            print(f"✅ Экспорт {format_type.upper()}: {file_path}")
    except Exception as e:
        print(f"❌ Ошибка экспорта: {e}")
    
    # Форматированные результаты статистических тестов
    try:
        table_results = comparator.hypothesis_test_results(
            comparison_result, format_type='table'
        )
        print(f"\n📊 Таблица статистических тестов создана ({len(table_results)} символов)")
        
        summary_results = comparator.hypothesis_test_results(
            comparison_result, format_type='summary'
        )
        print(f"📊 Сводка статистических тестов создана ({len(summary_results)} символов)")
    except Exception as e:
        print(f"❌ Ошибка форматирования результатов: {e}")


def main():
    """Главная функция демонстрации."""
    logger.info("Запуск демонстрации модуля сравнения экспериментов")
    
    try:
        # Создаем директорию для результатов
        Path("results").mkdir(exist_ok=True)
        
        print("🚀 Демонстрация модуля сравнения экспериментов")
        print("=" * 60)
        
        # Базовое сравнение
        comparison_result = demonstrate_basic_comparison()
        
        # Статистический анализ
        demonstrate_statistical_analysis()
        
        # Продвинутый анализ
        demonstrate_advanced_analysis()
        
        # Визуализация
        demonstrate_visualization()
        
        # Генерация отчетов
        demonstrate_report_generation()
        
        print("\n✅ Демонстрация завершена успешно!")
        print(f"📁 Результаты сохранены в директории: results/")
        
    except Exception as e:
        logger.error(f"Ошибка во время демонстрации: {e}")
        print(f"\n❌ Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())