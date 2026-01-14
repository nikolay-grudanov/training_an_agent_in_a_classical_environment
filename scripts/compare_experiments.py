#!/usr/bin/env python3
"""CLI скрипт для сравнения RL экспериментов.

Этот скрипт предоставляет удобный интерфейс командной строки
для сравнения и анализа результатов RL экспериментов.

Использование:
    python scripts/compare_experiments.py exp1.json exp2.json exp3.json
    python scripts/compare_experiments.py --dir results/experiments/
    python scripts/compare_experiments.py --config comparison_config.yaml exp1.json exp2.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.comparison import (
    ComparisonConfig,
    ExperimentComparator,
    compare_experiments_cli,
)
from src.experiments.experiment import Experiment

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: Path) -> ComparisonConfig:
    """Загрузить конфигурацию из файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Объект конфигурации сравнения
        
    Raises:
        ValueError: Если файл конфигурации невалиден
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат конфигурации: {config_path.suffix}")
        
        # Фильтруем только поддерживаемые параметры
        supported_params = {
            'significance_level', 'confidence_level', 'bootstrap_samples',
            'multiple_comparison_method', 'effect_size_method',
            'convergence_threshold', 'convergence_window', 'stability_window',
            'min_sample_size'
        }
        
        filtered_config = {
            k: v for k, v in config_data.items() 
            if k in supported_params
        }
        
        return ComparisonConfig(**filtered_config)
        
    except Exception as e:
        raise ValueError(f"Ошибка загрузки конфигурации из {config_path}: {e}")


def find_experiment_files(directory: Path) -> List[Path]:
    """Найти файлы экспериментов в директории.
    
    Args:
        directory: Директория для поиска
        
    Returns:
        Список путей к файлам экспериментов
    """
    experiment_files = []
    
    # Ищем файлы с паттернами experiment_*.json и experiment_*.pickle
    patterns = ['experiment_*.json', 'experiment_*.pickle', '*_experiment.json']
    
    for pattern in patterns:
        experiment_files.extend(directory.glob(pattern))
    
    # Удаляем дубликаты и сортируем
    experiment_files = sorted(set(experiment_files))
    
    logger.info(f"Найдено {len(experiment_files)} файлов экспериментов в {directory}")
    return experiment_files


def validate_experiment_files(file_paths: List[Path]) -> List[Path]:
    """Валидировать файлы экспериментов.
    
    Args:
        file_paths: Список путей к файлам
        
    Returns:
        Список валидных файлов
    """
    valid_files = []
    
    for file_path in file_paths:
        try:
            if not file_path.exists():
                logger.warning(f"Файл не найден: {file_path}")
                continue
            
            # Пытаемся загрузить эксперимент для проверки
            experiment = Experiment.load(file_path)
            
            # Проверяем наличие результатов
            if not experiment.results or 'baseline' not in experiment.results:
                logger.warning(f"Эксперимент {file_path} не содержит результатов")
                continue
            
            valid_files.append(file_path)
            logger.debug(f"Валидный файл эксперимента: {file_path}")
            
        except Exception as e:
            logger.warning(f"Ошибка валидации {file_path}: {e}")
            continue
    
    logger.info(f"Валидных файлов экспериментов: {len(valid_files)} из {len(file_paths)}")
    return valid_files


def create_comparison_summary(comparison_result) -> str:
    """Создать краткую сводку результатов сравнения.
    
    Args:
        comparison_result: Результат сравнения экспериментов
        
    Returns:
        Текстовая сводка
    """
    lines = []
    lines.append("=" * 60)
    lines.append("СВОДКА РЕЗУЛЬТАТОВ СРАВНЕНИЯ")
    lines.append("=" * 60)
    
    # Основная информация
    lines.append(f"Количество экспериментов: {len(comparison_result.experiment_ids)}")
    lines.append(f"Эксперименты: {', '.join(comparison_result.experiment_ids)}")
    lines.append("")
    
    # Лучший эксперимент
    if 'overall' in comparison_result.rankings:
        best_experiment = comparison_result.rankings['overall'][0]
        lines.append(f"🏆 Лучший эксперимент: {best_experiment}")
    
    # Метрики производительности
    lines.append("\n📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
    lines.append("-" * 40)
    
    for exp_id, metrics in comparison_result.performance_metrics.items():
        lines.append(f"\n{exp_id}:")
        lines.append(f"  Средняя награда: {metrics.mean_reward:.2f}")
        lines.append(f"  Стабильность: {metrics.stability_score:.3f}")
        lines.append(f"  Эффективность: {metrics.sample_efficiency:.3f}")
        if metrics.convergence_timesteps:
            lines.append(f"  Сходимость: {metrics.convergence_timesteps} шагов")
    
    # Статистическая значимость
    lines.append("\n🔬 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    lines.append("-" * 35)
    
    total_tests = 0
    significant_tests = 0
    
    for metric, tests in comparison_result.statistical_tests.items():
        metric_significant = sum(1 for test in tests.values() if test.significant)
        metric_total = len(tests)
        total_tests += metric_total
        significant_tests += metric_significant
        
        lines.append(f"{metric}: {metric_significant}/{metric_total} значимых различий")
    
    if total_tests > 0:
        significance_rate = significant_tests / total_tests * 100
        lines.append(f"\nОбщая значимость: {significant_tests}/{total_tests} ({significance_rate:.1f}%)")
    
    # Рейтинги
    lines.append("\n🏆 РЕЙТИНГИ:")
    lines.append("-" * 15)
    
    for metric, ranking in comparison_result.rankings.items():
        if metric != 'overall':
            lines.append(f"{metric}: {' > '.join(ranking)}")
    
    # Ключевые рекомендации
    lines.append("\n💡 КЛЮЧЕВЫЕ РЕКОМЕНДАЦИИ:")
    lines.append("-" * 30)
    
    for i, recommendation in enumerate(comparison_result.recommendations[:3], 1):
        lines.append(f"{i}. {recommendation}")
    
    if len(comparison_result.recommendations) > 3:
        lines.append(f"... и еще {len(comparison_result.recommendations) - 3} рекомендаций")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Сравнение и анализ RL экспериментов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Сравнить конкретные эксперименты
  python scripts/compare_experiments.py exp1.json exp2.json exp3.json

  # Сравнить все эксперименты в директории
  python scripts/compare_experiments.py --dir results/experiments/

  # Использовать кастомную конфигурацию
  python scripts/compare_experiments.py --config my_config.yaml exp1.json exp2.json

  # Сохранить результаты в конкретную директорию
  python scripts/compare_experiments.py --output results/comparison/ exp1.json exp2.json

  # Анализировать конкретные метрики
  python scripts/compare_experiments.py --metrics mean_reward stability_score exp1.json exp2.json

  # Генерировать только JSON отчет
  python scripts/compare_experiments.py --format json --no-plots exp1.json exp2.json
        """
    )
    
    # Основные аргументы
    parser.add_argument(
        'experiments',
        nargs='*',
        help='Пути к файлам экспериментов (.json или .pickle)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=Path,
        help='Директория с файлами экспериментов'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Файл конфигурации сравнения (.yaml или .json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Директория для сохранения результатов'
    )
    
    # Параметры анализа
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        default=['mean_reward', 'stability_score', 'sample_efficiency'],
        help='Метрики для сравнения'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['html', 'markdown', 'json'],
        default='html',
        help='Формат отчета'
    )
    
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Уровень значимости для статистических тестов'
    )
    
    # Флаги
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Не создавать графики'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Не экспортировать данные в CSV/JSON'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Минимальный вывод'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Определяем файлы экспериментов
        experiment_files = []
        
        if args.dir:
            if not args.dir.exists():
                logger.error(f"Директория не найдена: {args.dir}")
                return 1
            
            experiment_files.extend(find_experiment_files(args.dir))
        
        if args.experiments:
            experiment_files.extend([Path(f) for f in args.experiments])
        
        if not experiment_files:
            logger.error("Не указаны файлы экспериментов. Используйте --help для справки.")
            return 1
        
        # Валидируем файлы
        valid_files = validate_experiment_files(experiment_files)
        
        if len(valid_files) < 2:
            logger.error("Для сравнения необходимо минимум 2 валидных эксперимента")
            return 1
        
        # Загружаем конфигурацию
        if args.config:
            if not args.config.exists():
                logger.error(f"Файл конфигурации не найден: {args.config}")
                return 1
            
            try:
                config = load_config_from_file(args.config)
                logger.info(f"Загружена конфигурация из {args.config}")
            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации: {e}")
                return 1
        else:
            config = ComparisonConfig(significance_level=args.significance_level)
        
        # Определяем директорию вывода
        output_dir = args.output or Path("results/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем компаратор
        comparator = ExperimentComparator(config, output_dir)
        
        # Загружаем эксперименты
        logger.info(f"Загрузка {len(valid_files)} экспериментов...")
        experiments = []
        
        for file_path in valid_files:
            try:
                experiment = Experiment.load(file_path)
                experiments.append(experiment)
                logger.debug(f"Загружен эксперимент: {experiment.experiment_id}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки {file_path}: {e}")
                continue
        
        if len(experiments) < 2:
            logger.error("Недостаточно успешно загруженных экспериментов")
            return 1
        
        # Выполняем сравнение
        logger.info(f"Сравнение {len(experiments)} экспериментов по метрикам: {args.metrics}")
        comparison_result = comparator.compare_experiments(experiments, args.metrics, config)
        
        # Выводим краткую сводку
        if not args.quiet:
            summary = create_comparison_summary(comparison_result)
            print(summary)
        
        # Генерируем отчет
        logger.info(f"Генерация отчета в формате {args.format}...")
        report_path = comparator.generate_comparison_report(
            comparison_result,
            include_plots=not args.no_plots,
            output_format=args.format
        )
        
        print(f"\n📄 Отчет сохранен: {report_path}")
        
        # Создаем графики
        if not args.no_plots:
            logger.info("Создание графиков...")
            try:
                plots = comparator.generate_comparison_plots(comparison_result)
                for plot_type, plot_path in plots.items():
                    print(f"📈 График {plot_type}: {plot_path}")
            except Exception as e:
                logger.warning(f"Ошибка создания графиков: {e}")
        
        # Экспортируем данные
        if not args.no_export:
            logger.info("Экспорт данных...")
            try:
                exported_files = comparator.export_results(
                    comparison_result, formats=['csv', 'json']
                )
                for format_type, file_path in exported_files.items():
                    print(f"💾 Экспорт {format_type.upper()}: {file_path}")
            except Exception as e:
                logger.warning(f"Ошибка экспорта данных: {e}")
        
        print(f"\n✅ Сравнение завершено успешно!")
        print(f"📁 Все результаты сохранены в: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        return 1
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())