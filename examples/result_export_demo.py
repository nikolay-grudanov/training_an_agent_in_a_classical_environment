#!/usr/bin/env python3
"""Демонстрация функциональности экспорта результатов экспериментов.

Этот скрипт показывает, как использовать ResultExporter для экспорта
результатов RL экспериментов в различные форматы с интеграцией
снимков зависимостей и валидацией целостности.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment import Experiment, ExperimentStatus
from src.experiments.result_exporter import (
    CompressionType,
    ExportFormat,
    ResultExporter,
    export_experiment_results,
    export_multiple_experiments_results,
)
from src.utils.config import RLConfig


def create_mock_config(algorithm_name: str = "PPO") -> Mock:
    """Создать мок конфигурации для демонстрации."""
    config = Mock(spec=RLConfig)
    
    # Создаем мок алгоритма
    algorithm_mock = Mock()
    algorithm_mock.name = algorithm_name
    algorithm_mock.learning_rate = 0.001
    config.algorithm = algorithm_mock
    
    # Создаем мок среды
    environment_mock = Mock()
    environment_mock.name = "LunarLander-v2"
    config.environment = environment_mock
    
    # Создаем мок обучения
    training_mock = Mock()
    training_mock.total_timesteps = 100000
    config.training = training_mock
    
    return config


def create_mock_experiment(experiment_id: str, algorithm: str = "PPO") -> Mock:
    """Создать мок эксперимента для демонстрации."""
    experiment = Mock(spec=Experiment)
    experiment.experiment_id = experiment_id
    experiment.status = ExperimentStatus.COMPLETED
    experiment.hypothesis = f"Тестирование алгоритма {algorithm} на LunarLander-v2"
    
    baseline_config = create_mock_config(algorithm)
    variant_config = create_mock_config(algorithm)
    variant_config.algorithm.learning_rate = 0.002  # Изменяем learning rate
    
    experiment.baseline_config = baseline_config
    experiment.variant_config = variant_config
    
    # Создаем результаты эксперимента
    experiment.results = {
        "baseline": {
            "mean_reward": 150.5,
            "final_reward": 200.0,
            "convergence_timesteps": 50000,
            "training_time": 1800.0,
            "metrics_history": [
                {"timestep": 10000, "episode_reward": 80.0, "episode_length": 250},
                {"timestep": 20000, "episode_reward": 120.0, "episode_length": 220},
                {"timestep": 30000, "episode_reward": 160.0, "episode_length": 200},
                {"timestep": 40000, "episode_reward": 180.0, "episode_length": 180},
                {"timestep": 50000, "episode_reward": 200.0, "episode_length": 160}
            ]
        },
        "variant": {
            "mean_reward": 175.2,
            "final_reward": 220.0,
            "convergence_timesteps": 45000,
            "training_time": 1750.0,
            "metrics_history": [
                {"timestep": 10000, "episode_reward": 100.0, "episode_length": 240},
                {"timestep": 20000, "episode_reward": 140.0, "episode_length": 210},
                {"timestep": 30000, "episode_reward": 180.0, "episode_length": 190},
                {"timestep": 40000, "episode_reward": 200.0, "episode_length": 170},
                {"timestep": 45000, "episode_reward": 220.0, "episode_length": 150}
            ]
        },
        "comparison": {
            "performance_metrics": {
                "mean_reward": {
                    "baseline": 150.5,
                    "variant": 175.2,
                    "improvement": 24.7,
                    "improvement_percent": 16.4,
                    "better": "variant"
                }
            },
            "summary": {
                "overall_better": "variant",
                "reward_improvement": 24.7,
                "significant_improvement": True
            }
        }
    }
    
    # Временные метки
    experiment.created_at = datetime.now()
    experiment.started_at = datetime.now()
    experiment.completed_at = datetime.now()
    
    return experiment


def demo_single_experiment_export():
    """Демонстрация экспорта одного эксперимента."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ЭКСПОРТА ОДНОГО ЭКСПЕРИМЕНТА")
    print("=" * 60)
    
    # Создаем временную директорию для демонстрации
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "demo_exports"
        
        # Создаем мок эксперимента
        experiment = create_mock_experiment("demo_exp_001", "PPO")
        
        # Создаем экспортер
        exporter = ResultExporter(
            output_dir=output_dir,
            include_dependencies=True,  # Включаем снимки зависимостей
            validate_integrity=True,    # Включаем валидацию
            auto_compress=False         # Отключаем автосжатие для демонстрации
        )
        
        print(f"Экспортируем эксперимент: {experiment.experiment_id}")
        print(f"Директория экспорта: {output_dir}")
        
        # Экспортируем в различные форматы
        formats = [
            ExportFormat.JSON,
            ExportFormat.CSV,
            ExportFormat.PICKLE,
            ExportFormat.EXCEL
        ]
        
        try:
            result = exporter.export_experiment(
                experiment,
                formats=formats,
                include_raw_data=True,
                include_plots=False  # Отключаем графики для демонстрации
            )
            
            print(f"\n✅ Экспорт успешно завершен!")
            print(f"Тип экспорта: {result['export_type']}")
            print(f"ID эксперимента: {result['experiment_id']}")
            print(f"Экспортированные форматы: {result['exported_formats']}")
            
            # Показываем созданные файлы
            export_dir = Path(result['export_dir'])
            print(f"\nСозданные файлы:")
            for file_path in export_dir.rglob('*'):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  📄 {file_path.name} ({size_mb:.2f} MB)")
            
            # Валидация целостности
            if result.get('validation'):
                validation = result['validation']
                print(f"\n🔍 Валидация целостности:")
                print(f"  Валидность: {'✅ Да' if validation['valid'] else '❌ Нет'}")
                print(f"  Проверено файлов: {len(validation['checked_files'])}")
                if validation['errors']:
                    print(f"  Ошибки: {validation['errors']}")
            
            # Демонстрируем сжатие
            print(f"\n📦 Сжимаем экспорт...")
            compressed_path = exporter.compress_export(
                export_dir,
                compression_type=CompressionType.ZIP
            )
            
            compressed_size = Path(compressed_path).stat().st_size / (1024 * 1024)
            print(f"✅ Создан архив: {Path(compressed_path).name} ({compressed_size:.2f} MB)")
            
        except Exception as e:
            print(f"❌ Ошибка экспорта: {e}")


def demo_multiple_experiments_export():
    """Демонстрация экспорта нескольких экспериментов."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ЭКСПОРТА НЕСКОЛЬКИХ ЭКСПЕРИМЕНТОВ")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "multi_demo_exports"
        
        # Создаем несколько экспериментов
        experiments = [
            create_mock_experiment("multi_exp_001", "PPO"),
            create_mock_experiment("multi_exp_002", "A2C"),
            create_mock_experiment("multi_exp_003", "SAC")
        ]
        
        # Изменяем результаты для разнообразия
        experiments[1].results["baseline"]["mean_reward"] = 140.0
        experiments[1].results["variant"]["mean_reward"] = 165.0
        
        experiments[2].results["baseline"]["mean_reward"] = 160.0
        experiments[2].results["variant"]["mean_reward"] = 185.0
        
        print(f"Экспортируем {len(experiments)} экспериментов:")
        for exp in experiments:
            print(f"  - {exp.experiment_id} ({exp.baseline_config.algorithm.name})")
        
        # Создаем экспортер
        exporter = ResultExporter(
            output_dir=output_dir,
            include_dependencies=False,  # Отключаем для ускорения
            validate_integrity=True,
            auto_compress=True,          # Включаем автосжатие
            compression_type=CompressionType.ZIP
        )
        
        try:
            result = exporter.export_multiple_experiments(
                experiments,
                formats=[ExportFormat.JSON, ExportFormat.CSV, ExportFormat.EXCEL],
                include_comparison=True,  # Включаем сравнительный анализ
                include_summary=True      # Включаем сводку
            )
            
            print(f"\n✅ Экспорт множественных экспериментов завершен!")
            print(f"Количество экспериментов: {result['experiment_count']}")
            print(f"Экспортированные форматы: {result['exported_formats']}")
            
            # Показываем структуру экспорта
            export_dir = Path(result['export_dir'])
            print(f"\nСтруктура экспорта:")
            for item in export_dir.rglob('*'):
                if item.is_file():
                    relative_path = item.relative_to(export_dir)
                    size_kb = item.stat().st_size / 1024
                    print(f"  📄 {relative_path} ({size_kb:.1f} KB)")
            
            # Информация о сжатом архиве
            if result.get('compressed_archive'):
                archive_path = Path(result['compressed_archive'])
                archive_size = archive_path.stat().st_size / (1024 * 1024)
                print(f"\n📦 Сжатый архив: {archive_path.name} ({archive_size:.2f} MB)")
            
        except Exception as e:
            print(f"❌ Ошибка экспорта: {e}")


def demo_utility_functions():
    """Демонстрация удобных функций экспорта."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ УДОБНЫХ ФУНКЦИЙ ЭКСПОРТА")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "utility_demo"
        
        # Создаем эксперимент
        experiment = create_mock_experiment("utility_exp_001", "TD3")
        
        print(f"Используем удобную функцию export_experiment_results()")
        
        try:
            # Используем удобную функцию
            result = export_experiment_results(
                experiment,
                output_dir=output_dir,
                formats=[ExportFormat.JSON, ExportFormat.CSV],
                include_dependencies=False
            )
            
            print(f"✅ Экспорт через удобную функцию завершен!")
            print(f"Создано файлов: {len(result['exported_files'])}")
            
            # Демонстрируем экспорт нескольких экспериментов
            experiments = [
                create_mock_experiment("utility_multi_001", "PPO"),
                create_mock_experiment("utility_multi_002", "A2C")
            ]
            
            print(f"\nИспользуем export_multiple_experiments_results()")
            
            multi_result = export_multiple_experiments_results(
                experiments,
                output_dir=output_dir,
                formats=[ExportFormat.JSON],
                include_comparison=True
            )
            
            print(f"✅ Множественный экспорт завершен!")
            print(f"Экспериментов: {multi_result['experiment_count']}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def demo_export_management():
    """Демонстрация управления экспортами."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ УПРАВЛЕНИЯ ЭКСПОРТАМИ")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "management_demo"
        
        # Создаем экспортер
        exporter = ResultExporter(
            output_dir=output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Создаем несколько экспортов
        experiments = [
            create_mock_experiment(f"mgmt_exp_{i:03d}", "PPO")
            for i in range(1, 4)
        ]
        
        print("Создаем несколько экспортов...")
        
        for exp in experiments:
            try:
                exporter.export_experiment(exp, formats=[ExportFormat.JSON])
                print(f"  ✅ Экспортирован: {exp.experiment_id}")
            except Exception as e:
                print(f"  ❌ Ошибка экспорта {exp.experiment_id}: {e}")
        
        # Получаем список экспортов
        print(f"\n📋 Список экспортов:")
        exports_list = exporter.list_exports()
        
        for export_info in exports_list:
            print(f"  📦 {export_info['export_name']}")
            print(f"     Тип: {export_info['export_type']}")
            print(f"     Время: {export_info['timestamp']}")
            print(f"     Форматы: {export_info['formats']}")
            print(f"     Сжат: {'Да' if export_info['compressed'] else 'Нет'}")
            print(f"     Валидирован: {'Да' if export_info['validated'] else 'Нет'}")
        
        # Демонстрируем очистку
        print(f"\n🧹 Очистка старых экспортов (сохраняем только 2)...")
        cleanup_result = exporter.cleanup_old_exports(keep_count=2, keep_days=1)
        
        print(f"  Удалено экспортов: {cleanup_result['deleted_count']}")
        print(f"  Освобождено места: {cleanup_result['deleted_size_mb']:.2f} MB")
        
        if cleanup_result['errors']:
            print(f"  Ошибки: {cleanup_result['errors']}")
        
        # Генерируем сводный отчет
        print(f"\n📊 Генерируем сводный отчет...")
        
        remaining_exports = exporter.list_exports()
        export_dirs = [exp['export_dir'] for exp in remaining_exports]
        
        if export_dirs:
            try:
                report_path = exporter.generate_summary_report(
                    export_dirs,
                    include_statistics=True,
                    include_trends=False
                )
                
                report_size = Path(report_path).stat().st_size / 1024
                print(f"  ✅ Отчет создан: {Path(report_path).name} ({report_size:.1f} KB)")
                
                # Показываем начало отчета
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"\n📄 Предварительный просмотр отчета:")
                    print("-" * 40)
                    print(preview)
                    print("-" * 40)
                
            except Exception as e:
                print(f"  ❌ Ошибка создания отчета: {e}")


def main():
    """Главная функция демонстрации."""
    print("🚀 ДЕМОНСТРАЦИЯ СИСТЕМЫ ЭКСПОРТА РЕЗУЛЬТАТОВ RL ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print()
    print("Эта демонстрация показывает возможности ResultExporter:")
    print("• Экспорт в различные форматы (JSON, CSV, Excel, Pickle, HDF5)")
    print("• Интеграция снимков зависимостей")
    print("• Валидация целостности данных")
    print("• Сжатие и архивирование")
    print("• Инкрементальный экспорт")
    print("• Управление экспортами")
    print("• Генерация сводных отчетов")
    print()
    
    try:
        # Запускаем демонстрации
        demo_single_experiment_export()
        demo_multiple_experiments_export()
        demo_utility_functions()
        demo_export_management()
        
        print("\n" + "=" * 80)
        print("🎉 ВСЕ ДЕМОНСТРАЦИИ УСПЕШНО ЗАВЕРШЕНЫ!")
        print("=" * 80)
        print()
        print("Основные возможности продемонстрированы:")
        print("✅ Экспорт одиночных экспериментов")
        print("✅ Экспорт множественных экспериментов")
        print("✅ Различные форматы экспорта")
        print("✅ Сжатие и архивирование")
        print("✅ Валидация целостности")
        print("✅ Управление экспортами")
        print("✅ Генерация отчетов")
        print()
        print("Для использования в реальных проектах:")
        print("1. Импортируйте ResultExporter из src.experiments.result_exporter")
        print("2. Создайте экземпляр с нужными параметрами")
        print("3. Используйте методы export_experiment() или export_multiple_experiments()")
        print("4. При необходимости используйте удобные функции export_*_results()")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА ДЕМОНСТРАЦИИ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()