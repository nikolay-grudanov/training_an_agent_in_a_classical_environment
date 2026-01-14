"""Упрощенные интеграционные тесты воспроизводимости.

Этот модуль содержит упрощенные тесты для быстрой проверки
основной функциональности системы воспроизводимости.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig
from src.utils.dependency_tracker import DependencyTracker
from src.utils.reproducibility_checker import ReproducibilityChecker, StrictnessLevel
from src.utils.seeding import set_seed


def simple_deterministic_function(seed: int = 42) -> Dict[str, Any]:
    """Простая детерминированная функция для тестирования."""
    set_seed(seed)
    np.random.seed(seed)
    
    return {
        'random_value': float(np.random.randn()),
        'random_array': np.random.randn(3).tolist(),
        'computation': float(np.sum(np.random.randn(5))),
        'seed_used': seed
    }


@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def simple_config():
    """Простая конфигурация для тестов."""
    config = RLConfig(
        experiment_name="simple_test",
        seed=42,
        algorithm=AlgorithmConfig(name="PPO", seed=42, device="cpu"),  # Явно указываем CPU
        environment=EnvironmentConfig(name="TestEnv-v1"),
        training=TrainingConfig(total_timesteps=100)
    )
    # Настраиваем воспроизводимость для тестов
    config.reproducibility.use_cuda = False
    config.reproducibility.deterministic = True
    config.reproducibility.benchmark = False
    
    config.enforce_seed_consistency()
    return config


class TestSimpleReproducibility:
    """Упрощенные тесты воспроизводимости."""
    
    def test_basic_seed_consistency(self, simple_config):
        """Тест базовой консистентности сидов."""
        # Проверяем, что сиды синхронизированы
        assert simple_config.seed == 42
        assert simple_config.algorithm.seed == 42
        assert simple_config.reproducibility.seed == 42
        
        # Проверяем валидацию
        is_valid, warnings = simple_config.validate_reproducibility()
        assert is_valid
        # Предупреждения могут быть (например, о device='auto' или CUDA), но это не критично
        assert isinstance(warnings, list)
    
    def test_deterministic_function_reproducibility(self):
        """Тест воспроизводимости детерминированной функции."""
        # Выполняем функцию несколько раз с одним сидом
        results = []
        for _ in range(3):
            result = simple_deterministic_function(seed=42)
            results.append(result)
        
        # Все результаты должны быть идентичны
        first_result = results[0]
        for result in results[1:]:
            assert result['random_value'] == first_result['random_value']
            assert result['random_array'] == first_result['random_array']
            assert result['computation'] == first_result['computation']
            assert result['seed_used'] == first_result['seed_used']
    
    def test_different_seeds_produce_different_results(self):
        """Тест того, что разные сиды дают разные результаты."""
        result1 = simple_deterministic_function(seed=42)
        result2 = simple_deterministic_function(seed=123)
        
        # Результаты должны отличаться
        assert result1['random_value'] != result2['random_value']
        assert result1['random_array'] != result2['random_array']
        assert result1['computation'] != result2['computation']
        assert result1['seed_used'] != result2['seed_used']
    
    def test_dependency_tracker_basic_functionality(self, temp_dir):
        """Тест базовой функциональности трекера зависимостей."""
        tracker = DependencyTracker(temp_dir)
        
        # Создаем снимок
        snapshot = tracker.create_dependency_snapshot("test_snapshot")
        
        # Проверяем структуру снимка
        assert 'metadata' in snapshot
        assert 'system' in snapshot
        assert 'packages' in snapshot
        assert 'ml_libraries' in snapshot
        
        # Проверяем метаданные
        assert snapshot['metadata']['name'] == "test_snapshot"
        assert 'timestamp' in snapshot['metadata']
        assert 'hash' in snapshot['metadata']
        
        # Проверяем системную информацию
        assert 'python' in snapshot['system']
        assert 'platform' in snapshot['system']
        assert 'hardware' in snapshot['system']
    
    def test_reproducibility_checker_basic_workflow(self, temp_dir, simple_config):
        """Тест базового workflow проверщика воспроизводимости."""
        checker = ReproducibilityChecker(
            project_root=temp_dir,
            strictness_level=StrictnessLevel.MINIMAL
        )
        
        experiment_id = "basic_test"
        
        # Регистрируем два идентичных запуска
        for run_num in range(2):
            result = simple_deterministic_function(seed=42)
            
            run_id = checker.register_experiment_run(
                experiment_id=experiment_id,
                config=simple_config,
                results=result,
                metrics={'values': [result['random_value']]},
                metadata={'run_number': run_num}
            )
            
            assert run_id is not None
        
        # Проверяем воспроизводимость
        report = checker.check_reproducibility(experiment_id)
        
        assert report.experiment_id == experiment_id
        assert len(report.runs) == 2
        assert report.is_reproducible
        assert report.confidence_score >= 0.8
    
    def test_reproducibility_checker_detects_differences(self, temp_dir, simple_config):
        """Тест детектирования различий проверщиком воспроизводимости."""
        checker = ReproducibilityChecker(
            project_root=temp_dir,
            strictness_level=StrictnessLevel.STANDARD
        )
        
        experiment_id = "difference_test"
        
        # Первый запуск с сидом 42
        result1 = simple_deterministic_function(seed=42)
        run1_id = checker.register_experiment_run(
            experiment_id=experiment_id,
            config=simple_config,
            results=result1,
            metrics={'values': [result1['random_value']]},
            metadata={'run_number': 1}
        )
        
        # Второй запуск с другим сидом
        different_config = RLConfig(
            experiment_name="simple_test",
            seed=123,  # Другой сид!
            algorithm=AlgorithmConfig(name="PPO", seed=123),
            environment=EnvironmentConfig(name="TestEnv-v1"),
            training=TrainingConfig(total_timesteps=100)
        )
        
        result2 = simple_deterministic_function(seed=123)
        run2_id = checker.register_experiment_run(
            experiment_id=experiment_id,
            config=different_config,
            results=result2,
            metrics={'values': [result2['random_value']]},
            metadata={'run_number': 2}
        )
        
        # Проверяем воспроизводимость
        report = checker.check_reproducibility(experiment_id)
        
        # Должны быть обнаружены различия
        assert not report.is_reproducible
        assert len(report.issues) > 0
        assert len(report.recommendations) > 0
    
    def test_determinism_validation_simple(self, temp_dir):
        """Простой тест валидации детерминизма."""
        checker = ReproducibilityChecker(project_root=temp_dir)
        
        # Тест детерминированной функции
        validation_result = checker.validate_determinism(
            test_function=lambda: simple_deterministic_function(42),
            seed=42,
            num_runs=3
        )
        
        assert validation_result['is_deterministic']
        assert validation_result['unique_results'] == 1
        assert validation_result['success_rate'] == 1.0
        
        # Тест недетерминированной функции
        import time
        
        def non_deterministic_function():
            return {'timestamp': time.time()}
        
        validation_result = checker.validate_determinism(
            test_function=non_deterministic_function,
            seed=42,
            num_runs=3
        )
        
        assert not validation_result['is_deterministic']
        assert validation_result['unique_results'] > 1
    
    def test_config_reproducibility_report(self, simple_config):
        """Тест генерации отчета о воспроизводимости конфигурации."""
        report = simple_config.get_reproducibility_report()
        
        # Проверяем структуру отчета
        assert 'timestamp' in report
        assert 'experiment_name' in report
        assert 'is_valid' in report
        assert 'warnings' in report
        assert 'seeds' in report
        assert 'determinism' in report
        assert 'algorithm' in report
        assert 'system' in report
        assert 'recommendations' in report
        
        # Проверяем валидность
        assert report['is_valid']
        # Предупреждения могут быть (например, о CUDA), но это не критично для валидности
        assert isinstance(report['warnings'], list)
        
        # Проверяем информацию о сидах
        seeds_info = report['seeds']
        assert seeds_info['main_seed'] == 42
        assert seeds_info['seeds_consistent']
        # Проверяем, что есть информация о сидах
        assert 'reproducibility_seed' in seeds_info
        assert 'algorithm_seed' in seeds_info
    
    def test_automatic_reproducibility_test(self, temp_dir, simple_config):
        """Тест автоматического тестирования воспроизводимости."""
        checker = ReproducibilityChecker(project_root=temp_dir)
        
        # Запускаем автоматический тест
        report = checker.run_reproducibility_test(
            test_function=simple_deterministic_function,
            experiment_id="auto_test",
            seeds=[42, 42, 42],  # Три одинаковых сида
            config=simple_config
        )
        
        # Проверяем результаты
        assert report.experiment_id == "auto_test"
        assert len(report.runs) == 3
        assert report.is_reproducible
        assert report.confidence_score >= 0.9
        
        # Все запуски должны иметь идентичные результаты
        first_run = report.runs[0]
        for run in report.runs[1:]:
            assert run.results == first_run.results
    
    def test_snapshot_comparison(self, temp_dir):
        """Тест сравнения снимков зависимостей."""
        tracker = DependencyTracker(temp_dir)
        
        # Создаем два снимка
        snapshot1 = tracker.create_dependency_snapshot("snapshot1")
        snapshot2 = tracker.create_dependency_snapshot("snapshot2")
        
        # Сравниваем снимки
        comparison = tracker.compare_snapshots("snapshot1", "snapshot2")
        
        # Проверяем структуру сравнения
        assert 'metadata' in comparison
        assert 'changes' in comparison
        
        # Проверяем метаданные
        assert comparison['metadata']['snapshot1'] == "snapshot1"
        assert comparison['metadata']['snapshot2'] == "snapshot2"
        
        # Для идентичных снимков не должно быть изменений
        changes = comparison['changes']
        assert len(changes['packages_added']) == 0
        assert len(changes['packages_removed']) == 0
        assert len(changes['packages_updated']) == 0
    
    def test_export_with_dependencies(self, temp_dir, simple_config):
        """Тест экспорта с включением зависимостей."""
        from src.experiments.result_exporter import ResultExporter
        from unittest.mock import MagicMock
        
        exporter = ResultExporter(
            output_dir=temp_dir / "exports",
            include_dependencies=True,
            validate_integrity=True
        )
        
        # Создаем мок эксперимента
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "export_test"
        mock_experiment.status.value = "completed"
        mock_experiment.hypothesis = "Тест экспорта"
        mock_experiment.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_experiment.started_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_experiment.completed_at.isoformat.return_value = "2024-01-01T01:00:00"
        mock_experiment.results = {
            'baseline': simple_deterministic_function(42),
            'variant': simple_deterministic_function(42)
        }
        mock_experiment.baseline_config = simple_config
        mock_experiment.variant_config = simple_config
        
        # Экспортируем
        export_metadata = exporter.export_experiment(
            experiment=mock_experiment,
            formats=['json'],
            include_raw_data=True,
            include_plots=False
        )
        
        # Проверяем результаты экспорта
        assert export_metadata is not None
        assert 'exported_files' in export_metadata
        assert 'json' in export_metadata['exported_files']
        
        # Проверяем, что файл создан
        json_file = Path(export_metadata['exported_files']['json'])
        assert json_file.exists()
        
        # Проверяем содержимое
        with open(json_file, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['experiment_id'] == "export_test"
        assert 'dependencies' in exported_data
        assert 'results' in exported_data


if __name__ == "__main__":
    # Запуск тестов для отладки
    pytest.main([__file__, "-v", "-s"])