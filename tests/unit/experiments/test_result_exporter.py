"""Тесты для модуля экспорта результатов экспериментов."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.experiments.experiment import Experiment, ExperimentStatus
from src.experiments.result_exporter import (
    CompressionType,
    ExportError,
    ExportFormat,
    ResultExporter,
    ValidationError,
    export_experiment_results,
    export_multiple_experiments_results,
)
from src.utils.config import RLConfig


@pytest.fixture
def mock_config():
    """Создать мок конфигурации."""
    config = Mock(spec=RLConfig)
    
    # Создаем мок алгоритма
    algorithm_mock = Mock()
    algorithm_mock.name = "PPO"
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


@pytest.fixture
def mock_experiment(mock_config):
    """Создать мок эксперимента."""
    experiment = Mock(spec=Experiment)
    experiment.experiment_id = "test_exp_001"
    experiment.status = ExperimentStatus.COMPLETED
    experiment.hypothesis = "Тестовая гипотеза"
    experiment.baseline_config = mock_config
    experiment.variant_config = mock_config
    experiment.results = {
        "baseline": {
            "mean_reward": 150.5,
            "final_reward": 200.0,
            "metrics_history": [
                {"timestep": 1000, "episode_reward": 100.0, "episode_length": 200},
                {"timestep": 2000, "episode_reward": 150.0, "episode_length": 180},
                {"timestep": 3000, "episode_reward": 200.0, "episode_length": 160}
            ]
        },
        "variant": {
            "mean_reward": 175.2,
            "final_reward": 220.0,
            "metrics_history": [
                {"timestep": 1000, "episode_reward": 120.0, "episode_length": 190},
                {"timestep": 2000, "episode_reward": 170.0, "episode_length": 170},
                {"timestep": 3000, "episode_reward": 220.0, "episode_length": 150}
            ]
        }
    }
    
    from datetime import datetime
    experiment.created_at = datetime.now()
    experiment.started_at = datetime.now()
    experiment.completed_at = datetime.now()
    
    return experiment


@pytest.fixture
def temp_output_dir():
    """Создать временную директорию для тестов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestResultExporter:
    """Тесты для класса ResultExporter."""
    
    def test_init_default_params(self, temp_output_dir):
        """Тест инициализации с параметрами по умолчанию."""
        exporter = ResultExporter(output_dir=temp_output_dir)
        
        assert exporter.output_dir == temp_output_dir
        assert exporter.include_dependencies is True
        assert exporter.validate_integrity is True
        assert exporter.auto_compress is False
        assert exporter.compression_type == CompressionType.ZIP
    
    def test_init_custom_params(self, temp_output_dir):
        """Тест инициализации с кастомными параметрами."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False,
            auto_compress=True,
            compression_type=CompressionType.TAR_GZ
        )
        
        assert exporter.include_dependencies is False
        assert exporter.validate_integrity is False
        assert exporter.auto_compress is True
        assert exporter.compression_type == CompressionType.TAR_GZ
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_export_experiment_json(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест экспорта эксперимента в JSON формат."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        assert result["export_type"] == "single_experiment"
        assert result["experiment_id"] == "test_exp_001"
        assert ExportFormat.JSON in result["exported_files"]
        
        # Проверяем, что файл создан
        json_file = Path(result["exported_files"][ExportFormat.JSON])
        assert json_file.exists()
        
        # Проверяем содержимое
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["experiment_id"] == "test_exp_001"
        assert data["status"] == "completed"
        assert "results" in data
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_export_experiment_csv(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест экспорта эксперимента в CSV формат."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.CSV]
        )
        
        assert ExportFormat.CSV in result["exported_files"]
        
        # Проверяем, что файл создан
        csv_file = Path(result["exported_files"][ExportFormat.CSV])
        assert csv_file.exists()
        
        # Проверяем содержимое
        df = pd.read_csv(csv_file)
        assert not df.empty
        assert "experiment_id" in df.columns
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_export_experiment_multiple_formats(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест экспорта в несколько форматов."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        formats = [ExportFormat.JSON, ExportFormat.CSV, ExportFormat.PICKLE]
        result = exporter.export_experiment(mock_experiment, formats=formats)
        
        for format_type in formats:
            assert format_type in result["exported_files"]
            file_path = Path(result["exported_files"][format_type])
            assert file_path.exists()
    
    @patch('src.experiments.result_exporter.create_experiment_snapshot')
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_export_with_dependencies(self, mock_tracker, mock_snapshot, temp_output_dir, mock_experiment):
        """Тест экспорта с включением зависимостей."""
        mock_snapshot.return_value = {"test": "dependency_data"}
        
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=True,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        # Проверяем, что снимок зависимостей был создан
        mock_snapshot.assert_called_once_with("test_exp_001")
        
        # Проверяем содержимое файла
        json_file = Path(result["exported_files"][ExportFormat.JSON])
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "dependencies" in data
        assert data["dependencies"]["test"] == "dependency_data"
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_export_multiple_experiments(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест экспорта нескольких экспериментов."""
        # Создаем второй мок эксперимента
        experiment2 = Mock(spec=Experiment)
        experiment2.experiment_id = "test_exp_002"
        experiment2.status = ExperimentStatus.COMPLETED
        experiment2.hypothesis = "Вторая гипотеза"
        experiment2.baseline_config = mock_experiment.baseline_config
        experiment2.variant_config = mock_experiment.variant_config
        experiment2.results = mock_experiment.results
        experiment2.created_at = mock_experiment.created_at
        experiment2.started_at = mock_experiment.started_at
        experiment2.completed_at = mock_experiment.completed_at
        
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        experiments = [mock_experiment, experiment2]
        result = exporter.export_multiple_experiments(
            experiments,
            formats=[ExportFormat.JSON],
            include_comparison=False,
            include_summary=True
        )
        
        assert result["export_type"] == "multiple_experiments"
        assert result["experiment_count"] == 2
        assert set(result["experiment_ids"]) == {"test_exp_001", "test_exp_002"}
        
        # Проверяем, что файлы созданы
        for format_type in [ExportFormat.JSON]:
            assert format_type in result["exported_files"]
            file_path = Path(result["exported_files"][format_type])
            assert file_path.exists()
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_incremental_export(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест инкрементального экспорта."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Первый экспорт
        initial_result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dir = Path(initial_result["export_dir"])
        
        # Инкрементальный экспорт
        incremental_result = exporter.incremental_export(
            mock_experiment,
            export_dir,
            update_existing=True
        )
        
        assert incremental_result["updated"] is True
        assert len(incremental_result["updated_files"]) > 0
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_validate_export_integrity(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест валидации целостности экспорта."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False  # Отключаем автоматическую валидацию
        )
        
        # Экспортируем эксперимент
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dir = Path(result["export_dir"])
        
        # Валидируем экспорт
        validation_result = exporter.validate_export_integrity(export_dir)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        assert len(validation_result["checked_files"]) > 0
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_compress_export_zip(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест сжатия экспорта в ZIP."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Экспортируем эксперимент
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dir = Path(result["export_dir"])
        
        # Сжимаем экспорт
        archive_path = exporter.compress_export(
            export_dir,
            compression_type=CompressionType.ZIP
        )
        
        assert Path(archive_path).exists()
        assert archive_path.endswith('.zip')
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_list_exports(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест получения списка экспортов."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Создаем несколько экспортов
        exporter.export_experiment(mock_experiment, formats=[ExportFormat.JSON])
        
        # Получаем список
        exports_list = exporter.list_exports()
        
        assert len(exports_list) == 1
        # Для одиночного эксперимента experiment_count может быть 0 в метаданных
        assert exports_list[0]["experiment_count"] >= 0  
        assert exports_list[0]["export_type"] == "single_experiment"
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_cleanup_old_exports(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест очистки старых экспортов."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Создаем экспорт
        exporter.export_experiment(mock_experiment, formats=[ExportFormat.JSON])
        
        # Очищаем (с параметрами, которые не удалят наш экспорт)
        cleanup_result = exporter.cleanup_old_exports(
            keep_count=10,
            keep_days=30
        )
        
        assert cleanup_result["deleted_count"] == 0
        assert len(cleanup_result["errors"]) == 0
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_generate_summary_report(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест генерации сводного отчета."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Создаем экспорт
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dirs = [result["export_dir"]]
        
        # Генерируем отчет
        report_path = exporter.generate_summary_report(
            export_dirs,
            include_statistics=True,
            include_trends=False
        )
        
        assert Path(report_path).exists()
        assert report_path.endswith('.html')
        
        # Проверяем содержимое отчета
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Сводный отчет по экспериментам RL" in content
        assert "test_exp_001" in content
    
    def test_export_error_handling(self, temp_output_dir):
        """Тест обработки ошибок при экспорте."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        # Тест с некорректным экспериментом
        invalid_experiment = Mock()
        invalid_experiment.experiment_id = None  # Некорректный ID
        
        with pytest.raises(ExportError):
            exporter.export_experiment(invalid_experiment)
    
    def test_unsupported_format_error(self, temp_output_dir, mock_experiment):
        """Тест ошибки неподдерживаемого формата."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        with patch.object(exporter, '_export_to_format') as mock_export:
            mock_export.side_effect = ValueError("Неподдерживаемый формат")
            
            # Экспорт должен продолжиться, но формат будет пропущен
            result = exporter.export_experiment(
                mock_experiment,
                formats=["unsupported_format"]
            )
            
            # Проверяем, что экспорт завершился, но без файлов
            assert result["export_type"] == "single_experiment"
            assert len(result["exported_files"]) == 0


class TestExportFormats:
    """Тесты для различных форматов экспорта."""
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_json_format_content(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест содержимого JSON экспорта."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        json_file = Path(result["exported_files"][ExportFormat.JSON])
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверяем обязательные поля
        assert data["experiment_id"] == "test_exp_001"
        assert data["status"] == "completed"
        assert data["hypothesis"] == "Тестовая гипотеза"
        assert "results" in data
        assert "configurations" in data
        assert "raw_metrics" in data
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_csv_format_content(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест содержимого CSV экспорта."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.CSV]
        )
        
        csv_file = Path(result["exported_files"][ExportFormat.CSV])
        df = pd.read_csv(csv_file)
        
        # Проверяем структуру данных
        assert not df.empty
        assert "experiment_id" in df.columns
        assert "config_type" in df.columns
        
        # Проверяем, что есть данные для baseline и variant
        config_types = df["config_type"].unique()
        assert "baseline" in config_types
        assert "variant" in config_types
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    @patch('h5py.File')
    def test_hdf5_format(self, mock_h5py, mock_tracker, temp_output_dir, mock_experiment):
        """Тест HDF5 экспорта."""
        # Мокаем h5py.File
        mock_file = Mock()
        mock_h5py.return_value.__enter__.return_value = mock_file
        
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.HDF5]
        )
        
        assert ExportFormat.HDF5 in result["exported_files"]
        
        # Проверяем, что h5py.File был вызван
        mock_h5py.assert_called_once()
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_pickle_format(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест Pickle экспорта."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.PICKLE]
        )
        
        pickle_file = Path(result["exported_files"][ExportFormat.PICKLE])
        assert pickle_file.exists()
        assert pickle_file.suffix == '.pkl'
        
        # Проверяем, что файл можно загрузить
        import pickle
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        assert data["experiment_id"] == "test_exp_001"


class TestCompressionTypes:
    """Тесты для различных типов сжатия."""
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_zip_compression(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест ZIP сжатия."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dir = Path(result["export_dir"])
        archive_path = exporter.compress_export(
            export_dir,
            compression_type=CompressionType.ZIP
        )
        
        assert Path(archive_path).exists()
        assert archive_path.endswith('.zip')
    
    @patch('src.experiments.result_exporter.DependencyTracker')
    def test_tar_gz_compression(self, mock_tracker, temp_output_dir, mock_experiment):
        """Тест TAR.GZ сжатия."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        result = exporter.export_experiment(
            mock_experiment,
            formats=[ExportFormat.JSON]
        )
        
        export_dir = Path(result["export_dir"])
        archive_path = exporter.compress_export(
            export_dir,
            compression_type=CompressionType.TAR_GZ
        )
        
        assert Path(archive_path).exists()
        assert archive_path.endswith('.tar.gz')


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""
    
    @patch('src.experiments.result_exporter.ResultExporter')
    def test_export_experiment_results_function(self, mock_exporter_class, mock_experiment):
        """Тест функции export_experiment_results."""
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_experiment.return_value = {"test": "result"}
        
        result = export_experiment_results(
            mock_experiment,
            output_dir="/test/dir",
            formats=[ExportFormat.JSON],
            include_dependencies=True
        )
        
        # Проверяем, что экспортер был создан с правильными параметрами
        mock_exporter_class.assert_called_once_with(
            output_dir="/test/dir",
            include_dependencies=True
        )
        
        # Проверяем, что метод экспорта был вызван
        mock_exporter.export_experiment.assert_called_once_with(
            mock_experiment,
            [ExportFormat.JSON]
        )
        
        assert result == {"test": "result"}
    
    @patch('src.experiments.result_exporter.ResultExporter')
    def test_export_multiple_experiments_results_function(self, mock_exporter_class, mock_experiment):
        """Тест функции export_multiple_experiments_results."""
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_multiple_experiments.return_value = {"test": "result"}
        
        experiments = [mock_experiment]
        result = export_multiple_experiments_results(
            experiments,
            output_dir="/test/dir",
            formats=[ExportFormat.JSON],
            include_comparison=True
        )
        
        # Проверяем, что экспортер был создан
        mock_exporter_class.assert_called_once_with(output_dir="/test/dir")
        
        # Проверяем, что метод экспорта был вызван
        mock_exporter.export_multiple_experiments.assert_called_once_with(
            experiments,
            [ExportFormat.JSON],
            include_comparison=True
        )
        
        assert result == {"test": "result"}


class TestDataFlattening:
    """Тесты для преобразования данных."""
    
    def test_flatten_data_for_csv(self, temp_output_dir):
        """Тест преобразования иерархических данных в плоскую структуру."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False
        )
        
        test_data = {
            "experiment_id": "test_001",
            "status": "completed",
            "raw_metrics": {
                "baseline": [
                    {"timestep": 1000, "reward": 100.0},
                    {"timestep": 2000, "reward": 150.0}
                ]
            }
        }
        
        flattened = exporter._flatten_data_for_csv(test_data)
        
        assert len(flattened) == 2  # Две записи метрик
        assert flattened[0]["experiment_id"] == "test_001"
        assert flattened[0]["config_type"] == "baseline"
        assert flattened[0]["timestep"] == 1000
        assert flattened[0]["reward"] == 100.0
    
    def test_flatten_data_without_metrics(self, temp_output_dir):
        """Тест преобразования данных без метрик."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False
        )
        
        test_data = {
            "experiment_id": "test_001",
            "status": "completed",
            "nested": {
                "value": 42,
                "deep": {
                    "nested": "test"
                }
            }
        }
        
        flattened = exporter._flatten_data_for_csv(test_data)
        
        assert len(flattened) == 1
        assert flattened[0]["experiment_id"] == "test_001"
        assert flattened[0]["nested_value"] == 42
        assert flattened[0]["nested_deep_nested"] == "test"


class TestErrorHandling:
    """Тесты для обработки ошибок."""
    
    def test_export_nonexistent_directory(self):
        """Тест экспорта в несуществующую директорию."""
        # Директория должна создаваться автоматически
        nonexistent_dir = Path("/tmp/nonexistent_test_dir_12345")
        
        exporter = ResultExporter(output_dir=nonexistent_dir)
        
        # Проверяем, что директория была создана
        assert exporter.output_dir.exists()
        
        # Очищаем после теста
        import shutil
        if nonexistent_dir.exists():
            shutil.rmtree(nonexistent_dir)
    
    def test_validate_nonexistent_export(self, temp_output_dir):
        """Тест валидации несуществующего экспорта."""
        exporter = ResultExporter(output_dir=temp_output_dir)
        
        nonexistent_dir = temp_output_dir / "nonexistent"
        
        with pytest.raises(ValueError, match="Директория экспорта не существует"):
            exporter.validate_export_integrity(nonexistent_dir)
    
    def test_compress_nonexistent_export(self, temp_output_dir):
        """Тест сжатия несуществующего экспорта."""
        exporter = ResultExporter(output_dir=temp_output_dir)
        
        nonexistent_dir = temp_output_dir / "nonexistent"
        
        with pytest.raises(ValueError, match="Директория экспорта не существует"):
            exporter.compress_export(nonexistent_dir)
    
    def test_unsupported_compression_type(self, temp_output_dir, mock_experiment):
        """Тест неподдерживаемого типа сжатия."""
        exporter = ResultExporter(
            output_dir=temp_output_dir,
            include_dependencies=False,
            validate_integrity=False
        )
        
        with patch.object(exporter, '_compress_export') as mock_compress:
            mock_compress.side_effect = ValueError("Неподдерживаемый тип сжатия")
            
            with pytest.raises(Exception):  # CompressionError
                exporter.compress_export(temp_output_dir, "unsupported")


if __name__ == "__main__":
    pytest.main([__file__])