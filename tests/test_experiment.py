"""Тесты для класса Experiment."""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.experiments.experiment import (
    Experiment,
    ExperimentStatus,
    ExperimentError,
    InvalidStateTransitionError,
    ConfigurationError
)
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig


@pytest.fixture
def baseline_config():
    """Базовая конфигурация для тестов."""
    return RLConfig(
        experiment_name="test_baseline",
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64
        ),
        environment=EnvironmentConfig(name="LunarLander-v3"),
        training=TrainingConfig(total_timesteps=100000)
    )


@pytest.fixture
def variant_config():
    """Вариантная конфигурация для тестов."""
    return RLConfig(
        experiment_name="test_variant",
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=1e-3,  # Отличается от baseline
            n_steps=2048,
            batch_size=64
        ),
        environment=EnvironmentConfig(name="LunarLander-v3"),
        training=TrainingConfig(total_timesteps=100000)
    )


@pytest.fixture
def invalid_variant_config():
    """Невалидная вариантная конфигурация (другая среда)."""
    return RLConfig(
        experiment_name="test_invalid",
        algorithm=AlgorithmConfig(name="PPO"),
        environment=EnvironmentConfig(name="Pendulum-v1"),  # Другая среда
        training=TrainingConfig(total_timesteps=100000)
    )


@pytest.fixture
def identical_config():
    """Идентичная базовой конфигурация."""
    return RLConfig(
        experiment_name="test_identical",
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64
        ),
        environment=EnvironmentConfig(name="LunarLander-v3"),
        training=TrainingConfig(total_timesteps=100000)
    )


@pytest.fixture
def temp_output_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestExperimentInitialization:
    """Тесты инициализации эксперимента."""
    
    def test_valid_initialization(self, baseline_config, variant_config, temp_output_dir):
        """Тест успешной инициализации эксперимента."""
        hypothesis = "Увеличение learning rate улучшит производительность"
        
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis=hypothesis,
            output_dir=temp_output_dir
        )
        
        assert experiment.experiment_id is not None
        assert len(experiment.experiment_id) > 0
        assert experiment.baseline_config == baseline_config
        assert experiment.variant_config == variant_config
        assert experiment.hypothesis == hypothesis
        assert experiment.status == ExperimentStatus.CREATED
        assert experiment.created_at is not None
        assert experiment.started_at is None
        assert experiment.completed_at is None
        assert experiment.results == {'baseline': {}, 'variant': {}, 'comparison': {}}
        assert experiment.experiment_dir.exists()
    
    def test_custom_experiment_id(self, baseline_config, variant_config, temp_output_dir):
        """Тест инициализации с пользовательским ID."""
        custom_id = "custom-experiment-123"
        
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            experiment_id=custom_id,
            output_dir=temp_output_dir
        )
        
        assert experiment.experiment_id == custom_id
    
    def test_invalid_environment_mismatch(self, baseline_config, invalid_variant_config, temp_output_dir):
        """Тест ошибки при несовпадении сред."""
        with pytest.raises(ConfigurationError, match="Среды должны быть одинаковыми"):
            Experiment(
                baseline_config=baseline_config,
                variant_config=invalid_variant_config,
                hypothesis="Test hypothesis",
                output_dir=temp_output_dir
            )
    
    def test_identical_configurations(self, baseline_config, identical_config, temp_output_dir):
        """Тест ошибки при идентичных конфигурациях."""
        with pytest.raises(ConfigurationError, match="Конфигурации идентичны"):
            Experiment(
                baseline_config=baseline_config,
                variant_config=identical_config,
                hypothesis="Test hypothesis",
                output_dir=temp_output_dir
            )
    
    def test_empty_hypothesis(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при пустой гипотезе."""
        with pytest.raises(ConfigurationError, match="Гипотеза не может быть пустой"):
            Experiment(
                baseline_config=baseline_config,
                variant_config=variant_config,
                hypothesis="   ",  # Только пробелы
                output_dir=temp_output_dir
            )


class TestExperimentLifecycle:
    """Тесты жизненного цикла эксперимента."""
    
    def test_start_experiment(self, baseline_config, variant_config, temp_output_dir):
        """Тест запуска эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.started_at is not None
        assert isinstance(experiment.started_at, datetime)
    
    def test_start_already_running(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при повторном запуске."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        
        with pytest.raises(InvalidStateTransitionError):
            experiment.start()
    
    def test_pause_and_resume(self, baseline_config, variant_config, temp_output_dir):
        """Тест приостановки и возобновления эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        experiment.pause()
        
        assert experiment.status == ExperimentStatus.PAUSED
        assert experiment.paused_at is not None
        
        experiment.resume()
        
        assert experiment.status == ExperimentStatus.RUNNING
    
    def test_pause_not_running(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при приостановке не запущенного эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(InvalidStateTransitionError):
            experiment.pause()
    
    def test_stop_successful(self, baseline_config, variant_config, temp_output_dir):
        """Тест успешного завершения эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        experiment.stop()
        
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.completed_at is not None
        assert 'metadata' in experiment.results
    
    def test_stop_with_failure(self, baseline_config, variant_config, temp_output_dir):
        """Тест завершения эксперимента с ошибкой."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        error_message = "Test error occurred"
        experiment.stop(failed=True, error_message=error_message)
        
        assert experiment.status == ExperimentStatus.FAILED
        assert experiment.completed_at is not None
        assert experiment.results['error'] == error_message


class TestResultsManagement:
    """Тесты управления результатами."""
    
    def test_add_baseline_results(self, baseline_config, variant_config, temp_output_dir):
        """Тест добавления результатов baseline."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        results = {
            'mean_reward': 150.0,
            'final_reward': 200.0,
            'episode_length': 250,
            'training_time': 3600
        }
        
        experiment.add_result('baseline', results)
        
        assert experiment._baseline_completed is True
        assert experiment.results['baseline']['mean_reward'] == 150.0
        assert 'timestamp' in experiment.results['baseline']
        assert experiment.results['baseline']['config_type'] == 'baseline'
    
    def test_add_variant_results(self, baseline_config, variant_config, temp_output_dir):
        """Тест добавления результатов variant."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        results = {
            'mean_reward': 180.0,
            'final_reward': 220.0,
            'episode_length': 230,
            'training_time': 3400
        }
        
        experiment.add_result('variant', results)
        
        assert experiment._variant_completed is True
        assert experiment.results['variant']['mean_reward'] == 180.0
    
    def test_add_invalid_config_type(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при неверном типе конфигурации."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(ValueError, match="Неверный тип конфигурации"):
            experiment.add_result('invalid_type', {})
    
    def test_automatic_comparison(self, baseline_config, variant_config, temp_output_dir):
        """Тест автоматического сравнения при добавлении всех результатов."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        baseline_results = {'mean_reward': 150.0, 'episode_length': 250}
        variant_results = {'mean_reward': 180.0, 'episode_length': 230}
        
        experiment.add_result('baseline', baseline_results)
        experiment.add_result('variant', variant_results)
        
        # Проверяем, что сравнение выполнено
        assert 'comparison' in experiment.results
        assert experiment.results['comparison'] is not None
        comparison = experiment.results['comparison']
        
        assert 'performance_metrics' in comparison
        assert 'mean_reward' in comparison['performance_metrics']
        
        reward_metric = comparison['performance_metrics']['mean_reward']
        assert reward_metric['baseline'] == 150.0
        assert reward_metric['variant'] == 180.0
        assert reward_metric['improvement'] == 30.0
        assert reward_metric['better'] == 'variant'
    
    def test_compare_results_incomplete(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при сравнении неполных результатов."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.add_result('baseline', {'mean_reward': 150.0})
        
        with pytest.raises(ValueError, match="Невозможно провести сравнение"):
            experiment.compare_results()


class TestSerialization:
    """Тесты сериализации и десериализации."""
    
    def test_save_and_load_json(self, baseline_config, variant_config, temp_output_dir):
        """Тест сохранения и загрузки в JSON формате."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        # Добавляем некоторые результаты
        experiment.start()
        experiment.add_result('baseline', {'mean_reward': 150.0})
        experiment.add_result('variant', {'mean_reward': 180.0})
        experiment.stop()
        
        # Сохраняем
        filepath = experiment.save(format_type='json')
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Загружаем
        loaded_experiment = Experiment.load(filepath)
        
        assert loaded_experiment.experiment_id == experiment.experiment_id
        assert loaded_experiment.hypothesis == experiment.hypothesis
        assert loaded_experiment.status == ExperimentStatus.COMPLETED
        assert loaded_experiment.results['baseline']['mean_reward'] == 150.0
        assert loaded_experiment.results['variant']['mean_reward'] == 180.0
    
    def test_save_and_load_pickle(self, baseline_config, variant_config, temp_output_dir):
        """Тест сохранения и загрузки в pickle формате."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        experiment.stop()
        
        # Сохраняем
        filepath = experiment.save(format_type='pickle')
        assert filepath.exists()
        assert filepath.suffix == '.pickle'
        
        # Загружаем
        loaded_experiment = Experiment.load(filepath)
        
        assert loaded_experiment.experiment_id == experiment.experiment_id
        assert loaded_experiment.hypothesis == experiment.hypothesis
        assert loaded_experiment.status == ExperimentStatus.COMPLETED
    
    def test_save_invalid_format(self, baseline_config, variant_config, temp_output_dir):
        """Тест ошибки при неподдерживаемом формате."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            experiment.save(format_type='xml')
    
    def test_load_nonexistent_file(self):
        """Тест ошибки при загрузке несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            Experiment.load('/nonexistent/path/experiment.json')


class TestStatusAndSummary:
    """Тесты получения статуса и сводки."""
    
    def test_get_status(self, baseline_config, variant_config, temp_output_dir):
        """Тест получения статуса эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        status = experiment.get_status()
        
        assert status['experiment_id'] == experiment.experiment_id
        assert status['status'] == 'created'
        assert status['hypothesis'] == experiment.hypothesis
        assert status['created_at'] is not None
        assert status['started_at'] is None
        assert status['baseline_completed'] is False
        assert status['variant_completed'] is False
        assert status['results_available'] is False
    
    def test_get_status_with_duration(self, baseline_config, variant_config, temp_output_dir):
        """Тест получения статуса с длительностью."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        experiment.start()
        experiment.stop()
        
        status = experiment.get_status()
        
        assert status['duration_seconds'] is not None
        assert status['duration_seconds'] >= 0
    
    def test_get_summary(self, baseline_config, variant_config, temp_output_dir):
        """Тест получения сводки эксперимента."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        summary = experiment.get_summary()
        
        assert summary['experiment_id'] == experiment.experiment_id
        assert summary['hypothesis'] == experiment.hypothesis
        assert summary['status'] == 'created'
        assert 'configurations' in summary
        assert 'baseline' in summary['configurations']
        assert 'variant' in summary['configurations']
        
        baseline_config_summary = summary['configurations']['baseline']
        assert baseline_config_summary['algorithm'] == 'PPO'
        assert baseline_config_summary['environment'] == 'LunarLander-v3'
        assert baseline_config_summary['learning_rate'] == 3e-4
    
    def test_get_summary_with_results(self, baseline_config, variant_config, temp_output_dir):
        """Тест получения сводки с результатами."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        # Добавляем результаты для автоматического сравнения
        experiment.add_result('baseline', {'mean_reward': 150.0})
        experiment.add_result('variant', {'mean_reward': 180.0})
        
        summary = experiment.get_summary()
        
        assert 'results' in summary
        assert summary['results'] == experiment.results['comparison']


class TestStringRepresentation:
    """Тесты строкового представления."""
    
    def test_repr(self, baseline_config, variant_config, temp_output_dir):
        """Тест __repr__ метода."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        repr_str = repr(experiment)
        
        assert experiment.experiment_id in repr_str
        assert 'created' in repr_str
        assert 'PPO' in repr_str
    
    def test_str(self, baseline_config, variant_config, temp_output_dir):
        """Тест __str__ метода."""
        experiment = Experiment(
            baseline_config=baseline_config,
            variant_config=variant_config,
            hypothesis="Test hypothesis",
            output_dir=temp_output_dir
        )
        
        str_repr = str(experiment)
        
        assert experiment.experiment_id in str_repr
        assert experiment.hypothesis in str_repr
        assert 'PPO' in str_repr
        assert 'Статус: created' in str_repr


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_experiment_reproducibility(seed, baseline_config, variant_config, temp_output_dir):
    """Тест воспроизводимости экспериментов с разными seeds."""
    # Устанавливаем seed в конфигурациях
    baseline_config.seed = seed
    variant_config.seed = seed
    
    experiment = Experiment(
        baseline_config=baseline_config,
        variant_config=variant_config,
        hypothesis="Test reproducibility",
        output_dir=temp_output_dir
    )
    
    assert experiment.baseline_config.seed == seed
    assert experiment.variant_config.seed == seed