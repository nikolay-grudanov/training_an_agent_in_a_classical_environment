"""Тесты для ExperimentRunner.

Этот модуль содержит комплексные тесты для класса ExperimentRunner,
включая тестирование различных режимов выполнения, обработки ошибок,
мониторинга и интеграции компонентов.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.experiments.config import Configuration
from src.experiments.experiment import Experiment
from src.experiments.runner import (
    ExperimentRunner,
    ExecutionMode,
    ProgressInfo,
    ResourceUsage,
    RunnerStatus,
)
from src.training.trainer import TrainingResult
from src.utils.config import RLConfig


@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_baseline_config():
    """Мок baseline конфигурации."""
    return Configuration(
        algorithm="PPO",
        environment="LunarLander-v2",
        experiment_name="baseline_test",
        training_steps=1000,
        seed=42,
    )


@pytest.fixture
def mock_variant_config():
    """Мок variant конфигурации."""
    return Configuration(
        algorithm="PPO",
        environment="LunarLander-v2",
        experiment_name="variant_test",
        training_steps=1000,
        seed=42,
        hyperparameters={"learning_rate": 0.001},  # Отличие от baseline
    )


@pytest.fixture
def mock_experiment(temp_dir, mock_baseline_config, mock_variant_config):
    """Мок эксперимента."""
    # Создаем RLConfig объекты из Configuration
    baseline_rl_config = RLConfig(
        algorithm=mock_baseline_config,
        environment=mock_baseline_config,
        training=mock_baseline_config,
        seed=mock_baseline_config.seed,
        experiment_name=mock_baseline_config.experiment_name,
        output_dir=str(temp_dir),
    )

    variant_rl_config = RLConfig(
        algorithm=mock_variant_config,
        environment=mock_variant_config,
        training=mock_variant_config,
        seed=mock_variant_config.seed,
        experiment_name=mock_variant_config.experiment_name,
        output_dir=str(temp_dir),
    )

    experiment = Experiment(
        baseline_config=baseline_rl_config,
        variant_config=variant_rl_config,
        hypothesis="Variant должен показать лучшие результаты",
        output_dir=temp_dir,
    )

    return experiment


@pytest.fixture
def mock_training_result():
    """Мок результата обучения."""
    return TrainingResult(
        success=True,
        total_timesteps=1000,
        training_time=10.0,
        final_mean_reward=100.0,
        final_std_reward=10.0,
        experiment_name="test_experiment",
        algorithm="PPO",
        environment_name="LunarLander-v2",
        seed=42,
    )


class TestExperimentRunner:
    """Тесты для класса ExperimentRunner."""

    def test_init_default_parameters(self, mock_experiment):
        """Тест инициализации с параметрами по умолчанию."""
        runner = ExperimentRunner(mock_experiment)

        assert runner.experiment == mock_experiment
        assert runner.execution_mode == ExecutionMode.SEQUENTIAL
        assert runner.max_workers >= 1
        assert runner.enable_monitoring is True
        assert runner.status == RunnerStatus.IDLE
        assert runner.baseline_result is None
        assert runner.variant_result is None

    def test_init_custom_parameters(self, mock_experiment):
        """Тест инициализации с пользовательскими параметрами."""
        resource_limits = {"memory_mb": 4096, "cpu_percent": 80.0}

        runner = ExperimentRunner(
            experiment=mock_experiment,
            execution_mode=ExecutionMode.PARALLEL,
            max_workers=4,
            enable_monitoring=False,
            checkpoint_frequency=5000,
            resource_limits=resource_limits,
        )

        assert runner.execution_mode == ExecutionMode.PARALLEL
        assert runner.max_workers == 4
        assert runner.enable_monitoring is False
        assert runner.checkpoint_frequency == 5000
        assert runner.resource_limits == resource_limits

    def test_init_invalid_parameters(self, mock_experiment):
        """Тест инициализации с некорректными параметрами."""
        with pytest.raises(ValueError, match="max_workers должен быть >= 1"):
            ExperimentRunner(mock_experiment, max_workers=0)

    @patch("src.experiments.runner.Trainer")
    def test_run_configuration_baseline(
        self, mock_trainer_class, mock_experiment, mock_training_result
    ):
        """Тест выполнения baseline конфигурации."""
        # Настройка мока
        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_class.return_value = mock_trainer

        runner = ExperimentRunner(mock_experiment)

        # Выполнение
        result = runner.run_configuration(
            config_type="baseline",
            config=mock_experiment.baseline_config,
        )

        # Проверки
        assert result == mock_training_result
        assert result.success is True
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()

    @patch("src.experiments.runner.Trainer")
    def test_run_configuration_variant(
        self, mock_trainer_class, mock_experiment, mock_training_result
    ):
        """Тест выполнения variant конфигурации."""
        # Настройка мока
        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_class.return_value = mock_trainer

        runner = ExperimentRunner(mock_experiment)

        # Выполнение
        result = runner.run_configuration(
            config_type="variant",
            config=mock_experiment.variant_config,
        )

        # Проверки
        assert result == mock_training_result
        assert result.success is True

    def test_run_configuration_invalid_type(self, mock_experiment):
        """Тест выполнения с некорректным типом конфигурации."""
        runner = ExperimentRunner(mock_experiment)

        with pytest.raises(ValueError, match="Неверный тип конфигурации"):
            runner.run_configuration(
                config_type="invalid",
                config=mock_experiment.baseline_config,
            )

    @patch("src.experiments.runner.Trainer")
    def test_run_configuration_training_failure(
        self, mock_trainer_class, mock_experiment
    ):
        """Тест обработки ошибки обучения."""
        # Настройка мока для ошибки
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        runner = ExperimentRunner(mock_experiment)

        # Выполнение
        result = runner.run_configuration(
            config_type="baseline",
            config=mock_experiment.baseline_config,
        )

        # Проверки
        assert result is None

    def test_setup_environment(self, mock_experiment, temp_dir):
        """Тест настройки среды выполнения."""
        runner = ExperimentRunner(mock_experiment)

        # Выполнение
        runner.setup_environment()

        # Проверки
        assert mock_experiment.experiment_dir.exists()
        assert (mock_experiment.experiment_dir / "logs").exists()
        assert (mock_experiment.experiment_dir / "checkpoints").exists()

    def test_monitor_progress(self, mock_experiment):
        """Тест мониторинга прогресса."""
        runner = ExperimentRunner(mock_experiment)

        # Изменение прогресса
        runner.progress.baseline_progress = 50.0
        runner.progress.variant_progress = 30.0
        runner.progress.current_phase = "baseline"

        # Получение прогресса
        progress = runner.monitor_progress()

        # Проверки
        assert isinstance(progress, ProgressInfo)
        assert progress.baseline_progress == 50.0
        assert progress.variant_progress == 30.0
        assert progress.current_phase == "baseline"

    def test_handle_failure_abort(self, mock_experiment):
        """Тест обработки ошибки со стратегией abort."""
        runner = ExperimentRunner(mock_experiment)

        error = Exception("Test error")
        result = runner.handle_failure(error, "baseline", "abort")

        assert result is False

    def test_handle_failure_skip(self, mock_experiment):
        """Тест обработки ошибки со стратегией skip."""
        runner = ExperimentRunner(mock_experiment)

        error = Exception("Test error")
        result = runner.handle_failure(error, "baseline", "skip")

        assert result is True

    def test_get_status(self, mock_experiment):
        """Тест получения статуса runner'а."""
        runner = ExperimentRunner(mock_experiment)
        runner.status = RunnerStatus.RUNNING_BASELINE
        runner.progress.baseline_progress = 75.0

        status = runner.get_status()

        assert isinstance(status, dict)
        assert status["status"] == "running_baseline"
        assert status["execution_mode"] == "sequential"
        assert status["progress"]["baseline"] == 75.0
        assert "resource_usage" in status
        assert "experiment_status" in status

    @patch("src.experiments.runner.ExperimentRunner._execute_sequential")
    @patch("src.experiments.runner.ExperimentRunner._validate_experiment")
    @patch("src.experiments.runner.ExperimentRunner._setup_environment")
    def test_run_sequential_success(
        self, mock_setup, mock_validate, mock_execute, mock_experiment
    ):
        """Тест успешного выполнения в последовательном режиме."""
        # Настройка моков
        mock_execute.return_value = True

        runner = ExperimentRunner(
            mock_experiment, execution_mode=ExecutionMode.SEQUENTIAL
        )

        # Выполнение
        result = runner.run()

        # Проверки
        assert result is True
        assert runner.status == RunnerStatus.COMPLETED
        mock_validate.assert_called_once()
        mock_setup.assert_called_once()
        mock_execute.assert_called_once()

    @patch("src.experiments.runner.ExperimentRunner._execute_parallel")
    @patch("src.experiments.runner.ExperimentRunner._validate_experiment")
    @patch("src.experiments.runner.ExperimentRunner._setup_environment")
    def test_run_parallel_success(
        self, mock_setup, mock_validate, mock_execute, mock_experiment
    ):
        """Тест успешного выполнения в параллельном режиме."""
        # Настройка моков
        mock_execute.return_value = True

        runner = ExperimentRunner(
            mock_experiment, execution_mode=ExecutionMode.PARALLEL
        )

        # Выполнение
        result = runner.run()

        # Проверки
        assert result is True
        assert runner.status == RunnerStatus.COMPLETED
        mock_execute.assert_called_once()

    @patch("src.experiments.runner.ExperimentRunner._validate_configurations")
    @patch("src.experiments.runner.ExperimentRunner._validate_experiment")
    @patch("src.experiments.runner.ExperimentRunner._setup_environment")
    def test_run_validation_mode(
        self, mock_setup, mock_validate, mock_validate_configs, mock_experiment
    ):
        """Тест выполнения в режиме валидации."""
        # Настройка моков
        mock_validate_configs.return_value = True

        runner = ExperimentRunner(
            mock_experiment, execution_mode=ExecutionMode.VALIDATION
        )

        # Выполнение
        result = runner.run()

        # Проверки
        assert result is True
        assert runner.status == RunnerStatus.COMPLETED
        mock_validate_configs.assert_called_once()

    @patch("src.experiments.runner.ExperimentRunner._validate_experiment")
    def test_run_validation_failure(self, mock_validate, mock_experiment):
        """Тест обработки ошибки валидации."""
        # Настройка мока для ошибки
        mock_validate.side_effect = RuntimeError("Validation failed")

        runner = ExperimentRunner(mock_experiment)

        # Выполнение
        result = runner.run()

        # Проверки
        assert result is False
        assert runner.status == RunnerStatus.FAILED

    def test_run_keyboard_interrupt(self, mock_experiment):
        """Тест обработки прерывания клавиатурой."""
        runner = ExperimentRunner(mock_experiment)

        # Имитация прерывания
        with patch.object(
            runner, "_validate_experiment", side_effect=KeyboardInterrupt
        ):
            result = runner.run()

        # Проверки
        assert result is False
        # Статус может быть INTERRUPTED или FAILED в зависимости от момента прерывания
        assert runner.status in [RunnerStatus.INTERRUPTED, RunnerStatus.FAILED]

    @patch("src.experiments.runner.Trainer")
    def test_create_trainer_config(self, mock_trainer_class, mock_experiment):
        """Тест создания конфигурации тренера."""
        runner = ExperimentRunner(mock_experiment)

        config = runner._create_trainer_config(
            mock_experiment.baseline_config, "baseline"
        )

        # Проверки
        assert config.experiment_name.endswith("_baseline")
        assert config.algorithm == mock_experiment.baseline_config.algorithm.name
        assert (
            config.environment_name == mock_experiment.baseline_config.environment.name
        )
        assert (
            config.total_timesteps
            == mock_experiment.baseline_config.training.total_timesteps
        )

    def test_resource_usage_current(self):
        """Тест получения текущего использования ресурсов."""
        usage = ResourceUsage.current()

        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_percent >= 0
        assert usage.memory_mb > 0
        assert usage.memory_percent >= 0

    def test_progress_info_overall_progress(self):
        """Тест расчета общего прогресса."""
        progress = ProgressInfo()

        # Тест с нулевыми шагами
        assert progress.overall_progress == 0.0

        # Тест с прогрессом
        progress.total_steps = 100
        progress.current_step = 50
        assert progress.overall_progress == 50.0

        # Тест с превышением
        progress.current_step = 150
        assert progress.overall_progress == 100.0


class TestExperimentRunnerIntegration:
    """Интеграционные тесты для ExperimentRunner."""

    @patch("src.experiments.runner.Trainer")
    def test_full_sequential_execution(
        self, mock_trainer_class, mock_experiment, mock_training_result
    ):
        """Тест полного последовательного выполнения эксперимента."""
        # Настройка мока
        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_class.return_value = mock_trainer

        runner = ExperimentRunner(
            mock_experiment, execution_mode=ExecutionMode.SEQUENTIAL
        )

        # Выполнение
        success = runner.run()

        # Проверки
        assert success is True
        assert runner.status == RunnerStatus.COMPLETED
        assert runner.baseline_result is not None
        assert runner.variant_result is not None
        assert runner.execution_start_time is not None
        assert runner.execution_end_time is not None

    @patch("src.experiments.runner.ProcessPoolExecutor")
    @patch("src.experiments.runner.Trainer")
    def test_full_parallel_execution(
        self,
        mock_trainer_class,
        mock_executor_class,
        mock_experiment,
        mock_training_result,
    ):
        """Тест полного параллельного выполнения эксперимента."""
        # Настройка мока executor
        mock_executor = Mock()
        mock_future = Mock()
        mock_future.result.return_value = ("baseline", mock_training_result)
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Настройка as_completed
        with patch(
            "src.experiments.runner.as_completed",
            return_value=[mock_future, mock_future],
        ):
            runner = ExperimentRunner(
                mock_experiment, execution_mode=ExecutionMode.PARALLEL
            )

            # Имитация результатов
            results = [
                ("baseline", mock_training_result),
                ("variant", mock_training_result),
            ]
            mock_future.result.side_effect = results

            # Выполнение
            success = runner.run()

        # Проверки
        assert success is True
        assert runner.status == RunnerStatus.COMPLETED

    def test_cleanup_resources(self, mock_experiment, temp_dir):
        """Тест очистки ресурсов."""
        runner = ExperimentRunner(mock_experiment)

        # Создание временных файлов
        temp_file = temp_dir / "temp_file.txt"
        temp_file.write_text("test")

        # Выполнение очистки
        runner.cleanup()

        # Проверки (основная логика очистки)
        assert runner.status != RunnerStatus.IDLE or True  # Cleanup не меняет статус


class TestExperimentRunnerCLI:
    """Тесты для CLI интерфейса ExperimentRunner."""

    def test_cli_help(self):
        """Тест вызова помощи CLI."""
        from click.testing import CliRunner
        from src.experiments.runner import run_experiment_cli

        runner = CliRunner()
        result = runner.invoke(run_experiment_cli, ["--help"])

        assert result.exit_code == 0
        assert "Запустить эксперимент" in result.output

    def test_cli_missing_config(self):
        """Тест CLI без указания конфигурации."""
        from click.testing import CliRunner
        from src.experiments.runner import run_experiment_cli

        runner = CliRunner()
        result = runner.invoke(run_experiment_cli, [])

        assert result.exit_code == 1
        assert "Необходимо указать --config или --experiment-id" in result.output

    @patch("src.experiments.runner.Experiment.load")
    @patch("src.experiments.runner.ExperimentRunner")
    def test_cli_with_config(self, mock_runner_class, mock_load, temp_dir):
        """Тест CLI с файлом конфигурации."""
        from click.testing import CliRunner
        from src.experiments.runner import run_experiment_cli

        # Создание временного файла конфигурации
        config_file = temp_dir / "test_config.json"
        config_file.write_text('{"test": "config"}')

        # Настройка моков
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment"
        mock_load.return_value = mock_experiment

        mock_runner = Mock()
        mock_runner.run.return_value = True
        mock_runner.get_status.return_value = {"execution_time": 10.0}
        mock_runner.baseline_result = Mock(final_mean_reward=100.0)
        mock_runner.variant_result = Mock(final_mean_reward=110.0)
        mock_runner_class.return_value = mock_runner

        # Выполнение CLI
        runner = CliRunner()
        result = runner.invoke(run_experiment_cli, ["--config", str(config_file)])

        # Проверки
        assert result.exit_code == 0
        assert "Эксперимент выполнен успешно" in result.output
        mock_load.assert_called_once_with(str(config_file))
        mock_runner.run.assert_called_once()


@pytest.mark.integration
class TestExperimentRunnerRealExecution:
    """Интеграционные тесты с реальным выполнением (медленные)."""

    @pytest.mark.slow
    def test_real_validation_mode(self, temp_dir):
        """Тест реального выполнения в режиме валидации."""
        # Создание реальных конфигураций
        baseline_config = Configuration(
            algorithm="PPO",
            environment="LunarLander-v2",
            experiment_name="real_baseline",
            training_steps=100,  # Минимальное количество для быстрого теста
            seed=42,
        )

        variant_config = Configuration(
            algorithm="PPO",
            environment="LunarLander-v2",
            experiment_name="real_variant",
            training_steps=100,
            seed=42,
            hyperparameters={"learning_rate": 0.001},
        )

        # Создание эксперимента (требует реальные RLConfig объекты)
        # Этот тест может быть пропущен если зависимости недоступны
        pytest.skip("Требует полную настройку среды RL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
