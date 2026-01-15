"""Тесты для модуля проверки воспроизводимости RL экспериментов."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.utils.reproducibility_checker import (
    ReproducibilityChecker,
    StrictnessLevel,
    ReproducibilityIssueType,
    ReproducibilityIssue,
    ExperimentRun,
    ReproducibilityReport,
    create_simple_reproducibility_test,
    quick_reproducibility_check,
    validate_experiment_reproducibility,
)
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig


class TestReproducibilityChecker:
    """Тесты для класса ReproducibilityChecker."""

    @pytest.fixture
    def temp_project_root(self):
        """Временная директория проекта."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def checker(self, temp_project_root):
        """Экземпляр ReproducibilityChecker для тестов."""
        return ReproducibilityChecker(
            project_root=temp_project_root, strictness_level=StrictnessLevel.STANDARD
        )

    @pytest.fixture
    def sample_config(self):
        """Пример конфигурации для тестов."""
        return RLConfig(
            experiment_name="test_experiment",
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=42),
            environment=EnvironmentConfig(name="CartPole-v1"),
        )

    @pytest.fixture
    def sample_run_data(self):
        """Пример данных запуска."""
        return {
            "results": {"final_reward": 100.0, "episode_length": 50},
            "metrics": {
                "rewards": [10.0, 20.0, 30.0, 40.0, 50.0],
                "episode_lengths": [45, 48, 52, 49, 51],
            },
            "metadata": {"test_run": True},
        }

    def test_init(self, temp_project_root):
        """Тест инициализации ReproducibilityChecker."""
        checker = ReproducibilityChecker(
            project_root=temp_project_root, strictness_level=StrictnessLevel.STRICT
        )

        assert checker.project_root == temp_project_root
        assert checker.strictness_level == StrictnessLevel.STRICT
        assert checker.reports_dir.exists()
        assert checker.runs_dir.exists()
        assert checker.dependency_tracker is not None
        assert checker.seed_manager is not None

    def test_strictness_config(self, checker):
        """Тест конфигурации уровней строгости."""
        config = checker._get_strictness_config()

        assert isinstance(config, dict)
        assert "check_exact_match" in config
        assert "statistical_alpha" in config
        assert "tolerance_rtol" in config
        assert "min_runs_for_stats" in config

        # Проверяем, что STANDARD уровень имеет разумные значения
        assert config["check_exact_match"] is True
        assert config["check_statistical_equivalence"] is True
        assert config["statistical_alpha"] == 0.05

    def test_register_experiment_run(self, checker, sample_config, sample_run_data):
        """Тест регистрации запуска эксперимента."""
        run_id = checker.register_experiment_run(
            experiment_id="test_exp", config=sample_config, **sample_run_data
        )

        assert run_id.startswith("test_exp_")

        # Проверяем, что файл запуска создан
        run_files = list(checker.runs_dir.glob("test_exp_*.json"))
        assert len(run_files) == 1

        # Проверяем содержимое файла
        with open(run_files[0], "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["run_id"] == run_id
        assert saved_data["seed"] == 42
        assert saved_data["results"] == sample_run_data["results"]
        assert saved_data["metrics"] == sample_run_data["metrics"]

    def test_compute_config_hash(self, checker, sample_config):
        """Тест вычисления хеша конфигурации."""
        hash1 = checker._compute_config_hash(sample_config)
        hash2 = checker._compute_config_hash(sample_config)

        # Одинаковые конфигурации должны давать одинаковые хеши
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 хеш

        # Разные конфигурации должны давать разные хеши
        different_config = RLConfig(experiment_name="different_experiment", seed=123)
        hash3 = checker._compute_config_hash(different_config)
        assert hash1 != hash3

    @patch("src.utils.reproducibility_checker.DependencyTracker")
    def test_compute_environment_hash(self, mock_tracker_class, checker):
        """Тест вычисления хеша среды."""
        # Настраиваем мок
        mock_tracker = Mock()
        mock_tracker.get_system_info.return_value = {
            "python": {"version": "3.8.10"},
            "platform": {"system": "Linux"},
        }
        mock_tracker.get_ml_library_versions.return_value = {
            "torch": "1.12.0",
            "stable-baselines3": "1.6.0",
        }
        checker.dependency_tracker = mock_tracker

        hash1 = checker._compute_environment_hash()
        hash2 = checker._compute_environment_hash()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_load_experiment_runs(self, checker, sample_config, sample_run_data):
        """Тест загрузки запусков эксперимента."""
        # Регистрируем несколько запусков
        run_ids = []
        for i in range(3):
            run_id = checker.register_experiment_run(
                experiment_id="test_load", config=sample_config, **sample_run_data
            )
            run_ids.append(run_id)

        # Загружаем запуски
        runs = checker._load_experiment_runs("test_load")

        assert len(runs) == 3
        assert all(isinstance(run, ExperimentRun) for run in runs)
        assert all(run.run_id in run_ids for run in runs)

    def test_validate_determinism(self, checker):
        """Тест валидации детерминизма функции."""

        # Детерминистическая функция
        def deterministic_function():
            np.random.seed(42)
            return {"value": np.random.random()}

        result = checker.validate_determinism(
            test_function=deterministic_function, seed=42, num_runs=3
        )

        assert result["is_deterministic"] is True
        assert result["num_runs"] == 3
        assert result["unique_results"] == 1
        assert result["success_rate"] == 1.0

        # Недетерминистическая функция (используем время для гарантированной недетерминированности)
        import time

        def nondeterministic_function():
            time.sleep(0.001)  # Небольшая задержка для изменения времени
            return {"value": time.time()}

        result = checker.validate_determinism(
            test_function=nondeterministic_function, seed=42, num_runs=3
        )

        assert result["is_deterministic"] is False
        assert result["unique_results"] > 1

    def test_check_seed_consistency(self, checker):
        """Тест проверки консистентности сидов."""
        # Создаем тестовые запуски
        reference_run = ExperimentRun(
            run_id="ref",
            seed=42,
            timestamp="2023-01-01T00:00:00",
            config_hash="hash1",
            environment_hash="env1",
            results={},
            metrics={},
        )

        # Запуск с тем же сидом
        consistent_run = ExperimentRun(
            run_id="consistent",
            seed=42,
            timestamp="2023-01-01T00:01:00",
            config_hash="hash1",
            environment_hash="env1",
            results={},
            metrics={},
        )

        # Запуск с другим сидом
        inconsistent_run = ExperimentRun(
            run_id="inconsistent",
            seed=123,
            timestamp="2023-01-01T00:02:00",
            config_hash="hash1",
            environment_hash="env1",
            results={},
            metrics={},
        )

        # Тест с консистентными сидами
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
        )

        checker._check_seed_consistency(report, reference_run, [consistent_run])

        assert len(report.issues) == 0
        assert report.seed_analysis["seeds_consistent"] is True

        # Тест с несовместимыми сидами
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
        )

        checker._check_seed_consistency(report, reference_run, [inconsistent_run])

        assert len(report.issues) == 1
        assert report.issues[0].issue_type == ReproducibilityIssueType.SEED_MISMATCH
        assert report.seed_analysis["seeds_consistent"] is False

    def test_check_exact_results_match(self, checker):
        """Тест проверки точного совпадения результатов."""
        # Создаем тестовые запуски с одинаковыми результатами
        reference_run = ExperimentRun(
            run_id="ref",
            seed=42,
            timestamp="2023-01-01T00:00:00",
            config_hash="hash1",
            environment_hash="env1",
            results={"reward": 100.0, "length": 50},
            metrics={},
        )

        matching_run = ExperimentRun(
            run_id="match",
            seed=42,
            timestamp="2023-01-01T00:01:00",
            config_hash="hash1",
            environment_hash="env1",
            results={"reward": 100.0, "length": 50},
            metrics={},
        )

        different_run = ExperimentRun(
            run_id="different",
            seed=42,
            timestamp="2023-01-01T00:02:00",
            config_hash="hash1",
            environment_hash="env1",
            results={"reward": 95.0, "length": 48},
            metrics={},
        )

        # Тест с совпадающими результатами
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
        )

        checker._check_exact_results_match(report, reference_run, [matching_run])
        assert len(report.issues) == 0

        # Тест с различающимися результатами
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
        )

        checker._check_exact_results_match(report, reference_run, [different_run])
        assert len(report.issues) > 0
        assert any(
            issue.issue_type == ReproducibilityIssueType.STATISTICAL_DIFFERENCE
            for issue in report.issues
        )

    def test_compute_trend(self, checker):
        """Тест вычисления тренда временного ряда."""
        # Возрастающий тренд
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        trend = checker._compute_trend(values, window_size=3)

        assert len(trend) > 0
        assert np.all(trend >= 0)  # Возрастающий тренд

        # Убывающий тренд
        values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        trend = checker._compute_trend(values, window_size=3)

        assert len(trend) > 0
        assert np.all(trend <= 0)  # Убывающий тренд

        # Недостаточно данных
        values = [1.0, 2.0]
        trend = checker._compute_trend(values, window_size=5)

        assert len(trend) == 0

    def test_compute_final_assessment(self, checker):
        """Тест вычисления итоговой оценки воспроизводимости."""
        # Тест без проблем
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
            seed_analysis={"seeds_consistent": True},
            environment_analysis={"environment_consistent": True},
        )

        checker._compute_final_assessment(report)

        assert report.is_reproducible is True
        assert report.confidence_score > 0.8

        # Тест с критическими проблемами
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=1.0,
            issues=[
                ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.SEED_MISMATCH,
                    severity="critical",
                    description="Test issue",
                    recommendation="Fix it",
                )
            ],
        )

        checker._compute_final_assessment(report)

        assert report.is_reproducible is False
        assert report.confidence_score < 1.0

    def test_generate_recommendations(self, checker):
        """Тест генерации рекомендаций."""
        report = ReproducibilityReport(
            experiment_id="test",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=False,
            confidence_score=0.5,
            issues=[
                ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.SEED_MISMATCH,
                    severity="critical",
                    description="Seed mismatch",
                    recommendation="Fix seeds",
                ),
                ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.DEPENDENCY_CONFLICT,
                    severity="warning",
                    description="Dependency conflict",
                    recommendation="Fix dependencies",
                ),
            ],
        )

        checker._generate_recommendations(report)

        assert len(report.recommendations) > 0
        assert any("сид" in rec.lower() for rec in report.recommendations)
        assert any("зависимост" in rec.lower() for rec in report.recommendations)

    def test_run_reproducibility_test(self, checker, sample_config):
        """Тест автоматического теста воспроизводимости."""

        # Мок функции тестирования
        def mock_test_function(seed):
            return {
                "final_reward": 100.0 + seed,  # Зависит от сида для тестирования
                "episode_length": 50,
                "metrics": {
                    "rewards": [10.0, 20.0, 30.0],
                    "episode_lengths": [48, 50, 52],
                },
            }

        # Запускаем тест
        report = checker.run_reproducibility_test(
            test_function=mock_test_function,
            experiment_id="auto_test",
            seeds=[42, 42, 42],  # Одинаковые сиды
            config=sample_config,
        )

        assert isinstance(report, ReproducibilityReport)
        assert report.experiment_id == "auto_test"
        assert len(report.runs) == 3

    def test_diagnose_seed_issues(self, checker):
        """Тест диагностики проблем с сидами."""
        # Создаем запуски с разными сидами
        runs = [
            ExperimentRun(
                run_id="run1",
                seed=42,
                timestamp="2023-01-01T00:00:00",
                config_hash="hash1",
                environment_hash="env1",
                results={},
                metrics={},
            ),
            ExperimentRun(
                run_id="run2",
                seed=42,
                timestamp="2023-01-01T00:01:00",
                config_hash="hash1",
                environment_hash="env1",
                results={},
                metrics={},
            ),
            ExperimentRun(
                run_id="run3",
                seed=123,
                timestamp="2023-01-01T00:02:00",
                config_hash="hash1",
                environment_hash="env1",
                results={},
                metrics={},
            ),
        ]

        diagnosis = checker._diagnose_seed_issues(runs)

        assert diagnosis["total_runs"] == 3
        assert diagnosis["unique_seeds"] == 2
        assert diagnosis["consistent"] is False
        assert "issues" in diagnosis

    def test_generate_reproducibility_guide(self, checker):
        """Тест генерации руководства по воспроизводимости."""
        guide = checker.generate_reproducibility_guide()

        assert isinstance(guide, str)
        assert len(guide) > 1000  # Должно быть достаточно подробным
        assert "воспроизводимость" in guide.lower()
        assert "сид" in guide.lower()
        assert "зависимост" in guide.lower()
        assert "## " in guide  # Markdown заголовки
        assert "```python" in guide  # Примеры кода


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    def test_create_simple_reproducibility_test(self):
        """Тест создания простой функции тестирования воспроизводимости."""
        test_function = create_simple_reproducibility_test(
            algorithm_name="PPO", env_name="CartPole-v1", total_timesteps=100
        )

        assert callable(test_function)

        # Тестируем функцию (может потребовать установленных зависимостей)
        try:
            result = test_function(42)
            assert isinstance(result, dict)
            assert "total_reward" in result
            assert "episode_length" in result
            assert "metrics" in result
        except ImportError:
            # Пропускаем если нет зависимостей
            pytest.skip("Stable-Baselines3 или Gymnasium не установлены")

    @patch("src.utils.reproducibility_checker.ReproducibilityChecker")
    def test_quick_reproducibility_check(self, mock_checker_class):
        """Тест быстрой проверки воспроизводимости."""
        # Настраиваем мок
        mock_checker = Mock()
        mock_report = Mock()
        mock_report.is_reproducible = True
        mock_report.confidence_score = 0.95
        mock_report.issues = []

        mock_checker.run_reproducibility_test.return_value = mock_report
        mock_checker_class.return_value = mock_checker

        result = quick_reproducibility_check(
            experiment_id="quick_test", num_runs=3, seed=42
        )

        assert result is True
        mock_checker_class.assert_called_once()
        mock_checker.run_reproducibility_test.assert_called_once()

    @patch("src.utils.reproducibility_checker.ReproducibilityChecker")
    def test_validate_experiment_reproducibility(self, mock_checker_class):
        """Тест валидации воспроизводимости конфигурации эксперимента."""
        # Создаем тестовую конфигурацию
        config = RLConfig(
            experiment_name="test_validation",
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=42),
            environment=EnvironmentConfig(name="CartPole-v1"),
        )

        # Настраиваем мок
        mock_checker = Mock()
        mock_report = Mock()
        mock_report.is_reproducible = True
        mock_checker.run_reproducibility_test.return_value = mock_report
        mock_checker_class.return_value = mock_checker

        result = validate_experiment_reproducibility(
            config=config, num_validation_runs=3
        )

        assert result is True
        mock_checker_class.assert_called_once()


class TestReproducibilityIssue:
    """Тесты для класса ReproducibilityIssue."""

    def test_issue_creation(self):
        """Тест создания проблемы воспроизводимости."""
        issue = ReproducibilityIssue(
            issue_type=ReproducibilityIssueType.SEED_MISMATCH,
            severity="critical",
            description="Test issue",
            recommendation="Fix it",
            details={"key": "value"},
        )

        assert issue.issue_type == ReproducibilityIssueType.SEED_MISMATCH
        assert issue.severity == "critical"
        assert issue.description == "Test issue"
        assert issue.recommendation == "Fix it"
        assert issue.details == {"key": "value"}
        assert issue.timestamp is not None


class TestExperimentRun:
    """Тесты для класса ExperimentRun."""

    def test_run_creation(self):
        """Тест создания запуска эксперимента."""
        run = ExperimentRun(
            run_id="test_run_123",
            seed=42,
            timestamp="2023-01-01T00:00:00",
            config_hash="abc123",
            environment_hash="def456",
            results={"reward": 100.0},
            metrics={"rewards": [10.0, 20.0, 30.0]},
            metadata={"test": True},
        )

        assert run.run_id == "test_run_123"
        assert run.seed == 42
        assert run.timestamp == "2023-01-01T00:00:00"
        assert run.config_hash == "abc123"
        assert run.environment_hash == "def456"
        assert run.results == {"reward": 100.0}
        assert run.metrics == {"rewards": [10.0, 20.0, 30.0]}
        assert run.metadata == {"test": True}


class TestReproducibilityReport:
    """Тесты для класса ReproducibilityReport."""

    def test_report_creation(self):
        """Тест создания отчета о воспроизводимости."""
        report = ReproducibilityReport(
            experiment_id="test_experiment",
            timestamp="2023-01-01T00:00:00",
            strictness_level=StrictnessLevel.STANDARD,
            is_reproducible=True,
            confidence_score=0.95,
        )

        assert report.experiment_id == "test_experiment"
        assert report.timestamp == "2023-01-01T00:00:00"
        assert report.strictness_level == StrictnessLevel.STANDARD
        assert report.is_reproducible is True
        assert report.confidence_score == 0.95
        assert report.runs == []
        assert report.issues == []
        assert report.statistics == {}
        assert report.recommendations == []


class TestStrictnessLevel:
    """Тесты для enum StrictnessLevel."""

    def test_strictness_levels(self):
        """Тест уровней строгости."""
        assert StrictnessLevel.MINIMAL.value == "minimal"
        assert StrictnessLevel.STANDARD.value == "standard"
        assert StrictnessLevel.STRICT.value == "strict"
        assert StrictnessLevel.PARANOID.value == "paranoid"

        # Проверяем, что все уровни различны
        levels = [level.value for level in StrictnessLevel]
        assert len(levels) == len(set(levels))


class TestReproducibilityIssueType:
    """Тесты для enum ReproducibilityIssueType."""

    def test_issue_types(self):
        """Тест типов проблем воспроизводимости."""
        expected_types = [
            "seed_mismatch",
            "environment_difference",
            "dependency_conflict",
            "hardware_difference",
            "algorithm_nondeterminism",
            "statistical_difference",
            "trend_deviation",
            "configuration_mismatch",
        ]

        actual_types = [issue_type.value for issue_type in ReproducibilityIssueType]

        for expected_type in expected_types:
            assert expected_type in actual_types

        # Проверяем, что все типы различны
        assert len(actual_types) == len(set(actual_types))


@pytest.mark.integration
class TestReproducibilityCheckerIntegration:
    """Интеграционные тесты для ReproducibilityChecker."""

    @pytest.fixture
    def temp_project_root(self):
        """Временная директория проекта."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_full_reproducibility_workflow(self, temp_project_root):
        """Тест полного рабочего процесса проверки воспроизводимости."""
        checker = ReproducibilityChecker(
            project_root=temp_project_root, strictness_level=StrictnessLevel.MINIMAL
        )

        # Создаем простую тестовую функцию
        def simple_test(seed):
            np.random.seed(seed)
            return {
                "value": np.random.random(),
                "metrics": {"values": [np.random.random() for _ in range(5)]},
            }

        # Создаем конфигурацию
        config = RLConfig(experiment_name="integration_test", seed=42)

        # Регистрируем несколько запусков
        for i in range(3):
            results = simple_test(42)
            checker.register_experiment_run(
                experiment_id="integration_test",
                config=config,
                results={"value": results["value"]},
                metrics=results["metrics"],
                metadata={"run_number": i},
            )

        # Проверяем воспроизводимость
        report = checker.check_reproducibility("integration_test")

        assert isinstance(report, ReproducibilityReport)
        assert report.experiment_id == "integration_test"
        assert len(report.runs) == 3
        assert (
            report.is_reproducible is True
        )  # Должно быть воспроизводимо с одинаковыми сидами

        # Проверяем, что отчет сохранен
        report_files = list(checker.reports_dir.glob("report_integration_test_*.json"))
        assert len(report_files) == 1

    def test_reproducibility_with_different_seeds(self, temp_project_root):
        """Тест воспроизводимости с разными сидами."""
        checker = ReproducibilityChecker(
            project_root=temp_project_root, strictness_level=StrictnessLevel.STANDARD
        )

        def test_function(seed):
            np.random.seed(seed)
            return {
                "value": np.random.random(),
                "metrics": {"values": [np.random.random() for _ in range(3)]},
            }

        # Регистрируем запуски с разными сидами
        seeds = [42, 123, 456]
        for i, seed in enumerate(seeds):
            config = RLConfig(experiment_name="different_seeds_test", seed=seed)
            results = test_function(seed)

            checker.register_experiment_run(
                experiment_id="different_seeds_test",
                config=config,
                results={"value": results["value"]},
                metrics=results["metrics"],
                metadata={"run_number": i},
            )

        # Проверяем воспроизводимость
        report = checker.check_reproducibility("different_seeds_test")

        assert isinstance(report, ReproducibilityReport)
        assert len(report.runs) == 3
        assert (
            report.is_reproducible is False
        )  # Не должно быть воспроизводимо с разными сидами

        # Должны быть проблемы с сидами
        seed_issues = [
            issue
            for issue in report.issues
            if issue.issue_type == ReproducibilityIssueType.SEED_MISMATCH
        ]
        assert len(seed_issues) > 0
