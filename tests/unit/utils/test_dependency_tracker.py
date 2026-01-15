"""Тесты для модуля отслеживания зависимостей."""

import json
import platform
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.dependency_tracker import (
    DependencyTracker,
    create_experiment_snapshot,
    validate_environment_for_experiment,
)


class TestDependencyTracker:
    """Тесты для класса DependencyTracker."""

    @pytest.fixture
    def temp_project_root(self, tmp_path):
        """Временная директория проекта."""
        return tmp_path / "test_project"

    @pytest.fixture
    def tracker(self, temp_project_root):
        """Экземпляр DependencyTracker для тестов."""
        return DependencyTracker(temp_project_root)

    def test_init(self, tracker, temp_project_root):
        """Тест инициализации трекера."""
        assert tracker.project_root == temp_project_root
        assert tracker.snapshots_dir == temp_project_root / "results" / "dependencies"
        assert tracker.snapshots_dir.exists()
        assert "torch" in tracker.ml_libraries
        assert "stable-baselines3" in tracker.ml_libraries

    def test_get_system_info(self, tracker):
        """Тест получения информации о системе."""
        system_info = tracker.get_system_info()

        assert "timestamp" in system_info
        assert "python" in system_info
        assert "platform" in system_info
        assert "hardware" in system_info

        # Проверяем Python информацию
        python_info = system_info["python"]
        assert python_info["version"] == sys.version
        assert python_info["version_info"]["major"] == sys.version_info.major
        assert python_info["executable"] == sys.executable

        # Проверяем платформу
        platform_info = system_info["platform"]
        assert platform_info["system"] == platform.system()
        assert platform_info["machine"] == platform.machine()

    @patch("subprocess.run")
    def test_get_package_manager_info_pip_available(self, mock_run, tracker):
        """Тест определения доступности pip."""
        mock_run.return_value = MagicMock(stdout="pip 21.0.1", returncode=0)

        managers = tracker.get_package_manager_info()

        assert managers["pip"]["available"] is True
        assert "pip 21.0.1" in managers["pip"]["version"]
        mock_run.assert_called()

    @patch("subprocess.run")
    def test_get_package_manager_info_conda_available(self, mock_run, tracker):
        """Тест определения доступности conda."""

        def side_effect(*args, **kwargs):
            if "conda" in args[0] and "--version" in args[0]:
                return MagicMock(stdout="conda 4.10.3", returncode=0)
            elif "conda" in args[0] and "info" in args[0]:
                return MagicMock(
                    stdout='{"active_prefix_name": "test_env", "active_prefix": "/path/to/env"}',
                    returncode=0,
                )
            else:
                return MagicMock(stdout="pip 21.0.1", returncode=0)

        mock_run.side_effect = side_effect

        managers = tracker.get_package_manager_info()

        assert managers["conda"]["available"] is True
        assert "conda 4.10.3" in managers["conda"]["version"]
        assert managers["conda"]["active_env"] == "test_env"

    @patch("subprocess.run")
    def test_get_pip_packages(self, mock_run, tracker):
        """Тест получения pip пакетов."""
        mock_run.return_value = MagicMock(
            stdout="numpy==1.21.0\npandas==1.3.0\ntorch==1.9.0", returncode=0
        )

        packages = tracker.get_pip_packages()

        assert packages["numpy"] == "1.21.0"
        assert packages["pandas"] == "1.3.0"
        assert packages["torch"] == "1.9.0"
        mock_run.assert_called_with(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_conda_packages(self, mock_run, tracker):
        """Тест получения conda пакетов."""
        mock_packages = [
            {"name": "numpy", "version": "1.21.0"},
            {"name": "pandas", "version": "1.3.0"},
        ]
        mock_run.return_value = MagicMock(
            stdout=json.dumps(mock_packages), returncode=0
        )

        packages = tracker.get_conda_packages()

        assert packages["numpy"] == "1.21.0"
        assert packages["pandas"] == "1.3.0"

    @patch("builtins.__import__")
    def test_get_ml_library_versions(self, mock_import, tracker):
        """Тест получения версий ML библиотек."""
        # Мокаем импорт numpy
        mock_numpy = MagicMock()
        mock_numpy.__version__ = "1.21.0"

        def import_side_effect(name, *args, **kwargs):
            if name == "numpy":
                return mock_numpy
            else:
                raise ImportError(f"No module named '{name}'")

        mock_import.side_effect = import_side_effect

        versions = tracker.get_ml_library_versions()

        assert versions["numpy"] == "1.21.0"
        assert versions["torch"] is None  # Не установлен

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_create_dependency_snapshot(self, mock_json_dump, mock_file, tracker):
        """Тест создания снимка зависимостей."""
        with (
            patch.object(tracker, "get_system_info") as mock_system,
            patch.object(tracker, "get_package_manager_info") as mock_managers,
            patch.object(tracker, "get_pip_packages") as mock_pip,
            patch.object(tracker, "get_conda_packages") as mock_conda,
            patch.object(tracker, "get_ml_library_versions") as mock_ml,
        ):
            mock_system.return_value = {"python": {"version": "3.8.0"}}
            mock_managers.return_value = {"pip": {"available": True}}
            mock_pip.return_value = {"numpy": "1.21.0"}
            mock_conda.return_value = {}
            mock_ml.return_value = {"numpy": "1.21.0"}

            snapshot = tracker.create_dependency_snapshot("test_snapshot")

            assert snapshot["metadata"]["name"] == "test_snapshot"
            assert "timestamp" in snapshot["metadata"]
            assert "hash" in snapshot["metadata"]
            assert "system" in snapshot
            assert "packages" in snapshot
            assert "ml_libraries" in snapshot

            mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_snapshot(self, mock_json_load, mock_file, tracker):
        """Тест загрузки снимка."""
        mock_snapshot = {
            "metadata": {"name": "test", "timestamp": "2023-01-01T00:00:00"},
            "system": {},
            "packages": {},
        }
        mock_json_load.return_value = mock_snapshot

        # Создаем файл снимка
        snapshot_file = tracker.snapshots_dir / "snapshot_test.json"
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file.touch()

        snapshot = tracker.load_snapshot("test")

        assert snapshot == mock_snapshot
        mock_json_load.assert_called_once()

    def test_load_snapshot_not_found(self, tracker):
        """Тест загрузки несуществующего снимка."""
        snapshot = tracker.load_snapshot("nonexistent")
        assert snapshot is None

    def test_compare_snapshots(self, tracker):
        """Тест сравнения снимков."""
        snapshot1 = {
            "packages": {"pip": {"numpy": "1.20.0", "pandas": "1.2.0"}},
            "ml_libraries": {"numpy": "1.20.0"},
            "system": {"python": {"version": "3.8.0"}},
        }

        snapshot2 = {
            "packages": {"pip": {"numpy": "1.21.0", "scipy": "1.7.0"}},
            "ml_libraries": {"numpy": "1.21.0"},
            "system": {"python": {"version": "3.8.0"}},
        }

        with patch.object(tracker, "load_snapshot") as mock_load:
            mock_load.side_effect = [snapshot1, snapshot2]

            comparison = tracker.compare_snapshots("snap1", "snap2")

            assert "scipy" in comparison["changes"]["packages_added"]
            assert "pandas" in comparison["changes"]["packages_removed"]
            assert "numpy" in comparison["changes"]["packages_updated"]
            assert comparison["changes"]["packages_updated"]["numpy"]["old"] == "1.20.0"
            assert comparison["changes"]["packages_updated"]["numpy"]["new"] == "1.21.0"

    @patch("subprocess.run")
    def test_detect_dependency_conflicts_no_conflicts(self, mock_run, tracker):
        """Тест детектирования конфликтов - нет конфликтов."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        with patch.object(tracker, "get_pip_packages") as mock_packages:
            mock_packages.return_value = {"numpy": "1.21.0"}

            conflicts = tracker.detect_dependency_conflicts()

            assert len(conflicts) == 0

    @patch("subprocess.run")
    def test_detect_dependency_conflicts_pip_conflicts(self, mock_run, tracker):
        """Тест детектирования pip конфликтов."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="package-a 1.0 has requirement package-b>=2.0, but you have package-b 1.0",
        )

        with patch.object(tracker, "get_pip_packages") as mock_packages:
            mock_packages.return_value = {"numpy": "1.21.0"}

            conflicts = tracker.detect_dependency_conflicts()

            assert len(conflicts) == 1
            assert conflicts[0]["type"] == "pip_dependency_conflict"

    def test_detect_dependency_conflicts_gym_gymnasium(self, tracker):
        """Тест детектирования конфликта gym/gymnasium."""
        with patch.object(tracker, "get_pip_packages") as mock_packages:
            mock_packages.return_value = {"gym": "0.21.0", "gymnasium": "0.26.0"}

            conflicts = tracker.detect_dependency_conflicts()

            gym_conflict = next(
                (c for c in conflicts if c["type"] == "library_conflict"), None
            )
            assert gym_conflict is not None
            assert "gym" in gym_conflict["packages"]
            assert "gymnasium" in gym_conflict["packages"]

    def test_generate_compatibility_report(self, tracker):
        """Тест генерации отчета совместимости."""
        mock_snapshot = {
            "system": {
                "python": {"version_info": {"major": 3, "minor": 9, "micro": 0}},
                "platform": {"system": "Linux"},
                "hardware": {"memory_total": 8 * 1024**3},  # 8GB
            },
            "ml_libraries": {
                "numpy": "1.21.0",
                "stable-baselines3": "1.5.0",
                "gymnasium": "0.26.0",
            },
        }

        with (
            patch.object(tracker, "create_dependency_snapshot") as mock_create,
            patch.object(tracker, "detect_dependency_conflicts") as mock_conflicts,
        ):
            mock_create.return_value = mock_snapshot
            mock_conflicts.return_value = []

            report = tracker.generate_compatibility_report()

            assert report["system_compatibility"]["python_version_ok"] is True
            assert report["system_compatibility"]["platform_supported"] is True
            assert report["system_compatibility"]["memory_sufficient"] is True
            assert report["package_compatibility"]["ml_libraries_compatible"] is True

    def test_export_pip_requirements(self, tracker, tmp_path):
        """Тест экспорта pip requirements."""
        with patch.object(tracker, "get_pip_packages") as mock_packages:
            mock_packages.return_value = {"numpy": "1.21.0", "pandas": "1.3.0"}

            output_file = tmp_path / "requirements.txt"
            content = tracker.export_requirements("pip", output_file)

            assert "numpy==1.21.0" in content
            assert "pandas==1.3.0" in content
            assert output_file.exists()

    @patch("subprocess.run")
    def test_export_conda_environment(self, mock_run, tracker, tmp_path):
        """Тест экспорта conda environment."""
        mock_run.return_value = MagicMock(
            stdout="name: test\nchannels:\n  - defaults\ndependencies:\n  - numpy=1.21.0",
            returncode=0,
        )

        output_file = tmp_path / "environment.yml"
        content = tracker.export_requirements("conda", output_file)

        assert "numpy=1.21.0" in content
        assert output_file.exists()

    def test_validate_reproducibility_success(self, tracker):
        """Тест успешной валидации воспроизводимости."""
        reference_snapshot = {
            "system": {
                "python": {"version_info": {"major": 3, "minor": 9, "micro": 0}},
                "platform": {"system": "Linux"},
            },
            "ml_libraries": {"numpy": "1.21.0", "torch": "1.9.0"},
        }

        current_snapshot = {
            "system": {
                "python": {"version_info": {"major": 3, "minor": 9, "micro": 0}},
                "platform": {"system": "Linux"},
            },
            "ml_libraries": {"numpy": "1.21.0", "torch": "1.9.0"},
        }

        with (
            patch.object(tracker, "load_snapshot") as mock_load,
            patch.object(tracker, "create_dependency_snapshot") as mock_create,
        ):
            mock_load.return_value = reference_snapshot
            mock_create.return_value = current_snapshot

            validation = tracker.validate_reproducibility("reference")

            assert validation["reproducible"] is True
            assert len(validation["issues"]) == 0

    def test_validate_reproducibility_python_mismatch(self, tracker):
        """Тест валидации с несоответствием версии Python."""
        reference_snapshot = {
            "system": {
                "python": {"version_info": {"major": 3, "minor": 8, "micro": 0}},
                "platform": {"system": "Linux"},
            },
            "ml_libraries": {"numpy": "1.21.0"},
        }

        current_snapshot = {
            "system": {
                "python": {"version_info": {"major": 3, "minor": 9, "micro": 0}},
                "platform": {"system": "Linux"},
            },
            "ml_libraries": {"numpy": "1.21.0"},
        }

        with (
            patch.object(tracker, "load_snapshot") as mock_load,
            patch.object(tracker, "create_dependency_snapshot") as mock_create,
        ):
            mock_load.return_value = reference_snapshot
            mock_create.return_value = current_snapshot

            validation = tracker.validate_reproducibility("reference")

            assert validation["reproducible"] is False
            assert len(validation["issues"]) == 1
            assert validation["issues"][0]["type"] == "python_version_mismatch"

    def test_get_snapshots_list(self, tracker):
        """Тест получения списка снимков."""
        # Создаем тестовые файлы снимков
        snapshot1 = {
            "metadata": {
                "name": "snapshot1",
                "timestamp": "2023-01-01T00:00:00",
                "hash": "hash1",
            }
        }

        snapshot2 = {
            "metadata": {
                "name": "snapshot2",
                "timestamp": "2023-01-02T00:00:00",
                "hash": "hash2",
            }
        }

        # Создаем файлы
        tracker.snapshots_dir.mkdir(parents=True, exist_ok=True)

        with open(tracker.snapshots_dir / "snapshot_snapshot1.json", "w") as f:
            json.dump(snapshot1, f)

        with open(tracker.snapshots_dir / "snapshot_snapshot2.json", "w") as f:
            json.dump(snapshot2, f)

        snapshots = tracker.get_snapshots_list()

        assert len(snapshots) == 2
        # Проверяем сортировку по времени (новые первые)
        assert snapshots[0]["name"] == "snapshot2"
        assert snapshots[1]["name"] == "snapshot1"

    def test_cleanup_old_snapshots(self, tracker):
        """Тест очистки старых снимков."""
        # Создаем тестовые снимки
        tracker.snapshots_dir.mkdir(parents=True, exist_ok=True)

        for i in range(15):
            snapshot = {
                "metadata": {
                    "name": f"snapshot{i}",
                    "timestamp": f"2023-01-{i + 1:02d}T00:00:00",
                    "hash": f"hash{i}",
                }
            }

            with open(tracker.snapshots_dir / f"snapshot_snapshot{i}.json", "w") as f:
                json.dump(snapshot, f)

        deleted_count = tracker.cleanup_old_snapshots(keep_count=10)

        assert deleted_count == 5
        remaining_files = list(tracker.snapshots_dir.glob("snapshot_*.json"))
        assert len(remaining_files) == 10


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    @patch("src.utils.dependency_tracker.DependencyTracker")
    def test_create_experiment_snapshot(self, mock_tracker_class):
        """Тест создания снимка для эксперимента."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        mock_snapshot = {
            "metadata": {"name": "test_snapshot", "timestamp": "2023-01-01T00:00:00"}
        }
        mock_tracker.create_dependency_snapshot.return_value = mock_snapshot
        mock_tracker.snapshots_dir = Path("/tmp/snapshots")

        with patch("builtins.open", mock_open()), patch("json.dump") as mock_json_dump:
            snapshot = create_experiment_snapshot("exp123")

            assert snapshot["metadata"]["experiment_id"] == "exp123"
            mock_tracker.create_dependency_snapshot.assert_called_once()
            mock_json_dump.assert_called_once()

    @patch("src.utils.dependency_tracker.DependencyTracker")
    def test_validate_environment_for_experiment_success(self, mock_tracker_class):
        """Тест успешной валидации среды для эксперимента."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        mock_validation = {"reproducible": True, "issues": [], "warnings": []}
        mock_tracker.validate_reproducibility.return_value = mock_validation

        result = validate_environment_for_experiment("reference_snapshot")

        assert result is True
        mock_tracker.validate_reproducibility.assert_called_once_with(
            "reference_snapshot"
        )

    @patch("src.utils.dependency_tracker.DependencyTracker")
    def test_validate_environment_for_experiment_failure(self, mock_tracker_class):
        """Тест неуспешной валидации среды для эксперимента."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        mock_validation = {
            "reproducible": False,
            "issues": [{"type": "python_version_mismatch"}],
            "warnings": [],
        }
        mock_tracker.validate_reproducibility.return_value = mock_validation

        result = validate_environment_for_experiment("reference_snapshot")

        assert result is False


@pytest.mark.integration
class TestDependencyTrackerIntegration:
    """Интеграционные тесты для DependencyTracker."""

    def test_real_system_info(self, tmp_path):
        """Тест получения реальной информации о системе."""
        tracker = DependencyTracker(tmp_path)
        system_info = tracker.get_system_info()

        # Проверяем, что получили реальные данные
        assert system_info["python"]["version"] == sys.version
        assert system_info["platform"]["system"] == platform.system()
        assert system_info["hardware"]["cpu_count"] > 0
        assert system_info["hardware"]["memory_total"] > 0

    def test_real_package_detection(self, tmp_path):
        """Тест реального определения пакетов."""
        tracker = DependencyTracker(tmp_path)

        # Тестируем pip (должен быть доступен)
        managers = tracker.get_package_manager_info()
        assert managers["pip"]["available"] is True

        # Тестируем получение pip пакетов
        packages = tracker.get_pip_packages()
        assert isinstance(packages, dict)
        # Проверяем, что получили хотя бы некоторые пакеты
        assert len(packages) > 0
        # Проверяем наличие базовых пакетов Python
        common_packages = {"pip", "setuptools", "wheel", "numpy", "pytest"}
        found_packages = set(packages.keys()) & common_packages
        assert len(found_packages) > 0, (
            f"Не найдено ни одного из ожидаемых пакетов: {common_packages}"
        )

    def test_full_snapshot_creation(self, tmp_path):
        """Тест создания полного снимка."""
        tracker = DependencyTracker(tmp_path)
        snapshot = tracker.create_dependency_snapshot("integration_test")

        # Проверяем структуру снимка
        assert "metadata" in snapshot
        assert "system" in snapshot
        assert "packages" in snapshot
        assert "ml_libraries" in snapshot

        # Проверяем, что файл создан
        snapshot_file = tracker.snapshots_dir / "snapshot_integration_test.json"
        assert snapshot_file.exists()

        # Проверяем, что можем загрузить снимок
        loaded_snapshot = tracker.load_snapshot("integration_test")
        assert loaded_snapshot is not None
        assert loaded_snapshot["metadata"]["name"] == "integration_test"
