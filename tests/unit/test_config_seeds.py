"""Тесты для функциональности управления сидами в конфигурации.

Этот модуль тестирует новые возможности для обеспечения консистентности
сидов и воспроизводимости в системе обучения RL агентов.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.config import (
    RLConfig,
    AlgorithmConfig,
    ReproducibilityConfig,
    ConfigLoader,
    load_config_with_seeds,
    enforce_global_seed_consistency,
)


class TestRLConfigSeedMethods:
    """Тесты для методов управления сидами в RLConfig."""

    def test_enforce_seed_consistency(self):
        """Тест принудительной синхронизации сидов."""
        # Создаем конфигурацию без автоматической синхронизации
        config = RLConfig(
            seed=123,
            algorithm=AlgorithmConfig(name="PPO", seed=456),
            reproducibility=ReproducibilityConfig(seed=789, auto_propagate_seeds=False),
        )

        # Принудительно устанавливаем разные сиды
        config.algorithm.seed = 456
        config.reproducibility.seed = 789

        # Проверяем, что сиды изначально разные
        assert config.seed == 123
        assert config.algorithm.seed == 456
        assert config.reproducibility.seed == 789

        # Принудительная синхронизация
        config.enforce_seed_consistency()

        # Проверяем, что все сиды синхронизированы
        assert config.seed == 123
        assert config.algorithm.seed == 123
        assert config.reproducibility.seed == 123

    def test_validate_reproducibility_no_conflicts(self):
        """Тест валидации воспроизводимости без конфликтов."""
        config = RLConfig(
            seed=42,
            algorithm=AlgorithmConfig(
                name="PPO", seed=42, device="cpu"
            ),  # Явно указываем device
            reproducibility=ReproducibilityConfig(
                seed=42, deterministic=True, use_cuda=False
            ),
        )

        is_valid, warnings = config.validate_reproducibility()

        assert is_valid is True
        # Может быть несколько предупреждений о системе, но не о конфликтах сидов
        seed_conflicts = [w for w in warnings if "конфликт сидов" in w.lower()]
        assert len(seed_conflicts) == 0

    def test_validate_reproducibility_with_conflicts(self):
        """Тест валидации воспроизводимости с конфликтами."""
        # Создаем конфигурацию без автоматической синхронизации
        config = RLConfig(
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=123),
            reproducibility=ReproducibilityConfig(
                seed=456,
                deterministic=True,
                benchmark=True,  # Конфликт с deterministic
                auto_propagate_seeds=False,  # Отключаем автоматическую синхронизацию
            ),
        )

        # Принудительно устанавливаем разные сиды
        config.algorithm.seed = 123
        config.reproducibility.seed = 456

        is_valid, warnings = config.validate_reproducibility()

        assert len(warnings) > 0
        # Проверяем, что обнаружены конфликты сидов
        seed_conflict_found = any("конфликт сидов" in w.lower() for w in warnings)
        assert seed_conflict_found

        # Проверяем, что обнаружен конфликт deterministic/benchmark
        deterministic_conflict_found = any(
            "deterministic" in w.lower() and "benchmark" in w.lower() for w in warnings
        )
        assert deterministic_conflict_found

    @patch("src.utils.seeding.set_seed")
    @patch("src.utils.seeding.verify_reproducibility")
    def test_apply_seeds_success(self, mock_verify, mock_set_seed):
        """Тест успешного применения сидов."""
        mock_verify.return_value = True

        config = RLConfig(
            seed=42, reproducibility=ReproducibilityConfig(validate_determinism=True)
        )

        config.apply_seeds()

        mock_set_seed.assert_called_once_with(42)
        mock_verify.assert_called_once_with(42, test_operations=50)

    @patch("src.utils.seeding.set_seed")
    @patch("src.utils.seeding.verify_reproducibility")
    def test_apply_seeds_reproducibility_failure(self, mock_verify, mock_set_seed):
        """Тест применения сидов с ошибкой воспроизводимости."""
        mock_verify.return_value = False

        config = RLConfig(
            seed=42, reproducibility=ReproducibilityConfig(validate_determinism=True)
        )

        with pytest.raises(
            RuntimeError, match="Не удалось обеспечить воспроизводимость"
        ):
            config.apply_seeds()

    def test_get_reproducibility_report(self):
        """Тест генерации отчета о воспроизводимости."""
        config = RLConfig(
            experiment_name="test_experiment",
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=42),
            reproducibility=ReproducibilityConfig(seed=42),
        )

        report = config.get_reproducibility_report()

        # Проверяем структуру отчета
        assert "timestamp" in report
        assert "experiment_name" in report
        assert "is_valid" in report
        assert "warnings" in report
        assert "seeds" in report
        assert "determinism" in report
        assert "algorithm" in report
        assert "system" in report
        assert "recommendations" in report

        # Проверяем содержимое
        assert report["experiment_name"] == "test_experiment"
        assert report["seeds"]["main_seed"] == 42
        assert report["seeds"]["seeds_consistent"] is True

    def test_check_seed_conflicts(self):
        """Тест проверки конфликтов сидов."""
        # Создаем конфигурацию без автоматической синхронизации
        config = RLConfig(
            seed=42,
            algorithm=AlgorithmConfig(name="PPO", seed=123),
            reproducibility=ReproducibilityConfig(seed=456, auto_propagate_seeds=False),
        )

        # Принудительно устанавливаем разные сиды
        config.algorithm.seed = 123
        config.reproducibility.seed = 456

        conflicts = config._check_seed_conflicts()

        assert len(conflicts) == 2  # Два конфликта
        assert any("algorithm seed" in conflict for conflict in conflicts)
        assert any("reproducibility seed" in conflict for conflict in conflicts)

    def test_auto_propagate_seeds_on_init(self):
        """Тест автоматического распространения сидов при инициализации."""
        config = RLConfig(
            seed=999,
            algorithm=AlgorithmConfig(name="PPO", seed=123),
            reproducibility=ReproducibilityConfig(seed=456, auto_propagate_seeds=True),
        )

        # После инициализации сиды должны быть синхронизированы
        assert config.algorithm.seed == 999
        assert config.reproducibility.seed == 999


class TestConfigLoaderSeedMethods:
    """Тесты для новых методов ConfigLoader."""

    def setup_method(self):
        """Настройка для каждого теста."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.loader = ConfigLoader(self.config_dir)

    def teardown_method(self):
        """Очистка после каждого теста."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("src.utils.seeding.set_seed")
    @patch("src.utils.seeding.verify_reproducibility")
    def test_load_config_with_seed_validation(self, mock_verify, mock_set_seed):
        """Тест загрузки конфигурации с валидацией сидов."""
        mock_verify.return_value = True

        # Создаем тестовую конфигурацию
        config_data = {
            "seed": 42,
            "algorithm": {"name": "PPO", "seed": 42},
            "environment": {"name": "LunarLander-v3"},  # Добавляем обязательное поле
            "reproducibility": {"seed": 42},
        }

        config_file = self.config_dir / "test.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Загружаем с валидацией
        config = self.loader.load_config_with_seed_validation(
            config_path=config_file, apply_seeds=True, validate_reproducibility=True
        )

        assert config.seed == 42
        mock_set_seed.assert_called_once_with(42)

    def test_create_reproducibility_report(self):
        """Тест создания отчета о воспроизводимости."""
        config = RLConfig(seed=42)

        # Создаем отчет без сохранения
        report = self.loader.create_reproducibility_report(config)

        assert isinstance(report, dict)
        assert "seeds" in report
        assert "determinism" in report

        # Создаем отчет с сохранением
        report_file = self.config_dir / "reproducibility_report.json"
        report = self.loader.create_reproducibility_report(config, report_file)

        assert report_file.exists()

        # Проверяем содержимое файла
        with open(report_file, "r") as f:
            saved_report = json.load(f)

        assert saved_report["seeds"]["main_seed"] == 42

    def test_validate_seed_consistency_across_configs(self):
        """Тест валидации консистентности сидов между конфигурациями."""
        # Создаем несколько тестовых конфигураций
        configs_data = [
            {
                "seed": 42,
                "algorithm": {"name": "PPO", "seed": 42},
                "environment": {"name": "LunarLander-v3"},
            },
            {
                "seed": 42,
                "algorithm": {"name": "A2C", "seed": 42},
                "environment": {"name": "LunarLander-v3"},
            },
            {
                "seed": 123,
                "algorithm": {"name": "SAC", "seed": 123},
                "environment": {"name": "LunarLander-v3"},
            },  # Другой сид
        ]

        config_files = []
        for i, config_data in enumerate(configs_data):
            config_file = self.config_dir / f"config_{i}.yaml"
            with open(config_file, "w") as f:
                import yaml

                yaml.dump(config_data, f)
            config_files.append(config_file)

        # Проверяем консистентность
        report = self.loader.validate_seed_consistency_across_configs(config_files)

        assert report["total_configs"] == 3
        assert report["loaded_configs"] == 3
        assert len(report["consistency_issues"]) > 0  # Должен быть конфликт сидов

        # Проверяем, что обнаружен конфликт основных сидов
        main_seed_conflict = any(
            "различные основные сиды" in issue.lower()
            for issue in report["consistency_issues"]
        )
        assert main_seed_conflict


class TestUtilityFunctions:
    """Тесты для утилитарных функций."""

    @patch("src.utils.seeding.set_seed")
    @patch("src.utils.seeding.verify_reproducibility")
    def test_load_config_with_seeds(self, mock_verify, mock_set_seed):
        """Тест удобной функции загрузки конфигурации с сидами."""
        mock_verify.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Создаем тестовую конфигурацию
            config_data = {
                "seed": 42,
                "algorithm": {"name": "PPO"},
                "environment": {"name": "LunarLander-v3"},
                "reproducibility": {"seed": 42},
            }

            config_file = config_dir / "test.yaml"
            with open(config_file, "w") as f:
                import yaml

                yaml.dump(config_data, f)

            # Используем глобальную функцию
            with patch("src.utils.config.get_config_loader") as mock_get_loader:
                mock_loader = MagicMock()
                mock_loader.load_config_with_seed_validation.return_value = RLConfig(
                    seed=42
                )
                mock_get_loader.return_value = mock_loader

                config = load_config_with_seeds(config_path=config_file)

                assert config.seed == 42
                mock_loader.load_config_with_seed_validation.assert_called_once()

    def test_enforce_global_seed_consistency(self):
        """Тест принудительной синхронизации сидов между конфигурациями."""
        configs = [
            RLConfig(seed=42, algorithm=AlgorithmConfig(name="PPO", seed=123)),
            RLConfig(seed=456, algorithm=AlgorithmConfig(name="A2C", seed=789)),
            RLConfig(seed=999, algorithm=AlgorithmConfig(name="SAC", seed=111)),
        ]

        # Синхронизируем с мастер-сидом
        synced_configs = enforce_global_seed_consistency(configs, master_seed=777)

        # Проверяем, что все конфигурации имеют одинаковый сид
        for config in synced_configs:
            assert config.seed == 777
            assert config.algorithm.seed == 777
            assert config.reproducibility.seed == 777

    def test_enforce_global_seed_consistency_no_master(self):
        """Тест синхронизации без указания мастер-сида."""
        configs = [
            RLConfig(seed=42),
            RLConfig(seed=456),
        ]

        # Синхронизируем без мастер-сида (должен использоваться первый)
        synced_configs = enforce_global_seed_consistency(configs)

        for config in synced_configs:
            assert config.seed == 42

    def test_enforce_global_seed_consistency_empty_list(self):
        """Тест синхронизации пустого списка конфигураций."""
        configs = []
        result = enforce_global_seed_consistency(configs)
        assert result == []


class TestReproducibilityConfig:
    """Тесты для расширенной конфигурации воспроизводимости."""

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = ReproducibilityConfig()

        assert config.seed == 42
        assert config.deterministic is True
        assert config.benchmark is False
        assert config.use_cuda is False
        assert config.enforce_seed_consistency is True
        assert config.validate_determinism is True
        assert config.warn_on_seed_conflicts is True
        assert config.auto_propagate_seeds is True

    def test_custom_values(self):
        """Тест пользовательских значений."""
        config = ReproducibilityConfig(
            seed=123,
            deterministic=False,
            enforce_seed_consistency=False,
            auto_propagate_seeds=False,
        )

        assert config.seed == 123
        assert config.deterministic is False
        assert config.enforce_seed_consistency is False
        assert config.auto_propagate_seeds is False


@pytest.mark.parametrize("seed_value", [0, 42, 1000, 2**32 - 1])
def test_seed_validation_valid_values(seed_value):
    """Тест валидации корректных значений сидов."""
    config = RLConfig(seed=seed_value)
    loader = ConfigLoader()

    # Не должно вызывать исключений
    loader._validate_config(config)


@pytest.mark.parametrize("invalid_seed", [-1, 2**32, 2**33])
def test_seed_validation_invalid_values(invalid_seed):
    """Тест валидации некорректных значений сидов."""
    config = RLConfig(seed=invalid_seed)
    loader = ConfigLoader()

    with pytest.raises(ValueError, match="seed должен быть в диапазоне"):
        loader._validate_config(config)


class TestIntegrationWithSeeding:
    """Интеграционные тесты с модулем seeding."""

    @patch("src.utils.seeding.set_seed")
    @patch("src.utils.seeding.verify_reproducibility")
    def test_integration_apply_seeds(self, mock_verify, mock_set_seed):
        """Тест интеграции с модулем seeding."""
        mock_verify.return_value = True

        config = RLConfig(
            seed=42, reproducibility=ReproducibilityConfig(validate_determinism=True)
        )

        config.apply_seeds()

        # Проверяем, что вызваны функции из модуля seeding
        mock_set_seed.assert_called_once_with(42)
        mock_verify.assert_called_once_with(42, test_operations=50)

    @patch("src.utils.seeding.set_seed")
    def test_integration_without_validation(self, mock_set_seed):
        """Тест интеграции без валидации воспроизводимости."""
        config = RLConfig(
            seed=42, reproducibility=ReproducibilityConfig(validate_determinism=False)
        )

        config.apply_seeds()

        mock_set_seed.assert_called_once_with(42)
