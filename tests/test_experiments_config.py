"""Тесты для модуля src.experiments.config.

Этот модуль содержит комплексные тесты для класса Configuration,
включая валидацию, сериализацию, сравнение и создание конфигураций.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.experiments.config import (
    Configuration,
    ValidationError,
    compare_configs,
    create_a2c_config,
    create_ppo_config,
    create_sac_config,
    create_td3_config,
    validate_config_file,
)


class TestConfiguration:
    """Тесты для класса Configuration."""

    def test_init_minimal(self) -> None:
        """Тест создания конфигурации с минимальными параметрами."""
        config = Configuration(algorithm="PPO", environment="LunarLander-v2")

        assert config.algorithm == "PPO"
        assert config.environment == "LunarLander-v2"
        assert config.seed == 42
        assert config.training_steps == 100_000
        assert config.evaluation_frequency == 10_000
        assert config.experiment_name == "default_experiment"
        assert config.description == ""
        assert isinstance(config.hyperparameters, dict)
        assert len(config.hyperparameters) > 0  # Должны быть установлены по умолчанию

    def test_init_full(self) -> None:
        """Тест создания конфигурации со всеми параметрами."""
        hyperparams = {"learning_rate": 1e-3, "gamma": 0.95}

        config = Configuration(
            algorithm="A2C",
            environment="Pendulum-v1",
            hyperparameters=hyperparams,
            seed=123,
            training_steps=50_000,
            evaluation_frequency=5_000,
            experiment_name="test_experiment",
            description="Тестовый эксперимент",
        )

        assert config.algorithm == "A2C"
        assert config.environment == "Pendulum-v1"
        # Гиперпараметры должны быть дополнены значениями по умолчанию
        assert (
            config.hyperparameters["learning_rate"] == 1e-3
        )  # Пользовательское значение
        assert config.hyperparameters["gamma"] == 0.95  # Пользовательское значение
        assert "n_steps" in config.hyperparameters  # Должно быть добавлено из defaults
        assert config.seed == 123
        assert config.training_steps == 50_000
        assert config.evaluation_frequency == 5_000
        assert config.experiment_name == "test_experiment"
        assert config.description == "Тестовый эксперимент"

    def test_algorithm_normalization(self) -> None:
        """Тест нормализации названия алгоритма."""
        config = Configuration(algorithm="ppo", environment="LunarLander-v2")
        assert config.algorithm == "PPO"

        config = Configuration(algorithm="sac", environment="Pendulum-v1")
        assert config.algorithm == "SAC"

    @pytest.mark.parametrize("algorithm", ["PPO", "A2C", "SAC", "TD3"])
    def test_supported_algorithms(self, algorithm: str) -> None:
        """Тест поддерживаемых алгоритмов."""
        config = Configuration(algorithm=algorithm, environment="LunarLander-v2")
        assert config.algorithm == algorithm

    def test_unsupported_algorithm(self) -> None:
        """Тест неподдерживаемого алгоритма."""
        with pytest.raises(ValidationError, match="Неподдерживаемый алгоритм"):
            Configuration(algorithm="INVALID", environment="LunarLander-v2")

    @pytest.mark.parametrize(
        "environment",
        ["LunarLander-v2", "MountainCarContinuous-v0", "Acrobot-v1", "Pendulum-v1"],
    )
    def test_supported_environments(self, environment: str) -> None:
        """Тест поддерживаемых сред."""
        config = Configuration(algorithm="PPO", environment=environment)
        assert config.environment == environment

    def test_invalid_seed(self) -> None:
        """Тест невалидного seed."""
        with pytest.raises(ValidationError, match="seed должен быть в диапазоне"):
            Configuration(algorithm="PPO", environment="LunarLander-v2", seed=-1)

        with pytest.raises(ValidationError, match="seed должен быть в диапазоне"):
            Configuration(algorithm="PPO", environment="LunarLander-v2", seed=2**32)

    def test_invalid_training_steps(self) -> None:
        """Тест невалидного training_steps."""
        with pytest.raises(
            ValidationError, match="training_steps должен быть положительным"
        ):
            Configuration(
                algorithm="PPO", environment="LunarLander-v2", training_steps=0
            )

    def test_invalid_evaluation_frequency(self) -> None:
        """Тест невалидного evaluation_frequency."""
        with pytest.raises(
            ValidationError, match="evaluation_frequency должен быть положительным"
        ):
            Configuration(
                algorithm="PPO", environment="LunarLander-v2", evaluation_frequency=0
            )

        with pytest.raises(
            ValidationError, match="evaluation_frequency не может быть больше"
        ):
            Configuration(
                algorithm="PPO",
                environment="LunarLander-v2",
                training_steps=1000,
                evaluation_frequency=2000,
            )

    def test_empty_experiment_name(self) -> None:
        """Тест пустого experiment_name."""
        with pytest.raises(
            ValidationError, match="experiment_name не может быть пустым"
        ):
            Configuration(
                algorithm="PPO", environment="LunarLander-v2", experiment_name=""
            )

    def test_hyperparameter_validation(self) -> None:
        """Тест валидации гиперпараметров."""
        # Невалидный learning_rate
        with pytest.raises(
            ValidationError, match="learning_rate должен быть положительным"
        ):
            Configuration(
                algorithm="PPO",
                environment="LunarLander-v2",
                hyperparameters={"learning_rate": -0.1},
            )

        # Невалидный gamma
        with pytest.raises(ValidationError, match="gamma должен быть в диапазоне"):
            Configuration(
                algorithm="PPO",
                environment="LunarLander-v2",
                hyperparameters={"gamma": 1.5},
            )

    def test_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        config = Configuration(
            algorithm="PPO",
            environment="LunarLander-v2",
            seed=123,
            experiment_name="test",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["algorithm"] == "PPO"
        assert config_dict["environment"] == "LunarLander-v2"
        assert config_dict["seed"] == 123
        assert config_dict["experiment_name"] == "test"
        assert "hyperparameters" in config_dict

    def test_from_dict(self) -> None:
        """Тест создания из словаря."""
        data = {
            "algorithm": "A2C",
            "environment": "Pendulum-v1",
            "seed": 456,
            "training_steps": 75_000,
            "experiment_name": "from_dict_test",
        }

        config = Configuration.from_dict(data)

        assert config.algorithm == "A2C"
        assert config.environment == "Pendulum-v1"
        assert config.seed == 456
        assert config.training_steps == 75_000
        assert config.experiment_name == "from_dict_test"

    def test_from_dict_missing_required(self) -> None:
        """Тест создания из словаря с отсутствующими обязательными полями."""
        with pytest.raises(ValidationError, match="Отсутствует обязательное поле"):
            Configuration.from_dict({"algorithm": "PPO"})  # Нет environment

    def test_copy(self) -> None:
        """Тест создания копии конфигурации."""
        original = Configuration(algorithm="PPO", environment="LunarLander-v2")

        copy_config = original.copy()

        assert copy_config == original
        assert copy_config is not original
        assert copy_config.hyperparameters is not original.hyperparameters

        # Изменение копии не должно влиять на оригинал
        original_lr = original.hyperparameters["learning_rate"]
        copy_config.hyperparameters["learning_rate"] = 2e-3
        assert original.hyperparameters["learning_rate"] == original_lr

    def test_get_differences_identical(self) -> None:
        """Тест сравнения идентичных конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config2 = Configuration(algorithm="PPO", environment="LunarLander-v2")

        differences = config1.get_differences(config2)
        assert len(differences) == 0

    def test_get_differences_different(self) -> None:
        """Тест сравнения различных конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2", seed=42)
        config2 = Configuration(algorithm="A2C", environment="LunarLander-v2", seed=123)

        differences = config1.get_differences(config2)

        assert "algorithm" in differences
        assert differences["algorithm"]["self"] == "PPO"
        assert differences["algorithm"]["other"] == "A2C"

        assert "seed" in differences
        assert differences["seed"]["self"] == 42
        assert differences["seed"]["other"] == 123

    def test_merge(self) -> None:
        """Тест объединения конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2", seed=42)

        config2 = Configuration(algorithm="A2C", environment="Pendulum-v1", seed=123)

        # Изменяем гиперпараметры после создания для тестирования
        config1.hyperparameters["learning_rate"] = 1e-3
        config2.hyperparameters["learning_rate"] = 2e-3
        config2.hyperparameters["custom_param"] = "test"

        merged = config1.merge(config2)

        # config2 имеет приоритет для основных полей
        assert merged.algorithm == "A2C"
        assert merged.environment == "Pendulum-v1"
        assert merged.seed == 123

        # Гиперпараметры объединяются
        assert merged.hyperparameters["learning_rate"] == 2e-3  # Из config2
        assert merged.hyperparameters["custom_param"] == "test"  # Из config2

    def test_equality(self) -> None:
        """Тест проверки равенства конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config2 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config3 = Configuration(algorithm="A2C", environment="LunarLander-v2")

        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"

    def test_hash(self) -> None:
        """Тест хеширования конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config2 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config3 = Configuration(algorithm="A2C", environment="LunarLander-v2")

        # Одинаковые конфигурации должны иметь одинаковый хеш
        assert hash(config1) == hash(config2)

        # Разные конфигурации должны иметь разный хеш (с высокой вероятностью)
        assert hash(config1) != hash(config3)

        # Можно использовать в множествах
        config_set = {config1, config2, config3}
        assert len(config_set) == 2  # config1 и config2 одинаковые


class TestConfigurationSerialization:
    """Тесты сериализации конфигураций."""

    def test_save_load_yaml(self) -> None:
        """Тест сохранения и загрузки в формате YAML."""
        original = Configuration(
            algorithm="PPO",
            environment="LunarLander-v2",
            seed=123,
            experiment_name="yaml_test",
            description="Тест YAML сериализации",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Сохраняем
            original.save(filepath, format_type="yaml")
            assert filepath.exists()

            # Загружаем
            loaded = Configuration.load(filepath)

            assert loaded == original
            assert loaded.algorithm == original.algorithm
            assert loaded.environment == original.environment
            assert loaded.seed == original.seed
            assert loaded.experiment_name == original.experiment_name
            assert loaded.description == original.description

        finally:
            if filepath.exists():
                filepath.unlink()

    def test_save_load_json(self) -> None:
        """Тест сохранения и загрузки в формате JSON."""
        original = Configuration(
            algorithm="A2C",
            environment="Pendulum-v1",
            seed=456,
            experiment_name="json_test",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Сохраняем
            original.save(filepath, format_type="json")
            assert filepath.exists()

            # Загружаем
            loaded = Configuration.load(filepath)

            assert loaded == original

        finally:
            if filepath.exists():
                filepath.unlink()

    def test_load_nonexistent_file(self) -> None:
        """Тест загрузки несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            Configuration.load("nonexistent_file.yaml")

    def test_save_unsupported_format(self) -> None:
        """Тест сохранения в неподдерживаемом формате."""
        config = Configuration(algorithm="PPO", environment="LunarLander-v2")

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            config.save("test.txt", format_type="txt")

    def test_load_invalid_yaml(self) -> None:
        """Тест загрузки невалидного YAML файла."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            filepath = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="Ошибка парсинга файла"):
                Configuration.load(filepath)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_load_invalid_data(self) -> None:
        """Тест загрузки файла с невалидными данными."""
        invalid_data = {"algorithm": "INVALID_ALGO", "environment": "LunarLander-v2"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            filepath = Path(f.name)

        try:
            with pytest.raises(ValidationError):
                Configuration.load(filepath)
        finally:
            if filepath.exists():
                filepath.unlink()


class TestAlgorithmDefaults:
    """Тесты для получения настроек по умолчанию для алгоритмов."""

    @pytest.mark.parametrize("algorithm", ["PPO", "A2C", "SAC", "TD3"])
    def test_get_algorithm_defaults(self, algorithm: str) -> None:
        """Тест получения настроек по умолчанию для поддерживаемых алгоритмов."""
        defaults = Configuration.get_algorithm_defaults(algorithm)

        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        assert "learning_rate" in defaults
        assert "gamma" in defaults

        # Проверяем специфичные параметры
        if algorithm in ["PPO", "A2C"]:
            assert "n_steps" in defaults
        elif algorithm in ["SAC", "TD3"]:
            assert "buffer_size" in defaults

    def test_get_algorithm_defaults_invalid(self) -> None:
        """Тест получения настроек для неподдерживаемого алгоритма."""
        with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
            Configuration.get_algorithm_defaults("INVALID")

    def test_defaults_immutable(self) -> None:
        """Тест что изменение возвращенных настроек не влияет на оригинал."""
        defaults1 = Configuration.get_algorithm_defaults("PPO")
        defaults2 = Configuration.get_algorithm_defaults("PPO")

        # Изменяем первый словарь
        defaults1["learning_rate"] = 999.0

        # Второй словарь не должен измениться
        assert defaults2["learning_rate"] != 999.0


class TestConfigurationFactories:
    """Тесты для фабричных функций создания конфигураций."""

    def test_create_ppo_config(self) -> None:
        """Тест создания конфигурации PPO."""
        config = create_ppo_config(
            environment="LunarLander-v2", experiment_name="ppo_test"
        )

        assert config.algorithm == "PPO"
        assert config.environment == "LunarLander-v2"
        assert config.experiment_name == "ppo_test"
        assert "learning_rate" in config.hyperparameters
        assert "n_steps" in config.hyperparameters

    def test_create_a2c_config(self) -> None:
        """Тест создания конфигурации A2C."""
        config = create_a2c_config(
            environment="Acrobot-v1", experiment_name="a2c_test", seed=999
        )

        assert config.algorithm == "A2C"
        assert config.environment == "Acrobot-v1"
        assert config.experiment_name == "a2c_test"
        assert config.seed == 999

    def test_create_sac_config(self) -> None:
        """Тест создания конфигурации SAC."""
        config = create_sac_config(
            environment="Pendulum-v1", experiment_name="sac_test"
        )

        assert config.algorithm == "SAC"
        assert config.environment == "Pendulum-v1"
        assert config.experiment_name == "sac_test"
        assert "buffer_size" in config.hyperparameters

    def test_create_td3_config(self) -> None:
        """Тест создания конфигурации TD3."""
        config = create_td3_config(
            environment="Pendulum-v1", experiment_name="td3_test"
        )

        assert config.algorithm == "TD3"
        assert config.environment == "Pendulum-v1"
        assert config.experiment_name == "td3_test"
        assert "policy_delay" in config.hyperparameters


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    def test_validate_config_file_valid(self) -> None:
        """Тест валидации корректного файла конфигурации."""
        config = Configuration(algorithm="PPO", environment="LunarLander-v2")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            config.save(filepath)
            assert validate_config_file(filepath) is True
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_validate_config_file_invalid(self) -> None:
        """Тест валидации некорректного файла конфигурации."""
        invalid_data = {"algorithm": "INVALID", "environment": "LunarLander-v2"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            filepath = Path(f.name)

        try:
            assert validate_config_file(filepath) is False
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_compare_configs(self) -> None:
        """Тест сравнения конфигураций."""
        config1 = Configuration(
            algorithm="PPO", environment="LunarLander-v2", experiment_name="config1"
        )
        config2 = Configuration(
            algorithm="A2C", environment="LunarLander-v2", experiment_name="config2"
        )

        comparison = compare_configs(config1, config2)

        assert isinstance(comparison, dict)
        assert comparison["identical"] is False
        assert comparison["differences_count"] > 0
        assert "differences" in comparison
        assert comparison["summary"]["algorithm_same"] is False
        assert comparison["summary"]["environment_same"] is True

    def test_compare_configs_identical(self) -> None:
        """Тест сравнения идентичных конфигураций."""
        config1 = Configuration(algorithm="PPO", environment="LunarLander-v2")
        config2 = Configuration(algorithm="PPO", environment="LunarLander-v2")

        comparison = compare_configs(config1, config2)

        assert comparison["identical"] is True
        assert comparison["differences_count"] == 0
        assert comparison["summary"]["algorithm_same"] is True
        assert comparison["summary"]["environment_same"] is True
        assert comparison["summary"]["hyperparameters_same"] is True


class TestStringRepresentations:
    """Тесты для строковых представлений конфигурации."""

    def test_repr(self) -> None:
        """Тест __repr__ метода."""
        config = Configuration(
            algorithm="PPO", environment="LunarLander-v2", experiment_name="test_config"
        )

        repr_str = repr(config)
        assert "Configuration" in repr_str
        assert "PPO" in repr_str
        assert "LunarLander-v2" in repr_str
        assert "test_config" in repr_str

    def test_str(self) -> None:
        """Тест __str__ метода."""
        config = Configuration(
            algorithm="A2C",
            environment="Pendulum-v1",
            experiment_name="test_experiment",
            description="Тестовое описание",
            training_steps=50_000,
            evaluation_frequency=5_000,
            seed=123,
        )

        str_repr = str(config)
        assert "test_experiment" in str_repr
        assert "A2C" in str_repr
        assert "Pendulum-v1" in str_repr
        assert "50,000" in str_repr
        assert "5,000" in str_repr
        assert "123" in str_repr
        assert "Тестовое описание" in str_repr
