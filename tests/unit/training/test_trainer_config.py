"""Тесты для конфигурации тренера без внешних зависимостей."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Мокаем внешние зависимости перед импортом
with patch.dict(
    "sys.modules",
    {
        "gymnasium": MagicMock(),
        "stable_baselines3": MagicMock(),
        "stable_baselines3.common.callbacks": MagicMock(),
        "typer": MagicMock(),
        "rich": MagicMock(),
        "rich.console": MagicMock(),
        "rich.table": MagicMock(),
        "rich.progress": MagicMock(),
        "rich.panel": MagicMock(),
    },
):
    from src.training.trainer import TrainerConfig, TrainingMode


class TestTrainerConfig:
    """Тесты для TrainerConfig."""

    def test_default_config(self):
        """Тест создания конфигурации по умолчанию."""
        config = TrainerConfig()

        assert config.experiment_name == "default_experiment"
        assert config.algorithm == "PPO"
        assert config.environment_name == "LunarLander-v3"
        assert config.mode == TrainingMode.TRAIN
        assert config.total_timesteps == 100_000
        assert config.seed == 42
        assert config.agent_config is not None
        assert config.agent_config.algorithm == "PPO"

    def test_custom_config(self):
        """Тест создания пользовательской конфигурации."""
        config = TrainerConfig(
            experiment_name="test_experiment",
            algorithm="A2C",
            environment_name="CartPole-v1",
            total_timesteps=50_000,
            seed=123,
        )

        assert config.experiment_name == "test_experiment"
        assert config.algorithm == "A2C"
        assert config.environment_name == "CartPole-v1"
        assert config.total_timesteps == 50_000
        assert config.seed == 123
        assert config.agent_config.algorithm == "A2C"

    def test_invalid_algorithm(self):
        """Тест валидации неподдерживаемого алгоритма."""
        with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
            TrainerConfig(algorithm="INVALID")

    def test_invalid_timesteps(self):
        """Тест валидации некорректного количества шагов."""
        with pytest.raises(ValueError, match="total_timesteps должен быть > 0"):
            TrainerConfig(total_timesteps=0)

        with pytest.raises(ValueError, match="total_timesteps должен быть > 0"):
            TrainerConfig(total_timesteps=-1000)

    def test_invalid_eval_freq(self):
        """Тест валидации некорректной частоты оценки."""
        with pytest.raises(ValueError, match="eval_freq должен быть > 0"):
            TrainerConfig(eval_freq=0)

    def test_path_setup(self):
        """Тест настройки путей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainerConfig(
                experiment_name="test_exp",
                output_dir=temp_dir,
            )

            # Проверяем, что пути созданы
            assert config.model_save_path is not None
            assert config.logs_dir is not None
            assert config.tensorboard_log is not None

            # Проверяем, что директории существуют
            assert Path(config.logs_dir).exists()
            assert Path(config.tensorboard_log).exists()
            assert Path(config.model_save_path).parent.exists()


class TestTrainingMode:
    """Тесты для TrainingMode enum."""

    def test_training_mode_values(self):
        """Тест значений режимов обучения."""
        assert TrainingMode.TRAIN.value == "train"
        assert TrainingMode.RESUME.value == "resume"
        assert TrainingMode.EVALUATE.value == "evaluate"
        assert TrainingMode.FINETUNE.value == "finetune"

    def test_training_mode_from_string(self):
        """Тест создания режима из строки."""
        assert TrainingMode("train") == TrainingMode.TRAIN
        assert TrainingMode("resume") == TrainingMode.RESUME
        assert TrainingMode("evaluate") == TrainingMode.EVALUATE
        assert TrainingMode("finetune") == TrainingMode.FINETUNE


@pytest.mark.parametrize("algorithm", ["PPO", "A2C", "SAC", "TD3"])
def test_supported_algorithms(algorithm):
    """Тест поддерживаемых алгоритмов."""
    config = TrainerConfig(algorithm=algorithm)
    assert config.algorithm == algorithm


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_training_modes(mode):
    """Тест всех режимов обучения."""
    config = TrainerConfig(mode=mode)
    assert config.mode == mode


def test_config_normalization():
    """Тест нормализации конфигурации."""
    # Тест нормализации алгоритма в верхний регистр
    config = TrainerConfig(algorithm="ppo")
    assert config.algorithm == "PPO"

    config = TrainerConfig(algorithm="a2c")
    assert config.algorithm == "A2C"


def test_config_validation_edge_cases():
    """Тест граничных случаев валидации."""
    # Минимальные валидные значения
    config = TrainerConfig(
        total_timesteps=1,
        eval_freq=1,
        n_eval_episodes=1,
    )
    assert config.total_timesteps == 1
    assert config.eval_freq == 1
    assert config.n_eval_episodes == 1

    # Проверка больших значений
    config = TrainerConfig(
        total_timesteps=10_000_000,
        eval_freq=100_000,
    )
    assert config.total_timesteps == 10_000_000
    assert config.eval_freq == 100_000
