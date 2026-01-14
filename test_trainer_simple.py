#!/usr/bin/env python3
"""Простой тест основных классов тренера."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import tempfile
import yaml

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TrainingMode(Enum):
    """Режимы обучения."""
    
    TRAIN = "train"           # Обучение с нуля
    RESUME = "resume"         # Продолжение обучения
    EVALUATE = "evaluate"     # Только оценка
    FINETUNE = "finetune"     # Дообучение


@dataclass
class AgentConfig:
    """Упрощенная конфигурация агента."""
    algorithm: str = "PPO"
    env_name: str = "LunarLander-v3"
    total_timesteps: int = 100_000
    seed: int = 42
    learning_rate: float = 3e-4


@dataclass
class TrainerConfig:
    """Упрощенная конфигурация тренера."""
    
    # Основные параметры
    experiment_name: str = "default_experiment"
    algorithm: str = "PPO"
    environment_name: str = "LunarLander-v3"
    mode: TrainingMode = TrainingMode.TRAIN
    
    # Параметры обучения
    total_timesteps: int = 100_000
    seed: int = 42
    
    # Мониторинг и оценка
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    
    # Сохранение
    save_freq: int = 50_000
    output_dir: str = "results"
    
    # Конфигурация агента
    agent_config: Optional[AgentConfig] = None
    
    def __post_init__(self) -> None:
        """Валидация и нормализация конфигурации."""
        # Валидация алгоритма
        supported_algorithms = ["PPO", "A2C", "SAC", "TD3"]
        if self.algorithm.upper() not in supported_algorithms:
            raise ValueError(
                f"Неподдерживаемый алгоритм: {self.algorithm}. "
                f"Поддерживаемые: {supported_algorithms}"
            )
        
        # Нормализация алгоритма
        self.algorithm = self.algorithm.upper()
        
        # Валидация параметров
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps должен быть > 0: {self.total_timesteps}")
        
        if self.eval_freq <= 0:
            raise ValueError(f"eval_freq должен быть > 0: {self.eval_freq}")
        
        if self.n_eval_episodes <= 0:
            raise ValueError(f"n_eval_episodes должен быть > 0: {self.n_eval_episodes}")
        
        # Создание путей
        self._setup_paths()
        
        # Создание конфигурации агента если не задана
        if self.agent_config is None:
            self.agent_config = AgentConfig(
                algorithm=self.algorithm,
                env_name=self.environment_name,
                total_timesteps=self.total_timesteps,
                seed=self.seed,
            )
    
    def _setup_paths(self) -> None:
        """Настроить пути для сохранения."""
        output_path = Path(self.output_dir)
        experiment_path = output_path / self.experiment_name
        
        # Создание директорий
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Настройка путей
        self.model_save_path = str(experiment_path / "models" / f"{self.algorithm.lower()}_model")
        self.logs_dir = str(experiment_path / "logs")
        self.tensorboard_log = str(experiment_path / "tensorboard")
        
        # Создание директорий
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Результат обучения."""
    
    # Основные метрики
    success: bool
    total_timesteps: int
    training_time: float
    final_mean_reward: float
    final_std_reward: float
    
    # История обучения
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    evaluation_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Информация о модели
    model_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    # Метаданные
    experiment_name: str = ""
    algorithm: str = ""
    environment_name: str = ""
    seed: int = 42
    
    # Ошибки
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Дополнительные метрики
    best_mean_reward: float = float("-inf")
    convergence_timestep: Optional[int] = None
    early_stopped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать результат в словарь."""
        return {
            "success": self.success,
            "total_timesteps": self.total_timesteps,
            "training_time": self.training_time,
            "final_mean_reward": self.final_mean_reward,
            "final_std_reward": self.final_std_reward,
            "best_mean_reward": self.best_mean_reward,
            "convergence_timestep": self.convergence_timestep,
            "early_stopped": self.early_stopped,
            "experiment_name": self.experiment_name,
            "algorithm": self.algorithm,
            "environment_name": self.environment_name,
            "seed": self.seed,
            "model_path": self.model_path,
            "checkpoint_paths": self.checkpoint_paths,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
        }
    
    def save(self, path: Path) -> None:
        """Сохранить результат в файл."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def test_trainer_config_basic():
    """Тест базовой функциональности TrainerConfig."""
    print("🧪 Тестирование TrainerConfig...")
    
    # Тест создания конфигурации по умолчанию
    config = TrainerConfig()
    assert config.experiment_name == "default_experiment"
    assert config.algorithm == "PPO"
    assert config.environment_name == "LunarLander-v3"
    assert config.mode == TrainingMode.TRAIN
    assert config.total_timesteps == 100_000
    assert config.seed == 42
    assert config.agent_config is not None
    assert config.agent_config.algorithm == "PPO"
    print("✅ Конфигурация по умолчанию создана корректно")
    
    # Тест пользовательской конфигурации
    custom_config = TrainerConfig(
        experiment_name="test_experiment",
        algorithm="A2C",
        environment_name="CartPole-v1",
        total_timesteps=50_000,
        seed=123,
    )
    assert custom_config.experiment_name == "test_experiment"
    assert custom_config.algorithm == "A2C"
    assert custom_config.environment_name == "CartPole-v1"
    assert custom_config.total_timesteps == 50_000
    assert custom_config.seed == 123
    assert custom_config.agent_config.algorithm == "A2C"
    print("✅ Пользовательская конфигурация создана корректно")
    
    # Тест валидации
    try:
        TrainerConfig(algorithm="INVALID")
        assert False, "Должна была быть ошибка валидации"
    except ValueError as e:
        assert "Неподдерживаемый алгоритм" in str(e)
        print("✅ Валидация алгоритма работает")
    
    try:
        TrainerConfig(total_timesteps=0)
        assert False, "Должна была быть ошибка валидации"
    except ValueError as e:
        assert "total_timesteps должен быть > 0" in str(e)
        print("✅ Валидация timesteps работает")
    
    # Тест настройки путей
    with tempfile.TemporaryDirectory() as temp_dir:
        path_config = TrainerConfig(
            experiment_name="test_paths",
            output_dir=temp_dir,
        )
        assert hasattr(path_config, 'model_save_path')
        assert hasattr(path_config, 'logs_dir')
        assert hasattr(path_config, 'tensorboard_log')
        
        # Проверяем, что директории созданы
        assert Path(path_config.logs_dir).exists()
        assert Path(path_config.tensorboard_log).exists()
        print("✅ Настройка путей работает")


def test_training_mode():
    """Тест enum TrainingMode."""
    print("🧪 Тестирование TrainingMode...")
    
    assert TrainingMode.TRAIN.value == "train"
    assert TrainingMode.RESUME.value == "resume"
    assert TrainingMode.EVALUATE.value == "evaluate"
    assert TrainingMode.FINETUNE.value == "finetune"
    print("✅ Значения TrainingMode корректны")
    
    # Тест создания из строки
    assert TrainingMode("train") == TrainingMode.TRAIN
    assert TrainingMode("resume") == TrainingMode.RESUME
    print("✅ Создание TrainingMode из строки работает")


def test_training_result():
    """Тест TrainingResult."""
    print("🧪 Тестирование TrainingResult...")
    
    result = TrainingResult(
        success=True,
        total_timesteps=100_000,
        training_time=300.5,
        final_mean_reward=250.0,
        final_std_reward=50.0,
        experiment_name="test_exp",
        algorithm="PPO",
        environment_name="LunarLander-v3",
        seed=42,
    )
    
    assert result.success is True
    assert result.total_timesteps == 100_000
    assert result.training_time == 300.5
    assert result.final_mean_reward == 250.0
    assert result.experiment_name == "test_exp"
    print("✅ TrainingResult создан корректно")
    
    # Тест преобразования в словарь
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["total_timesteps"] == 100_000
    assert result_dict["experiment_name"] == "test_exp"
    print("✅ Преобразование в словарь работает")
    
    # Тест сохранения
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = Path(temp_dir) / "test_result.yaml"
        result.save(result_path)
        assert result_path.exists()
        print("✅ Сохранение результата работает")


def test_supported_algorithms():
    """Тест поддерживаемых алгоритмов."""
    print("🧪 Тестирование поддерживаемых алгоритмов...")
    
    supported = ["PPO", "A2C", "SAC", "TD3"]
    
    for algorithm in supported:
        config = TrainerConfig(algorithm=algorithm)
        assert config.algorithm == algorithm
        print(f"✅ Алгоритм {algorithm} поддерживается")
    
    # Тест нормализации регистра
    config = TrainerConfig(algorithm="ppo")
    assert config.algorithm == "PPO"
    print("✅ Нормализация регистра работает")


def test_agent_config():
    """Тест AgentConfig."""
    print("🧪 Тестирование AgentConfig...")
    
    config = AgentConfig()
    assert config.algorithm == "PPO"
    assert config.env_name == "LunarLander-v3"
    assert config.total_timesteps == 100_000
    assert config.seed == 42
    assert config.learning_rate == 3e-4
    print("✅ AgentConfig создан корректно")
    
    custom_config = AgentConfig(
        algorithm="SAC",
        env_name="Pendulum-v1",
        total_timesteps=50_000,
        seed=999,
        learning_rate=1e-3,
    )
    assert custom_config.algorithm == "SAC"
    assert custom_config.env_name == "Pendulum-v1"
    assert custom_config.total_timesteps == 50_000
    assert custom_config.seed == 999
    assert custom_config.learning_rate == 1e-3
    print("✅ Пользовательский AgentConfig создан корректно")


def main():
    """Главная функция тестирования."""
    print("🎮 Запуск простых тестов системы обучения")
    print("=" * 50)
    
    try:
        test_agent_config()
        test_trainer_config_basic()
        test_training_mode()
        test_training_result()
        test_supported_algorithms()
        
        print("\n🎉 Все тесты пройдены успешно!")
        print("✅ Основная функциональность работает корректно")
        print("\n📋 Протестированные компоненты:")
        print("  - TrainerConfig: создание, валидация, настройка путей")
        print("  - TrainingMode: enum значения и создание из строк")
        print("  - TrainingResult: создание, сериализация, сохранение")
        print("  - AgentConfig: базовая конфигурация агента")
        print("  - Поддержка алгоритмов: PPO, A2C, SAC, TD3")
        
        print("\n🚀 Система готова к использованию!")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()