"""Комплексный класс Configuration для управления конфигурацией RL экспериментов.

Этот модуль предоставляет класс Configuration для создания, валидации, сравнения
и сериализации конфигураций экспериментов с поддержкой различных алгоритмов RL,
гиперпараметров и форматов сохранения.
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Базовое исключение для ошибок конфигурации."""
    pass


class ValidationError(ConfigurationError):
    """Исключение для ошибок валидации конфигурации."""
    pass


@dataclass
class Configuration:
    """Комплексный класс для управления конфигурацией RL экспериментов.
    
    Поддерживает создание, валидацию, сравнение и сериализацию конфигураций
    с различными алгоритмами RL, гиперпараметрами и настройками обучения.
    
    Attributes:
        algorithm: Выбранный алгоритм RL (PPO, A2C, SAC, TD3)
        environment: Название среды для обучения
        hyperparameters: Словарь с гиперпараметрами алгоритма
        seed: Случайное зерно для воспроизводимости
        training_steps: Общее количество шагов обучения
        evaluation_frequency: Частота оценки (каждые N шагов)
        experiment_name: Человекочитаемое название эксперимента
        description: Подробное описание эксперимента
    """
    
    algorithm: str
    environment: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    training_steps: int = 100_000
    evaluation_frequency: int = 10_000
    experiment_name: str = "default_experiment"
    description: str = ""
    
    # Поддерживаемые алгоритмы и среды
    _SUPPORTED_ALGORITHMS: Set[str] = field(
        default_factory=lambda: {"PPO", "A2C", "SAC", "TD3"},
        init=False,
        repr=False
    )
    
    _SUPPORTED_ENVIRONMENTS: Set[str] = field(
        default_factory=lambda: {
            "LunarLander-v2",
            "MountainCarContinuous-v0", 
            "Acrobot-v1",
            "Pendulum-v1"
        },
        init=False,
        repr=False
    )
    
    def __post_init__(self) -> None:
        """Инициализация после создания объекта."""
        # Нормализуем название алгоритма
        self.algorithm = self.algorithm.upper()
        
        # Устанавливаем гиперпараметры по умолчанию если не заданы
        if not self.hyperparameters:
            try:
                self.hyperparameters = self.get_algorithm_defaults(self.algorithm)
            except ValueError:
                # Если алгоритм неподдерживаемый, оставляем пустой словарь
                # Ошибка будет поймана в validate()
                self.hyperparameters = {}
        else:
            # Если гиперпараметры заданы частично, дополняем их значениями по умолчанию
            try:
                defaults = self.get_algorithm_defaults(self.algorithm)
                # Объединяем: пользовательские параметры имеют приоритет
                merged_params = defaults.copy()
                merged_params.update(self.hyperparameters)
                self.hyperparameters = merged_params
            except ValueError:
                # Если алгоритм неподдерживаемый, оставляем как есть
                pass
        
        # Валидируем конфигурацию
        self.validate()
        
        logger.debug(
            f"Создана конфигурация: {self.experiment_name} "
            f"({self.algorithm} на {self.environment})"
        )
    
    def validate(self) -> None:
        """Комплексная валидация всех параметров конфигурации.
        
        Raises:
            ValidationError: Если найдены ошибки валидации
        """
        errors = []
        
        # Валидация алгоритма
        if not isinstance(self.algorithm, str):
            errors.append("algorithm должен быть строкой")
        elif self.algorithm not in self._SUPPORTED_ALGORITHMS:
            errors.append(
                f"Неподдерживаемый алгоритм: {self.algorithm}. "
                f"Поддерживаются: {', '.join(sorted(self._SUPPORTED_ALGORITHMS))}"
            )
        
        # Валидация среды
        if not isinstance(self.environment, str):
            errors.append("environment должен быть строкой")
        elif self.environment not in self._SUPPORTED_ENVIRONMENTS:
            logger.warning(
                f"Среда {self.environment} может быть не поддержана. "
                f"Рекомендуемые: {', '.join(sorted(self._SUPPORTED_ENVIRONMENTS))}"
            )
        
        # Валидация seed
        if not isinstance(self.seed, int):
            errors.append("seed должен быть целым числом")
        elif not (0 <= self.seed <= 2**32 - 1):
            errors.append("seed должен быть в диапазоне [0, 2^32-1]")
        
        # Валидация training_steps
        if not isinstance(self.training_steps, int):
            errors.append("training_steps должен быть целым числом")
        elif self.training_steps <= 0:
            errors.append("training_steps должен быть положительным")
        
        # Валидация evaluation_frequency
        if not isinstance(self.evaluation_frequency, int):
            errors.append("evaluation_frequency должен быть целым числом")
        elif self.evaluation_frequency <= 0:
            errors.append("evaluation_frequency должен быть положительным")
        elif self.evaluation_frequency > self.training_steps:
            errors.append(
                "evaluation_frequency не может быть больше training_steps"
            )
        
        # Валидация experiment_name
        if not isinstance(self.experiment_name, str):
            errors.append("experiment_name должен быть строкой")
        elif not self.experiment_name.strip():
            errors.append("experiment_name не может быть пустым")
        
        # Валидация description
        if not isinstance(self.description, str):
            errors.append("description должен быть строкой")
        
        # Валидация гиперпараметров
        if not isinstance(self.hyperparameters, dict):
            errors.append("hyperparameters должен быть словарем")
        else:
            self._validate_hyperparameters(errors)
        
        if errors:
            raise ValidationError(f"Ошибки валидации: {'; '.join(errors)}")
    
    def _validate_hyperparameters(self, errors: List[str]) -> None:
        """Валидация гиперпараметров для конкретного алгоритма.
        
        Args:
            errors: Список для добавления ошибок валидации
        """
        # Проверяем только если алгоритм поддерживается
        if self.algorithm not in self._SUPPORTED_ALGORITHMS:
            return
        
        required_params = self._get_required_hyperparameters(self.algorithm)
        
        # Проверяем наличие обязательных параметров только если гиперпараметры не пустые
        # и не были установлены автоматически
        if self.hyperparameters:
            for param in required_params:
                if param not in self.hyperparameters:
                    errors.append(f"Отсутствует обязательный гиперпараметр: {param}")
        
        # Валидация конкретных параметров
        hp = self.hyperparameters
        
        # learning_rate
        if "learning_rate" in hp:
            lr = hp["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("learning_rate должен быть положительным числом")
        
        # gamma
        if "gamma" in hp:
            gamma = hp["gamma"]
            if not isinstance(gamma, (int, float)) or not (0 <= gamma <= 1):
                errors.append("gamma должен быть в диапазоне [0, 1]")
        
        # batch_size
        if "batch_size" in hp:
            bs = hp["batch_size"]
            if not isinstance(bs, int) or bs <= 0:
                errors.append("batch_size должен быть положительным целым числом")
        
        # n_steps (для on-policy алгоритмов)
        if "n_steps" in hp:
            ns = hp["n_steps"]
            if not isinstance(ns, int) or ns <= 0:
                errors.append("n_steps должен быть положительным целым числом")
        
        # buffer_size (для off-policy алгоритмов)
        if "buffer_size" in hp:
            bs = hp["buffer_size"]
            if not isinstance(bs, int) or bs <= 0:
                errors.append("buffer_size должен быть положительным целым числом")
    
    def _get_required_hyperparameters(self, algorithm: str) -> Set[str]:
        """Получить список обязательных гиперпараметров для алгоритма.
        
        Args:
            algorithm: Название алгоритма
            
        Returns:
            Множество обязательных параметров
        """
        common_params = {"learning_rate", "gamma"}
        
        algorithm_specific = {
            "PPO": {"n_steps", "batch_size", "n_epochs", "clip_range"},
            "A2C": {"n_steps"},
            "SAC": {"buffer_size", "tau"},
            "TD3": {"buffer_size", "tau", "policy_delay"}
        }
        
        return common_params | algorithm_specific.get(algorithm, set())
    
    def get_differences(self, other: "Configuration") -> Dict[str, Dict[str, Any]]:
        """Сравнить с другой конфигурацией и найти различия.
        
        Args:
            other: Другая конфигурация для сравнения
            
        Returns:
            Словарь с различиями в формате:
            {
                "field_name": {
                    "self": value_in_self,
                    "other": value_in_other
                }
            }
        """
        if not isinstance(other, Configuration):
            raise TypeError("Сравнение возможно только с другим объектом Configuration")
        
        differences = {}
        
        # Сравниваем основные поля
        fields_to_compare = [
            "algorithm", "environment", "seed", "training_steps",
            "evaluation_frequency", "experiment_name", "description"
        ]
        
        for field_name in fields_to_compare:
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            
            if self_value != other_value:
                differences[field_name] = {
                    "self": self_value,
                    "other": other_value
                }
        
        # Сравниваем гиперпараметры
        hp_diff = self._compare_hyperparameters(other.hyperparameters)
        if hp_diff:
            differences["hyperparameters"] = hp_diff
        
        return differences
    
    def _compare_hyperparameters(
        self, other_hp: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Сравнить гиперпараметры с другой конфигурацией.
        
        Args:
            other_hp: Гиперпараметры другой конфигурации
            
        Returns:
            Словарь с различиями в гиперпараметрах
        """
        differences = {}
        
        # Все уникальные ключи
        all_keys = set(self.hyperparameters.keys()) | set(other_hp.keys())
        
        for key in all_keys:
            self_value = self.hyperparameters.get(key, "<отсутствует>")
            other_value = other_hp.get(key, "<отсутствует>")
            
            if self_value != other_value:
                differences[key] = {
                    "self": self_value,
                    "other": other_value
                }
        
        return differences
    
    def merge(self, other: "Configuration") -> "Configuration":
        """Объединить с другой конфигурацией (другая имеет приоритет).
        
        Args:
            other: Конфигурация для объединения
            
        Returns:
            Новая конфигурация с объединенными параметрами
        """
        if not isinstance(other, Configuration):
            raise TypeError("Объединение возможно только с другим объектом Configuration")
        
        # Создаем копию текущей конфигурации
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Объединяем словари (other имеет приоритет)
        for key, value in other_dict.items():
            if key == "hyperparameters":
                # Для гиперпараметров делаем глубокое объединение
                merged_dict[key].update(value)
            else:
                merged_dict[key] = value
        
        return Configuration.from_dict(merged_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь.
        
        Returns:
            Словарь с данными конфигурации
        """
        return {
            "algorithm": self.algorithm,
            "environment": self.environment,
            "hyperparameters": copy.deepcopy(self.hyperparameters),
            "seed": self.seed,
            "training_steps": self.training_steps,
            "evaluation_frequency": self.evaluation_frequency,
            "experiment_name": self.experiment_name,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Configuration":
        """Создать конфигурацию из словаря.
        
        Args:
            data: Словарь с данными конфигурации
            
        Returns:
            Объект конфигурации
            
        Raises:
            ValidationError: Если данные невалидны
        """
        try:
            # Извлекаем обязательные поля
            algorithm = data["algorithm"]
            environment = data["environment"]
            
            # Извлекаем опциональные поля с значениями по умолчанию
            hyperparameters = data.get("hyperparameters", {})
            seed = data.get("seed", 42)
            training_steps = data.get("training_steps", 100_000)
            evaluation_frequency = data.get("evaluation_frequency", 10_000)
            experiment_name = data.get("experiment_name", "default_experiment")
            description = data.get("description", "")
            
            return cls(
                algorithm=algorithm,
                environment=environment,
                hyperparameters=hyperparameters,
                seed=seed,
                training_steps=training_steps,
                evaluation_frequency=evaluation_frequency,
                experiment_name=experiment_name,
                description=description
            )
            
        except KeyError as e:
            raise ValidationError(f"Отсутствует обязательное поле: {e}")
        except Exception as e:
            raise ValidationError(f"Ошибка создания конфигурации из словаря: {e}")
    
    def save(self, filepath: Union[str, Path], format_type: str = "yaml") -> None:
        """Сохранить конфигурацию в файл.
        
        Args:
            filepath: Путь к файлу для сохранения
            format_type: Формат файла ("yaml" или "json")
            
        Raises:
            ValueError: Если формат не поддерживается
            IOError: Если ошибка записи файла
        """
        if format_type not in ["yaml", "json"]:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        try:
            if format_type == "yaml":
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(
                        data, f, 
                        default_flow_style=False, 
                        allow_unicode=True,
                        sort_keys=False
                    )
            else:  # json
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Конфигурация сохранена в {filepath}")
            
        except Exception as e:
            error_msg = f"Ошибка при сохранении конфигурации: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Configuration":
        """Загрузить конфигурацию из файла.
        
        Args:
            filepath: Путь к файлу конфигурации
            
        Returns:
            Объект конфигурации
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValidationError: Если файл содержит невалидные данные
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif filepath.suffix.lower() == ".json":
                    data = json.load(f)
                else:
                    # Пытаемся определить формат по содержимому
                    content = f.read()
                    f.seek(0)
                    
                    if content.strip().startswith("{"):
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                raise ValidationError("Файл должен содержать объект (словарь)")
            
            config = cls.from_dict(data)
            logger.info(f"Конфигурация загружена из {filepath}")
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValidationError(f"Ошибка парсинга файла {filepath}: {e}")
        except Exception as e:
            raise ValidationError(f"Ошибка загрузки конфигурации: {e}")
    
    def copy(self) -> "Configuration":
        """Создать глубокую копию конфигурации.
        
        Returns:
            Копия конфигурации
        """
        return Configuration.from_dict(self.to_dict())
    
    @staticmethod
    def get_algorithm_defaults(algorithm: str) -> Dict[str, Any]:
        """Получить гиперпараметры по умолчанию для алгоритма.
        
        Args:
            algorithm: Название алгоритма
            
        Returns:
            Словарь с гиперпараметрами по умолчанию
            
        Raises:
            ValueError: Если алгоритм не поддерживается
        """
        algorithm = algorithm.upper()
        
        defaults = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": -1,
                "target_kl": None
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "vf_coef": 0.25,
                "max_grad_norm": 0.5,
                "use_rms_prop": True,
                "rms_prop_eps": 1e-5,
                "use_sde": False,
                "sde_sample_freq": -1
            },
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "learning_starts": 100,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "target_entropy": "auto",
                "use_sde": False,
                "sde_sample_freq": -1
            },
            "TD3": {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "learning_starts": 100,
                "batch_size": 100,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": -1,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "action_noise": None
            }
        }
        
        if algorithm not in defaults:
            raise ValueError(
                f"Неподдерживаемый алгоритм: {algorithm}. "
                f"Поддерживаются: {', '.join(defaults.keys())}"
            )
        
        return copy.deepcopy(defaults[algorithm])
    
    def __repr__(self) -> str:
        """Строковое представление конфигурации."""
        return (
            f"Configuration(algorithm={self.algorithm}, "
            f"environment={self.environment}, "
            f"experiment_name='{self.experiment_name}')"
        )
    
    def __str__(self) -> str:
        """Человекочитаемое представление конфигурации."""
        return (
            f"Конфигурация эксперимента: {self.experiment_name}\n"
            f"Алгоритм: {self.algorithm}\n"
            f"Среда: {self.environment}\n"
            f"Шагов обучения: {self.training_steps:,}\n"
            f"Частота оценки: {self.evaluation_frequency:,}\n"
            f"Зерно: {self.seed}\n"
            f"Описание: {self.description or 'Не указано'}"
        )
    
    def __eq__(self, other: object) -> bool:
        """Проверка равенства конфигураций."""
        if not isinstance(other, Configuration):
            return False
        
        return self.to_dict() == other.to_dict()
    
    def __hash__(self) -> int:
        """Хеш конфигурации для использования в множествах и словарях."""
        # Создаем хеш на основе основных параметров
        hashable_data = (
            self.algorithm,
            self.environment,
            self.seed,
            self.training_steps,
            self.evaluation_frequency,
            self.experiment_name,
            # Для гиперпараметров создаем хеш от отсортированных элементов
            tuple(sorted(self.hyperparameters.items()))
        )
        return hash(hashable_data)


def create_ppo_config(
    environment: str = "LunarLander-v2",
    experiment_name: str = "ppo_experiment",
    **kwargs: Any
) -> Configuration:
    """Создать конфигурацию PPO с настройками по умолчанию.
    
    Args:
        environment: Название среды
        experiment_name: Название эксперимента
        **kwargs: Дополнительные параметры для переопределения
        
    Returns:
        Конфигурация PPO
    """
    config_data = {
        "algorithm": "PPO",
        "environment": environment,
        "experiment_name": experiment_name,
        "hyperparameters": Configuration.get_algorithm_defaults("PPO")
    }
    
    # Переопределяем параметры если переданы
    config_data.update(kwargs)
    
    return Configuration.from_dict(config_data)


def create_a2c_config(
    environment: str = "LunarLander-v2",
    experiment_name: str = "a2c_experiment",
    **kwargs: Any
) -> Configuration:
    """Создать конфигурацию A2C с настройками по умолчанию.
    
    Args:
        environment: Название среды
        experiment_name: Название эксперимента
        **kwargs: Дополнительные параметры для переопределения
        
    Returns:
        Конфигурация A2C
    """
    config_data = {
        "algorithm": "A2C",
        "environment": environment,
        "experiment_name": experiment_name,
        "hyperparameters": Configuration.get_algorithm_defaults("A2C")
    }
    
    config_data.update(kwargs)
    return Configuration.from_dict(config_data)


def create_sac_config(
    environment: str = "Pendulum-v1",
    experiment_name: str = "sac_experiment",
    **kwargs: Any
) -> Configuration:
    """Создать конфигурацию SAC с настройками по умолчанию.
    
    Args:
        environment: Название среды
        experiment_name: Название эксперимента
        **kwargs: Дополнительные параметры для переопределения
        
    Returns:
        Конфигурация SAC
    """
    config_data = {
        "algorithm": "SAC",
        "environment": environment,
        "experiment_name": experiment_name,
        "hyperparameters": Configuration.get_algorithm_defaults("SAC")
    }
    
    config_data.update(kwargs)
    return Configuration.from_dict(config_data)


def create_td3_config(
    environment: str = "Pendulum-v1",
    experiment_name: str = "td3_experiment",
    **kwargs: Any
) -> Configuration:
    """Создать конфигурацию TD3 с настройками по умолчанию.
    
    Args:
        environment: Название среды
        experiment_name: Название эксперимента
        **kwargs: Дополнительные параметры для переопределения
        
    Returns:
        Конфигурация TD3
    """
    config_data = {
        "algorithm": "TD3",
        "environment": environment,
        "experiment_name": experiment_name,
        "hyperparameters": Configuration.get_algorithm_defaults("TD3")
    }
    
    config_data.update(kwargs)
    return Configuration.from_dict(config_data)


def validate_config_file(filepath: Union[str, Path]) -> bool:
    """Проверить валидность файла конфигурации.
    
    Args:
        filepath: Путь к файлу конфигурации
        
    Returns:
        True если конфигурация валидна, False иначе
    """
    try:
        Configuration.load(filepath)
        logger.info(f"Конфигурация {filepath} валидна")
        return True
    except Exception as e:
        logger.error(f"Конфигурация {filepath} невалидна: {e}")
        return False


def compare_configs(
    config1: Configuration, 
    config2: Configuration
) -> Dict[str, Any]:
    """Сравнить две конфигурации и вернуть подробный отчет.
    
    Args:
        config1: Первая конфигурация
        config2: Вторая конфигурация
        
    Returns:
        Словарь с результатами сравнения
    """
    differences = config1.get_differences(config2)
    
    return {
        "identical": len(differences) == 0,
        "differences_count": len(differences),
        "differences": differences,
        "config1_name": config1.experiment_name,
        "config2_name": config2.experiment_name,
        "summary": {
            "algorithm_same": config1.algorithm == config2.algorithm,
            "environment_same": config1.environment == config2.environment,
            "hyperparameters_same": config1.hyperparameters == config2.hyperparameters
        }
    }