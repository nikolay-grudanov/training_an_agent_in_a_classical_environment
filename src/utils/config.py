"""Утилиты для управления конфигурацией в RL экспериментах.

Этот модуль предоставляет интеграцию с Hydra для загрузки конфигураций,
валидацию параметров, подстановку переменных окружения, слияние конфигураций
и типобезопасный доступ к настройкам. Включает расширенные возможности
для обеспечения консистентности сидов и воспроизводимости экспериментов.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """Конфигурация алгоритма RL."""

    name: str
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    device: str = "auto"
    verbose: int = 1
    seed: Optional[int] = None
    policy_kwargs: Optional[Dict[str, Any]] = None
    tensorboard_log: Optional[str] = None


@dataclass
class EnvironmentConfig:
    """Конфигурация среды RL."""

    name: str
    render_mode: Optional[str] = None
    max_episode_steps: Optional[int] = None
    reward_threshold: Optional[float] = None
    wrappers: List[str] = field(default_factory=list)
    wrapper_kwargs: Dict[str, Any] = field(default_factory=dict)
    record_video: bool = False
    video_folder: Optional[str] = None
    video_length: int = 0


@dataclass
class TrainingConfig:
    """Конфигурация обучения."""

    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    eval_log_path: Optional[str] = None
    save_freq: int = 20000
    save_path: Optional[str] = None
    save_replay_buffer: bool = False
    monitor_training: bool = True
    log_interval: int = 1000
    early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.01


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента."""

    name: str
    description: str = ""
    hypothesis: str = ""
    runs_per_algorithm: int = 1
    seeds: List[int] = field(default_factory=lambda: [42])
    algorithms: List[str] = field(default_factory=lambda: ["ppo"])
    metrics: List[str] = field(
        default_factory=lambda: ["mean_reward", "episode_length"]
    )
    save_results: bool = True
    generate_plots: bool = True
    create_videos: bool = True


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""

    level: str = "INFO"
    log_to_file: bool = True
    log_dir: Optional[str] = None
    json_format: bool = False
    console_output: bool = True
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class ReproducibilityConfig:
    """Конфигурация воспроизводимости."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    use_cuda: bool = False
    enforce_seed_consistency: bool = True
    validate_determinism: bool = True
    warn_on_seed_conflicts: bool = True
    auto_propagate_seeds: bool = True


@dataclass
class RLConfig:
    """Основная конфигурация RL системы."""

    experiment_name: str = "default_experiment"
    output_dir: str = "results"
    seed: int = 42

    algorithm: AlgorithmConfig = field(
        default_factory=lambda: AlgorithmConfig(name="PPO")
    )
    environment: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(name="LunarLander-v3")
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: Optional[ExperimentConfig] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    def __post_init__(self) -> None:
        """Пост-инициализация для автоматической синхронизации сидов."""
        if self.reproducibility.auto_propagate_seeds:
            self.enforce_seed_consistency()

    def enforce_seed_consistency(self) -> None:
        """Принудительная синхронизация сидов между всеми компонентами.

        Устанавливает основной seed во всех подкомпонентах конфигурации
        для обеспечения полной воспроизводимости.
        """
        logger.info(f"Принудительная синхронизация сидов с основным seed: {self.seed}")

        # Синхронизируем seed в конфигурации воспроизводимости
        if self.reproducibility.seed != self.seed:
            logger.debug(
                f"Обновление seed в reproducibility: {self.reproducibility.seed} -> {self.seed}"
            )
            self.reproducibility.seed = self.seed

        # Синхронизируем seed в алгоритме
        if self.algorithm.seed != self.seed:
            logger.debug(
                f"Обновление seed в algorithm: {self.algorithm.seed} -> {self.seed}"
            )
            self.algorithm.seed = self.seed

        # Синхронизируем seeds в эксперименте
        if self.experiment and self.experiment.seeds:
            if self.seed not in self.experiment.seeds:
                logger.debug(
                    f"Добавление основного seed в experiment.seeds: {self.seed}"
                )
                if self.seed not in self.experiment.seeds:
                    self.experiment.seeds.insert(0, self.seed)

        logger.debug("Синхронизация сидов завершена")

    def validate_reproducibility(self) -> Tuple[bool, List[str]]:
        """Проверка настроек воспроизводимости.

        Returns:
            Tuple[bool, List[str]]: (валидность, список предупреждений)
        """
        warnings_list = []
        is_valid = True

        logger.info("Валидация настроек воспроизводимости")

        # Проверка консистентности сидов
        if self.reproducibility.enforce_seed_consistency:
            seed_conflicts = self._check_seed_conflicts()
            if seed_conflicts:
                warnings_list.extend(seed_conflicts)
                if self.reproducibility.warn_on_seed_conflicts:
                    for warning_msg in seed_conflicts:
                        logger.warning(warning_msg)

        # Проверка детерминистических настроек
        if self.reproducibility.validate_determinism:
            determinism_issues = self._check_determinism_settings()
            if determinism_issues:
                warnings_list.extend(determinism_issues)
                for issue in determinism_issues:
                    logger.warning(issue)

        # Проверка CUDA настроек
        cuda_warnings = self._check_cuda_settings()
        if cuda_warnings:
            warnings_list.extend(cuda_warnings)
            for warning_msg in cuda_warnings:
                logger.warning(warning_msg)

        # Проверка алгоритм-специфичных настроек
        algo_warnings = self._check_algorithm_reproducibility()
        if algo_warnings:
            warnings_list.extend(algo_warnings)
            for warning_msg in algo_warnings:
                logger.warning(warning_msg)

        if warnings_list:
            logger.warning(
                f"Обнаружено {len(warnings_list)} предупреждений о воспроизводимости"
            )
        else:
            logger.info("Все настройки воспроизводимости корректны")

        return is_valid, warnings_list

    def apply_seeds(self) -> None:
        """Применение сидов ко всем компонентам системы.

        Использует модуль seeding для установки глобальных сидов
        и настройки детерминистического поведения.
        """
        try:
            from src.utils.seeding import set_seed, verify_reproducibility

            logger.info(f"Применение глобального seed: {self.seed}")

            # Устанавливаем глобальный seed
            set_seed(self.seed)

            # Проверяем воспроизводимость если требуется
            if self.reproducibility.validate_determinism:
                logger.info("Проверка воспроизводимости...")
                is_reproducible = verify_reproducibility(self.seed, test_operations=50)
                if not is_reproducible:
                    logger.error("Воспроизводимость не обеспечена!")
                    raise RuntimeError("Не удалось обеспечить воспроизводимость")
                else:
                    logger.info("Воспроизводимость подтверждена")

            # Настраиваем PyTorch детерминизм
            if self.reproducibility.deterministic:
                import torch

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = self.reproducibility.benchmark
                logger.debug("Настроен детерминистический режим PyTorch")

            logger.info("Применение сидов завершено успешно")

        except ImportError as e:
            logger.error(f"Не удалось импортировать модуль seeding: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при применении сидов: {e}")
            raise

    def get_reproducibility_report(self) -> Dict[str, Any]:
        """Получить отчет о настройках воспроизводимости.

        Returns:
            Словарь с детальной информацией о настройках воспроизводимости
        """
        logger.info("Генерация отчета о воспроизводимости")

        # Проверяем валидность настроек
        is_valid, warnings_list = self.validate_reproducibility()

        # Собираем информацию о сидах
        seeds_info = {
            "main_seed": self.seed,
            "reproducibility_seed": self.reproducibility.seed,
            "algorithm_seed": self.algorithm.seed,
            "experiment_seeds": self.experiment.seeds if self.experiment else None,
            "seeds_consistent": len(self._check_seed_conflicts()) == 0,
        }

        # Информация о детерминистических настройках
        determinism_info = {
            "deterministic_mode": self.reproducibility.deterministic,
            "benchmark_mode": self.reproducibility.benchmark,
            "cuda_enabled": self.reproducibility.use_cuda,
            "enforce_consistency": self.reproducibility.enforce_seed_consistency,
            "validate_determinism": self.reproducibility.validate_determinism,
        }

        # Информация об алгоритме
        algorithm_info = {
            "algorithm_name": self.algorithm.name,
            "use_sde": self.algorithm.use_sde,
            "device": self.algorithm.device,
            "policy_kwargs": self.algorithm.policy_kwargs,
        }

        # Системная информация
        system_info = self._get_system_reproducibility_info()

        report = {
            "timestamp": self._get_timestamp(),
            "experiment_name": self.experiment_name,
            "is_valid": is_valid,
            "warnings": warnings_list,
            "seeds": seeds_info,
            "determinism": determinism_info,
            "algorithm": algorithm_info,
            "system": system_info,
            "recommendations": self._get_reproducibility_recommendations(warnings_list),
        }

        logger.info(
            f"Отчет о воспроизводимости сгенерирован. Статус: {'✓' if is_valid else '✗'}"
        )
        return report

    def _check_seed_conflicts(self) -> List[str]:
        """Проверить конфликты сидов между компонентами."""
        conflicts = []

        # Проверяем основной seed vs reproducibility seed
        if self.reproducibility.seed != self.seed:
            conflicts.append(
                f"Конфликт сидов: основной seed ({self.seed}) != "
                f"reproducibility seed ({self.reproducibility.seed})"
            )

        # Проверяем основной seed vs algorithm seed
        if self.algorithm.seed is not None and self.algorithm.seed != self.seed:
            conflicts.append(
                f"Конфликт сидов: основной seed ({self.seed}) != "
                f"algorithm seed ({self.algorithm.seed})"
            )

        # Проверяем experiment seeds
        if self.experiment and self.experiment.seeds:
            if self.seed not in self.experiment.seeds:
                conflicts.append(
                    f"Основной seed ({self.seed}) отсутствует в experiment.seeds: "
                    f"{self.experiment.seeds}"
                )

        return conflicts

    def _check_determinism_settings(self) -> List[str]:
        """Проверить настройки детерминизма."""
        issues = []

        # Проверяем противоречивые настройки
        if self.reproducibility.deterministic and self.reproducibility.benchmark:
            issues.append(
                "Противоречие: deterministic=True и benchmark=True одновременно. "
                "Это может нарушить воспроизводимость"
            )

        # Проверяем SDE настройки
        if self.algorithm.use_sde:
            issues.append(
                "Использование SDE (Stochastic Differential Equations) может "
                "снизить воспроизводимость результатов"
            )

        # Проверяем device настройки
        if self.algorithm.device == "auto":
            issues.append(
                "device='auto' может привести к различному поведению "
                "на разных системах. Рекомендуется явно указать 'cpu' или 'cuda'"
            )

        return issues

    def _check_cuda_settings(self) -> List[str]:
        """Проверить CUDA настройки."""
        warnings_list = []

        try:
            import torch

            if torch.cuda.is_available() and not self.reproducibility.use_cuda:
                warnings_list.append(
                    "CUDA доступна, но use_cuda=False. Убедитесь, что это намеренно"
                )

            if self.reproducibility.use_cuda and not torch.cuda.is_available():
                warnings_list.append("use_cuda=True, но CUDA недоступна в системе")

        except ImportError:
            warnings_list.append("PyTorch не установлен, проверка CUDA невозможна")

        return warnings_list

    def _check_algorithm_reproducibility(self) -> List[str]:
        """Проверить настройки воспроизводимости алгоритма."""
        warnings_list = []

        # Проверяем специфичные для алгоритма настройки
        if self.algorithm.name in ["SAC", "TD3"]:
            if self.algorithm.use_sde:
                warnings_list.append(
                    f"Алгоритм {self.algorithm.name} с SDE может иметь "
                    "пониженную воспроизводимость"
                )

        # Проверяем policy_kwargs
        if self.algorithm.policy_kwargs:
            if "activation_fn" in self.algorithm.policy_kwargs:
                activation = self.algorithm.policy_kwargs["activation_fn"]
                if (
                    hasattr(activation, "__name__")
                    and "dropout" in activation.__name__.lower()
                ):
                    warnings_list.append(
                        "Использование dropout в policy может снизить воспроизводимость"
                    )

        return warnings_list

    def _get_system_reproducibility_info(self) -> Dict[str, Any]:
        """Получить системную информацию для воспроизводимости."""
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
        }

        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["cudnn_version"] = torch.backends.cudnn.version()
        except ImportError:
            info["torch_version"] = "not installed"

        try:
            import numpy as np

            info["numpy_version"] = np.__version__
        except ImportError:
            info["numpy_version"] = "not installed"

        try:
            import gymnasium

            info["gymnasium_version"] = gymnasium.__version__
        except ImportError:
            info["gymnasium_version"] = "not installed"

        return info

    def _get_reproducibility_recommendations(
        self, warnings_list: List[str]
    ) -> List[str]:
        """Получить рекомендации по улучшению воспроизводимости."""
        recommendations = []

        if any("конфликт сидов" in w.lower() for w in warnings_list):
            recommendations.append(
                "Используйте enforce_seed_consistency() для автоматической "
                "синхронизации сидов"
            )

        if any(
            "deterministic" in w.lower() and "benchmark" in w.lower()
            for w in warnings_list
        ):
            recommendations.append(
                "Установите benchmark=False для полной воспроизводимости"
            )

        if any("sde" in w.lower() for w in warnings_list):
            recommendations.append(
                "Отключите SDE (use_sde=False) для детерминистического поведения"
            )

        if any("device" in w.lower() and "auto" in w.lower() for w in warnings_list):
            recommendations.append(
                "Явно укажите device ('cpu' или 'cuda') вместо 'auto'"
            )

        if not recommendations:
            recommendations.append("Настройки воспроизводимости оптимальны")

        return recommendations

    def _get_timestamp(self) -> str:
        """Получить текущий timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()


class ConfigLoader:
    """Загрузчик конфигураций с поддержкой Hydra."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Инициализация загрузчика конфигураций.

        Args:
            config_dir: Директория с конфигурационными файлами
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Инициализирован ConfigLoader с директорией: {self.config_dir}")

    def load_config(
        self,
        config_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
    ) -> RLConfig:
        """Загрузить конфигурацию.

        Args:
            config_name: Имя конфигурационного файла (без расширения)
            config_path: Путь к конкретному файлу конфигурации
            overrides: Список переопределений в формате key=value

        Returns:
            Загруженная конфигурация

        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            ValueError: Если конфигурация невалидна
        """
        try:
            if config_path:
                # Загружаем конкретный файл
                config_dict = self._load_yaml_file(config_path)
            elif config_name:
                # Загружаем по имени из директории
                config_file = self.config_dir / f"{config_name}.yaml"
                config_dict = self._load_yaml_file(config_file)
            else:
                # Загружаем конфигурацию по умолчанию
                config_dict = self._get_default_config()

            # Применяем переопределения
            if overrides:
                config_dict = self._apply_overrides(config_dict, overrides)

            # Подставляем переменные окружения
            config_dict = self._substitute_env_vars(config_dict)

            # Валидируем и создаем объект конфигурации
            config = self._create_config_object(config_dict)

            # Дополнительная валидация
            self._validate_config(config)

            logger.info("Конфигурация успешно загружена")
            return config

        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise

    def save_config(self, config: RLConfig, output_path: Union[str, Path]) -> None:
        """Сохранить конфигурацию в файл.

        Args:
            config: Конфигурация для сохранения
            output_path: Путь для сохранения
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Преобразуем в словарь
            config_dict = self._config_to_dict(config)

            # Сохраняем в YAML
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Конфигурация сохранена в {output_path}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {e}")
            raise

    def merge_configs(
        self, base_config: RLConfig, override_config: Dict[str, Any]
    ) -> RLConfig:
        """Объединить конфигурации.

        Args:
            base_config: Базовая конфигурация
            override_config: Переопределения

        Returns:
            Объединенная конфигурация
        """
        # Преобразуем базовую конфигурацию в словарь
        base_dict = self._config_to_dict(base_config)

        # Глубоко объединяем словари
        merged_dict = self._deep_merge(base_dict, override_config)

        # Создаем новый объект конфигурации
        return self._create_config_object(merged_dict)

    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """Проверить валидность файла конфигурации.

        Args:
            config_path: Путь к файлу конфигурации

        Returns:
            True если конфигурация валидна
        """
        try:
            config = self.load_config(config_path=config_path)
            self._validate_config(config)
            return True
        except Exception as e:
            logger.error(f"Конфигурация невалидна: {e}")
            return False

    def get_available_configs(self) -> List[str]:
        """Получить список доступных конфигураций.

        Returns:
            Список имен конфигурационных файлов
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)

        return sorted(config_files)

    def _load_yaml_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Загрузить YAML файл."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML файла {file_path}: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Получить конфигурацию по умолчанию."""
        return {
            "experiment_name": "default_experiment",
            "output_dir": "results",
            "seed": 42,
            "algorithm": {
                "name": "PPO",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "seed": 42,
            },
            "environment": {"name": "LunarLander-v3"},
            "training": {"total_timesteps": 100000, "eval_freq": 10000},
            "logging": {"level": "INFO", "log_to_file": True},
            "reproducibility": {
                "seed": 42,
                "deterministic": True,
                "benchmark": False,
                "use_cuda": False,
                "enforce_seed_consistency": True,
                "validate_determinism": True,
                "warn_on_seed_conflicts": True,
                "auto_propagate_seeds": True,
            },
        }

    def _apply_overrides(
        self, config_dict: Dict[str, Any], overrides: List[str]
    ) -> Dict[str, Any]:
        """Применить переопределения конфигурации."""
        for override in overrides:
            if "=" not in override:
                logger.warning(f"Неверный формат переопределения: {override}")
                continue

            key, value = override.split("=", 1)

            # Преобразуем значение в правильный тип
            parsed_value = self._parse_value(value)

            # Устанавливаем значение по ключу (поддерживаем вложенные ключи)
            self._set_nested_value(config_dict, key, parsed_value)

        return config_dict

    def _substitute_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Подставить переменные окружения."""

        def substitute_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Подстановка переменных окружения в формате ${VAR_NAME}
                import re

                pattern = r"\$\{([^}]+)\}"

                def replace_env_var(match: Any) -> str:
                    var_name = match.group(1)
                    default_value = None

                    # Поддержка значений по умолчанию: ${VAR_NAME:default}
                    if ":" in var_name:
                        var_name, default_value = var_name.split(":", 1)

                    return os.getenv(var_name, default_value or match.group(0))

                return re.sub(pattern, replace_env_var, obj)
            else:
                return obj

        result = substitute_recursive(config_dict)
        # Гарантируем, что возвращаем Dict[str, Any]
        if isinstance(result, dict):
            return result
        else:
            # Это не должно произойти, но для безопасности типов
            return config_dict

    def _create_config_object(self, config_dict: Dict[str, Any]) -> RLConfig:
        """Создать объект конфигурации из словаря."""
        try:
            # Создаем подконфигурации
            algorithm_config = AlgorithmConfig(**config_dict.get("algorithm", {}))
            environment_config = EnvironmentConfig(**config_dict.get("environment", {}))
            training_config = TrainingConfig(**config_dict.get("training", {}))
            logging_config = LoggingConfig(**config_dict.get("logging", {}))
            reproducibility_config = ReproducibilityConfig(
                **config_dict.get("reproducibility", {})
            )

            # Создаем конфигурацию эксперимента если есть
            experiment_config = None
            if "experiment" in config_dict:
                experiment_config = ExperimentConfig(**config_dict["experiment"])

            # Создаем основную конфигурацию
            main_config = RLConfig(
                experiment_name=config_dict.get(
                    "experiment_name", "default_experiment"
                ),
                output_dir=config_dict.get("output_dir", "results"),
                seed=config_dict.get("seed", 42),
                algorithm=algorithm_config,
                environment=environment_config,
                training=training_config,
                experiment=experiment_config,
                logging=logging_config,
                reproducibility=reproducibility_config,
            )

            return main_config

        except Exception as e:
            raise ValueError(f"Ошибка создания объекта конфигурации: {e}")

    def _validate_config(self, config: RLConfig) -> None:
        """Валидировать конфигурацию."""
        # Проверяем алгоритм
        supported_algorithms = ["PPO", "A2C", "SAC", "TD3"]
        if config.algorithm.name not in supported_algorithms:
            raise ValueError(f"Неподдерживаемый алгоритм: {config.algorithm.name}")

        # Проверяем среду
        supported_environments = [
            "LunarLander-v2",
            "LunarLander-v3",
            "MountainCarContinuous-v0",
            "Acrobot-v1",
            "Pendulum-v1",
        ]
        if config.environment.name not in supported_environments:
            logger.warning(f"Среда {config.environment.name} может быть не поддержана")

        # Проверяем параметры обучения
        if config.training.total_timesteps <= 0:
            raise ValueError("total_timesteps должен быть положительным")

        if config.training.eval_freq <= 0:
            raise ValueError("eval_freq должен быть положительным")

        # Проверяем seed
        if not (0 <= config.seed <= 2**32 - 1):
            raise ValueError("seed должен быть в диапазоне [0, 2^32-1]")

        # Дополнительная валидация воспроизводимости
        self._validate_reproducibility_config(config)

        # Проверяем директории
        output_dir = Path(config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(
                f"Не удается создать выходную директорию {output_dir}: {e}"
            )

    def _validate_reproducibility_config(self, config: RLConfig) -> None:
        """Дополнительная валидация настроек воспроизводимости."""
        # Проверяем настройки воспроизводимости
        is_valid, warnings_list = config.validate_reproducibility()

        if warnings_list:
            logger.info(
                f"Обнаружено {len(warnings_list)} предупреждений о воспроизводимости"
            )
            for warning_msg in warnings_list:
                logger.warning(f"Воспроизводимость: {warning_msg}")

        # Принудительная синхронизация сидов если включена
        if config.reproducibility.enforce_seed_consistency:
            config.enforce_seed_consistency()

    def load_config_with_seed_validation(
        self,
        config_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
        apply_seeds: bool = True,
        validate_reproducibility: bool = True,
    ) -> RLConfig:
        """Загрузить конфигурацию с расширенной валидацией сидов.

        Args:
            config_name: Имя конфигурационного файла
            config_path: Путь к файлу конфигурации
            overrides: Список переопределений
            apply_seeds: Применить сиды сразу после загрузки
            validate_reproducibility: Валидировать настройки воспроизводимости

        Returns:
            Загруженная и валидированная конфигурация
        """
        # Загружаем базовую конфигурацию
        config = self.load_config(config_name, config_path, overrides)

        # Дополнительная валидация воспроизводимости
        if validate_reproducibility:
            logger.info("Выполнение расширенной валидации воспроизводимости")
            is_valid, warnings_list = config.validate_reproducibility()

            if warnings_list:
                logger.warning(
                    f"Обнаружено {len(warnings_list)} проблем с воспроизводимостью"
                )
                for warning_msg in warnings_list:
                    logger.warning(warning_msg)

        # Применяем сиды если требуется
        if apply_seeds:
            logger.info("Применение сидов к системе")
            config.apply_seeds()

        return config

    def create_reproducibility_report(
        self, config: RLConfig, output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Создать отчет о воспроизводимости и сохранить его.

        Args:
            config: Конфигурация для анализа
            output_path: Путь для сохранения отчета (опционально)

        Returns:
            Словарь с отчетом о воспроизводимости
        """
        logger.info("Создание отчета о воспроизводимости")

        # Генерируем отчет
        report = config.get_reproducibility_report()

        # Сохраняем отчет если указан путь
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                import json

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

                logger.info(f"Отчет о воспроизводимости сохранен: {output_path}")

            except Exception as e:
                logger.error(f"Ошибка при сохранении отчета: {e}")
                raise

        return report

    def validate_seed_consistency_across_configs(
        self, config_paths: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Проверить консистентность сидов между несколькими конфигурациями.

        Args:
            config_paths: Список путей к конфигурационным файлам

        Returns:
            Отчет о консистентности сидов
        """
        logger.info(
            f"Проверка консистентности сидов для {len(config_paths)} конфигураций"
        )

        configs = []
        seeds_info = []

        # Загружаем все конфигурации
        for config_path in config_paths:
            try:
                config = self.load_config(config_path=config_path)
                configs.append((config_path, config))

                seeds_info.append(
                    {
                        "config_path": str(config_path),
                        "main_seed": config.seed,
                        "reproducibility_seed": config.reproducibility.seed,
                        "algorithm_seed": config.algorithm.seed,
                        "experiment_seeds": config.experiment.seeds
                        if config.experiment
                        else None,
                    }
                )

            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации {config_path}: {e}")
                seeds_info.append({"config_path": str(config_path), "error": str(e)})

        # Анализируем консистентность
        consistency_report = {
            "total_configs": len(config_paths),
            "loaded_configs": len(configs),
            "seeds_info": seeds_info,
            "consistency_issues": [],
            "recommendations": [],
        }

        # Проверяем конфликты между конфигурациями
        main_seeds = [
            info.get("main_seed") for info in seeds_info if "main_seed" in info
        ]
        unique_main_seeds = set(main_seeds)

        if len(unique_main_seeds) > 1:
            consistency_report["consistency_issues"].append(
                f"Различные основные сиды между конфигурациями: {unique_main_seeds}"
            )
            consistency_report["recommendations"].append(
                "Используйте одинаковый основной seed для всех связанных экспериментов"
            )

        # Проверяем внутренние конфликты в каждой конфигурации
        for config_path, config in configs:
            conflicts = config._check_seed_conflicts()
            if conflicts:
                consistency_report["consistency_issues"].extend(
                    [f"{config_path}: {conflict}" for conflict in conflicts]
                )

        if not consistency_report["consistency_issues"]:
            consistency_report["recommendations"].append(
                "Все конфигурации имеют консистентные настройки сидов"
            )

        logger.info(
            f"Проверка консистентности завершена. "
            f"Обнаружено {len(consistency_report['consistency_issues'])} проблем"
        )

        return consistency_report

    def _config_to_dict(self, config: RLConfig) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь."""
        from dataclasses import asdict

        return asdict(config)

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Глубоко объединить два словаря."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _parse_value(self, value: str) -> Any:
        """Парсить строковое значение в правильный тип."""
        # Булевы значения
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # None
        if value.lower() in ("null", "none"):
            return None

        # Числа
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Списки (простые, разделенные запятыми)
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [self._parse_value(item.strip()) for item in items if item.strip()]

        # Строки
        return value

    def _set_nested_value(
        self, config_dict: Dict[str, Any], key: str, value: Any
    ) -> None:
        """Установить значение по вложенному ключу."""
        keys = key.split(".")
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value


# Глобальный экземпляр загрузчика
_config_loader = None


def get_config_loader(config_dir: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """Получить глобальный экземпляр загрузчика конфигураций.

    Args:
        config_dir: Директория с конфигурациями (только при первом вызове)

    Returns:
        Экземпляр ConfigLoader
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)

    return _config_loader


def load_config(
    config_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> RLConfig:
    """Удобная функция для загрузки конфигурации.

    Args:
        config_name: Имя конфигурационного файла
        config_path: Путь к файлу конфигурации
        overrides: Список переопределений

    Returns:
        Загруженная конфигурация
    """
    loader = get_config_loader()
    return loader.load_config(config_name, config_path, overrides)


def load_config_with_seeds(
    config_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
    apply_seeds: bool = True,
    validate_reproducibility: bool = True,
) -> RLConfig:
    """Удобная функция для загрузки конфигурации с применением сидов.

    Args:
        config_name: Имя конфигурационного файла
        config_path: Путь к файлу конфигурации
        overrides: Список переопределений
        apply_seeds: Применить сиды сразу после загрузки
        validate_reproducibility: Валидировать настройки воспроизводимости

    Returns:
        Загруженная конфигурация с примененными сидами
    """
    loader = get_config_loader()
    return loader.load_config_with_seed_validation(
        config_name, config_path, overrides, apply_seeds, validate_reproducibility
    )


def create_reproducibility_report(
    config: RLConfig, output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Создать отчет о воспроизводимости для конфигурации.

    Args:
        config: Конфигурация для анализа
        output_path: Путь для сохранения отчета

    Returns:
        Отчет о воспроизводимости
    """
    loader = get_config_loader()
    return loader.create_reproducibility_report(config, output_path)


def validate_configs_seed_consistency(
    config_paths: List[Union[str, Path]],
) -> Dict[str, Any]:
    """Проверить консистентность сидов между конфигурациями.

    Args:
        config_paths: Список путей к конфигурационным файлам

    Returns:
        Отчет о консистентности сидов
    """
    loader = get_config_loader()
    return loader.validate_seed_consistency_across_configs(config_paths)


def enforce_global_seed_consistency(
    configs: List[RLConfig], master_seed: Optional[int] = None
) -> List[RLConfig]:
    """Принудительно синхронизировать сиды между несколькими конфигурациями.

    Args:
        configs: Список конфигураций для синхронизации
        master_seed: Основной сид для всех конфигураций (если None, используется первый)

    Returns:
        Список конфигураций с синхронизированными сидами
    """
    if not configs:
        return configs

    # Определяем основной сид
    if master_seed is None:
        master_seed = configs[0].seed

    logger.info(
        f"Принудительная синхронизация {len(configs)} конфигураций с сидом {master_seed}"
    )

    # Синхронизируем все конфигурации
    for config in configs:
        config.seed = master_seed
        config.enforce_seed_consistency()

    logger.info("Синхронизация сидов завершена")
    return configs


def create_default_configs(config_dir: Union[str, Path]) -> None:
    """Создать файлы конфигураций по умолчанию.

    Args:
        config_dir: Директория для создания конфигураций
    """
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    loader = ConfigLoader(config_dir)

    # Создаем конфигурации по умолчанию
    configs = {
        "default": RLConfig(),
        "ppo_lunarlander": RLConfig(
            experiment_name="ppo_lunarlander",
            algorithm=AlgorithmConfig(name="PPO"),
            environment=EnvironmentConfig(name="LunarLander-v3"),
            training=TrainingConfig(total_timesteps=200000),
        ),
        "a2c_lunarlander": RLConfig(
            experiment_name="a2c_lunarlander",
            algorithm=AlgorithmConfig(name="A2C", learning_rate=7e-4, n_steps=5),
            environment=EnvironmentConfig(name="LunarLander-v3"),
            training=TrainingConfig(total_timesteps=200000),
        ),
    }

    for name, config in configs.items():
        config_file = config_dir / f"{name}.yaml"
        if not config_file.exists():
            loader.save_config(config, config_file)
            logger.info(f"Создан файл конфигурации: {config_file}")


def validate_config_directory(config_dir: Union[str, Path]) -> bool:
    """Проверить валидность всех конфигураций в директории.

    Args:
        config_dir: Директория с конфигурациями

    Returns:
        True если все конфигурации валидны
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        logger.error(f"Директория конфигураций не существует: {config_dir}")
        return False

    loader = ConfigLoader(config_dir)
    all_valid = True

    for config_file in config_dir.glob("*.yaml"):
        try:
            loader.load_config(config_path=config_file)
            logger.info(f"✓ {config_file.name} - валидна")
        except Exception as e:
            logger.error(f"✗ {config_file.name} - ошибка: {e}")
            all_valid = False

    return all_valid
