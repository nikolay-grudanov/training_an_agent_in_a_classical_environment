"""Конфигурация для FastAPI приложения системы обучения RL агентов.

Этот модуль содержит настройки API, включая параметры сервера, CORS,
аутентификации, логирования и интеграции с существующей RL системой.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from pydantic import validator
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """Настройки API из переменных окружения."""
    
    # Основные настройки сервера
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    
    # Настройки приложения
    title: str = "RL Agent Training API"
    description: str = "API для управления экспериментами обучения RL агентов"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # CORS настройки
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Настройки логирования
    log_level: str = "INFO"
    log_dir: str = "logs/api"
    log_format: str = "json"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    # Настройки безопасности
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Настройки базы данных/хранилища
    data_dir: str = "data"
    experiments_dir: str = "results/experiments"
    models_dir: str = "results/models"
    logs_dir: str = "results/logs"
    
    # Настройки RL системы
    max_concurrent_experiments: int = 3
    default_timeout_minutes: int = 60
    cleanup_interval_hours: int = 24
    
    # Настройки мониторинга
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Настройки кэширования
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    class Config:
        env_prefix = "RL_API_"
        env_file = ".env"
        case_sensitive = False
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Парсинг CORS origins из строки."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("cors_methods", pre=True)
    def parse_cors_methods(cls, v):
        """Парсинг CORS methods из строки."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_headers", pre=True)
    def parse_cors_headers(cls, v):
        """Парсинг CORS headers из строки."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v


@dataclass
class APIConfig:
    """Конфигурация API приложения."""
    
    # Настройки из переменных окружения
    settings: APISettings = field(default_factory=APISettings)
    
    # Пути к директориям
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    experiments_dir: Path = field(default_factory=lambda: Path("results/experiments"))
    models_dir: Path = field(default_factory=lambda: Path("results/models"))
    logs_dir: Path = field(default_factory=lambda: Path("results/logs"))
    
    # Настройки интеграции с RL системой
    config_loader_dir: Optional[Path] = None
    default_config_name: str = "default"
    supported_algorithms: List[str] = field(
        default_factory=lambda: ["PPO", "A2C", "SAC", "TD3"]
    )
    supported_environments: List[str] = field(
        default_factory=lambda: [
            "LunarLander-v2", 
            "LunarLander-v3",
            "MountainCarContinuous-v0", 
            "Acrobot-v1", 
            "Pendulum-v1"
        ]
    )
    
    # Настройки валидации
    max_experiment_name_length: int = 100
    max_description_length: int = 1000
    max_hypothesis_length: int = 500
    min_total_timesteps: int = 1000
    max_total_timesteps: int = 10_000_000
    
    def __post_init__(self) -> None:
        """Пост-инициализация для настройки путей."""
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Настройка директорий на основе настроек."""
        # Преобразуем относительные пути в абсолютные
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        
        if not self.experiments_dir.is_absolute():
            self.experiments_dir = self.base_dir / self.experiments_dir
        
        if not self.models_dir.is_absolute():
            self.models_dir = self.base_dir / self.models_dir
        
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.base_dir / self.logs_dir
        
        # Директория конфигураций
        if self.config_loader_dir is None:
            self.config_loader_dir = self.base_dir / "configs"
        
        # Создание директорий
        directories = [
            self.data_dir,
            self.experiments_dir, 
            self.models_dir,
            self.logs_dir,
            self.config_loader_dir,
            self.base_dir / self.settings.log_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Получить директорию для конкретного эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Путь к директории эксперимента
        """
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_model_path(self, experiment_id: str, model_name: str = "model") -> Path:
        """Получить путь к модели эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            model_name: Имя файла модели
            
        Returns:
            Путь к файлу модели
        """
        return self.get_experiment_dir(experiment_id) / f"{model_name}.zip"
    
    def get_metrics_path(self, experiment_id: str) -> Path:
        """Получить путь к файлу метрик эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Путь к файлу метрик
        """
        return self.get_experiment_dir(experiment_id) / "metrics.jsonl"
    
    def get_results_path(self, experiment_id: str) -> Path:
        """Получить путь к файлу результатов эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Путь к файлу результатов
        """
        return self.get_experiment_dir(experiment_id) / "results.json"
    
    def validate_algorithm(self, algorithm: str) -> bool:
        """Проверить поддержку алгоритма.
        
        Args:
            algorithm: Название алгоритма
            
        Returns:
            True если алгоритм поддерживается
        """
        return algorithm in self.supported_algorithms
    
    def validate_environment(self, environment: str) -> bool:
        """Проверить поддержку среды.
        
        Args:
            environment: Название среды
            
        Returns:
            True если среда поддерживается
        """
        return environment in self.supported_environments
    
    def validate_timesteps(self, timesteps: int) -> bool:
        """Проверить валидность количества временных шагов.
        
        Args:
            timesteps: Количество временных шагов
            
        Returns:
            True если количество валидно
        """
        return self.min_total_timesteps <= timesteps <= self.max_total_timesteps
    
    def to_dict(self) -> dict:
        """Преобразовать конфигурацию в словарь.
        
        Returns:
            Словарь с настройками конфигурации
        """
        return {
            "server": {
                "host": self.settings.host,
                "port": self.settings.port,
                "debug": self.settings.debug,
                "workers": self.settings.workers,
            },
            "application": {
                "title": self.settings.title,
                "description": self.settings.description,
                "version": self.settings.version,
            },
            "directories": {
                "base_dir": str(self.base_dir),
                "data_dir": str(self.data_dir),
                "experiments_dir": str(self.experiments_dir),
                "models_dir": str(self.models_dir),
                "logs_dir": str(self.logs_dir),
            },
            "rl_system": {
                "supported_algorithms": self.supported_algorithms,
                "supported_environments": self.supported_environments,
                "max_concurrent_experiments": self.settings.max_concurrent_experiments,
                "default_timeout_minutes": self.settings.default_timeout_minutes,
            },
            "validation": {
                "max_experiment_name_length": self.max_experiment_name_length,
                "max_description_length": self.max_description_length,
                "min_total_timesteps": self.min_total_timesteps,
                "max_total_timesteps": self.max_total_timesteps,
            }
        }


# Глобальный экземпляр конфигурации
_api_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Получить глобальный экземпляр конфигурации API.
    
    Returns:
        Экземпляр APIConfig
    """
    global _api_config
    
    if _api_config is None:
        _api_config = APIConfig()
    
    return _api_config


def create_api_config(
    base_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> APIConfig:
    """Создать новый экземпляр конфигурации API.
    
    Args:
        base_dir: Базовая директория проекта
        **kwargs: Дополнительные параметры конфигурации
        
    Returns:
        Новый экземпляр APIConfig
    """
    config_kwargs = {}
    
    if base_dir:
        config_kwargs["base_dir"] = Path(base_dir)
    
    config_kwargs.update(kwargs)
    
    return APIConfig(**config_kwargs)


def setup_api_config(config: APIConfig) -> None:
    """Установить глобальную конфигурацию API.
    
    Args:
        config: Экземпляр конфигурации для установки
    """
    global _api_config
    _api_config = config


def load_api_config_from_env() -> APIConfig:
    """Загрузить конфигурацию API из переменных окружения.
    
    Returns:
        Экземпляр APIConfig с настройками из окружения
    """
    settings = APISettings()
    return APIConfig(settings=settings)


def get_cors_config() -> dict:
    """Получить конфигурацию CORS.
    
    Returns:
        Словарь с настройками CORS
    """
    config = get_api_config()
    
    return {
        "allow_origins": config.settings.cors_origins,
        "allow_credentials": config.settings.cors_credentials,
        "allow_methods": config.settings.cors_methods,
        "allow_headers": config.settings.cors_headers,
    }


def get_logging_config() -> dict:
    """Получить конфигурацию логирования.
    
    Returns:
        Словарь с настройками логирования
    """
    config = get_api_config()
    
    return {
        "level": config.settings.log_level,
        "log_dir": config.base_dir / config.settings.log_dir,
        "console_output": True,
        "json_format": config.settings.log_format == "json",
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
    }


def get_security_config() -> dict:
    """Получить конфигурацию безопасности.
    
    Returns:
        Словарь с настройками безопасности
    """
    config = get_api_config()
    
    return {
        "secret_key": config.settings.secret_key,
        "algorithm": config.settings.algorithm,
        "access_token_expire_minutes": config.settings.access_token_expire_minutes,
    }