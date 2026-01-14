"""API модуль для системы обучения RL агентов.

Этот модуль предоставляет FastAPI приложение для управления экспериментами,
обучения агентов и получения результатов через REST API.
"""

from .app import create_app
from .config import APIConfig, get_api_config
from .dependencies import get_experiment_service, get_environment_service, get_algorithm_service

__all__ = [
    "create_app",
    "APIConfig", 
    "get_api_config",
    "get_experiment_service",
    "get_environment_service", 
    "get_algorithm_service",
]