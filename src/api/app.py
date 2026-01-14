"""Основное FastAPI приложение для системы обучения RL агентов.

Этот модуль содержит создание и настройку FastAPI приложения с интеграцией
в существующую архитектуру RL системы, включая роутеры, middleware,
обработку ошибок и документацию API.
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel, Field, field_validator
from pydantic.functional_validators import field_validator as validator

from .config import APIConfig, get_api_config, get_cors_config, get_logging_config
from .dependencies import (
    ExperimentService,
    EnvironmentService, 
    AlgorithmService,
    get_experiment_service,
    get_environment_service,
    get_algorithm_service,
    get_current_user
)
from src.utils.logging import setup_logging, get_experiment_logger
from src.experiments.experiment import ExperimentStatus

# Настройка логирования
logger = logging.getLogger(__name__)


# Pydantic модели для API
class ExperimentConfig(BaseModel):
    """Модель конфигурации эксперимента."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Название эксперимента")
    algorithm: str = Field(..., description="Алгоритм RL")
    environment: str = Field(..., description="Среда обучения")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Гиперпараметры алгоритма")
    seed: int = Field(42, ge=0, le=2**32-1, description="Сид для воспроизводимости")
    description: str = Field("", max_length=1000, description="Описание эксперимента")
    hypothesis: str = Field("", max_length=500, description="Гипотеза эксперимента")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Валидация названия эксперимента."""
        if not v.strip():
            raise ValueError("Название эксперимента не может быть пустым")
        return v.strip()
    
    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v):
        """Валидация алгоритма."""
        supported = ["PPO", "A2C", "SAC", "TD3"]
        if v not in supported:
            raise ValueError(f"Алгоритм должен быть одним из: {supported}")
        return v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Валидация среды."""
        supported = [
            "LunarLander-v2", "LunarLander-v3",
            "MountainCarContinuous-v0", "Acrobot-v1", "Pendulum-v1"
        ]
        if v not in supported:
            raise ValueError(f"Среда должна быть одной из: {supported}")
        return v


class ExperimentUpdate(BaseModel):
    """Модель обновления эксперимента."""
    
    status: Optional[str] = Field(None, description="Новый статус эксперимента")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Обновленные гиперпараметры")
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Валидация статуса."""
        if v is not None:
            allowed = ["running", "paused", "cancelled"]
            if v not in allowed:
                raise ValueError(f"Статус должен быть одним из: {allowed}")
        return v


class ExperimentResponse(BaseModel):
    """Модель ответа с информацией об эксперименте."""
    
    id: str = Field(..., description="Идентификатор эксперимента")
    name: str = Field(..., description="Название эксперимента")
    status: str = Field(..., description="Статус эксперимента")
    algorithm: str = Field(..., description="Алгоритм RL")
    environment: str = Field(..., description="Среда обучения")
    hyperparameters: Dict[str, Any] = Field(..., description="Гиперпараметры")
    seed: int = Field(..., description="Сид")
    created_at: str = Field(..., description="Время создания")
    started_at: Optional[str] = Field(None, description="Время запуска")
    completed_at: Optional[str] = Field(None, description="Время завершения")
    description: str = Field("", description="Описание")
    hypothesis: str = Field("", description="Гипотеза")


class TrainingResponse(BaseModel):
    """Модель ответа на запуск обучения."""
    
    experiment_id: str = Field(..., description="Идентификатор эксперимента")
    status: str = Field(..., description="Статус обучения")
    message: str = Field(..., description="Сообщение о статусе")


class MetricsResponse(BaseModel):
    """Модель ответа с метриками."""
    
    experiment_id: str = Field(..., description="Идентификатор эксперимента")
    metrics: List[Dict[str, Any]] = Field(..., description="Список метрик")
    summary: Dict[str, Any] = Field(..., description="Сводка метрик")


class ResultsResponse(BaseModel):
    """Модель ответа с результатами."""
    
    experiment_id: str = Field(..., description="Идентификатор эксперимента")
    final_evaluation: Dict[str, Any] = Field(..., description="Финальная оценка")
    model_path: Optional[str] = Field(None, description="Путь к модели")
    video_path: Optional[str] = Field(None, description="Путь к видео")
    graph_path: Optional[str] = Field(None, description="Путь к графику")


class EnvironmentInfo(BaseModel):
    """Модель информации о среде."""
    
    name: str = Field(..., description="Название среды")
    description: str = Field(..., description="Описание среды")
    observation_space: Dict[str, Any] = Field(..., description="Пространство наблюдений")
    action_space: Dict[str, Any] = Field(..., description="Пространство действий")


class AlgorithmInfo(BaseModel):
    """Модель информации об алгоритме."""
    
    name: str = Field(..., description="Название алгоритма")
    description: str = Field(..., description="Описание алгоритма")
    default_hyperparameters: Dict[str, Any] = Field(..., description="Гиперпараметры по умолчанию")


class HealthResponse(BaseModel):
    """Модель ответа healthcheck."""
    
    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Время проверки")
    version: str = Field(..., description="Версия API")
    uptime_seconds: float = Field(..., description="Время работы в секундах")
    active_experiments: int = Field(..., description="Количество активных экспериментов")


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой."""
    
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")
    timestamp: str = Field(..., description="Время ошибки")


# Глобальные переменные для отслеживания состояния
app_start_time = time.time()
shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    logger.info("Запуск FastAPI приложения")
    
    # Настройка логирования
    log_config = get_logging_config()
    setup_logging(**log_config)
    
    # Инициализация сервисов
    config = get_api_config()
    logger.info(f"Инициализация с конфигурацией: {config.to_dict()}")
    
    # Настройка graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Получен сигнал {signum}, инициируется graceful shutdown")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Shutdown
    logger.info("Завершение работы FastAPI приложения")
    
    # Ожидание завершения активных экспериментов
    experiment_service = get_experiment_service()
    if experiment_service.running_experiments:
        logger.info(f"Ожидание завершения {len(experiment_service.running_experiments)} экспериментов")
        
        # Даем 30 секунд на graceful shutdown
        try:
            await asyncio.wait_for(
                asyncio.gather(*experiment_service.running_experiments.values(), return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout при ожидании завершения экспериментов")
    
    logger.info("Приложение завершено")


def create_app() -> FastAPI:
    """Создать и настроить FastAPI приложение.
    
    Returns:
        Настроенное FastAPI приложение
    """
    config = get_api_config()
    
    # Создание приложения
    app = FastAPI(
        title=config.settings.title,
        description=config.settings.description,
        version=config.settings.version,
        docs_url=config.settings.docs_url,
        redoc_url=config.settings.redoc_url,
        openapi_url=config.settings.openapi_url,
        lifespan=lifespan
    )
    
    # Настройка CORS
    cors_config = get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        **cors_config
    )
    
    # Middleware для доверенных хостов
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # В продакшене следует ограничить
    )
    
    # Middleware для логирования запросов
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Логирование HTTP запросов."""
        start_time = time.time()
        
        # Логируем входящий запрос
        logger.info(
            f"Входящий запрос: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        # Выполняем запрос
        response = await call_next(request)
        
        # Логируем ответ
        process_time = time.time() - start_time
        logger.info(
            f"Ответ: {response.status_code} за {process_time:.3f}s",
            extra={
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        # Добавляем заголовок с временем обработки
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Обработчики ошибок
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """Кастомный обработчик HTTP исключений."""
        error_response = ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        )
        
        logger.warning(
            f"HTTP исключение: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Обработчик общих исключений."""
        error_response = ErrorResponse(
            error="InternalServerError",
            message="Внутренняя ошибка сервера",
            details={"exception_type": type(exc).__name__},
            timestamp=datetime.now().isoformat()
        )
        
        logger.error(
            f"Необработанное исключение: {exc}",
            extra={
                "exception_type": type(exc).__name__,
                "path": request.url.path,
                "method": request.method
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )
    
    # Healthcheck эндпоинт
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(
        experiment_service: ExperimentService = Depends(get_experiment_service)
    ):
        """Проверка состояния сервиса."""
        uptime = time.time() - app_start_time
        active_experiments = len(experiment_service.running_experiments)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version=config.settings.version,
            uptime_seconds=uptime,
            active_experiments=active_experiments
        )
    
    # Эндпоинты для экспериментов
    @app.post("/experiments", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED, tags=["Experiments"])
    async def create_experiment(
        experiment_config: ExperimentConfig,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Создать новый эксперимент."""
        experiment = await experiment_service.create_experiment(
            name=experiment_config.name,
            algorithm=experiment_config.algorithm,
            environment=experiment_config.environment,
            hyperparameters=experiment_config.hyperparameters,
            seed=experiment_config.seed,
            description=experiment_config.description,
            hypothesis=experiment_config.hypothesis
        )
        
        return ExperimentResponse(
            id=experiment.experiment_id,
            name=experiment_config.name,
            status=experiment.status.value,
            algorithm=experiment.baseline_config.algorithm.name,
            environment=experiment.baseline_config.environment.name,
            hyperparameters=experiment_config.hyperparameters,
            seed=experiment_config.seed,
            created_at=experiment.created_at.isoformat(),
            started_at=experiment.started_at.isoformat() if experiment.started_at else None,
            completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
            description=experiment_config.description,
            hypothesis=experiment_config.hypothesis
        )
    
    @app.get("/experiments", response_model=List[ExperimentResponse], tags=["Experiments"])
    async def list_experiments(
        status_filter: Optional[str] = None,
        algorithm: Optional[str] = None,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить список экспериментов."""
        experiments = await experiment_service.list_experiments(
            status_filter=status_filter,
            algorithm_filter=algorithm
        )
        
        return [
            ExperimentResponse(
                id=exp.experiment_id,
                name=exp.baseline_config.experiment_name,
                status=exp.status.value,
                algorithm=exp.baseline_config.algorithm.name,
                environment=exp.baseline_config.environment.name,
                hyperparameters={},  # Можно добавить извлечение гиперпараметров
                seed=exp.baseline_config.seed,
                created_at=exp.created_at.isoformat(),
                started_at=exp.started_at.isoformat() if exp.started_at else None,
                completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
                description="",
                hypothesis=exp.hypothesis
            )
            for exp in experiments
        ]
    
    @app.get("/experiments/{experiment_id}", response_model=ExperimentResponse, tags=["Experiments"])
    async def get_experiment(
        experiment_id: str,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить информацию об эксперименте."""
        experiment = await experiment_service.get_experiment(experiment_id)
        
        return ExperimentResponse(
            id=experiment.experiment_id,
            name=experiment.baseline_config.experiment_name,
            status=experiment.status.value,
            algorithm=experiment.baseline_config.algorithm.name,
            environment=experiment.baseline_config.environment.name,
            hyperparameters={},
            seed=experiment.baseline_config.seed,
            created_at=experiment.created_at.isoformat(),
            started_at=experiment.started_at.isoformat() if experiment.started_at else None,
            completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
            description="",
            hypothesis=experiment.hypothesis
        )
    
    @app.put("/experiments/{experiment_id}", response_model=ExperimentResponse, tags=["Experiments"])
    async def update_experiment(
        experiment_id: str,
        update_data: ExperimentUpdate,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Обновить эксперимент."""
        experiment = await experiment_service.update_experiment(
            experiment_id=experiment_id,
            status_update=update_data.status,
            hyperparameters=update_data.hyperparameters
        )
        
        return ExperimentResponse(
            id=experiment.experiment_id,
            name=experiment.baseline_config.experiment_name,
            status=experiment.status.value,
            algorithm=experiment.baseline_config.algorithm.name,
            environment=experiment.baseline_config.environment.name,
            hyperparameters={},
            seed=experiment.baseline_config.seed,
            created_at=experiment.created_at.isoformat(),
            started_at=experiment.started_at.isoformat() if experiment.started_at else None,
            completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
            description="",
            hypothesis=experiment.hypothesis
        )
    
    @app.post("/experiments/{experiment_id}/train", response_model=TrainingResponse, tags=["Training"])
    async def start_training(
        experiment_id: str,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Запустить обучение для эксперимента."""
        result = await experiment_service.start_training(experiment_id)
        
        return TrainingResponse(**result)
    
    @app.get("/experiments/{experiment_id}/metrics", response_model=MetricsResponse, tags=["Metrics"])
    async def get_experiment_metrics(
        experiment_id: str,
        from_timestep: Optional[int] = None,
        to_timestep: Optional[int] = None,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить метрики эксперимента."""
        metrics_data = await experiment_service.get_experiment_metrics(
            experiment_id=experiment_id,
            from_timestep=from_timestep,
            to_timestep=to_timestep
        )
        
        return MetricsResponse(**metrics_data)
    
    @app.get("/experiments/{experiment_id}/results", response_model=ResultsResponse, tags=["Results"])
    async def get_experiment_results(
        experiment_id: str,
        experiment_service: ExperimentService = Depends(get_experiment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить результаты эксперимента."""
        results_data = await experiment_service.get_experiment_results(experiment_id)
        
        return ResultsResponse(**results_data)
    
    # Эндпоинты для сред
    @app.get("/environments", response_model=List[EnvironmentInfo], tags=["Environments"])
    async def list_environments(
        environment_service: EnvironmentService = Depends(get_environment_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить список доступных сред."""
        environments = await environment_service.list_environments()
        
        return [EnvironmentInfo(**env) for env in environments]
    
    # Эндпоинты для алгоритмов
    @app.get("/algorithms", response_model=List[AlgorithmInfo], tags=["Algorithms"])
    async def list_algorithms(
        algorithm_service: AlgorithmService = Depends(get_algorithm_service),
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить список доступных алгоритмов."""
        algorithms = await algorithm_service.list_algorithms()
        
        return [AlgorithmInfo(**algo) for algo in algorithms]
    
    # Информационные эндпоинты
    @app.get("/", tags=["Info"])
    async def root():
        """Корневой эндпоинт с информацией об API."""
        return {
            "message": "RL Agent Training API",
            "version": config.settings.version,
            "docs": config.settings.docs_url,
            "health": "/health"
        }
    
    @app.get("/config", tags=["Info"])
    async def get_config_info(
        current_user: Optional[str] = Depends(get_current_user)
    ):
        """Получить информацию о конфигурации API."""
        return config.to_dict()
    
    return app


def run_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    reload: bool = False,
    workers: int = 1
) -> None:
    """Запустить FastAPI приложение.
    
    Args:
        host: Хост для привязки
        port: Порт для привязки
        debug: Режим отладки
        reload: Автоперезагрузка при изменениях
        workers: Количество worker процессов
    """
    config = get_api_config()
    
    # Переопределяем настройки если переданы
    if host != "0.0.0.0":
        config.settings.host = host
    if port != 8000:
        config.settings.port = port
    if debug:
        config.settings.debug = debug
    if reload:
        config.settings.reload = reload
    if workers != 1:
        config.settings.workers = workers
    
    # Создание приложения
    app = create_app()
    
    # Настройки uvicorn
    uvicorn_config = {
        "app": app,
        "host": config.settings.host,
        "port": config.settings.port,
        "debug": config.settings.debug,
        "reload": config.settings.reload,
        "access_log": True,
        "log_level": config.settings.log_level.lower(),
    }
    
    # В продакшене используем несколько workers
    if not config.settings.debug and not config.settings.reload and config.settings.workers > 1:
        uvicorn_config["workers"] = config.settings.workers
    
    logger.info(f"Запуск сервера на {config.settings.host}:{config.settings.port}")
    
    # Запуск сервера
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    # Запуск из командной строки
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Agent Training API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    run_app(
        host=args.host,
        port=args.port,
        debug=args.debug,
        reload=args.reload,
        workers=args.workers
    )