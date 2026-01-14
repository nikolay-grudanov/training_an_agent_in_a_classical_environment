"""Зависимости FastAPI для системы обучения RL агентов.

Этот модуль содержит функции зависимостей для внедрения сервисов,
конфигураций и других компонентов в эндпоинты API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import APIConfig, get_api_config
from src.experiments.experiment import Experiment, ExperimentStatus
from src.experiments.runner import ExperimentRunner
from src.utils.config import ConfigLoader, RLConfig, AlgorithmConfig, EnvironmentConfig
from src.utils.logging import get_experiment_logger

# Настройка логирования
logger = logging.getLogger(__name__)

# Схема безопасности (опциональная)
security = HTTPBearer(auto_error=False)


class ExperimentService:
    """Сервис для управления экспериментами."""
    
    def __init__(self, config: APIConfig):
        """Инициализация сервиса экспериментов.
        
        Args:
            config: Конфигурация API
        """
        self.config = config
        self.experiments: Dict[str, Experiment] = {}
        self.running_experiments: Dict[str, asyncio.Task] = {}
        self.config_loader = ConfigLoader(config.config_loader_dir)
        
        logger.info("Инициализирован сервис экспериментов")
    
    async def create_experiment(
        self,
        name: str,
        algorithm: str,
        environment: str,
        hyperparameters: Dict[str, Any],
        seed: int,
        description: str = "",
        hypothesis: str = ""
    ) -> Experiment:
        """Создать новый эксперимент.
        
        Args:
            name: Название эксперимента
            algorithm: Алгоритм RL
            environment: Среда обучения
            hyperparameters: Гиперпараметры алгоритма
            seed: Сид для воспроизводимости
            description: Описание эксперимента
            hypothesis: Гипотеза эксперимента
            
        Returns:
            Созданный эксперимент
            
        Raises:
            HTTPException: Если параметры невалидны
        """
        # Валидация параметров
        if not self.config.validate_algorithm(algorithm):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый алгоритм: {algorithm}"
            )
        
        if not self.config.validate_environment(environment):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемая среда: {environment}"
            )
        
        # Создание конфигураций
        try:
            # Базовая конфигурация (для сравнения)
            baseline_config = self._create_rl_config(
                name=f"{name}_baseline",
                algorithm=algorithm,
                environment=environment,
                hyperparameters=hyperparameters,
                seed=seed
            )
            
            # Вариантная конфигурация (модифицированная)
            variant_hyperparams = hyperparameters.copy()
            # Небольшое изменение learning_rate для создания варианта
            if "learning_rate" in variant_hyperparams:
                variant_hyperparams["learning_rate"] *= 1.1
            else:
                variant_hyperparams["learning_rate"] = 3e-4 * 1.1
            
            variant_config = self._create_rl_config(
                name=f"{name}_variant",
                algorithm=algorithm,
                environment=environment,
                hyperparameters=variant_hyperparams,
                seed=seed
            )
            
            # Создание эксперимента
            experiment = Experiment(
                baseline_config=baseline_config,
                variant_config=variant_config,
                hypothesis=hypothesis or f"Сравнение конфигураций {algorithm} на {environment}",
                output_dir=self.config.experiments_dir
            )
            
            # Сохранение эксперимента
            self.experiments[experiment.experiment_id] = experiment
            
            logger.info(f"Создан эксперимент {experiment.experiment_id}")
            return experiment
            
        except Exception as e:
            logger.error(f"Ошибка создания эксперимента: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка создания эксперимента: {str(e)}"
            )
    
    def _create_rl_config(
        self,
        name: str,
        algorithm: str,
        environment: str,
        hyperparameters: Dict[str, Any],
        seed: int
    ) -> RLConfig:
        """Создать конфигурацию RL.
        
        Args:
            name: Название конфигурации
            algorithm: Алгоритм
            environment: Среда
            hyperparameters: Гиперпараметры
            seed: Сид
            
        Returns:
            Конфигурация RL
        """
        # Создание конфигурации алгоритма
        algo_config = AlgorithmConfig(
            name=algorithm,
            seed=seed,
            **hyperparameters
        )
        
        # Создание конфигурации среды
        env_config = EnvironmentConfig(name=environment)
        
        # Создание основной конфигурации
        return RLConfig(
            experiment_name=name,
            seed=seed,
            algorithm=algo_config,
            environment=env_config,
            output_dir=str(self.config.experiments_dir)
        )
    
    async def get_experiment(self, experiment_id: str) -> Experiment:
        """Получить эксперимент по ID.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Эксперимент
            
        Raises:
            HTTPException: Если эксперимент не найден
        """
        if experiment_id not in self.experiments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Эксперимент {experiment_id} не найден"
            )
        
        return self.experiments[experiment_id]
    
    async def list_experiments(
        self,
        status_filter: Optional[str] = None,
        algorithm_filter: Optional[str] = None
    ) -> List[Experiment]:
        """Получить список экспериментов с фильтрацией.
        
        Args:
            status_filter: Фильтр по статусу
            algorithm_filter: Фильтр по алгоритму
            
        Returns:
            Список экспериментов
        """
        experiments = list(self.experiments.values())
        
        # Фильтрация по статусу
        if status_filter:
            try:
                status_enum = ExperimentStatus(status_filter)
                experiments = [exp for exp in experiments if exp.status == status_enum]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Неверный статус: {status_filter}"
                )
        
        # Фильтрация по алгоритму
        if algorithm_filter:
            experiments = [
                exp for exp in experiments 
                if exp.baseline_config.algorithm.name == algorithm_filter
            ]
        
        return experiments
    
    async def update_experiment(
        self,
        experiment_id: str,
        status_update: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """Обновить эксперимент.
        
        Args:
            experiment_id: Идентификатор эксперимента
            status_update: Новый статус
            hyperparameters: Обновленные гиперпараметры
            
        Returns:
            Обновленный эксперимент
            
        Raises:
            HTTPException: Если обновление невозможно
        """
        experiment = await self.get_experiment(experiment_id)
        
        # Обновление статуса
        if status_update:
            try:
                if status_update == "running":
                    experiment.start()
                elif status_update == "paused":
                    experiment.pause()
                elif status_update == "cancelled":
                    experiment.stop(failed=True, error_message="Отменен пользователем")
                else:
                    raise ValueError(f"Неподдерживаемое обновление статуса: {status_update}")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ошибка обновления статуса: {str(e)}"
                )
        
        # Обновление гиперпараметров (только если эксперимент не запущен)
        if hyperparameters and experiment.status == ExperimentStatus.CREATED:
            # Обновляем гиперпараметры в конфигурациях
            for key, value in hyperparameters.items():
                if hasattr(experiment.baseline_config.algorithm, key):
                    setattr(experiment.baseline_config.algorithm, key, value)
                if hasattr(experiment.variant_config.algorithm, key):
                    setattr(experiment.variant_config.algorithm, key, value)
        
        return experiment
    
    async def start_training(self, experiment_id: str) -> Dict[str, Any]:
        """Запустить обучение для эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Информация о запуске обучения
            
        Raises:
            HTTPException: Если запуск невозможен
        """
        experiment = await self.get_experiment(experiment_id)
        
        # Проверка возможности запуска
        if experiment.status != ExperimentStatus.CREATED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Нельзя запустить эксперимент в статусе {experiment.status.value}"
            )
        
        # Проверка лимита одновременных экспериментов
        if len(self.running_experiments) >= self.config.settings.max_concurrent_experiments:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Превышен лимит одновременных экспериментов"
            )
        
        # Запуск обучения в фоновой задаче
        task = asyncio.create_task(self._run_experiment(experiment))
        self.running_experiments[experiment_id] = task
        
        # Обновление статуса эксперимента
        experiment.start()
        
        return {
            "experiment_id": experiment_id,
            "status": "running",
            "message": "Обучение запущено"
        }
    
    async def _run_experiment(self, experiment: Experiment) -> None:
        """Выполнить эксперимент в фоновом режиме.
        
        Args:
            experiment: Эксперимент для выполнения
        """
        try:
            logger.info(f"Запуск эксперимента {experiment.experiment_id}")
            
            # Создание runner'а для эксперимента
            runner = ExperimentRunner(
                baseline_config=experiment.baseline_config,
                variant_config=experiment.variant_config,
                output_dir=self.config.get_experiment_dir(experiment.experiment_id)
            )
            
            # Выполнение эксперимента
            results = await asyncio.to_thread(runner.run_experiment)
            
            # Сохранение результатов
            experiment.add_result("baseline", results.get("baseline", {}))
            experiment.add_result("variant", results.get("variant", {}))
            
            # Завершение эксперимента
            experiment.stop()
            
            logger.info(f"Эксперимент {experiment.experiment_id} завершен успешно")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения эксперимента {experiment.experiment_id}: {e}")
            experiment.stop(failed=True, error_message=str(e))
        
        finally:
            # Удаление из списка выполняющихся
            if experiment.experiment_id in self.running_experiments:
                del self.running_experiments[experiment.experiment_id]
    
    async def get_experiment_metrics(
        self,
        experiment_id: str,
        from_timestep: Optional[int] = None,
        to_timestep: Optional[int] = None
    ) -> Dict[str, Any]:
        """Получить метрики эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            from_timestep: Начальный временной шаг
            to_timestep: Конечный временной шаг
            
        Returns:
            Метрики эксперимента
        """
        experiment = await self.get_experiment(experiment_id)
        
        # Загрузка метрик из файла
        metrics_file = self.config.get_metrics_path(experiment_id)
        metrics = []
        
        if metrics_file.exists():
            try:
                import json
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        metric = json.loads(line.strip())
                        
                        # Фильтрация по временным шагам
                        timestep = metric.get('timestep', 0)
                        if from_timestep and timestep < from_timestep:
                            continue
                        if to_timestep and timestep > to_timestep:
                            continue
                        
                        metrics.append(metric)
            except Exception as e:
                logger.error(f"Ошибка загрузки метрик: {e}")
        
        # Создание сводки
        summary = {}
        if metrics:
            rewards = [m.get('reward', 0) for m in metrics if 'reward' in m]
            if rewards:
                summary = {
                    "avg_reward_last_100": sum(rewards[-100:]) / min(len(rewards), 100),
                    "total_timesteps": max(m.get('timestep', 0) for m in metrics),
                    "best_reward": max(rewards),
                    "total_episodes": len(rewards)
                }
        
        return {
            "experiment_id": experiment_id,
            "metrics": metrics,
            "summary": summary
        }
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Получить результаты эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            
        Returns:
            Результаты эксперимента
            
        Raises:
            HTTPException: Если результаты недоступны
        """
        experiment = await self.get_experiment(experiment_id)
        
        if experiment.status != ExperimentStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Эксперимент не завершен или завершился с ошибкой"
            )
        
        # Загрузка результатов из файла
        results_file = self.config.get_results_path(experiment_id)
        if results_file.exists():
            try:
                import json
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                return results
            except Exception as e:
                logger.error(f"Ошибка загрузки результатов: {e}")
        
        # Возврат результатов из объекта эксперимента
        return {
            "experiment_id": experiment_id,
            "final_evaluation": experiment.results.get("comparison", {}),
            "model_path": str(self.config.get_model_path(experiment_id)),
            "baseline_results": experiment.results.get("baseline", {}),
            "variant_results": experiment.results.get("variant", {})
        }


class EnvironmentService:
    """Сервис для управления средами обучения."""
    
    def __init__(self, config: APIConfig):
        """Инициализация сервиса сред.
        
        Args:
            config: Конфигурация API
        """
        self.config = config
        logger.info("Инициализирован сервис сред")
    
    async def list_environments(self) -> List[Dict[str, Any]]:
        """Получить список доступных сред.
        
        Returns:
            Список информации о средах
        """
        environments = []
        
        for env_name in self.config.supported_environments:
            try:
                import gymnasium as gym
                env = gym.make(env_name)
                
                env_info = {
                    "name": env_name,
                    "description": f"Среда {env_name}",
                    "observation_space": {
                        "type": str(type(env.observation_space).__name__),
                        "shape": getattr(env.observation_space, 'shape', None),
                        "dtype": str(getattr(env.observation_space, 'dtype', None))
                    },
                    "action_space": {
                        "type": str(type(env.action_space).__name__),
                        "shape": getattr(env.action_space, 'shape', None),
                        "n": getattr(env.action_space, 'n', None)
                    }
                }
                
                env.close()
                environments.append(env_info)
                
            except Exception as e:
                logger.warning(f"Не удалось загрузить информацию о среде {env_name}: {e}")
                environments.append({
                    "name": env_name,
                    "description": f"Среда {env_name} (информация недоступна)",
                    "observation_space": {},
                    "action_space": {}
                })
        
        return environments


class AlgorithmService:
    """Сервис для управления алгоритмами RL."""
    
    def __init__(self, config: APIConfig):
        """Инициализация сервиса алгоритмов.
        
        Args:
            config: Конфигурация API
        """
        self.config = config
        logger.info("Инициализирован сервис алгоритмов")
    
    async def list_algorithms(self) -> List[Dict[str, Any]]:
        """Получить список доступных алгоритмов.
        
        Returns:
            Список информации об алгоритмах
        """
        algorithms_info = {
            "PPO": {
                "name": "PPO",
                "description": "Proximal Policy Optimization - стабильный on-policy алгоритм",
                "default_hyperparameters": {
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2
                }
            },
            "A2C": {
                "name": "A2C",
                "description": "Advantage Actor-Critic - быстрый on-policy алгоритм",
                "default_hyperparameters": {
                    "learning_rate": 7e-4,
                    "n_steps": 5,
                    "gamma": 0.99,
                    "gae_lambda": 1.0,
                    "ent_coef": 0.01,
                    "vf_coef": 0.25
                }
            },
            "SAC": {
                "name": "SAC",
                "description": "Soft Actor-Critic - эффективный off-policy алгоритм",
                "default_hyperparameters": {
                    "learning_rate": 3e-4,
                    "buffer_size": 1000000,
                    "learning_starts": 100,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99
                }
            },
            "TD3": {
                "name": "TD3",
                "description": "Twin Delayed Deep Deterministic Policy Gradient",
                "default_hyperparameters": {
                    "learning_rate": 1e-3,
                    "buffer_size": 1000000,
                    "learning_starts": 100,
                    "batch_size": 100,
                    "tau": 0.005,
                    "gamma": 0.99
                }
            }
        }
        
        return [
            algorithms_info[algo] 
            for algo in self.config.supported_algorithms 
            if algo in algorithms_info
        ]


# Глобальные экземпляры сервисов
_experiment_service: Optional[ExperimentService] = None
_environment_service: Optional[EnvironmentService] = None
_algorithm_service: Optional[AlgorithmService] = None


def get_experiment_service(config: APIConfig = Depends(get_api_config)) -> ExperimentService:
    """Получить сервис экспериментов.
    
    Args:
        config: Конфигурация API
        
    Returns:
        Экземпляр ExperimentService
    """
    global _experiment_service
    
    if _experiment_service is None:
        _experiment_service = ExperimentService(config)
    
    return _experiment_service


def get_environment_service(config: APIConfig = Depends(get_api_config)) -> EnvironmentService:
    """Получить сервис сред.
    
    Args:
        config: Конфигурация API
        
    Returns:
        Экземпляр EnvironmentService
    """
    global _environment_service
    
    if _environment_service is None:
        _environment_service = EnvironmentService(config)
    
    return _environment_service


def get_algorithm_service(config: APIConfig = Depends(get_api_config)) -> AlgorithmService:
    """Получить сервис алгоритмов.
    
    Args:
        config: Конфигурация API
        
    Returns:
        Экземпляр AlgorithmService
    """
    global _algorithm_service
    
    if _algorithm_service is None:
        _algorithm_service = AlgorithmService(config)
    
    return _algorithm_service


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Проверить токен аутентификации (опционально).
    
    Args:
        credentials: Учетные данные HTTP Bearer
        
    Returns:
        Идентификатор пользователя или None
    """
    # Простая заглушка для аутентификации
    # В продакшене здесь должна быть полноценная проверка JWT токена
    if credentials is None:
        return None
    
    # Проверка токена (заглушка)
    if credentials.credentials == "test-token":
        return "test-user"
    
    return None


def get_current_user(user_id: Optional[str] = Depends(verify_token)) -> Optional[str]:
    """Получить текущего пользователя.
    
    Args:
        user_id: Идентификатор пользователя из токена
        
    Returns:
        Идентификатор пользователя или None
    """
    return user_id


def require_auth(user_id: Optional[str] = Depends(get_current_user)) -> str:
    """Требовать аутентификации.
    
    Args:
        user_id: Идентификатор пользователя
        
    Returns:
        Идентификатор пользователя
        
    Raises:
        HTTPException: Если пользователь не аутентифицирован
    """
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется аутентификация",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id