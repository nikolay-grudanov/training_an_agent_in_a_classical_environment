"""Утилиты для структурированного логирования в RL экспериментах.

Этот модуль предоставляет настройку логирования с интеграцией в Stable-Baselines3,
поддержку JSON-структурированных логов для экспериментов, ротацию логов и
различные уровни логирования для разных компонентов.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    # Заглушка для случая, когда SB3 не установлен
    class BaseCallback:
        def __init__(self):
            self.num_timesteps = 0
            self.model = None
        
        def _on_step(self) -> bool:
            return True
        
        def _on_training_end(self) -> None:
            pass
    
    SB3_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """Форматтер для JSON-структурированных логов."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматировать запись лога в JSON.
        
        Args:
            record: Запись лога
            
        Returns:
            JSON-строка с информацией о логе
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Добавляем дополнительные поля если они есть
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = getattr(record, 'experiment_id')
        if hasattr(record, 'episode'):
            log_entry['episode'] = getattr(record, 'episode')
        if hasattr(record, 'timestep'):
            log_entry['timestep'] = getattr(record, 'timestep')
        if hasattr(record, 'reward'):
            log_entry['reward'] = getattr(record, 'reward')
        if hasattr(record, 'loss'):
            log_entry['loss'] = getattr(record, 'loss')
            
        # Добавляем информацию об исключении если есть
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


class ExperimentLoggerAdapter(logging.LoggerAdapter):
    """Адаптер логгера для добавления контекста эксперимента."""
    
    def __init__(self, logger: logging.Logger, experiment_id: str):
        """Инициализация адаптера.
        
        Args:
            logger: Базовый логгер
            experiment_id: Идентификатор эксперимента
        """
        super().__init__(logger, {'experiment_id': experiment_id})
        self.experiment_id = experiment_id
    
    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        """Обработать сообщение лога, добавив контекст эксперимента.
        
        Args:
            msg: Сообщение лога
            kwargs: Дополнительные аргументы
            
        Returns:
            Обработанное сообщение и аргументы
        """
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs
    
    def log_training_step(self, timestep: int, episode: int, reward: float, 
                         loss: Optional[float] = None, **kwargs: Any) -> None:
        """Логировать шаг обучения.
        
        Args:
            timestep: Номер временного шага
            episode: Номер эпизода
            reward: Полученная награда
            loss: Значение функции потерь (опционально)
            **kwargs: Дополнительные метрики
        """
        extra = {
            'timestep': timestep,
            'episode': episode,
            'reward': reward,
        }
        if loss is not None:
            extra['loss'] = loss
        extra.update(kwargs)
        
        self.info(f"Training step - Episode: {episode}, Timestep: {timestep}, "
                 f"Reward: {reward:.4f}", extra=extra)


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    json_format: bool = False,
    experiment_id: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Настроить систему логирования для RL экспериментов.
    
    Args:
        log_level: Уровень логирования
        log_dir: Директория для файлов логов
        console_output: Выводить ли логи в консоль
        json_format: Использовать ли JSON формат для файловых логов
        experiment_id: Идентификатор эксперимента
        max_bytes: Максимальный размер файла лога в байтах
        backup_count: Количество резервных копий логов
        
    Returns:
        Настроенный логгер
    """
    # Создаем основной логгер
    logger = logging.getLogger('rl_training')
    logger.setLevel(log_level)
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Настройка форматтеров
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = JSONFormatter() if json_format else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный вывод
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Основной файл логов
        log_filename = f"training_{experiment_id}.log" if experiment_id else "training.log"
        log_file = log_dir / log_filename
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Отдельный файл для ошибок
        error_file = log_dir / f"errors_{experiment_id}.log" if experiment_id else log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    logger.info(f"Логирование настроено. Уровень: {logging.getLevelName(log_level)}")
    if log_dir:
        logger.info(f"Логи сохраняются в: {log_dir}")
    
    return logger


def get_experiment_logger(experiment_id: str, 
                         base_logger: Optional[logging.Logger] = None) -> ExperimentLoggerAdapter:
    """Получить логгер для конкретного эксперимента.
    
    Args:
        experiment_id: Идентификатор эксперимента
        base_logger: Базовый логгер (если None, используется корневой)
        
    Returns:
        Адаптер логгера с контекстом эксперимента
    """
    if base_logger is None:
        base_logger = logging.getLogger('rl_training')
    
    return ExperimentLoggerAdapter(base_logger, experiment_id)


class TrainingCallback(BaseCallback):
    """Callback для интеграции логирования с Stable-Baselines3."""
    
    def __init__(self, logger: ExperimentLoggerAdapter, log_freq: int = 1000):
        """Инициализация callback'а.
        
        Args:
            logger: Логгер для записи метрик
            log_freq: Частота логирования (каждые N шагов)
        """
        super().__init__()
        self.logger = logger
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Вызывается на каждом шаге обучения.
        
        Returns:
            True для продолжения обучения
        """
        # Логируем каждые log_freq шагов
        if self.num_timesteps % self.log_freq == 0:
            # Получаем метрики из буфера модели
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                reward = ep_info.get('r', 0.0)
                length = ep_info.get('l', 0)
                
                self.logger.log_training_step(
                    timestep=self.num_timesteps,
                    episode=len(self.model.ep_info_buffer),
                    reward=reward,
                    episode_length=length
                )
        
        return True
    
    def _on_training_end(self) -> None:
        """Вызывается в конце обучения."""
        self.logger.info(f"Обучение завершено. Всего шагов: {self.num_timesteps}")
        """Callback для интеграции логирования с Stable-Baselines3."""
        
        def __init__(self, logger: ExperimentLoggerAdapter, log_freq: int = 1000):
            """Инициализация callback'а.
            
            Args:
                logger: Логгер для записи метрик
                log_freq: Частота логирования (каждые N шагов)
            """
            super().__init__()
            self.logger = logger
            self.log_freq = log_freq
            self.episode_rewards = []
            self.episode_lengths = []
        
        def _on_step(self) -> bool:
            """Вызывается на каждом шаге обучения.
            
            Returns:
                True для продолжения обучения
            """
            # Логируем каждые log_freq шагов
            if self.num_timesteps % self.log_freq == 0:
                # Получаем метрики из буфера модели
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    ep_info = self.model.ep_info_buffer[-1]
                    reward = ep_info.get('r', 0.0)
                    length = ep_info.get('l', 0)
                    
                    self.logger.log_training_step(
                        timestep=self.num_timesteps,
                        episode=len(self.model.ep_info_buffer),
                        reward=reward,
                        episode_length=length
                    )
            
            return True
        
        def _on_training_end(self) -> None:
            """Вызывается в конце обучения."""
            self.logger.info(f"Обучение завершено. Всего шагов: {self.num_timesteps}")


class MetricsLogger:
    """Логгер для метрик обучения с поддержкой различных форматов."""
    
    def __init__(self, log_dir: Union[str, Path], experiment_id: str):
        """Инициализация логгера метрик.
        
        Args:
            log_dir: Директория для сохранения метрик
            experiment_id: Идентификатор эксперимента
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        
        # Файлы для различных типов метрик
        self.metrics_file = self.log_dir / f"metrics_{experiment_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{experiment_id}.json"
        
        self.metrics: list[Dict[str, Any]] = []
    
    def log_metric(self, timestep: int, **metrics: Any) -> None:
        """Записать метрику.
        
        Args:
            timestep: Временной шаг
            **metrics: Метрики для записи
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'timestep': timestep,
            'experiment_id': self.experiment_id,
            **metrics
        }
        
        self.metrics.append(entry)
        
        # Записываем в файл (JSONL формат)
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def save_summary(self, **summary_data: Any) -> None:
        """Сохранить итоговую сводку эксперимента.
        
        Args:
            **summary_data: Данные для сводки
        """
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(self.metrics),
            **summary_data
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def get_metrics(self) -> list[Dict[str, Any]]:
        """Получить все записанные метрики.
        
        Returns:
            Список всех метрик
        """
        return self.metrics.copy()


def cleanup_old_logs(log_dir: Union[str, Path], days_to_keep: int = 30) -> None:
    """Очистить старые файлы логов.
    
    Args:
        log_dir: Директория с логами
        days_to_keep: Количество дней для хранения логов
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return
    
    import time
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    deleted_count = 0
    for log_file in log_dir.glob('*.log*'):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        logger = logging.getLogger('rl_training')
        logger.info(f"Удалено {deleted_count} старых файлов логов")


def get_logger(name: str) -> logging.Logger:
    """Получить логгер с указанным именем.
    
    Args:
        name: Имя логгера
        
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)





# Настройка логирования по умолчанию
def configure_default_logging() -> logging.Logger:
    """Настроить логирование по умолчанию для быстрого старта.
    
    Returns:
        Настроенный логгер
    """
    return setup_logging(
        log_level=logging.INFO,
        console_output=True,
        json_format=False
    )