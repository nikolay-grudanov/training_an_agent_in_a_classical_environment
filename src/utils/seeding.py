"""Утилиты для обеспечения воспроизводимости экспериментов RL.

Этот модуль предоставляет функции для установки глобальных seed'ов
для всех используемых библиотек: NumPy, PyTorch, Gymnasium, и Python random.
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Установить глобальный seed для воспроизводимости экспериментов.
    
    Устанавливает seed для всех основных библиотек:
    - Python random
    - NumPy
    - PyTorch (CPU и CUDA)
    - Gymnasium (через переменную окружения)
    
    Args:
        seed: Значение seed (0-2**32-1)
        
    Raises:
        ValueError: Если seed выходит за допустимые пределы
        
    Example:
        >>> set_seed(42)
        >>> # Теперь все случайные операции будут воспроизводимы
    """
    if not isinstance(seed, int):
        raise ValueError(f"Seed должен быть целым числом, получен {type(seed)}")
    
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(f"Seed должен быть в диапазоне [0, 2^32-1], получен {seed}")
    
    logger.info(f"Установка глобального seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Для детерминистических операций PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Gymnasium через переменную окружения
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.debug("Глобальный seed успешно установлен")


def get_random_seed() -> int:
    """Получить случайный seed для использования в экспериментах.
    
    Returns:
        Случайное значение seed в допустимом диапазоне
    """
    return random.randint(0, 2**32 - 1)


def create_deterministic_env_seed(base_seed: int, env_index: int = 0) -> int:
    """Создать детерминистический seed для конкретной среды.
    
    Полезно при параллельном обучении нескольких агентов или
    при создании множественных экземпляров среды.
    
    Args:
        base_seed: Базовый seed эксперимента
        env_index: Индекс среды (для различения параллельных сред)
        
    Returns:
        Детерминистический seed для конкретной среды
        
    Example:
        >>> base = 42
        >>> env1_seed = create_deterministic_env_seed(base, 0)  # 42
        >>> env2_seed = create_deterministic_env_seed(base, 1)  # 43
    """
    return (base_seed + env_index) % (2**32)


def verify_reproducibility(seed: int, test_operations: int = 100) -> bool:
    """Проверить, что установка seed обеспечивает воспроизводимость.
    
    Выполняет серию случайных операций дважды с одним seed'ом
    и проверяет идентичность результатов.
    
    Args:
        seed: Seed для тестирования
        test_operations: Количество тестовых операций
        
    Returns:
        True если воспроизводимость обеспечена, False иначе
    """
    # Первый прогон
    set_seed(seed)
    results1 = []
    for _ in range(test_operations):
        results1.append(np.random.random())
        results1.append(random.random())
        results1.append(torch.rand(1).item())
    
    # Второй прогон с тем же seed
    set_seed(seed)
    results2 = []
    for _ in range(test_operations):
        results2.append(np.random.random())
        results2.append(random.random())
        results2.append(torch.rand(1).item())
    
    # Проверка идентичности
    is_reproducible = np.allclose(results1, results2, rtol=1e-15, atol=1e-15)
    
    if is_reproducible:
        logger.info(f"Воспроизводимость подтверждена для seed {seed}")
    else:
        logger.error(f"Воспроизводимость НЕ обеспечена для seed {seed}")
    
    return is_reproducible


class SeedManager:
    """Менеджер для управления seed'ами в экспериментах."""
    
    def __init__(self, base_seed: Optional[int] = None):
        """Инициализация менеджера seed'ов.
        
        Args:
            base_seed: Базовый seed. Если None, будет сгенерирован случайный
        """
        self.base_seed = base_seed or get_random_seed()
        self._current_seed = self.base_seed
        self._seed_history: list[int] = []
        
        logger.info(f"Инициализирован SeedManager с базовым seed: {self.base_seed}")
    
    def set_experiment_seed(self, experiment_id: Optional[str] = None) -> int:
        """Установить seed для конкретного эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента (для логирования)
            
        Returns:
            Установленный seed
        """
        seed = self._current_seed
        set_seed(seed)
        self._seed_history.append(seed)
        
        exp_info = f" для эксперимента '{experiment_id}'" if experiment_id else ""
        logger.info(f"Установлен seed {seed}{exp_info}")
        
        return seed
    
    def get_next_seed(self) -> int:
        """Получить следующий детерминистический seed.
        
        Returns:
            Следующий seed в последовательности
        """
        self._current_seed = (self._current_seed + 1) % (2**32)
        return self._current_seed
    
    def reset_to_base(self) -> None:
        """Сбросить текущий seed к базовому значению."""
        self._current_seed = self.base_seed
        logger.debug(f"Seed сброшен к базовому значению: {self.base_seed}")
    
    def get_seed_history(self) -> list[int]:
        """Получить историю использованных seed'ов.
        
        Returns:
            Список всех использованных seed'ов
        """
        return self._seed_history.copy()