"""Тесты для модуля seeding.py."""

import pytest
import numpy as np
import torch
import random

from src.utils.seeding import (
    set_seed, 
    get_random_seed, 
    create_deterministic_env_seed,
    verify_reproducibility,
    SeedManager
)


def test_set_seed():
    """Тест установки глобального seed."""
    seed = 42
    set_seed(seed)
    
    # Проверяем, что генераторы дают одинаковые результаты
    np_val1 = np.random.random()
    py_val1 = random.random()
    torch_val1 = torch.rand(1).item()
    
    # Устанавливаем тот же seed снова
    set_seed(seed)
    
    np_val2 = np.random.random()
    py_val2 = random.random()
    torch_val2 = torch.rand(1).item()
    
    # Значения должны быть идентичными
    assert np_val1 == np_val2
    assert py_val1 == py_val2
    assert torch_val1 == torch_val2


def test_set_seed_invalid():
    """Тест обработки невалидных seed."""
    with pytest.raises(ValueError):
        set_seed(-1)
    
    with pytest.raises(ValueError):
        set_seed(2**32)
    
    with pytest.raises(ValueError):
        set_seed("invalid")


def test_get_random_seed():
    """Тест генерации случайного seed."""
    seed1 = get_random_seed()
    seed2 = get_random_seed()
    
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)
    assert 0 <= seed1 <= 2**32 - 1
    assert 0 <= seed2 <= 2**32 - 1
    # Вероятность получить одинаковые seed очень мала
    assert seed1 != seed2


def test_create_deterministic_env_seed():
    """Тест создания детерминистических seed для сред."""
    base_seed = 42
    
    seed1 = create_deterministic_env_seed(base_seed, 0)
    seed2 = create_deterministic_env_seed(base_seed, 1)
    seed3 = create_deterministic_env_seed(base_seed, 2)
    
    assert seed1 == 42
    assert seed2 == 43
    assert seed3 == 44
    
    # Проверяем переполнение
    large_base = 2**32 - 1
    seed_overflow = create_deterministic_env_seed(large_base, 1)
    assert seed_overflow == 0  # Должно быть 0 из-за модуля


def test_verify_reproducibility():
    """Тест проверки воспроизводимости."""
    seed = 123
    
    # Должно вернуть True для корректного seed
    assert verify_reproducibility(seed, test_operations=10)
    
    # Проверяем с большим количеством операций
    assert verify_reproducibility(seed, test_operations=100)


def test_seed_manager():
    """Тест класса SeedManager."""
    base_seed = 42
    manager = SeedManager(base_seed)
    
    assert manager.base_seed == base_seed
    
    # Тест установки seed для эксперимента
    seed1 = manager.set_experiment_seed("exp1")
    assert seed1 == base_seed
    
    # Тест получения следующего seed
    next_seed = manager.get_next_seed()
    assert next_seed == base_seed + 1
    
    seed2 = manager.set_experiment_seed("exp2")
    assert seed2 == base_seed + 1
    
    # Тест сброса к базовому
    manager.reset_to_base()
    assert manager._current_seed == base_seed
    
    # Тест истории
    history = manager.get_seed_history()
    assert len(history) == 2
    assert history == [base_seed, base_seed + 1]


def test_seed_manager_auto_seed():
    """Тест SeedManager с автоматическим seed."""
    manager = SeedManager()
    
    assert isinstance(manager.base_seed, int)
    assert 0 <= manager.base_seed <= 2**32 - 1


def test_reproducibility_across_modules():
    """Тест воспроизводимости между различными модулями."""
    seed = 999
    
    # Первый прогон
    set_seed(seed)
    results1 = []
    for _ in range(10):
        results1.append(np.random.randint(0, 1000))
        results1.append(random.randint(0, 1000))
        results1.append(int(torch.randint(0, 1000, (1,)).item()))
    
    # Второй прогон с тем же seed
    set_seed(seed)
    results2 = []
    for _ in range(10):
        results2.append(np.random.randint(0, 1000))
        results2.append(random.randint(0, 1000))
        results2.append(int(torch.randint(0, 1000, (1,)).item()))
    
    # Результаты должны быть идентичными
    assert results1 == results2