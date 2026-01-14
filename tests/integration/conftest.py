"""Конфигурация для интеграционных тестов генерации выходных данных.

Этот файл содержит общие фикстуры и настройки для интеграционных тестов,
включая настройку логирования, временных директорий и мок-объектов.
"""

import logging
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Настройка логирования для тестов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Отключаем избыточное логирование для тестов
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_session_dir() -> Generator[Path, None, None]:
    """Создание временной директории для всей сессии тестов."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_session_output_generation_"))
    yield temp_dir
    # Очистка после сессии (можно закомментировать для отладки)
    # if temp_dir.exists():
    #     shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Автоматическая настройка окружения для каждого теста."""
    # Устанавливаем семя для воспроизводимости
    import numpy as np
    np.random.seed(42)
    
    # Настройка matplotlib для headless режима
    import matplotlib
    matplotlib.use('Agg')
    
    yield
    
    # Очистка после теста
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture
def mock_logger():
    """Мок логгера для тестирования."""
    import logging
    from unittest.mock import MagicMock
    
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()
    
    return mock_logger


# Маркеры для pytest
def pytest_configure(config):
    """Конфигурация pytest с пользовательскими маркерами."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


# Параметры для пропуска тестов
def pytest_collection_modifyitems(config, items):
    """Модификация коллекции тестов для условного пропуска."""
    # Пропускаем GPU тесты если CUDA недоступна
    try:
        import torch
        if not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="CUDA not available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
    except ImportError:
        skip_gpu = pytest.mark.skip(reason="PyTorch not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)