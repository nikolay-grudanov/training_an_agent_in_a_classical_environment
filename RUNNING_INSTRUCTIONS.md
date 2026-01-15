# Инструкция по запуску приложения RL Agent Training System

## Содержание
1. [Требования к системе](#требования-к-системе)
2. [Установка и настройка окружения](#установка-и-настройка-окружения)
3. [Запуск API сервера](#запуск-api-сервера)
4. [Запуск обучения агента](#запуск-обучения-агента)
5. [Использование Jupyter notebook](#использование-jupyter-notebook)
6. [Проверка результатов](#проверка-результатов)
7. [Примеры использования API](#примеры-использования-api)
8. [Возможные проблемы и их решения](#возможные-проблемы-и-их-решения)

## Требования к системе

- Linux (проверено на Ubuntu 20.04+)
- Python 3.10.14
- Conda (рекомендуется Miniconda или Anaconda)
- Свободное место на диске: минимум 2 ГБ
- RAM: минимум 4 ГБ (рекомендуется 8 ГБ)
- CPU с поддержкой AVX (для PyTorch)

## Установка и настройка окружения

### 1. Клонирование репозитория (если применимо)
```bash
git clone <repository-url>
cd training_an_agent_in_a_classical_environment
```

### 2. Установка зависимостей через conda
```bash
# Убедитесь, что conda установлен
conda --version

# Создание и активация окружения
conda env create -f environment.yml
conda activate rocm
```

### 3. Проверка установки зависимостей
```bash
# Проверка версий основных библиотек
python -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)"
python -c "import stable_baselines3; print('SB3 version:', stable_baselines3.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Запуск API сервера

### 1. Запуск сервера в консоли
```bash
# Активация окружения
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rocm
cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment

# Запуск API сервера
python -m src.api.app --host 0.0.0.0 --port 8000 --debug
```

### 2. Проверка работоспособности API
Откройте в браузере: `http://localhost:8000/health`

Вы должны увидеть JSON-ответ с информацией о здоровье сервера:
```json
{
  "status": "healthy",
  "timestamp": "...",
  "version": "1.0.0",
  "uptime_seconds": ...,
  "active_experiments": 0
}
```

### 3. Документация API
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Запуск обучения агента

### 1. Запуск примера базового использования
```bash
# В новом терминале (не закрывая API сервер)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rocm
cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment

# Запуск базового примера
python examples/basic_usage.py
```

### 2. Запуск эксперимента
```bash
# Запуск примера эксперимента
PYTHONPATH=. python examples/experiment_example.py
```

### 3. Запуск обучения конкретного агента (например, PPO)
```bash
# Убедитесь, что PYTHONPATH установлен
export PYTHONPATH=.

# Запуск примера с PPO (если доступен)
# Обратите внимание: могут быть ошибки в некоторых примерах из-за проблем с импортом
python examples/basic_usage.py
```

## Использование Jupyter notebook

### 1. Запуск Jupyter сервера
```bash
# В новом терминале
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rocm
cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment

# Запуск Jupyter
jupyter lab
# или
jupyter notebook
```

### 2. Открытие и использование ноутбука
1. В браузере откройте URL, который выдал Jupyter (обычно `http://localhost:8888`)
2. Перейдите в директорию `notebooks/`
3. Откройте файл `1.ipynb`
4. Выполните ячейки по порядку (Shift+Enter для выполнения одной ячейки)

### 3. Создание нового ноутбука для экспериментов
1. В Jupyter интерфейсе нажмите "New" → "Python 3"
2. Импортируйте необходимые модули:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from src.agents.ppo_agent import PPOAgent, PPOConfig
```

## Проверка результатов

### 1. Проверка сгенерированных файлов
```bash
# Проверка директории с результатами
ls -la results/

# Проверка экспериментов
ls -la results/experiments/

# Проверка моделей
ls -la results/models/

# Проверка логов
ls -la results/logs/
```

### 2. Просмотр результатов в браузере
Некоторые результаты могут быть представлены в виде HTML файлов:
```bash
# Найти HTML файлы с результатами
find results/ -name "*.html" -type f
```

### 3. Проверка метрик
```bash
# Проверка файлов метрик
find results/ -name "*metrics*" -type f
```

## Примеры использования API

### 1. Создание эксперимента через API
```bash
curl -X POST "http://localhost:8000/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_ppo_experiment",
    "algorithm": "PPO",
    "environment": "LunarLander-v3",
    "hyperparameters": {
      "learning_rate": 0.0003,
      "n_steps": 2048
    },
    "seed": 42,
    "description": "Тестовый эксперимент с PPO",
    "hypothesis": "PPO покажет хорошую производительность на LunarLander-v3"
  }'
```

### 2. Получение списка экспериментов
```bash
curl "http://localhost:8000/experiments"
```

### 3. Запуск обучения эксперимента
```bash
# Замените {experiment_id} на фактический ID эксперимента
curl -X POST "http://localhost:8000/experiments/{experiment_id}/train"
```

### 4. Получение результатов эксперимента
```bash
# Замените {experiment_id} на фактический ID эксперимента
curl "http://localhost:8000/experiments/{experiment_id}/results"
```

## Возможные проблемы и их решения

### 1. Ошибка при запуске визуализации
**Проблема**: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`
**Решение**: Визуализация требует GUI и не работает в консольных системах. Используйте только для генерации файлов, а не для отображения в реальном времени.

### 2. Ошибка импорта модулей
**Проблема**: `ModuleNotFoundError: No module named 'src'`
**Решение**: Убедитесь, что переменная `PYTHONPATH` включает текущую директорию:
```bash
export PYTHONPATH=.:$PYTHONPATH
```

### 3. Ошибка с CheckpointManager
**Проблема**: `TypeError: CheckpointManager.__init__() missing 1 required positional argument: 'experiment_id'`
**Решение**: Это внутренняя ошибка в тестах, не влияющая на основную функциональность.

### 4. Ошибка с Box2D
**Проблема**: `ModuleNotFoundError: No module named 'Box2D'`
**Решение**: Установите зависимости для Gymnasium:
```bash
pip install "gymnasium[box2d]"
```

### 5. Ошибка с GPU
**Проблема**: Ошибки, связанные с CUDA или ROCm
**Решение**: Приложение настроено для работы с ROCm, но может работать и на CPU. Убедитесь, что PyTorch корректно установлен для вашей системы.

### 6. Ошибка с портом
**Проблема**: `OSError: [Errno 98] Address already in use`
**Решение**: Используйте другой порт или убейте процесс, занимающий порт:
```bash
# Найти процесс, использующий порт 8000
lsof -i :8000
# Убить процесс
kill -9 <PID>
```

### 7. Ошибка с зависимостями
**Проблема**: `ImportError` или `ModuleNotFoundError`
**Решение**: Убедитесь, что вы активировали правильное окружение conda:
```bash
conda activate rocm
conda list  # Проверить установленные пакеты
```

## Дополнительные советы

1. **Для разработки**: используйте флаг `--debug` при запуске API для получения более подробных сообщений об ошибках.

2. **Для мониторинга**: проверяйте логи в `results/logs/` для отслеживания прогресса обучения.

3. **Для воспроизводимости**: всегда используйте фиксированные seed значения в конфигурациях.

4. **Для производительности**: убедитесь, что у вас достаточно RAM и CPU ресурсов для обучения RL моделей.

5. **Для экспериментов**: используйте систему экспериментов для проведения контролируемых сравнений между разными конфигурациями.

## Заключение

После выполнения всех шагов вы должны иметь полностью работающее приложение для обучения RL агентов. API сервер будет доступен для управления экспериментами, а примеры кода позволят начать обучение собственных агентов.