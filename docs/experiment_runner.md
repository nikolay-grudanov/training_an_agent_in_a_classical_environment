# ExperimentRunner: Оркестратор RL Экспериментов

## Обзор

`ExperimentRunner` - это комплексный оркестратор для выполнения контролируемых экспериментов в области обучения с подкреплением (RL). Он управляет полным жизненным циклом эксперимента, включая настройку, выполнение, мониторинг и сбор результатов для baseline и variant конфигураций.

## Основные возможности

### 🚀 Режимы выполнения
- **Последовательный**: Baseline → Variant (безопасно, меньше ресурсов)
- **Параллельный**: Baseline ∥ Variant (быстрее, больше ресурсов)
- **Валидация**: Проверка конфигураций без обучения

### 📊 Мониторинг и контроль
- Мониторинг ресурсов в реальном времени (CPU, память)
- Прогресс-бары и статистика выполнения
- Автоматическое создание чекпоинтов
- Обработка прерываний и восстановление

### 🛠️ Обработка ошибок
- Graceful handling неудачных обучений
- Стратегии восстановления (abort, retry, skip)
- Детальное логирование ошибок
- Сохранение промежуточных результатов

### 🔧 Интеграция
- Полная интеграция с `Trainer` и `Experiment`
- CLI интерфейс для запуска из командной строки
- Поддержка конфигурационных файлов
- Экспорт результатов в различных форматах

## Быстрый старт

### Базовое использование

```python
from src.experiments.experiment import Experiment
from src.experiments.runner import ExperimentRunner, ExecutionMode

# Создание эксперимента (baseline + variant конфигурации)
experiment = Experiment(
    baseline_config=baseline_config,
    variant_config=variant_config,
    hypothesis="Variant покажет лучшие результаты"
)

# Создание и запуск runner'а
runner = ExperimentRunner(
    experiment=experiment,
    execution_mode=ExecutionMode.SEQUENTIAL
)

success = runner.run()
```

### CLI использование

```bash
# Запуск эксперимента из файла
python -m src.experiments.runner --config experiment_config.json

# Параллельное выполнение
python -m src.experiments.runner \
    --config experiment_config.json \
    --mode parallel \
    --max-workers 2

# Режим валидации
python -m src.experiments.runner \
    --config experiment_config.json \
    --mode validation
```

## Архитектура

### Основные компоненты

```
ExperimentRunner
├── Experiment (управление экспериментом)
├── Trainer (выполнение обучения)
├── ProgressInfo (отслеживание прогресса)
├── ResourceUsage (мониторинг ресурсов)
└── CheckpointManager (управление чекпоинтами)
```

### Жизненный цикл выполнения

```
1. Инициализация
   ├── Валидация эксперимента
   ├── Настройка среды
   └── Запуск мониторинга

2. Выполнение конфигураций
   ├── Baseline обучение
   ├── Variant обучение
   └── Сбор результатов

3. Анализ и завершение
   ├── Сравнение результатов
   ├── Генерация отчетов
   └── Очистка ресурсов
```

## Подробное API

### ExperimentRunner

#### Инициализация

```python
runner = ExperimentRunner(
    experiment: Experiment,                    # Эксперимент для выполнения
    execution_mode: ExecutionMode = SEQUENTIAL, # Режим выполнения
    max_workers: Optional[int] = None,         # Количество воркеров
    enable_monitoring: bool = True,            # Мониторинг ресурсов
    checkpoint_frequency: int = 10000,         # Частота чекпоинтов
    resource_limits: Optional[Dict] = None     # Лимиты ресурсов
)
```

#### Основные методы

```python
# Выполнение полного эксперимента
success: bool = runner.run()

# Выполнение одной конфигурации
result: TrainingResult = runner.run_configuration(
    config_type: str,           # "baseline" или "variant"
    config: Configuration,      # Конфигурация для выполнения
    trainer_config: Optional[TrainerConfig] = None
)

# Настройка среды
runner.setup_environment()

# Мониторинг прогресса
progress: ProgressInfo = runner.monitor_progress()

# Обработка ошибок
success: bool = runner.handle_failure(
    error: Exception,
    config_type: Optional[str] = None,
    recovery_strategy: str = "abort"  # "abort", "retry", "skip"
)

# Получение статуса
status: Dict[str, Any] = runner.get_status()

# Очистка ресурсов
runner.cleanup()
```

### ExecutionMode

```python
class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"    # Последовательное выполнение
    PARALLEL = "parallel"       # Параллельное выполнение
    VALIDATION = "validation"   # Режим валидации
```

### ProgressInfo

```python
@dataclass
class ProgressInfo:
    current_step: int = 0                    # Текущий шаг
    total_steps: int = 0                     # Общее количество шагов
    current_phase: str = "idle"              # Текущая фаза
    baseline_progress: float = 0.0           # Прогресс baseline (%)
    variant_progress: float = 0.0            # Прогресс variant (%)
    estimated_time_remaining: Optional[float] = None  # Оставшееся время
    current_config: Optional[str] = None     # Текущая конфигурация
    
    @property
    def overall_progress(self) -> float:     # Общий прогресс (%)
```

### ResourceUsage

```python
@dataclass
class ResourceUsage:
    cpu_percent: float = 0.0      # Использование CPU (%)
    memory_percent: float = 0.0   # Использование памяти (%)
    memory_mb: float = 0.0        # Память в MB
    disk_usage_mb: float = 0.0    # Использование диска в MB
    gpu_usage: Optional[float] = None  # Использование GPU (%)
    
    @classmethod
    def current(cls) -> "ResourceUsage":  # Текущее использование
```

## Примеры использования

### 1. Последовательное выполнение

```python
from src.experiments.runner import ExperimentRunner, ExecutionMode

runner = ExperimentRunner(
    experiment=experiment,
    execution_mode=ExecutionMode.SEQUENTIAL,
    enable_monitoring=True
)

# Выполнение с отслеживанием прогресса
success = runner.run()

if success:
    print(f"Baseline: {runner.baseline_result.final_mean_reward:.2f}")
    print(f"Variant: {runner.variant_result.final_mean_reward:.2f}")
```

### 2. Параллельное выполнение

```python
runner = ExperimentRunner(
    experiment=experiment,
    execution_mode=ExecutionMode.PARALLEL,
    max_workers=2,
    resource_limits={
        "memory_mb": 8192,
        "cpu_percent": 80.0
    }
)

success = runner.run()
```

### 3. Мониторинг в реальном времени

```python
import threading
import time

def monitor_progress(runner):
    while runner.status != RunnerStatus.COMPLETED:
        progress = runner.monitor_progress()
        status = runner.get_status()
        
        print(f"Phase: {progress.current_phase}")
        print(f"Baseline: {progress.baseline_progress:.1f}%")
        print(f"Variant: {progress.variant_progress:.1f}%")
        print(f"CPU: {status['resource_usage']['cpu_percent']:.1f}%")
        print(f"Memory: {status['resource_usage']['memory_mb']:.1f}MB")
        
        time.sleep(5)

# Запуск мониторинга в отдельном потоке
monitor_thread = threading.Thread(target=monitor_progress, args=(runner,))
monitor_thread.start()

# Выполнение эксперимента
success = runner.run()

monitor_thread.join()
```

### 4. Обработка ошибок

```python
class CustomExperimentRunner(ExperimentRunner):
    def handle_failure(self, error, config_type=None, recovery_strategy="retry"):
        self.logger.error(f"Ошибка в {config_type}: {error}")
        
        if isinstance(error, MemoryError):
            # Уменьшаем batch_size и повторяем
            return self._retry_with_smaller_batch(config_type)
        elif isinstance(error, TimeoutError):
            # Увеличиваем timeout и повторяем
            return self._retry_with_longer_timeout(config_type)
        else:
            # Стандартная обработка
            return super().handle_failure(error, config_type, recovery_strategy)
```

### 5. Режим валидации

```python
# Быстрая проверка конфигураций без обучения
runner = ExperimentRunner(
    experiment=experiment,
    execution_mode=ExecutionMode.VALIDATION
)

is_valid = runner.run()

if is_valid:
    print("✅ Конфигурации валидны, можно запускать обучение")
else:
    print("❌ Найдены ошибки в конфигурациях")
```

## CLI интерфейс

### Основные команды

```bash
# Базовый запуск
python -m src.experiments.runner --config experiment.json

# Параллельное выполнение с 4 воркерами
python -m src.experiments.runner \
    --config experiment.json \
    --mode parallel \
    --max-workers 4

# Валидация конфигураций
python -m src.experiments.runner \
    --config experiment.json \
    --mode validation

# Запуск с отключенным мониторингом
python -m src.experiments.runner \
    --config experiment.json \
    --no-monitoring

# Запуск существующего эксперимента по ID
python -m src.experiments.runner \
    --experiment-id abc123-def456 \
    --output-dir results/experiments

# Детальный вывод
python -m src.experiments.runner \
    --config experiment.json \
    --verbose -v
```

### Параметры CLI

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--config` | Путь к файлу конфигурации | - |
| `--experiment-id` | ID существующего эксперимента | - |
| `--mode` | Режим выполнения (sequential/parallel/validation) | sequential |
| `--max-workers` | Количество воркеров для параллельного режима | 2 |
| `--no-monitoring` | Отключить мониторинг ресурсов | False |
| `--output-dir` | Директория для результатов | results/experiments |
| `--verbose, -v` | Уровень детализации (можно повторять) | 0 |

## Конфигурация

### Файл конфигурации эксперимента

```json
{
  "experiment_id": "ppo_learning_rate_comparison",
  "hypothesis": "Увеличенный learning rate ускорит обучение",
  "baseline_config": {
    "algorithm": "PPO",
    "environment": "LunarLander-v2",
    "hyperparameters": {
      "learning_rate": 3e-4,
      "n_steps": 2048,
      "batch_size": 64
    },
    "training_steps": 100000,
    "seed": 42
  },
  "variant_config": {
    "algorithm": "PPO",
    "environment": "LunarLander-v2",
    "hyperparameters": {
      "learning_rate": 1e-3,
      "n_steps": 2048,
      "batch_size": 64
    },
    "training_steps": 100000,
    "seed": 42
  }
}
```

### Лимиты ресурсов

```python
resource_limits = {
    "memory_mb": 8192,      # Максимум 8GB памяти
    "cpu_percent": 90.0,    # Максимум 90% CPU
    "disk_gb": 10.0,        # Максимум 10GB диска
}

runner = ExperimentRunner(
    experiment=experiment,
    resource_limits=resource_limits
)
```

## Мониторинг и логирование

### Структура логов

```
results/experiments/{experiment_id}/
├── logs/
│   ├── experiment.log          # Основной лог эксперимента
│   ├── baseline_training.log   # Лог обучения baseline
│   ├── variant_training.log    # Лог обучения variant
│   └── runner.log             # Лог runner'а
├── checkpoints/
│   ├── runner_checkpoint_10000.pkl
│   └── runner_checkpoint_20000.pkl
├── models/
│   ├── baseline_model_final.zip
│   └── variant_model_final.zip
└── runner_final_state.yaml    # Финальное состояние
```

### Метрики мониторинга

```python
# Получение детальных метрик
status = runner.get_status()

print(f"Статус: {status['status']}")
print(f"Режим: {status['execution_mode']}")
print(f"Прогресс: {status['progress']['overall']:.1f}%")
print(f"CPU: {status['resource_usage']['cpu_percent']:.1f}%")
print(f"Память: {status['resource_usage']['memory_mb']:.1f}MB")
print(f"Время выполнения: {status['execution_time']:.1f}с")
```

## Производительность и оптимизация

### Рекомендации по производительности

1. **Параллельное выполнение**
   - Используйте для независимых конфигураций
   - Учитывайте ограничения памяти
   - Оптимально: 2 воркера (baseline + variant)

2. **Управление памятью**
   - Устанавливайте разумные лимиты
   - Мониторьте использование в реальном времени
   - Используйте чекпоинты для больших экспериментов

3. **Оптимизация I/O**
   - Используйте SSD для чекпоинтов
   - Настройте частоту сохранения
   - Очищайте временные файлы

### Типичные проблемы и решения

| Проблема | Причина | Решение |
|----------|---------|---------|
| Out of Memory | Большие модели/батчи | Уменьшить batch_size, использовать чекпоинты |
| Медленное выполнение | CPU bottleneck | Параллельное выполнение, оптимизация гиперпараметров |
| Прерывание обучения | Нестабильность среды | Автоматические перезапуски, валидация конфигураций |
| Потеря результатов | Отсутствие чекпоинтов | Увеличить частоту сохранения |

## Расширение функциональности

### Кастомные стратегии восстановления

```python
class CustomRecoveryRunner(ExperimentRunner):
    def handle_failure(self, error, config_type=None, recovery_strategy="custom"):
        if recovery_strategy == "custom":
            # Кастомная логика восстановления
            return self._custom_recovery(error, config_type)
        return super().handle_failure(error, config_type, recovery_strategy)
    
    def _custom_recovery(self, error, config_type):
        # Анализ ошибки и принятие решения
        if "memory" in str(error).lower():
            return self._reduce_memory_usage(config_type)
        elif "timeout" in str(error).lower():
            return self._extend_timeout(config_type)
        return False
```

### Кастомные метрики мониторинга

```python
class EnhancedRunner(ExperimentRunner):
    def _start_monitoring(self):
        super()._start_monitoring()
        # Добавление кастомных метрик
        self._start_gpu_monitoring()
        self._start_network_monitoring()
    
    def _start_gpu_monitoring(self):
        # Мониторинг GPU если доступен
        try:
            import GPUtil
            # Логика мониторинга GPU
        except ImportError:
            pass
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/test_experiment_runner.py -v

# Быстрые тесты (без интеграционных)
pytest tests/test_experiment_runner.py -v -m "not slow"

# Интеграционные тесты
pytest tests/test_experiment_runner.py -v -m "integration"

# Тесты с покрытием
pytest tests/test_experiment_runner.py --cov=src.experiments.runner --cov-report=html
```

### Мокирование для тестов

```python
@patch('src.experiments.runner.Trainer')
def test_custom_scenario(mock_trainer_class):
    # Настройка мока
    mock_trainer = Mock()
    mock_trainer.train.return_value = mock_training_result
    mock_trainer_class.return_value = mock_trainer
    
    # Тестирование
    runner = ExperimentRunner(experiment)
    result = runner.run_configuration("baseline", config)
    
    assert result.success
```

## Заключение

`ExperimentRunner` предоставляет мощный и гибкий инструмент для проведения контролируемых RL экспериментов. Он объединяет все компоненты системы в единый, надежный оркестратор, обеспечивающий:

- ✅ Надежное выполнение экспериментов
- ✅ Комплексный мониторинг и контроль
- ✅ Гибкие стратегии восстановления
- ✅ Простой и мощный API
- ✅ CLI интерфейс для автоматизации
- ✅ Полную интеграцию с экосистемой проекта

Для получения дополнительной информации см. примеры в `examples/experiment_runner_example.py` и тесты в `tests/test_experiment_runner.py`.