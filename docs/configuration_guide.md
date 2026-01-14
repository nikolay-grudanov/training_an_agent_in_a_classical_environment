# Руководство по Configuration для RL экспериментов

## Обзор

Класс `Configuration` предоставляет комплексное решение для управления конфигурациями экспериментов в области обучения с подкреплением (RL). Он включает валидацию, сериализацию, сравнение и объединение конфигураций.

## Основные возможности

- ✅ **Валидация**: Комплексная проверка параметров алгоритмов и сред
- ✅ **Сериализация**: Сохранение/загрузка в форматах YAML и JSON
- ✅ **Сравнение**: Детальное сравнение конфигураций для анализа экспериментов
- ✅ **Объединение**: Слияние конфигураций для создания вариантов
- ✅ **Типобезопасность**: Полные type hints и валидация типов
- ✅ **Алгоритмы**: Поддержка PPO, A2C, SAC, TD3 с настройками по умолчанию
- ✅ **Среды**: Поддержка LunarLander-v2, Pendulum-v1, Acrobot-v1, MountainCarContinuous-v0

## Быстрый старт

### Создание базовой конфигурации

```python
from src.experiments.config import Configuration

# Минимальная конфигурация
config = Configuration(
    algorithm="PPO",
    environment="LunarLander-v2"
)

print(config)
# Конфигурация эксперимента: default_experiment
# Алгоритм: PPO
# Среда: LunarLander-v2
# Шагов обучения: 100,000
# Частота оценки: 10,000
# Зерно: 42
```

### Использование фабричных функций

```python
from src.experiments.config import create_ppo_config, create_sac_config

# PPO для дискретных сред
ppo_config = create_ppo_config(
    environment="LunarLander-v2",
    experiment_name="ppo_baseline",
    training_steps=200_000
)

# SAC для непрерывных сред
sac_config = create_sac_config(
    environment="Pendulum-v1",
    experiment_name="sac_continuous",
    training_steps=100_000
)
```

## Детальное использование

### Кастомная конфигурация

```python
config = Configuration(
    algorithm="A2C",
    environment="Pendulum-v1",
    hyperparameters={
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "n_steps": 10  # Переопределяем значение по умолчанию
    },
    seed=123,
    training_steps=200_000,
    evaluation_frequency=20_000,
    experiment_name="custom_a2c_experiment",
    description="Эксперимент с кастомными параметрами A2C"
)
```

### Сериализация

```python
# Сохранение в YAML
config.save("experiments/my_config.yaml", format_type="yaml")

# Сохранение в JSON
config.save("experiments/my_config.json", format_type="json")

# Загрузка
loaded_config = Configuration.load("experiments/my_config.yaml")
```

### Сравнение конфигураций

```python
from src.experiments.config import compare_configs

baseline = create_ppo_config(experiment_name="baseline")
variant = create_ppo_config(experiment_name="variant")

# Изменяем параметры варианта
variant.hyperparameters["learning_rate"] = 1e-3
variant.training_steps = 200_000

# Сравниваем
differences = baseline.get_differences(variant)
comparison = compare_configs(baseline, variant)

print(f"Различий: {comparison['differences_count']}")
print(f"Идентичные: {comparison['identical']}")
```

### Объединение конфигураций

```python
base_config = create_ppo_config(experiment_name="base")
override_config = Configuration(
    algorithm="PPO",
    environment="LunarLander-v2",
    experiment_name="override",
    training_steps=300_000,
    seed=999
)

# Объединяем (override имеет приоритет)
merged_config = base_config.merge(override_config)
```

## Поддерживаемые алгоритмы

### PPO (Proximal Policy Optimization)
```python
ppo_defaults = Configuration.get_algorithm_defaults("PPO")
# learning_rate: 3e-4, n_steps: 2048, batch_size: 64, etc.
```

### A2C (Advantage Actor-Critic)
```python
a2c_defaults = Configuration.get_algorithm_defaults("A2C")
# learning_rate: 7e-4, n_steps: 5, gamma: 0.99, etc.
```

### SAC (Soft Actor-Critic)
```python
sac_defaults = Configuration.get_algorithm_defaults("SAC")
# learning_rate: 3e-4, buffer_size: 1_000_000, tau: 0.005, etc.
```

### TD3 (Twin Delayed DDPG)
```python
td3_defaults = Configuration.get_algorithm_defaults("TD3")
# learning_rate: 3e-4, buffer_size: 1_000_000, policy_delay: 2, etc.
```

## Валидация

Класс автоматически валидирует:

- **Алгоритм**: Должен быть одним из поддерживаемых (PPO, A2C, SAC, TD3)
- **Среда**: Предупреждение для неподдерживаемых сред
- **Seed**: Должен быть в диапазоне [0, 2^32-1]
- **Training steps**: Должен быть положительным
- **Evaluation frequency**: Должен быть положительным и ≤ training_steps
- **Experiment name**: Не может быть пустым
- **Гиперпараметры**: Проверка типов и диапазонов значений

### Обработка ошибок валидации

```python
from src.experiments.config import ValidationError

try:
    invalid_config = Configuration(
        algorithm="PPO",
        environment="LunarLander-v2",
        seed=-1,  # Невалидный seed
        training_steps=0  # Невалидное количество шагов
    )
except ValidationError as e:
    print(f"Ошибка валидации: {e}")
```

## Интеграция с существующим кодом

### Совместимость с src.utils.config

```python
from src.utils.config import RLConfig
from src.experiments.config import Configuration

# Преобразование между форматами
def config_to_rl_config(config: Configuration) -> RLConfig:
    # Логика преобразования
    pass

def rl_config_to_config(rl_config: RLConfig) -> Configuration:
    # Логика преобразования
    pass
```

### Использование с Experiment класом

```python
from src.experiments.experiment import Experiment
from src.experiments.config import Configuration

baseline_config = create_ppo_config(experiment_name="baseline")
variant_config = create_ppo_config(experiment_name="variant")
variant_config.hyperparameters["learning_rate"] = 1e-3

# Создаем эксперимент
experiment = Experiment(
    baseline_config=baseline_config,  # Нужна конвертация в RLConfig
    variant_config=variant_config,    # Нужна конвертация в RLConfig
    hypothesis="Уменьшение learning rate улучшит стабильность обучения"
)
```

## Лучшие практики

1. **Используйте фабричные функции** для создания стандартных конфигураций
2. **Валидируйте конфигурации** перед запуском экспериментов
3. **Сохраняйте конфигурации** в системе контроля версий
4. **Используйте описательные имена** для экспериментов
5. **Документируйте гипотезы** в поле description
6. **Сравнивайте конфигурации** перед запуском экспериментов

## Примеры использования

Полные примеры доступны в файле `examples/configuration_usage.py`:

```bash
cd /path/to/project
PYTHONPATH=/path/to/project python examples/configuration_usage.py
```

## API Reference

### Класс Configuration

#### Основные методы:
- `validate()` - Валидация конфигурации
- `get_differences(other)` - Сравнение с другой конфигурацией
- `merge(other)` - Объединение конфигураций
- `to_dict()` / `from_dict()` - Сериализация в словарь
- `save()` / `load()` - Сохранение/загрузка файлов
- `copy()` - Создание глубокой копии

#### Статические методы:
- `get_algorithm_defaults(algorithm)` - Получение настроек по умолчанию

### Фабричные функции:
- `create_ppo_config()` - Создание конфигурации PPO
- `create_a2c_config()` - Создание конфигурации A2C
- `create_sac_config()` - Создание конфигурации SAC
- `create_td3_config()` - Создание конфигурации TD3

### Утилиты:
- `validate_config_file(filepath)` - Валидация файла конфигурации
- `compare_configs(config1, config2)` - Сравнение двух конфигураций

## Исключения

- `ConfigurationError` - Базовое исключение для ошибок конфигурации
- `ValidationError` - Ошибки валидации параметров

## Тестирование

Запуск тестов:

```bash
pytest tests/test_experiments_config.py -v
```

Покрытие: 48 тестов, 100% покрытие основного функционала.