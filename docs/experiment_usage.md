# Использование класса Experiment

Класс `Experiment` предоставляет комплексное решение для проведения контролируемых экспериментов в области обучения с подкреплением.

## Основные возможности

- ✅ Управление жизненным циклом эксперимента (создание, запуск, пауза, завершение)
- ✅ Сравнение двух конфигураций (baseline vs variant)
- ✅ Автоматический анализ результатов и статистическое сравнение
- ✅ Сериализация в JSON и Pickle форматах
- ✅ Структурированное логирование с контекстом эксперимента
- ✅ Валидация конфигураций и состояний
- ✅ Полная типизация и обработка ошибок

## Быстрый старт

```python
from src.experiments.experiment import Experiment
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig

# 1. Создание конфигураций
baseline_config = RLConfig(
    algorithm=AlgorithmConfig(name="PPO", learning_rate=3e-4),
    environment=EnvironmentConfig(name="LunarLander-v3"),
    training=TrainingConfig(total_timesteps=100000)
)

variant_config = RLConfig(
    algorithm=AlgorithmConfig(name="PPO", learning_rate=1e-3),  # Отличие
    environment=EnvironmentConfig(name="LunarLander-v3"),
    training=TrainingConfig(total_timesteps=100000)
)

# 2. Создание эксперимента
experiment = Experiment(
    baseline_config=baseline_config,
    variant_config=variant_config,
    hypothesis="Увеличение learning rate улучшит производительность"
)

# 3. Жизненный цикл
experiment.start()

# 4. Добавление результатов (после обучения)
experiment.add_result('baseline', {
    'mean_reward': 150.0,
    'episode_length': 250,
    'training_time': 3600
})

experiment.add_result('variant', {
    'mean_reward': 175.0,
    'episode_length': 230,
    'training_time': 3400
})

# 5. Автоматическое сравнение
comparison = experiment.compare_results()
print(f"Лучший результат: {comparison['summary']['overall_better']}")

# 6. Завершение и сохранение
experiment.stop()
filepath = experiment.save(format_type='json')
```

## Структура данных

### Статусы эксперимента
- `CREATED` - эксперимент создан
- `RUNNING` - эксперимент выполняется
- `PAUSED` - эксперимент приостановлен
- `COMPLETED` - эксперимент успешно завершен
- `FAILED` - эксперимент завершился с ошибкой

### Результаты эксперимента
```python
{
    'baseline': {
        'mean_reward': float,
        'final_reward': float,
        'episode_length': int,
        'convergence_timesteps': int,
        'training_time': float,
        'metrics_history': List[Dict]  # опционально
    },
    'variant': {
        # аналогичная структура
    },
    'comparison': {
        'performance_metrics': {
            'mean_reward': {
                'baseline': float,
                'variant': float,
                'improvement': float,
                'improvement_percent': float,
                'better': str  # 'baseline' или 'variant'
            }
            # ... другие метрики
        },
        'summary': {
            'overall_better': str,
            'reward_improvement': float,
            'significant_improvement': bool
        }
    }
}
```

## Методы класса

### Управление жизненным циклом
- `start()` - запустить эксперимент
- `pause()` - приостановить эксперимент
- `resume()` - возобновить эксперимент
- `stop(failed=False, error_message=None)` - завершить эксперимент

### Работа с результатами
- `add_result(config_type, results, metrics=None)` - добавить результаты
- `compare_results()` - получить сравнение результатов

### Сериализация
- `save(format_type='json')` - сохранить эксперимент
- `load(filepath, format_type=None)` - загрузить эксперимент (класс-метод)

### Информация о состоянии
- `get_status()` - получить детальный статус
- `get_summary()` - получить краткую сводку

## Валидация

Класс автоматически валидирует:
- ✅ Совместимость сред (должны быть одинаковыми)
- ✅ Различия в конфигурациях (не должны быть идентичными)
- ✅ Корректность переходов состояний
- ✅ Поддерживаемые алгоритмы (PPO, A2C, SAC, TD3)
- ✅ Валидность параметров (положительные значения для timesteps, learning_rate)

## Обработка ошибок

```python
from src.experiments.experiment import (
    ExperimentError,
    InvalidStateTransitionError,
    ConfigurationError
)

try:
    experiment = Experiment(baseline_config, variant_config, hypothesis)
    experiment.start()
    # ... работа с экспериментом
except ConfigurationError as e:
    print(f"Ошибка конфигурации: {e}")
except InvalidStateTransitionError as e:
    print(f"Недопустимый переход состояния: {e}")
except ExperimentError as e:
    print(f"Общая ошибка эксперимента: {e}")
```

## Интеграция с логированием

Класс автоматически создает структурированные логи с контекстом эксперимента:

```python
# Логи содержат experiment_id и дополнительную информацию
2026-01-14 17:51:53 - rl_training - INFO - Создан эксперимент e8e074ec-440b-43bb-bc76-67a514318492
2026-01-14 17:51:53 - rl_training - INFO - Добавлены результаты для baseline
2026-01-14 17:51:53 - rl_training - INFO - Выполнено сравнение результатов
```

## Примеры использования

Полный пример использования доступен в файле `examples/experiment_example.py`.

### Интеграция с реальным обучением

```python
# Псевдокод интеграции с SB3
from stable_baselines3 import PPO

def run_training(config, experiment, config_type):
    env = gym.make(config.environment.name)
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=config.algorithm.learning_rate,
        # ... другие параметры
    )
    
    model.learn(total_timesteps=config.training.total_timesteps)
    
    # Оценка модели
    mean_reward, episode_length = evaluate_policy(model, env)
    
    # Добавление результатов в эксперимент
    results = {
        'mean_reward': mean_reward,
        'episode_length': episode_length,
        'training_time': time.time() - start_time
    }
    
    experiment.add_result(config_type, results)
```

## Лучшие практики

1. **Четкие гипотезы**: Формулируйте конкретные, проверяемые гипотезы
2. **Одинаковые среды**: Всегда используйте одну среду для baseline и variant
3. **Значимые различия**: Убедитесь, что конфигурации действительно отличаются
4. **Сохранение результатов**: Всегда сохраняйте эксперименты для воспроизводимости
5. **Структурированные метрики**: Используйте стандартный набор метрик для сравнения

## Расширение функциональности

Класс можно легко расширить для:
- Поддержки более двух конфигураций
- Статистических тестов значимости
- Интеграции с системами отслеживания экспериментов (MLflow, Weights & Biases)
- Автоматической генерации отчетов и визуализаций