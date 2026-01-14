# Руководство по базовому классу Agent

## Обзор

Базовый класс `Agent` предоставляет единый интерфейс для всех RL агентов в проекте. Он обеспечивает интеграцию с Stable-Baselines3, систему логирования, отслеживание метрик и управление воспроизводимостью.

## Основные компоненты

### 1. AgentConfig

Класс конфигурации для настройки агента:

```python
from src.agents.base import AgentConfig

config = AgentConfig(
    algorithm="PPO",
    env_name="CartPole-v1", 
    total_timesteps=100_000,
    learning_rate=3e-4,
    seed=42,
    # ... другие параметры
)
```

**Основные параметры:**
- `algorithm`: Алгоритм RL (PPO, A2C, SAC, TD3, DQN)
- `env_name`: Название среды Gymnasium
- `total_timesteps`: Количество шагов обучения
- `learning_rate`: Скорость обучения
- `seed`: Seed для воспроизводимости

### 2. TrainingResult

Результат обучения агента:

```python
result = agent.train()
print(f"Время обучения: {result.training_time:.2f} сек")
print(f"Средняя награда: {result.final_mean_reward:.2f}")
```

### 3. Agent (абстрактный базовый класс)

Основные методы:
- `train()`: Обучение агента
- `predict()`: Предсказание действий
- `save()`: Сохранение модели
- `load()`: Загрузка модели
- `evaluate()`: Оценка производительности

## Создание собственного агента

### Шаг 1: Наследование от Agent

```python
from src.agents.base import Agent, AgentConfig, TrainingResult
from stable_baselines3 import PPO

class PPOAgent(Agent):
    def _create_model(self):
        return PPO(
            policy=self.config.policy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            # ... другие параметры
        )
```

### Шаг 2: Реализация обязательных методов

```python
def train(self, total_timesteps=None, callback=None, **kwargs):
    if self.model is None:
        self.model = self._create_model()
    
    timesteps = total_timesteps or self.config.total_timesteps
    
    # Обучение
    self.model.learn(total_timesteps=timesteps, callback=callback)
    self.is_trained = True
    
    # Создание результата
    eval_metrics = self.evaluate()
    return TrainingResult(
        total_timesteps=timesteps,
        training_time=training_time,
        final_mean_reward=eval_metrics["mean_reward"],
        # ...
    )

def predict(self, observation, deterministic=True, **kwargs):
    if not self.is_trained:
        raise RuntimeError("Модель не обучена")
    return self.model.predict(observation, deterministic)

@classmethod
def load(cls, path, env=None, **kwargs):
    # Загрузка конфигурации и модели
    # ...
    return agent
```

## Интеграция с системой проекта

### Логирование

Агент автоматически интегрируется с системой логирования:

```python
agent = PPOAgent(config, experiment_name="my_experiment")
# Логи будут содержать experiment_id="my_experiment"
```

### Метрики

Автоматическое отслеживание метрик:

```python
# Метрики автоматически собираются во время обучения
metrics = agent.metrics_tracker.get_summary("episode_reward")
```

### Воспроизводимость

Автоматическая установка seed:

```python
config = AgentConfig(seed=42)  # Seed устанавливается автоматически
agent = PPOAgent(config)
```

## Проверка совместимости

Агент автоматически проверяет совместимость алгоритма и среды:

- **Дискретные действия**: PPO, A2C, DQN
- **Непрерывные действия**: PPO, A2C, SAC, TD3

```python
# Это вызовет ValueError
config = AgentConfig(algorithm="SAC", env_name="CartPole-v1")  # Дискретные действия
agent = PPOAgent(config)  # Ошибка!
```

## Сохранение и загрузка

### Сохранение

```python
agent.train()
agent.save("models/my_agent.zip")
# Создаются файлы:
# - models/my_agent.zip (модель)
# - models/my_agent.yaml (конфигурация)
```

### Загрузка

```python
agent = PPOAgent.load("models/my_agent.zip")
action, _ = agent.predict(observation)
```

## Оценка производительности

```python
# Оценка на 10 эпизодах
metrics = agent.evaluate(n_episodes=10, deterministic=True)
print(f"Средняя награда: {metrics['mean_reward']:.2f}")
print(f"Стандартное отклонение: {metrics['std_reward']:.2f}")
```

## Управление чекпоинтами

При указании `model_save_path` в конфигурации:

```python
config = AgentConfig(
    model_save_path="checkpoints/agent.zip",
    save_freq=50_000,  # Сохранение каждые 50k шагов
)
```

## Пример полного использования

```python
from src.agents.base import AgentConfig
from examples.agent_base_usage import PPOAgent

# 1. Конфигурация
config = AgentConfig(
    algorithm="PPO",
    env_name="CartPole-v1",
    total_timesteps=50_000,
    learning_rate=3e-4,
    seed=42,
)

# 2. Создание агента
agent = PPOAgent(config, experiment_name="cartpole_experiment")

# 3. Обучение
result = agent.train()
print(f"Обучение завершено за {result.training_time:.2f} сек")

# 4. Оценка
metrics = agent.evaluate(n_episodes=10)
print(f"Средняя награда: {metrics['mean_reward']:.2f}")

# 5. Сохранение
agent.save("models/cartpole_agent.zip")

# 6. Загрузка (в другой сессии)
loaded_agent = PPOAgent.load("models/cartpole_agent.zip")
action, _ = loaded_agent.predict(observation)
```

## Лучшие практики

### 1. Конфигурация
- Всегда используйте `AgentConfig` для параметров
- Устанавливайте `seed` для воспроизводимости
- Валидируйте параметры в `__post_init__`

### 2. Обучение
- Обрабатывайте исключения в `train()`
- Возвращайте подробный `TrainingResult`
- Используйте callbacks для мониторинга

### 3. Логирование
- Используйте `self.logger` для логов
- Добавляйте контекст в логи через `extra`
- Логируйте важные события (начало/конец обучения)

### 4. Тестирование
- Создавайте mock-реализации для тестов
- Тестируйте все публичные методы
- Проверяйте совместимость алгоритмов

### 5. Производительность
- Используйте `device="cpu"` для простых политик
- Кэшируйте дорогие операции
- Мониторьте использование памяти

## Расширение функциональности

### Добавление новых алгоритмов

1. Обновите проверку совместимости в `_validate_env_algorithm_compatibility`
2. Создайте новый класс агента
3. Добавьте тесты для нового алгоритма

### Кастомные метрики

```python
def train(self, **kwargs):
    # Во время обучения
    self.metrics_tracker.add_metric(
        name="custom_metric",
        value=custom_value,
        timestep=timestep,
        episode=episode,
    )
```

### Кастомные callbacks

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
    def _on_step(self):
        # Кастомная логика
        return True

# Использование
callback = CustomCallback(agent)
agent.train(callback=callback)
```

## Отладка

### Включение подробного логирования

```python
import logging
logging.getLogger('rl_training').setLevel(logging.DEBUG)
```

### Проверка состояния агента

```python
info = agent.get_model_info()
print(f"Агент обучен: {info['is_trained']}")
print(f"Алгоритм: {info['algorithm']}")
print(f"Среда: {info['env_name']}")
```

### Сброс для переобучения

```python
agent.reset_model()  # Сбрасывает состояние обучения
```

## Заключение

Базовый класс `Agent` обеспечивает:
- ✅ Единый интерфейс для всех RL алгоритмов
- ✅ Интеграцию с Stable-Baselines3
- ✅ Автоматическое логирование и метрики
- ✅ Управление воспроизводимостью
- ✅ Проверку совместимости алгоритмов
- ✅ Удобное сохранение/загрузку моделей

Используйте этот класс как основу для всех ваших RL агентов в проекте!