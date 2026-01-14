# TD3 Agent Guide

## Обзор

TD3 (Twin Delayed Deep Deterministic Policy Gradient) - это алгоритм обучения с подкреплением для непрерывных пространств действий. Он является улучшенной версией DDPG с тремя ключевыми инновациями:

1. **Twin Critics** - использование двух критиков для уменьшения переоценки Q-значений
2. **Delayed Policy Updates** - обновление политики реже, чем критиков
3. **Target Policy Smoothing** - добавление шума к целевой политике для регуляризации

## Особенности реализации

### Ключевые возможности

- ✅ **Только непрерывные действия** - TD3 предназначен исключительно для Box action spaces
- ✅ **Различные типы шума** - Normal, Ornstein-Uhlenbeck или без шума
- ✅ **Адаптивные расписания** - learning rate и шума
- ✅ **Нормализация наблюдений** - VecNormalize для стабильности обучения
- ✅ **Ранняя остановка** - при достижении целевой производительности
- ✅ **Комплексный мониторинг** - интеграция с системой метрик проекта
- ✅ **Автоматические чекпоинты** - сохранение лучших моделей

### Оптимизированные среды

Агент оптимизирован для следующих сред:

- **LunarLander-v3** (continuous) - посадка лунного модуля
- **Pendulum-v1** - управление маятником
- **BipedalWalker-v3** - ходьба двуногого робота
- **HalfCheetah-v4** - управление гепардом
- **Ant-v4** - управление муравьем

## Быстрый старт

### Базовое использование

```python
from src.agents.td3_agent import TD3Agent, TD3Config

# Создание конфигурации
config = TD3Config(
    env_name="LunarLander-v3",
    total_timesteps=200_000,
    learning_rate=1e-3,
    action_noise_type="normal",
    action_noise_std=0.1,
)

# Создание и обучение агента
agent = TD3Agent(config=config)
result = agent.train()

print(f"Финальная награда: {result.final_mean_reward:.2f}")
```

### Продвинутая конфигурация

```python
config = TD3Config(
    env_name="BipedalWalker-v3",
    total_timesteps=500_000,
    
    # Гиперпараметры TD3
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    policy_delay=2,
    target_policy_noise=0.2,
    
    # Шум для исследования
    action_noise_type="ornstein_uhlenbeck",
    ou_sigma=0.2,
    ou_theta=0.15,
    
    # Расписания
    use_lr_schedule=True,
    lr_schedule_type="linear",
    lr_final_ratio=0.1,
    
    use_noise_schedule=True,
    noise_schedule_type="linear",
    noise_final_ratio=0.01,
    
    # Ранняя остановка
    early_stopping=True,
    target_reward=300.0,
    patience_episodes=100,
    
    # Мониторинг
    eval_freq=10_000,
    save_freq=50_000,
    use_tensorboard=True,
)

agent = TD3Agent(config=config)
result = agent.train()
```

## Конфигурация

### Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `learning_rate` | Скорость обучения | 1e-3 |
| `buffer_size` | Размер replay buffer | 1,000,000 |
| `batch_size` | Размер батча | 256 |
| `tau` | Коэффициент мягкого обновления | 0.005 |
| `gamma` | Фактор дисконтирования | 0.99 |

### TD3-специфичные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `policy_delay` | Задержка обновления политики | 2 |
| `target_policy_noise` | Шум для целевой политики | 0.2 |
| `target_noise_clip` | Ограничение шума | 0.5 |

### Параметры шума

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `action_noise_type` | Тип шума: "normal", "ornstein_uhlenbeck", "none" | "normal" |
| `action_noise_std` | Стандартное отклонение шума | 0.1 |
| `ou_theta` | Параметр OU шума (скорость возврата) | 0.15 |
| `ou_sigma` | Параметр OU шума (волатильность) | 0.2 |

## Примеры использования

### 1. Обучение для LunarLander-v3

```python
from src.agents.td3_agent import TD3Agent, TD3Config

config = TD3Config(
    env_name="LunarLander-v3",
    total_timesteps=200_000,
    target_reward=200.0,
    model_save_path="models/td3_lunarlander.zip",
)

agent = TD3Agent(config=config)
result = agent.train()

# Оценка агента
metrics = agent.evaluate(n_episodes=10)
print(f"Средняя награда: {metrics['mean_reward']:.2f}")
```

### 2. Сравнение типов шума

```python
noise_types = ["normal", "ornstein_uhlenbeck", "none"]
results = {}

for noise_type in noise_types:
    config = TD3Config(
        env_name="Pendulum-v1",
        total_timesteps=50_000,
        action_noise_type=noise_type,
        early_stopping=False,
    )
    
    agent = TD3Agent(config=config)
    result = agent.train()
    results[noise_type] = result.final_mean_reward

print("Результаты по типам шума:")
for noise_type, reward in results.items():
    print(f"{noise_type}: {reward:.2f}")
```

### 3. Загрузка и тестирование модели

```python
# Загрузка сохраненной модели
agent = TD3Agent.load("models/td3_lunarlander.zip")

# Демонстрация работы
obs, _ = agent.env.reset()
for step in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = agent.env.step(action)
    
    if terminated or truncated:
        obs, _ = agent.env.reset()
```

## Мониторинг и отладка

### TensorBoard

```python
config = TD3Config(
    env_name="Pendulum-v1",
    use_tensorboard=True,
    tensorboard_log="logs/td3_pendulum",
)

agent = TD3Agent(config=config)
agent.train()

# Запуск TensorBoard
# tensorboard --logdir logs/td3_pendulum
```

### Кастомные колбэки

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"Шаг {self.n_calls}: награда = {self.locals.get('rewards', [0])[-1]}")
        return True

agent = TD3Agent(config=config)
result = agent.train(callback=CustomCallback())
```

## Оптимизация производительности

### Рекомендации по гиперпараметрам

1. **Размер буфера**: Используйте большие буферы (1M+) для стабильности
2. **Batch size**: 256-512 для больших сетей, 64-128 для простых задач
3. **Learning rate**: Начните с 1e-3, уменьшайте при нестабильности
4. **Policy delay**: 2-3 для большинства задач
5. **Шум**: Normal шум для простых задач, OU для сложных

### Настройка для конкретных сред

```python
# Для простых сред (Pendulum)
simple_config = TD3Config(
    buffer_size=200_000,
    batch_size=128,
    learning_starts=1_000,
)

# Для сложных сред (BipedalWalker)
complex_config = TD3Config(
    buffer_size=1_000_000,
    batch_size=256,
    learning_starts=10_000,
    action_noise_type="ornstein_uhlenbeck",
)
```

## Устранение неполадок

### Частые проблемы

1. **Нестабильное обучение**
   - Уменьшите learning rate
   - Увеличьте policy_delay
   - Используйте нормализацию наблюдений

2. **Медленная сходимость**
   - Увеличьте размер буфера
   - Настройте параметры шума
   - Проверьте архитектуру сети

3. **Переобучение**
   - Используйте раннюю остановку
   - Увеличьте регуляризацию шума
   - Уменьшите частоту обновлений

### Отладка

```python
# Включение детального логирования
config = TD3Config(
    env_name="Pendulum-v1",
    verbose=2,  # Максимальная детализация
)

# Получение информации о модели
agent = TD3Agent(config=config)
model_info = agent.get_model_info()
print(f"Информация о модели: {model_info}")
```

## Сравнение с другими алгоритмами

| Алгоритм | Тип действий | Стабильность | Эффективность выборки | Сложность |
|----------|--------------|--------------|----------------------|-----------|
| TD3 | Непрерывные | Высокая | Высокая | Средняя |
| PPO | Любые | Высокая | Низкая | Низкая |
| SAC | Непрерывные | Высокая | Высокая | Высокая |
| DDPG | Непрерывные | Средняя | Высокая | Средняя |

## Заключение

TD3 агент предоставляет мощный и стабильный инструмент для обучения в непрерывных средах. Благодаря продуманной архитектуре и интеграции с системой проекта, он подходит как для исследований, так и для практических применений.

Для получения дополнительной информации см.:
- [Примеры использования](../examples/td3_agent_usage.py)
- [Тесты](../tests/unit/test_td3_agent.py)
- [Конфигурация](../configs/td3_config.yaml)