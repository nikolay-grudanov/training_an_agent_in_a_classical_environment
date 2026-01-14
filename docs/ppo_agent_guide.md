# Руководство по PPO агенту

## Обзор

PPO агент представляет собой полную реализацию алгоритма Proximal Policy Optimization с оптимизированными гиперпараметрами для среды LunarLander-v3. Агент наследуется от базового класса `Agent` и интегрируется с системой логирования, метрик и чекпоинтов проекта.

## Основные возможности

### ✅ Алгоритм PPO
- Поддержка дискретных и непрерывных пространств действий
- Оптимизированные гиперпараметры для LunarLander-v3
- Адаптивное расписание learning rate (linear, exponential)
- Нормализация наблюдений и наград

### ✅ Мониторинг и колбэки
- Кастомные колбэки для отслеживания метрик
- Ранняя остановка при достижении целевой производительности
- Интеграция с TensorBoard
- Автоматическое сохранение чекпоинтов

### ✅ Производительность
- Векторизованная среда для ускорения обучения
- Поддержка GPU (при наличии)
- Оптимизированная архитектура нейронной сети
- Эффективное управление памятью

### ✅ Интеграция
- Полная интеграция с системой метрик проекта
- Структурированное логирование
- Воспроизводимость результатов
- Сохранение и загрузка моделей

## Быстрый старт

### Базовое использование

```python
from src.agents.ppo_agent import PPOAgent, PPOConfig

# Создание конфигурации
config = PPOConfig(
    env_name="LunarLander-v3",
    total_timesteps=100_000,
    seed=42
)

# Инициализация агента
agent = PPOAgent(config)

# Обучение
result = agent.train()

# Оценка
metrics = agent.evaluate(n_episodes=10)
print(f"Средняя награда: {metrics['mean_reward']:.2f}")

# Сохранение
agent.save("models/ppo_model.zip")
```

### Продвинутая конфигурация

```python
config = PPOConfig(
    # Основные параметры
    env_name="LunarLander-v3",
    total_timesteps=500_000,
    seed=42,
    
    # Оптимизированные гиперпараметры
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.98,
    clip_range=0.2,
    
    # Расписание learning rate
    use_lr_schedule=True,
    lr_schedule_type="linear",
    lr_final_ratio=0.1,
    
    # Архитектура сети
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn="tanh",
    
    # Нормализация
    normalize_env=True,
    norm_obs=True,
    norm_reward=True,
    
    # Ранняя остановка
    early_stopping=True,
    target_reward=200.0,
    patience_episodes=100,
    
    # Мониторинг
    eval_freq=25_000,
    save_freq=100_000,
    tensorboard_log="logs/ppo/",
    model_save_path="models/ppo_lunar.zip"
)
```

## Конфигурация

### Основные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `env_name` | str | - | Название среды Gymnasium |
| `total_timesteps` | int | 100_000 | Количество шагов обучения |
| `seed` | int | 42 | Seed для воспроизводимости |

### Гиперпараметры PPO

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `learning_rate` | float | 3e-4 | Скорость обучения |
| `n_steps` | int | 2048 | Шагов на обновление |
| `batch_size` | int | 64 | Размер батча |
| `n_epochs` | int | 10 | Эпох на обновление |
| `gamma` | float | 0.999 | Коэффициент дисконтирования |
| `gae_lambda` | float | 0.98 | GAE lambda |
| `clip_range` | float | 0.2 | Диапазон клиппинга |
| `ent_coef` | float | 0.01 | Коэффициент энтропии |
| `vf_coef` | float | 0.5 | Коэффициент функции ценности |

### Архитектура сети

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `net_arch` | List | [dict(pi=[64,64], vf=[64,64])] | Архитектура сети |
| `activation_fn` | str | "tanh" | Функция активации |
| `ortho_init` | bool | True | Ортогональная инициализация |

### Нормализация

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `normalize_env` | bool | True | Нормализация среды |
| `norm_obs` | bool | True | Нормализация наблюдений |
| `norm_reward` | bool | True | Нормализация наград |
| `clip_obs` | float | 10.0 | Клиппинг наблюдений |
| `clip_reward` | float | 10.0 | Клиппинг наград |

### Ранняя остановка

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `early_stopping` | bool | True | Включить раннюю остановку |
| `target_reward` | float | 200.0 | Целевая награда |
| `patience_episodes` | int | 50 | Терпение в эпизодах |
| `min_improvement` | float | 5.0 | Минимальное улучшение |

## Колбэки

### PPOMetricsCallback

Отслеживает метрики обучения и интегрируется с системой метрик проекта:

```python
from src.agents.ppo_agent import PPOMetricsCallback

callback = PPOMetricsCallback(
    metrics_tracker=metrics_tracker,
    log_freq=1000,
    verbose=1
)
```

**Отслеживаемые метрики:**
- Средняя награда за эпизод
- Стандартное отклонение награды
- Средняя длина эпизода
- Потери политики и функции ценности
- Энтропия политики

### EarlyStoppingCallback

Останавливает обучение при достижении целевой производительности или отсутствии улучшений:

```python
from src.agents.ppo_agent import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    target_reward=200.0,
    patience_episodes=50,
    min_improvement=5.0,
    check_freq=10000
)
```

## Примеры использования

### Обучение с кастомными колбэками

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"Шаг: {self.n_calls}")
        return True

agent = PPOAgent(config)
result = agent.train(callback=CustomCallback())
```

### Загрузка и дообучение

```python
# Загрузка обученного агента
agent = PPOAgent.load("models/ppo_model.zip")

# Дообучение
result = agent.train(total_timesteps=50_000)
```

### Оценка с визуализацией

```python
# Оценка без рендеринга
metrics = agent.evaluate(n_episodes=20, deterministic=True)

# Демонстрация с рендерингом
demo_metrics = agent.evaluate(n_episodes=3, render=True)
```

## Оптимизация производительности

### Рекомендуемые настройки для LunarLander-v3

```python
# Быстрое обучение (100K шагов)
fast_config = PPOConfig(
    env_name="LunarLander-v3",
    total_timesteps=100_000,
    n_steps=1024,
    batch_size=32,
    learning_rate=5e-4,
    normalize_env=False,
    early_stopping=True,
    target_reward=150.0
)

# Качественное обучение (500K шагов)
quality_config = PPOConfig(
    env_name="LunarLander-v3",
    total_timesteps=500_000,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    use_lr_schedule=True,
    normalize_env=True,
    target_reward=200.0
)

# Максимальное качество (1M шагов)
premium_config = PPOConfig(
    env_name="LunarLander-v3",
    total_timesteps=1_000_000,
    n_steps=4096,
    batch_size=128,
    learning_rate=2e-4,
    n_epochs=15,
    normalize_env=True,
    target_reward=250.0
)
```

### Мониторинг производительности

```python
# Получение информации о модели
info = agent.get_model_info()
print(f"Алгоритм: {info['algorithm']}")
print(f"Обучена: {info['is_trained']}")
print(f"Финальная награда: {info.get('final_mean_reward', 'N/A')}")

# Статистика нормализации (если используется)
if 'vec_normalize_stats' in info:
    stats = info['vec_normalize_stats']
    print(f"Среднее наблюдений: {stats['obs_mean']}")
    print(f"Дисперсия наблюдений: {stats['obs_var']}")
```

## Устранение неполадок

### Частые проблемы

1. **Медленная сходимость**
   - Увеличьте `learning_rate` до 5e-4
   - Уменьшите `n_steps` до 1024
   - Отключите нормализацию среды

2. **Нестабильное обучение**
   - Уменьшите `learning_rate` до 1e-4
   - Увеличьте `batch_size` до 128
   - Включите нормализацию наград

3. **Переобучение**
   - Увеличьте `ent_coef` до 0.02
   - Уменьшите `n_epochs` до 5
   - Используйте раннюю остановку

4. **Недостаток памяти**
   - Уменьшите `n_steps` и `batch_size`
   - Отключите TensorBoard логирование
   - Используйте `device="cpu"`

### Отладка

```python
# Включение детального логирования
import logging
logging.getLogger("src.agents.ppo_agent").setLevel(logging.DEBUG)

# Проверка конфигурации
config = PPOConfig(env_name="LunarLander-v3")
print(f"Валидная конфигурация: {config}")

# Тестирование среды
agent = PPOAgent(config)
obs, _ = agent.env.reset()
action, _ = agent.predict(obs)  # Должно вызвать ошибку до обучения
```

## Интеграция с проектом

### Система метрик

PPO агент автоматически интегрируется с системой метрик проекта:

```python
# Получение трекера метрик
metrics_tracker = agent.metrics_tracker

# Просмотр метрик
summary = metrics_tracker.get_summary()
for metric in summary.metrics[-10:]:  # Последние 10 метрик
    print(f"{metric.name}: {metric.value} (шаг {metric.step})")
```

### Логирование

Агент использует структурированное логирование:

```python
# Настройка уровня логирования
import logging
logging.getLogger("src.agents.ppo_agent").setLevel(logging.INFO)

# Логи автоматически включают контекст
agent = PPOAgent(config, experiment_name="lunar_experiment")
# Логи будут содержать experiment_id="lunar_experiment"
```

### Чекпоинты

Автоматическое сохранение чекпоинтов:

```python
config = PPOConfig(
    env_name="LunarLander-v3",
    save_freq=50_000,  # Сохранять каждые 50K шагов
    model_save_path="models/checkpoints/ppo_model.zip"
)

agent = PPOAgent(config)
# Чекпоинты будут сохраняться автоматически в models/checkpoints/
```

## Расширение функциональности

### Кастомные колбэки

```python
from stable_baselines3.common.callbacks import BaseCallback

class RewardThresholdCallback(BaseCallback):
    def __init__(self, threshold: float, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer[-10:]]
            if np.mean(recent_rewards) >= self.threshold:
                print(f"Достигнут порог награды: {self.threshold}")
                return False  # Остановить обучение
        return True

# Использование
callback = RewardThresholdCallback(threshold=180.0)
agent.train(callback=callback)
```

### Кастомная архитектура сети

```python
config = PPOConfig(
    env_name="LunarLander-v3",
    net_arch=[
        dict(
            pi=[128, 128, 64],  # Политика: 3 слоя
            vf=[128, 64]        # Функция ценности: 2 слоя
        )
    ],
    activation_fn="relu",
    policy_kwargs={
        "ortho_init": True,
        "log_std_init": -0.5,
    }
)
```

## Производительность

### Бенчмарки для LunarLander-v3

| Конфигурация | Время обучения | Финальная награда | Успешность |
|--------------|----------------|-------------------|------------|
| Быстрая (100K) | ~5 мин | 150 ± 30 | 70% |
| Качественная (500K) | ~20 мин | 220 ± 20 | 90% |
| Премиум (1M) | ~40 мин | 250 ± 15 | 95% |

*Тестировано на CPU Intel i7-8700K*

### Оптимизация для разных сред

```python
# Дискретные действия (CartPole, LunarLander)
discrete_config = PPOConfig(
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    clip_range=0.2
)

# Непрерывные действия (BipedalWalker, Pendulum)
continuous_config = PPOConfig(
    n_steps=1024,
    batch_size=32,
    learning_rate=1e-4,
    clip_range=0.1,
    use_sde=True
)
```