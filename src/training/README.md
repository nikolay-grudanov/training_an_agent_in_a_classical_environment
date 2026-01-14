# 🎮 Модуль обучения RL агентов

Комплексная система обучения агентов обучения с подкреплением с поддержкой всех современных алгоритмов, конфигурационного управления, мониторинга и восстановления сессий.

## 🚀 Основные возможности

### ✨ Поддерживаемые алгоритмы
- **PPO** (Proximal Policy Optimization) - для дискретных и непрерывных действий
- **A2C** (Advantage Actor-Critic) - быстрый on-policy алгоритм  
- **SAC** (Soft Actor-Critic) - для непрерывных действий
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient) - для непрерывных действий

### 🛠️ Режимы работы
- **TRAIN** - обучение с нуля
- **RESUME** - восстановление прерванного обучения
- **EVALUATE** - только оценка обученной модели
- **FINETUNE** - дообучение существующей модели

### 📊 Мониторинг и трекинг
- Автоматическое логирование метрик
- Интеграция с TensorBoard
- Система чекпоинтов
- Раннее остановка по критериям
- Детальная история обучения

### ⚙️ Конфигурация
- YAML конфигурационные файлы
- Интеграция с Hydra
- Переопределения через командную строку
- Валидация параметров

## 📦 Структура модуля

```
src/training/
├── __init__.py          # Экспорты модуля
├── trainer.py           # Основной класс Trainer
├── cli.py              # CLI интерфейс
└── README.md           # Документация
```

## 🎯 Быстрый старт

### Базовое использование

```python
from src.training import Trainer, TrainerConfig

# Создание конфигурации
config = TrainerConfig(
    experiment_name="my_experiment",
    algorithm="PPO",
    environment_name="LunarLander-v3",
    total_timesteps=100_000,
    seed=42,
)

# Обучение
with Trainer(config) as trainer:
    result = trainer.train()
    
    if result.success:
        print(f"Награда: {result.final_mean_reward:.2f}")
        print(f"Модель: {result.model_path}")
```

### Использование CLI

```bash
# Базовое обучение
python -m src.training.cli train --algorithm PPO --env LunarLander-v3 --timesteps 100000

# Обучение из конфигурации
python -m src.training.cli train --config configs/ppo_lunarlander.yaml

# Восстановление обучения
python -m src.training.cli resume checkpoints/checkpoint_50000.pkl

# Оценка модели
python -m src.training.cli evaluate models/ppo_final.zip --episodes 10 --render

# Сравнение алгоритмов
python -m src.training.cli compare --algorithm PPO A2C --timesteps 50000 --runs 3
```

### Конфигурационный файл

```yaml
# configs/my_config.yaml
experiment_name: "advanced_ppo"
output_dir: "results"
seed: 42

algorithm:
  name: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95

environment:
  name: "LunarLander-v3"

training:
  total_timesteps: 200000
  eval_freq: 10000
  n_eval_episodes: 10
  save_freq: 25000
  early_stopping: true
  patience: 5
```

## 📚 Подробная документация

### TrainerConfig

Основной класс конфигурации с полной настройкой обучения:

```python
@dataclass
class TrainerConfig:
    # Основные параметры
    experiment_name: str = "default_experiment"
    algorithm: str = "PPO"  # PPO, A2C, SAC, TD3
    environment_name: str = "LunarLander-v3"
    mode: TrainingMode = TrainingMode.TRAIN
    
    # Параметры обучения
    total_timesteps: int = 100_000
    seed: int = 42
    
    # Мониторинг
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    
    # Сохранение
    save_freq: int = 50_000
    checkpoint_freq: int = 25_000
    max_checkpoints: int = 5
    
    # Раннее остановка
    early_stopping: bool = False
    patience: int = 5
    min_improvement: float = 0.01
```

### Trainer

Главный оркестратор обучения:

```python
class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        """Инициализация тренера с конфигурацией."""
    
    def setup(self) -> None:
        """Настройка всех компонентов."""
    
    def train(self) -> TrainingResult:
        """Выполнение обучения."""
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Оценка агента."""
    
    def save_checkpoint(self, timestep: int) -> str:
        """Сохранение чекпоинта."""
    
    def load_checkpoint(self, path: str) -> None:
        """Загрузка чекпоинта."""
```

### TrainingResult

Результат обучения с полной информацией:

```python
@dataclass
class TrainingResult:
    success: bool
    total_timesteps: int
    training_time: float
    final_mean_reward: float
    final_std_reward: float
    
    # История и метаданные
    training_history: Dict[str, List[float]]
    evaluation_history: Dict[str, List[float]]
    model_path: Optional[str]
    checkpoint_paths: List[str]
    
    # Дополнительная информация
    best_mean_reward: float
    convergence_timestep: Optional[int]
    early_stopped: bool
    error_message: Optional[str]
```

## 🔧 Продвинутое использование

### Кастомная конфигурация агента

```python
from src.agents.base import AgentConfig

# Детальная настройка алгоритма
agent_config = AgentConfig(
    algorithm="PPO",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs={
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": "tanh",
    },
)

config = TrainerConfig(
    experiment_name="custom_ppo",
    agent_config=agent_config,
    # ... другие параметры
)
```

### Восстановление обучения

```python
# Автоматическое восстановление
config = TrainerConfig(
    mode=TrainingMode.RESUME,
    resume_from_checkpoint="checkpoints/checkpoint_50000.pkl",
    total_timesteps=100_000,  # Общее количество шагов
)

with Trainer(config) as trainer:
    result = trainer.train()  # Продолжит с 50000 шагов
```

### Раннее остановка

```python
config = TrainerConfig(
    early_stopping=True,
    patience=5,  # Остановка после 5 оценок без улучшения
    min_improvement=10.0,  # Минимальное улучшение награды
    eval_freq=5000,  # Частая оценка для раннего остановки
)
```

### Создание из RLConfig

```python
from src.utils.config import load_config

# Загрузка из YAML
rl_config = load_config(config_name="my_config")

# Преобразование в TrainerConfig
trainer_config = TrainerConfig.from_rl_config(rl_config)

trainer = Trainer(trainer_config)
```

## 🎮 Примеры использования

### Сравнение алгоритмов

```python
algorithms = ["PPO", "A2C", "SAC"]
results = {}

for algorithm in algorithms:
    config = TrainerConfig(
        experiment_name=f"comparison_{algorithm}",
        algorithm=algorithm,
        total_timesteps=50_000,
        seed=42,
    )
    
    with Trainer(config) as trainer:
        result = trainer.train()
        results[algorithm] = result

# Анализ результатов
for alg, result in results.items():
    if result.success:
        print(f"{alg}: {result.final_mean_reward:.2f}")
```

### Гиперпараметрический поиск

```python
learning_rates = [1e-4, 3e-4, 1e-3]
best_result = None
best_reward = float("-inf")

for lr in learning_rates:
    config = TrainerConfig(
        experiment_name=f"hypersearch_lr_{lr}",
        algorithm="PPO",
        total_timesteps=30_000,
    )
    
    # Переопределение learning_rate
    config.agent_config.learning_rate = lr
    
    with Trainer(config) as trainer:
        result = trainer.train()
        
        if result.success and result.final_mean_reward > best_reward:
            best_reward = result.final_mean_reward
            best_result = result

print(f"Лучший результат: {best_reward:.2f}")
```

### Мониторинг в реальном времени

```python
class CustomCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
    
    def _on_step(self) -> bool:
        # Кастомная логика мониторинга
        if self.n_calls % 1000 == 0:
            print(f"Шаг {self.n_calls}")
        return True

config = TrainerConfig(
    experiment_name="monitored_training",
    total_timesteps=100_000,
)

with Trainer(config) as trainer:
    # Добавление кастомного callback
    callback = CustomCallback()
    result = trainer.train()
```

## 🔍 Отладка и мониторинг

### Логирование

```python
import logging
from src.utils.logging import setup_logging

# Детальное логирование
setup_logging(level=logging.DEBUG)

config = TrainerConfig(
    verbose=2,  # Максимальная детализация
    log_interval=100,  # Частое логирование
)
```

### Анализ результатов

```python
# После обучения
result = trainer.train()

if result.success:
    # История обучения
    rewards = result.evaluation_history.get("mean_rewards", [])
    timesteps = result.evaluation_history.get("timesteps", [])
    
    # Построение графика
    import matplotlib.pyplot as plt
    plt.plot(timesteps, rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Training Progress")
    plt.show()
    
    # Сохранение результата
    result.save("results/training_result.yaml")
```

## 🚨 Обработка ошибок

```python
try:
    with Trainer(config) as trainer:
        result = trainer.train()
        
        if not result.success:
            print(f"Ошибка обучения: {result.error_message}")
            
            # Анализ предупреждений
            for warning in result.warnings:
                print(f"Предупреждение: {warning}")
                
except KeyboardInterrupt:
    print("Обучение прервано пользователем")
    # Автоматическое сохранение чекпоинта
    
except Exception as e:
    print(f"Критическая ошибка: {e}")
    # Логирование для отладки
```

## 📈 Интеграция с экспериментами

```python
# Интеграция с системой экспериментов
config = TrainerConfig(
    track_experiment=True,
    experiment_tags=["baseline", "ppo", "lunarlander"],
)

# Автоматическое логирование в систему трекинга
with Trainer(config) as trainer:
    result = trainer.train()
    # Метрики автоматически сохраняются
```

## 🎯 Лучшие практики

1. **Всегда используйте seed** для воспроизводимости
2. **Настройте eval_freq** для мониторинга прогресса
3. **Включите чекпоинты** для длительного обучения
4. **Используйте раннее остановка** для предотвращения переобучения
5. **Сохраняйте конфигурации** для воспроизведения экспериментов
6. **Мониторьте ресурсы** при длительном обучении

## 🔗 Связанные модули

- `src.agents` - Реализации RL алгоритмов
- `src.environments` - Обертки для сред
- `src.utils` - Утилиты (логирование, метрики, конфигурация)
- `src.experiments` - Система экспериментов

## 📝 Changelog

### v1.0.0
- Первая версия с поддержкой PPO, A2C, SAC, TD3
- CLI интерфейс
- Система чекпоинтов
- Конфигурационное управление
- Интеграция с экспериментами