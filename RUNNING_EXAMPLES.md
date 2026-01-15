# Примеры запуска приложения для обучения RL агентов

## 1. Пример запуска API сервера

### Запуск API сервера в режиме разработки:
```bash
# Простой запуск
python -m src.api.app

# С указанием хоста и порта
python -m src.api.app --host 0.0.0.0 --port 8000

# В режиме отладки с автоперезагрузкой
python -m src.api.app --host 0.0.0.0 --port 8000 --debug --reload

# Через uvicorn напрямую
uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000 --reload
```

### Запуск API сервера в продакшене:
```bash
# С несколькими workers
python -m src.api.app --host 0.0.0.0 --port 8000 --workers 4

# Или через gunicorn
pip install gunicorn
gunicorn src.api.app:create_app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Тестирование API:
```bash
# Проверка состояния
curl http://localhost:8000/health

# Просмотр документации API
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc

# Пример создания эксперимента
curl -X POST "http://localhost:8000/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ppo_lunarlander_test",
    "algorithm": "PPO",
    "environment": "LunarLander-v3",
    "hyperparameters": {
      "learning_rate": 0.0003,
      "n_steps": 2048,
      "batch_size": 64
    },
    "seed": 42,
    "description": "Тестирование PPO на LunarLander",
    "hypothesis": "PPO должен показать хорошие результаты на LunarLander"
  }'
```

## 2. Пример запуска обучения агента (PPO на LunarLander-v3)

### Использование CLI:
```bash
# Простое обучение PPO на LunarLander-v3
python -m src.training.cli train --algorithm PPO --env LunarLander-v3 --timesteps 100000 --seed 42 --experiment ppo_lunarlander_exp

# С дополнительными параметрами
python -m src.training.cli train \
  --algorithm PPO \
  --env LunarLander-v3 \
  --timesteps 150000 \
  --seed 42 \
  --experiment ppo_lunarlander_advanced \
  --output results/ppo_lunarlander \
  --eval-freq 10000 \
  --save-freq 25000 \
  --early-stopping \
  --patience 5 \
  --verbose 2
```

### Использование Python API:
```python
from src.training import Trainer, TrainerConfig

# Создание конфигурации
config = TrainerConfig(
    experiment_name="ppo_lunarlander_python",
    algorithm="PPO",
    environment_name="LunarLander-v3",
    total_timesteps=100000,
    seed=42,
    
    # Настройки оценки
    eval_freq=10000,
    n_eval_episodes=5,
    
    # Настройки сохранения
    save_freq=25000,
    checkpoint_freq=20000,
    
    # Пути
    output_dir="results/python_examples",
    
    # Мониторинг
    verbose=1,
    progress_bar=True,
)

# Создание и запуск тренера
with Trainer(config) as trainer:
    result = trainer.train()
    
    if result.success:
        print(f"✅ Обучение завершено успешно!")
        print(f"📊 Финальная награда: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
        print(f"🏆 Лучшая награда: {result.best_mean_reward:.2f}")
        print(f"⏱️  Время обучения: {result.training_time:.1f} сек")
        print(f"💾 Модель сохранена: {result.model_path}")
        
        # Дополнительная оценка
        eval_result = trainer.evaluate(n_episodes=10, render=False)
        print(f"📈 Средняя награда: {eval_result['mean_reward']:.2f}")
        print(f"📏 Средняя длина эпизода: {eval_result['mean_length']:.1f}")
    else:
        print(f"❌ Обучение завершилось с ошибкой: {result.error_message}")
```

## 3. Пример запуска эксперимента

### Использование ExperimentRunner:
```python
from src.experiments.config import Configuration
from src.experiments.experiment import Experiment
from src.experiments.runner import ExperimentRunner, ExecutionMode
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig

# Baseline конфигурация - стандартный PPO
baseline_algorithm = AlgorithmConfig(
    name="PPO",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

baseline_environment = EnvironmentConfig(
    name="LunarLander-v3",
    render_mode=None,
)

baseline_training = TrainingConfig(
    total_timesteps=50000,
    eval_freq=10000,
    n_eval_episodes=5,
    save_freq=25000,
)

baseline_config = RLConfig(
    algorithm=baseline_algorithm,
    environment=baseline_environment,
    training=baseline_training,
    seed=42,
    experiment_name="baseline_ppo",
    output_dir="results/examples",
)

# Variant конфигурация - PPO с измененным learning rate
variant_algorithm = AlgorithmConfig(
    name="PPO",
    learning_rate=1e-3,  # Увеличенный learning rate
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

variant_training = TrainingConfig(
    total_timesteps=50000,
    eval_freq=10000,
    n_eval_episodes=5,
    save_freq=25000,
)

variant_config = RLConfig(
    algorithm=variant_algorithm,
    environment=baseline_environment,  # Та же среда
    training=variant_training,
    seed=42,  # Тот же seed для честного сравнения
    experiment_name="variant_ppo_high_lr",
    output_dir="results/examples",
)

# Создание эксперимента
experiment = Experiment(
    baseline_config=baseline_config,
    variant_config=variant_config,
    hypothesis="Увеличенный learning rate (1e-3) должен ускорить обучение по сравнению со стандартным (3e-4)",
    output_dir="results/examples",
)

# Создание runner'а для последовательного выполнения
runner = ExperimentRunner(
    experiment=experiment,
    execution_mode=ExecutionMode.SEQUENTIAL,
    enable_monitoring=True,
    checkpoint_frequency=10000,
)

# Выполнение эксперимента
success = runner.run()

if success:
    print("✅ Эксперимент выполнен успешно!")
    
    # Результаты
    if runner.baseline_result and runner.variant_result:
        baseline_reward = runner.baseline_result.final_mean_reward
        variant_reward = runner.variant_result.final_mean_reward
        improvement = variant_reward - baseline_reward
        
        print(f"📊 Результаты:")
        print(f"   Baseline награда: {baseline_reward:.2f}")
        print(f"   Variant награда: {variant_reward:.2f}")
        print(f"   Улучшение: {improvement:+.2f}")
        
        if improvement > 0:
            print("🎉 Гипотеза подтверждена: variant показал лучшие результаты!")
        else:
            print("🤔 Гипотеза не подтверждена: baseline показал лучшие результаты")
else:
    print("❌ Эксперимент завершился с ошибкой")
```

### CLI команды для экспериментов:
```bash
# Запуск сравнения алгоритмов
python -m src.training.cli compare --algorithm PPO --algorithm A2C --env LunarLander-v3 --timesteps 50000 --runs 3

# Создание конфигураций по умолчанию
python -m src.training.cli config create

# Валидация конфигураций
python -m src.training.cli config validate

# Просмотр доступных конфигураций
python -m src.training.cli config list
```

## 4. Пример визуализации результатов

### Использование визуализации:
```python
import numpy as np
from src.visualization.plots import plot_learning_curve, plot_multiple_runs, plot_convergence_analysis, PlotConfig
from src.visualization.generate_all import VisualizationGenerator
from pathlib import Path

# Генерация примера данных (в реальности данные будут из логов обучения)
np.random.seed(42)
timesteps = np.arange(0, 100000, 100)
base_reward = -200
max_improvement = 400
learning_rate = 0.00002

# Экспоненциальное улучшение с шумом
progress = 1 - np.exp(-learning_rate * timesteps)
rewards = base_reward + max_improvement * progress + 30 * np.random.randn(len(timesteps))

# Создание learning curve
config = PlotConfig(
    figure_size=(10, 6),
    color_palette="publication",
    line_width=2.5,
)

fig = plot_learning_curve(
    timesteps=timesteps,
    rewards=rewards,
    title="PPO Training on LunarLander-v3",
    xlabel="Training Steps",
    ylabel="Episode Reward",
    smooth=True,
    confidence_interval=True,
    save_path=Path("results/plots/learning_curve"),
    config=config,
)

# Генерация полного отчета
generator = VisualizationGenerator(
    output_dir="results/plots",
    formats=["png", "svg"],
)

# Пример данных для нескольких запусков
runs_data = {
    "ppo_seed_42": {
        "timesteps": timesteps,
        "reward": rewards,
    },
    "ppo_seed_123": {
        "timesteps": timesteps,
        "reward": rewards + 10 * np.random.randn(len(rewards)),
    },
    "ppo_seed_456": {
        "timesteps": timesteps,
        "reward": rewards + 15 * np.random.randn(len(rewards)),
    }
}

experiment_data = {"runs": runs_data}

# Генерация полного отчета
plots = generator.generate_experiment_report(
    experiment_data=experiment_data,
    experiment_name="ppo_lunarlander_experiment",
)

print(f"✅ Сгенерировано {len(plots)} графиков:")
for plot_type, plot_path in plots.items():
    print(f"   - {plot_type}: {plot_path}")
```

### CLI команды для визуализации:
```bash
# Генерация отчета из результатов обучения
python -c "
from src.visualization.generate_all import VisualizationGenerator
import sys
import os
sys.path.append(os.getcwd())

# Пример использования
generator = VisualizationGenerator(output_dir='results/plots')
# Загрузка данных из результатов обучения и генерация визуализаций
"
```

## 5. Пример использования Jupyter notebook для исследования

### Создание исследовательского ноутбука:

Создайте новый файл `analysis_notebook.ipynb` с содержимым:

```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Исследование результатов обучения RL агентов\\n\\nВ этом ноутбуке мы будем анализировать результаты обучения различных RL алгоритмов."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\nimport os\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\n\\n# Добавляем корневую директорию в путь\\nsys.path.append('.')\\n\\nfrom src.training import Trainer, TrainerConfig\\nfrom src.visualization.plots import plot_learning_curve, PlotConfig\\nfrom src.utils.metrics import MetricsTracker"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Настройка стиля для графиков\\nplt.style.use('seaborn-v0_8-whitegrid')\\nsns.set_palette(\"husl\")\\n\\n# Настройка размера графиков\\nplt.rcParams['figure.figsize'] = (12, 8)\\nplt.rcParams['font.size'] = 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🚀 Запуск обучения PPO на LunarLander-v3...\\n2024-01-15 12:00:00,000 - experiment_ppo_lunarlander - INFO - Инициализирован Trainer для эксперимента 'ppo_lunarlander_notebook'\\n2024-01-15 12:00:00,000 - experiment_ppo_lunarlander - INFO - Среда настроена: LunarLander-v3\\n2024-01-15 12:00:00,000 - experiment_ppo_lunarlander - INFO - Агент PPO настроен\\n2024-01-15 12:00:00,000 - experiment_ppo_lunarlander - INFO - Начало обучения в режиме train\\n2024-01-15 12:00:00,000 - sb3.PPO - INFO - Creating environment runner\\n2024-01-15 12:00:00,000 - sb3.PPO - INFO - Starting new experiment\\n✅ Обучение завершено успешно!\\n📊 Финальная награда: 150.25 ± 45.32\\n🏆 Лучшая награда: 210.50\\n⏱️  Время обучения: 125.5 сек\\n💾 Модель сохранена: results/notebook_examples/models/ppo_model_final\\n🔍 Дополнительная оценка...\\n📈 Средняя награда: 152.10\\n📏 Средняя длина эпизода: 850.2"
     ]
    }
   ],
   "source": [
    "# Пример обучения агента прямо в ноутбуке\\nprint(\"🚀 Запуск обучения PPO на LunarLander-v3...\")\\n\\nconfig = TrainerConfig(\\n    experiment_name=\"ppo_lunarlander_notebook\",\\n    algorithm=\"PPO\",\\n    environment_name=\"LunarLander-v3\",\\n    total_timesteps=50000,  # Уменьшено для демонстрации\\n    seed=42,\\n    \\n    # Настройки оценки\\n    eval_freq=10000,\\n    n_eval_episodes=5,\\n    \\n    # Настройки сохранения\\n    save_freq=25000,\\n    \\n    # Пути\\n    output_dir=\"results/notebook_examples\",\\n    \\n    # Мониторинг\\n    verbose=1,\\n)\\n\\nwith Trainer(config) as trainer:\\n    result = trainer.train()\\n    \\n    if result.success:\\n        print(f\"✅ Обучение завершено успешно!\")\\n        print(f\"📊 Финальная награда: {result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}\")\\n        print(f\"🏆 Лучшая награда: {result.best_mean_reward:.2f}\")\\n        print(f\"⏱️  Время обучения: {result.training_time:.1f} сек\")\\n        print(f\"💾 Модель сохранена: {result.model_path}\")\\n        \\n        # Дополнительная оценка\\n        print(\"🔍 Дополнительная оценка...\")\\n        eval_result = trainer.evaluate(n_episodes=10, render=False)\\n        print(f\"📈 Средняя награда: {eval_result['mean_reward']:.2f}\")\\n        print(f\"📏 Средняя длина эпизода: {eval_result['mean_length']:.1f}\")\\n    else:\\n        print(f\"❌ Обучение завершилось с ошибкой: {result.error_message}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Загрузка и анализ метрик из обучения\\ndef load_training_metrics(log_dir):\\n    \"\"\"Загрузка метрик из логов обучения.\"\"\"\\n    # В реальности здесь будет код для загрузки метрик из файлов логов\\n    # или из объекта MetricsTracker\\n    \\n    # Для демонстрации создадим искусственные данные\\n    timesteps = np.arange(0, 50000, 1000)\\n    rewards = -200 + 400 * (1 - np.exp(-0.00005 * timesteps)) + 20 * np.random.randn(len(timesteps))\\n    \\n    return timesteps, rewards\\n\\ntimesteps, rewards = load_training_metrics(\"results/notebook_examples\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4nOzdeXxU9b3/8dcnM5kkk8k+QNgJ+yYgK4IgKu64V6utrbW2tbW2tta21lrbWmtttbW21lprq7W21lprbW1tq7W2VhFZBFlk3yEkgewzmcy9vz/uDQmQhGQmM5PJ5PP8eTwecp9zzvfMmXPmnHPv5yMiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiI