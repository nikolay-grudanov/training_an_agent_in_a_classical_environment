# Эксперимент PPO Seed 42 - 500K Timesteps

**Дата:** 5 февраля 2026 г.
**Статус:** ✅ УСПЕШНО ЗАВЕРШЁН
**Convergence:** ДА (>200 reward)

---

## 📊 Результаты

### Финальная модель (500K timesteps):
```
Mean Reward: 225.59 ± 22.18
Convergence: YES (>200)
Duration: 190.0s (3.17 минуты)
Speed: 3,025 it/s
```

### Лучшая модель (400K timesteps):
```
Mean Reward: 235.24 ± 25.52
Convergence: YES (>200)
```

**Лучший чекпоинт:** `best_model.zip` (checkpoint_400000.zip)

---

## 🎯 Параметры обучения

| Параметр | Значение |
|----------|----------|
| Алгоритм | PPO |
| Timesteps | 500,000 |
| Seed | 42 |
| Gamma | 0.999 |
| Entropy Coefficient | 0.01 |
| GAE Lambda | 0.98 |
| N Steps | 1024 |
| N Epochs | 4 |
| Batch Size | 64 |
| Learning Rate | 3e-4 |
| Device | CPU |

---

## 📈 Кривая обучения

| Timesteps | Mean Reward | Std Reward | Статус |
|-----------|-------------|-------------|----------|
| 50K | -442.30 | 119.41 | ❌ NO |
| 100K | -79.26 | 28.49 | ❌ NO |
| 150K | -109.41 | 29.62 | ❌ NO |
| 200K | 28.23 | 69.89 | ❌ NO |
| 250K | -12.90 | 79.32 | ❌ NO |
| 300K | -36.50 | 22.50 | ❌ NO |
| 350K | 143.86 | 117.56 | ❌ NO |
| **400K** | **243.45** | **22.85** | **✅ YES** |
| 450K | 205.41 | 23.54 | ✅ YES |
| 500K (финал) | 238.33 | 20.60 | ✅ YES |

**Вывод:** Модель достигла пика на 400K timesteps (243.45), на 500K немного снизилась до 238.33.

---

## 📁 Файлы эксперимента

### Основные файлы:
```
ppo_seed42_500K/
├── ppo_seed42_500K_model.zip  ← Финальная модель (500K timesteps)
├── best_model.zip               ← Лучшая модель (400K timesteps, 243.45 reward)
├── config.json                 ← Конфигурация эксперимента
├── metrics.csv                 ← Метрики обучения
├── eval_log.csv               ← Логи оценки (каждые 5K timesteps)
└── checkpoints/                ← Все чекпоинты
    ├── checkpoint_50000.zip
    ├── checkpoint_100000.zip
    ├── ...
    └── checkpoint_500000.zip
```

### Рекомендации по использованию:

1. **Для максимальной производительности:**
   ```bash
   # Использовать лучший чекпоинт
   python -c "from src.training.evaluation import evaluate_agent; result = evaluate_agent('results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip', n_eval_episodes=20); print(result)"
   ```

2. **Для стабильности:**
   ```bash
   # Использовать финальную модель
   python -c "from src.training.evaluation import evaluate_agent; result = evaluate_agent('results/experiments/ppo_seed42/ppo_seed42_500K/ppo_seed42_500K_model.zip', n_eval_episodes=20); print(result)"
   ```

---

## 🔍 Анализ

### Что хорошо:
- ✅ Достигли цели >200 reward
- ✅ Стабильность отличная (Std ~20-25)
- ✅ Обучение прошло быстро (3.17 минуты)
- ✅ Утилизация CPU полная (32 потока)

### Что можно улучшить:
- ⚠️ Обучение было нестабильным до 350K timesteps
- ⚠️ Модель достигла пика на 400K, затем немного снизилась

### Рекомендации:
1. ❌ **НЕ дообучать дальше** (лучший результат уже на 400K)
2. ✅ **Использовать checkpoint_400000.zip** как основную модель
3. ✅ **Пробовать другие seeds** для усреднения результатов

---

## 📚 Связанные документы

- **[EXPERIMENT_ORGANIZATION.md](../../EXPERIMENT_ORGANIZATION.md)** - Организация экспериментов
- **[SYSTEM_SPECS.md](../../SYSTEM_SPECS.md)** - Характеристики системы
- **[GRID_SEARCH_RESULTS.md](../../GRID_SEARCH_RESULTS.md)** - Оптимизация гиперпараметров

---

## 💡 Быстрый старт

### Оценить лучшую модель:
```bash
python -c "
from src.training.evaluation import evaluate_agent
result = evaluate_agent('results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip', n_eval_episodes=20)
print(f'Mean: {result[\"mean_reward\"]:.2f} ± {result[\"std_reward\"]:.2f}')
print(f'Convergence: {\"YES\" if result[\"convergence_achieved\"] else \"NO\"}')
"
```

### Сравнить с другими моделями:
```bash
python -c "
from src.training.evaluation import evaluate_agent

models = [
    'results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip',
    'results/experiments/ppo_seed999/ppo_seed999_model.zip',
]

for model in models:
    result = evaluate_agent(model, n_eval_episodes=10)
    name = model.split('/')[-2]
    print(f'{name}: {result[\"mean_reward\"]:.2f} ± {result[\"std_reward\"]:.2f}')
"
```

---

**Создано:** 5 февраля 2026 г.
