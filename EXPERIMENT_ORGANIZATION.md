# 📦 Организация экспериментов с timesteps в названии

**Дата:** 5 февраля 2026 г.
**Обновлено:** baseline_training.py теперь автоматически включает timesteps в название эксперимента

---

## 🎯 Проблема

Ранее эксперименты назывались только по seed:
```
results/experiments/
├── ppo_seed42/
├── ppo_seed123/
└── ppo_seed999/
```

**Проблемы:**
- ❌ Не понятно сколько шагов использовалось в каждом эксперименте
- ❌ Невозможно сравнить модели с разным числом шагов
- ❌ Приходилось переименовывать вручную

---

## ✅ Решение

### Автоматическое включение timesteps в название

**Изменено:** `src/experiments/completion/baseline_training.py`

**Как работает:**
```python
# Раньше (строка 273)
exp_id = experiment_id or f"{algo.value.lower()}_seed{seed}"
# Результат: ppo_seed42

# Теперь (строка 273-274)
exp_id = experiment_id or f"{algo.value.lower()}_seed{seed}_{timesteps//1000}K"
# Результат: ppo_seed42_500K
```

**Формат названия:** `{алгоритм}_seed{seed}_{timesteps//1000}K`

### Примеры новых названий:

| Timesteps | Seed | Название эксперимента |
|-----------|-------|---------------------|
| 50,000 | 42 | `ppo_seed42_50K` |
| 100,000 | 42 | `ppo_seed42_100K` |
| 150,000 | 42 | `ppo_seed42_150K` |
| 500,000 | 42 | `ppo_seed42_500K` |
| 1,000,000 | 999 | `ppo_seed999_1000K` |

---

## 📁 Новая структура директорий

### До изменений:
```
results/experiments/
├── ppo_seed42/
│   ├── ppo_seed42_model.zip
│   ├── config.json (timesteps: ???)
│   ├── metrics.csv
│   └── checkpoints/
│       ├── checkpoint_50000.zip
│       └── ...
└── ppo_seed999/
    └── ...
```

### После изменений:
```
results/experiments/
├── ppo_seed42_50K/
│   ├── ppo_seed42_50K_model.zip
│   ├── config.json (timesteps: 50000)
│   ├── metrics.csv
│   └── checkpoints/
│       └── checkpoint_50000.zip
│
├── ppo_seed42_100K/
│   ├── ppo_seed42_100K_model.zip
│   ├── config.json (timesteps: 100000)
│   ├── metrics.csv
│   └── checkpoints/
│       └── checkpoint_100000.zip
│
├── ppo_seed42_500K/
│   ├── ppo_seed42_500K_model.zip  ← **Финальная модель**
│   ├── best_model.zip               ← **Лучший чекпоинт**
│   ├── config.json (timesteps: 500000)
│   ├── metrics.csv
│   ├── eval_log.csv
│   └── checkpoints/
│       ├── checkpoint_50000.zip
│       ├── checkpoint_100000.zip
│       ├── ...
│       └── checkpoint_500000.zip
│
└── ppo_seed999_1000K/
    └── ...
```

---

## 🎯 Использование

### Стандартный запуск:

```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 500000 \
    --seed 42 \
    --gamma 0.999 \
    --ent-coef 0.01 \
    --gae-lambda 0.98 \
    --n-steps 1024 \
    --n-epochs 4 \
    --device cpu
```

**Результат:**
- Директория: `results/experiments/ppo_seed42_500K/`
- Модель: `ppo_seed42_500K_model.zip`
- Timesteps: 500,000 (записано в config.json)

### Обучение нескольких моделей с разными timesteps:

```bash
# Быстрая проверка (50K)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 50000 --seed 42 --device cpu

# Обычный эксперимент (500K)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --device cpu

# Долгий эксперимент (1M)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 1000000 --seed 42 --device cpu
```

**Результат:**
```
results/experiments/
├── ppo_seed42_50K/
├── ppo_seed42_500K/
└── ppo_seed42_1000K/
```

---

## 📊 Анализ результатов

### Сравнение разных timesteps:

```bash
# Оценить все модели
for dir in results/experiments/ppo_seed42_*/; do
    model="$dir/$(basename $dir)_model.zip"
    if [ -f "$model" ]; then
        echo "Evaluating: $(basename $dir)"
        python -c "from src.training.evaluation import evaluate_agent; result = evaluate_agent('$model', n_eval_episodes=10); print(f'Reward: {result[\"mean_reward\"]:.2f} ± {result[\"std_reward\"]:.2f}')"
    fi
done
```

**Результат:**
```
Evaluating: ppo_seed42_50K
Reward: -50.23 ± 120.45

Evaluating: ppo_seed42_500K
Reward: 225.59 ± 22.18

Evaluating: ppo_seed42_1000K
Reward: 230.12 ± 25.34
```

---

## 📋 Рекомендации по использованию

### 1. **Создание серии экспериментов**

```bash
# Серия с разными timesteps (для поиска оптимального)
for timesteps in 100000 150000 200000 300000 500000; do
    python -m src.experiments.completion.baseline_training \
        --algo ppo --timesteps $timesteps --seed 42 \
        --gamma 0.999 --ent-coef 0.01 --device cpu
done
```

**Результат:**
```
results/experiments/
├── ppo_seed42_100K/
├── ppo_seed42_150K/
├── ppo_seed42_200K/
├── ppo_seed42_300K/
└── ppo_seed42_500K/
```

### 2. **Сравнение результатов**

```bash
# Создать таблицу сравнения
echo "Timesteps | Mean Reward | Std Reward"
echo "---------|-------------|-----------"
for timesteps in 100000 150000 200000 300000 500000; do
    dir="results/experiments/ppo_seed42_${timesteps//1000}K"
    model="$dir/ppo_seed42_${timesteps//1000}K_model.zip"
    if [ -f "$model" ]; then
        result=$(python -c "from src.training.evaluation import evaluate_agent; r = evaluate_agent('$model', n_eval_episodes=10); print(f'{r[\"mean_reward\"]:.2f} {r[\"std_reward\"]:.2f}')")
        echo "$timesteps | $result"
    fi
done
```

### 3. **Поиск лучшего чекпоинта**

Для каждого эксперимента сохраняются все чекпоинты. Лучшая модель обычно НЕ финальная:

```bash
# Оценить все чекпоинты в эксперименте
for checkpoint in results/experiments/ppo_seed42_500K/checkpoints/*.zip; do
    name=$(basename $checkpoint .zip)
    result=$(python -c "from src.training.evaluation import evaluate_agent; r = evaluate_agent('$checkpoint', n_eval_episodes=10); print(f'{r[\"mean_reward\"]:.2f}')")
    echo "$name | $result"
done | sort -t '|' -k2 -rn
```

**Результат:**
```
checkpoint_400000.zip | 243.45  ← ЛУЧШИЙ!
checkpoint_500000.zip | 238.33  ← Финальный
checkpoint_450000.zip | 205.41
...
```

---

## 🔧 Возврат к старому поведению

Если нужно вернуть старое поведение (без timesteps в названии):

**Редактировать:** `src/experiments/completion/baseline_training.py`

**Строка 273-274:**
```python
# Было (новое поведение)
exp_id = experiment_id or f"{algo.value.lower()}_seed{seed}_{timesteps//1000}K"

# Вернуть (старое поведение)
exp_id = experiment_id or f"{algo.value.lower()}_seed{seed}"
```

---

## 📈 Примеры использования

### Пример 1: Найти оптимальное количество шагов

```bash
# Обучить с разными timesteps
for timesteps in 100000 150000 200000 300000 500000; do
    python -m src.experiments.completion.baseline_training \
        --algo ppo --timesteps $timesteps --seed 42 \
        --gamma 0.999 --ent-coef 0.01 --device cpu
done

# Построить график: reward vs timesteps
python -c "
import matplotlib.pyplot as plt
import pandas as pd

# Данные (замените на ваши реальные данные)
data = {
    'timesteps': [100K, 150K, 200K, 300K, 500K],
    'reward': [50, 120, 180, 220, 235]
}
df = pd.DataFrame(data)

plt.plot(df['timesteps'], df['reward'], marker='o')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Learning Curve: Reward vs Timesteps')
plt.grid(True)
plt.savefig('reward_vs_timesteps.png')
print('Graph saved: reward_vs_timesteps.png')
"
```

### Пример 2: Сравнение разных seeds

```bash
# Обучить с разными seeds
for seed in 42 123 999; do
    python -m src.experiments.completion.baseline_training \
        --algo ppo --timesteps 500000 --seed $seed \
        --gamma 0.999 --ent-coef 0.01 --device cpu
done

# Сравнить результаты
for seed in 42 123 999; do
    model="results/experiments/ppo_seed${seed}_500K/ppo_seed${seed}_500K_model.zip"
    result=$(python -c "from src.training.evaluation import evaluate_agent; r = evaluate_agent('$model', n_eval_episodes=10); print(f'{r[\"mean_reward\"]:.2f} {r[\"std_reward\"]:.2f}')")
    echo "Seed $seed: $result"
done
```

---

## 📚 Связанные документы

- **[SYSTEM_SPECS.md](SYSTEM_SPECS.md)** - Характеристики вашей системы
- **[CPU_vs_GPU_Comparison.md](CPU_vs_GPU_Comparison.md)** - Сравнение обучения CPU vs GPU
- **[GRID_SEARCH_RESULTS.md](GRID_SEARCH_RESULTS.md)** - Результаты оптимизации гиперпараметров
- **[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)** - Текущее состояние проекта

---

## ✅ Итог

**Что сделано:**
- ✅ Обновлён `baseline_training.py` для автоматического включения timesteps в название
- ✅ Новая структура директорий: `{алгоритм}_seed{seed}_{timesteps}K`
- ✅ Улучшенная организация экспериментов
- ✅ Легкое сравнение результатов с разными timesteps

**Использование:**
```bash
# Просто запустите обучение - имя эксперимента будет автоматически включать timesteps
python -m src.experiments.completion.baseline_training --algo ppo --timesteps 500000 --seed 42 --device cpu

# Результат: results/experiments/ppo_seed42_500K/
```

---

**Обновлено:** 5 февраля 2026 г.
