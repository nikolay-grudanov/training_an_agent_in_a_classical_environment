# Рекомендации по улучшению RL проекта

**Автор:** ML Theory Agent
**Дата:** 17 февраля 2026
**Цель:** Исправить методологические ошибки и улучшить качество проекта

---

## 🔴 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (НЕОБХОДИМО ДО СДАЧИ)

### 1. Исправить Cherry-Picking в анализе метрик

**Проблема:** Код использует `best_eval_reward` (максимальную награду за всё обучение) вместо `final_eval_reward` (финальную награду после обучения).

**Где исправить:** `src/reporting/analyze_models.py`, строки 367-372

**Было:**
```python
if best_reward >= REWARD_THRESHOLD:
    convergence_status = STATUS_CONVERGED
elif best_reward > 0:
    convergence_status = STATUS_NOT_CONVERGED
else:
    convergence_status = STATUS_UNKNOWN
```

**Должно быть:**
```python
if final_reward >= REWARD_THRESHOLD:
    convergence_status = STATUS_CONVERGED
elif final_reward > 0:
    convergence_status = STATUS_NOT_CONVERGED
else:
    convergence_status = STATUS_UNKNOWN
```

**Последствия после исправления:**
- ppo_seed123: CONVERGED → NOT_CONVERGED (final_reward 89.88 < 200)
- ppo_seed999: CONVERGED → NOT_CONVERGED (final_reward 188.83 < 200)
- Реальный процент сходимости PPO (seed=42): 66.7% (вместо 72.7%)

**Как проверить после исправления:**
```bash
python -m src.reporting.analyze_models --check-hypotheses
# Результат: 4/6 моделей PPO сошлись (seed=42), вместо 6/11
```

---

### 2. Пересмотреть A2C эксперименты

**Проблема:** A2C показывает финальные награды от -1863 до 116, что значительно ниже порога 200.

**Возможные причины:**
- Неправильные гиперпараметры A2C для LunarLander-v3
- Недостаточное количество шагов обучения (500K может быть мало для A2C)
- A2C менее стабильный алгоритм для этой среды

**Рекомендации:**
1. Увеличить количество шагов для A2C до 1M+:
   ```bash
   python -m src.experiments.completion.baseline_training \
       --algo a2c \
       --timesteps 1000000 \
       --seed 42 \
       --gamma 0.99
   ```

2. Попробовать разные learning rates для A2C:
   ```bash
   # lr=1e-4 (более агрессивный)
   python -m src.experiments.completion.baseline_training \
       --algo a2c \
       --timesteps 1000000 \
       --seed 42 \
       --learning-rate 1e-4

   # lr=1e-3 (более консервативный)
   python -m src.experiments.completion.baseline_training \
       --algo a2c \
       --timesteps 1000000 \
       --seed 42 \
       --learning-rate 1e-3
   ```

3. Увеличить n_steps для лучшего сэмплирования:
   ```bash
   python -m src.experiments.completion.baseline_training \
       --algo a2c \
       --timesteps 1000000 \
       --seed 42 \
       --n-steps 4096
   ```

---

## 🟡 ВАЖНЫЕ УЛУЧШЕНИЯ (РЕКОМЕНДУЕТСЯ)

### 3. Улучшить стабильность по seed'ам

**Проблема:** Высокая дисперсия по seed'ам (seed=42: 224-247, seed=123: 89.88, seed=999: 188.83)

**Рекомендация 3.1: Увеличить n_steps**
Изменить n_steps с 2048 до 4096 в `src/agents/ppo_agent.py`:
```python
# Было:
n_steps: int = 2048

# Должно быть:
n_steps: int = 4096  # Более устойчивое сэмплирование
```

**Рекомендация 3.2: Использовать learning rate schedule**
Добавить linear decay для learning rate:
```python
# В src/agents/ppo_agent.py
use_lr_schedule: bool = True
lr_schedule_type: str = "linear"
lr_final_ratio: float = 0.1  # LR уменьшится до 10% от начального
```

**Рекомендация 3.3: Добавить больше экспериментов с разными seed'ами**
```bash
# Запуск с seed=50, 100, 200 для усреднения
for seed in 50 100 200; do
    python -m src.experiments.completion.baseline_training \
        --algo ppo \
        --timesteps 500000 \
        --seed $seed \
        --gamma 0.999
done
```

---

### 4. Объединить разрозненные MD файлы

**Проблема:** Слишком много похожих отчётов (FINAL_REPORT.md, PROJECT_COMPLETION_REPORT.md, EXPERIMENT_ORGANIZATION.md)

**Рекомендация:**
1. Оставить README.md как основной документ проекта
2. Оставить PROJECT_STRUCTURE.md как техническую документацию
3. Создать один итоговый REPORT.md с кратким анализом
4. Архивировать или удалить устаревшие файлы

**Как объединить:**
```bash
# Создать архив для старых файлов
mkdir -p archive
mv FINAL_REPORT.md PROJECT_COMPLETION_REPORT.md EXPERIMENT_ORGANIZATION.md archive/

# Создать единый REPORT.md
cat > REPORT.md << 'EOF'
# Отчёт по RL проекту: LunarLander-v3

## Краткое описание
... (краткое содержание из README.md) ...

## Результаты
... (результаты из AUDIT_REPORT.md) ...

## Анализ
... (анализ из AUDIT_REPORT.md) ...

## Гипотезы
... (гипотезы из AUDIT_REPORT.md) ...
EOF
```

---

### 5. Добавить графики обучения для каждого эксперимента

**Проблема:** Текущие графики показывают только финальные результаты, без динамики обучения.

**Рекомендация:** Генерировать reward vs timestep графики для каждого эксперимента:
```bash
# Создать директорию для графиков
mkdir -p results/plots

# Для каждого эксперимента создать график
for exp_dir in results/experiments/*/; do
    exp_name=$(basename $exp_dir)
    if [ -f "$exp_dir/eval_log.csv" ]; then
        python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('$exp_dir/eval_log.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['timesteps'], df['mean_reward'])
plt.fill_between(
    df['timesteps'],
    df['mean_reward'] - df['std_reward'],
    df['mean_reward'] + df['std_reward'],
    alpha=0.2
)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('$exp_name')
plt.grid(True)
plt.savefig('results/plots/${exp_name}_reward_curve.png')
plt.close()
"
    fi
done
```

---

## 🟢 ПОЛЕЗНЫЕ УЛУЧШЕНИЯ (ОПЦИОНАЛЬНО)

### 6. Добавить документацию по воспроизводимости

**Рекомендация:** Создать `REPRODUCIBILITY.md` с инструкциями:
```markdown
# Воспроизводимость экспериментов

## Установка окружения
```bash
conda env create -f environment.yml
conda activate rocm
```

## Запуск эксперимента PPO
```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 500000 \
    --seed 42 \
    --gamma 0.999 \
    --ent-coef 0.01 \
    --gae-lambda 0.98 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --n-epochs 10 \
    --batch-size 64 \
    --device cpu
```

## Ожидаемые результаты
- Финальная награда: >200 для seed=42
- Время обучения: ~3-5 минут для 500K шагов на CPU

## Проверка воспроизводимости
```bash
# Запустить два раза с одинаковым seed
python -m src.experiments.completion.baseline_training --algo ppo --seed 42
python -m src.experiments.completion.baseline_training --algo ppo --seed 42

# Сравнить eval_log.csv (должны быть идентичны)
diff results/experiments/ppo_seed42/eval_log.csv \
     results/experiments/ppo_seed42_2/eval_log.csv
```
```

---

### 7. Добавить unit тесты для критических функций

**Рекомендация:** Добавить тесты в `tests/unit/`:
```python
# tests/unit/test_analyze_models.py
import pytest
from pathlib import Path
from src.reporting.analyze_models import extract_best_metrics, extract_final_metrics

def test_extract_best_metrics_returns_max():
    """Проверить, что extract_best_metrics возвращает максимум."""
    df = pd.DataFrame({
        'timesteps': [1000, 2000, 3000],
        'mean_reward': [100, 200, 150]
    })
    best_reward, best_std, best_timesteps = extract_best_metrics(df)

    assert best_reward == 200.0, f"Ожидалось 200, получено {best_reward}"
    assert best_timesteps == 2000, f"Ожидалось 2000, получено {best_timesteps}"

def test_extract_final_metrics_returns_last():
    """Проверить, что extract_final_metrics возвращает последнее значение."""
    df = pd.DataFrame({
        'timesteps': [1000, 2000, 3000],
        'mean_reward': [100, 200, 150]
    })
    final_reward, final_std = extract_final_metrics(df)

    assert final_reward == 150.0, f"Ожидалось 150, получено {final_reward}"
    assert final_std == pytest.approx(0.0), "Std должен быть примерно 0 для одного значения"
```

**Запуск тестов:**
```bash
pytest tests/unit/test_analyze_models.py -v
```

---

### 8. Оптимизировать структуру проекта

**Рекомендация:** Удалить или архивировать старые файлы:
```bash
# Создать архив для старых экспериментов
mkdir -p archive/old_experiments
mv results/demo_experiment archive/old_experiments/
mv results/test_rl_experiment archive/old_experiments/
mv results/test_experiment archive/old_experiments/

# Удалить временные файлы
rm -rf results/reproducibility/runs/
rm -rf results/dependencies/snapshot_experiment_*/

# Удалить пустые директории
find results -type d -empty -delete
```

---

## 📋 ПРОВЕРОЧНЫЙ СПИСОК ДЛЯ СДАЧИ

### Минимум (обязательно):
- [ ] Исправлена cherry-picking ошибка в `analyze_models.py`
- [ ] Исправленные результаты экспериментов (final_reward вместо best_reward)
- [ ] Обновлён финальный отчёт с корректными метриками
- [ ] README.md содержит краткое описание задачи и подхода

### Рекомендуется:
- [ ] Добавлены графики reward vs timestep для каждого эксперимента
- [ ] Создан `REPRODUCIBILITY.md` с инструкциями
- [ ] Объединены разрозненные MD файлы
- [ ] Удалены лишние файлы из `results/`

### Опционально:
- [ ] Добавлены unit тесты для критических функций
- [ ] A2C эксперименты пересмотрены и улучшены
- [ ] Добавлены эксперименты с разными seed'ами (50, 100, 200)
- [ ] Увеличен n_steps с 2048 до 4096

---

## 🎯 СРОКИ

- **Исправление критических ошибок:** 1-2 часа
- **Выполнение рекомендуемых улучшений:** 2-4 часа
- **Выполнение опциональных улучшений:** 4-8 часов

**Общее время:** 7-14 часов для полной доработки проекта

---

## 📞 ПОМОЩЬ

Если возникнут вопросы при исправлении ошибок, обратитесь к:
- Документация Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Документация Gymnasium: https://gymnasium.farama.org/
- Форум PyTorch: https://discuss.pytorch.org/

---

**Документ создан:** 17 февраля 2026
**Автор:** ML Theory Agent
