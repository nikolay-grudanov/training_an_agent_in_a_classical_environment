# Quickstart: Финальный отчёт и документация

**Feature**: 005-final-report | **Date**: 2026-02-05
**Purpose**: Быстрый старт для создания финального отчёта, визуализаций и демо-видео
**Time Estimate**: ~15-30 минут (без повторного обучения)

---

## 📋 ПЕРЕД НАЧАЛОМ

### 1. Активация окружения

```bash
conda activate rocm
```

### 2. Проверка зависимостей

```bash
python --version              # Expected: Python 3.10.14
pip list | grep torch        # Expected: torch 2.5.1+rocm6.2
pip list | grep stable-baselines3  # Expected: stable-baselines3 2.7.1
pip list | grep matplotlib   # Expected: matplotlib 3.9.4
pip list | grep pandas       # Expected: pandas 2.2.2
pip list | grep imageio      # Expected: imageio 2.35.1
```

### 3. Переход на ветку (если нужно)

```bash
git checkout 005-final-report
git pull origin 005-final-report
```

### 4. Проверка существующих моделей

```bash
# Проверить, что модели существуют
ls -lh results/experiments/ppo_seed42/ppo_seed42_500K/
ls -lh results/experiments/ppo_seed999/
ls -lh results/experiments/gamma_*/

# Лучший чекпоинт должен существовать
test -f results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip && echo "✅ Best model OK"
```

---

## 🚀 БЫСТРЫЙ СТАРТ (ВСЕ ЭТАПЫ)

Если вы хотите создать все артефакты финального отчёта последовательно, выполните команды ниже:

```bash
# ========================================
# ЭТАП 1: Создание директории для отчётов
# ========================================
echo "=== ЭТАП 1: Создание директории для отчётов ==="
mkdir -p results/reports

# ========================================
# ЭТАП 2: Анализ всех моделей
# ========================================
echo "=== ЭТАП 2: Анализ всех моделей ==="
python -m src.reporting.analyze_models \
    --experiments-dir results/experiments \
    --output-dir results/reports \
    --verbose

# Проверка результатов
echo "=== Таблица сравнения ==="
cat results/reports/model_comparison.csv | head -20

# ========================================
# ЭТАП 3: Генерация графиков
# ========================================
echo "=== ЭТАП 3: Генерация графиков ==="

# Кривая обучения
python -m src.reporting.generate_plots learning-curve \
    --metrics results/experiments/ppo_seed42/ppo_seed42_500K/metrics.csv \
    --output results/reports/reward_vs_timesteps.png \
    --title "Кривая обучения PPO (seed=42, gamma=0.999)" \
    --dpi 300

# Сравнительная диаграмма
python -m src.reporting.generate_plots comparison \
    --comparison results/reports/model_comparison.csv \
    --output results/reports/agent_comparison.png \
    --title "Сравнение итоговых наград агентов" \
    --dpi 300

# Проверка результатов
ls -lh results/reports/*.png

# ========================================
# ЭТАП 4: Генерация демо-видео (топ-3 модели)
# ========================================
echo "=== ЭТАП 4: Генерация демо-видео ==="
python -m src.reporting.generate_videos top-n \
    --comparison results/reports/model_comparison.csv \
    --output-dir results/reports \
    --top-n 3 \
    --episodes 5 \
    --fps 30

# Проверка результатов
ls -lh results/reports/demo_*.mp4

# ========================================
# ЭТАП 5: Сохранение зависимостей
# ========================================
echo "=== ЭТАП 5: Сохранение зависимостей ==="
pip freeze > results/reports/requirements.txt

# Проверка результата
echo "=== Первые 10 строк requirements.txt ==="
head -10 results/reports/requirements.txt

# ========================================
# ЭТАП 6: Создание финального отчёта
# ========================================
echo "=== ЭТАП 6: Создание финального отчёта ==="
python -m src.reporting.generate_report \
    --comparison results/reports/model_comparison.csv \
    --learning-curve results/reports/reward_vs_timesteps.png \
    --comparison-chart results/reports/agent_comparison.png \
    --videos results/reports/demo_best_model.mp4 \
    --videos results/reports/demo_second_best.mp4 \
    --videos results/reports/demo_third_best.mp4 \
    --output results/reports/FINAL_REPORT.md \
    --seed 42

# Проверка результата
echo "=== Финальный отчёт создан ==="
wc -l results/reports/FINAL_REPORT.md

# ========================================
# ЭТАП 7: Копирование отчёта в корень
# ========================================
echo "=== ЭТАП 7: Копирование отчёта в корень ==="
cp results/reports/FINAL_REPORT.md FINAL_REPORT.md
echo "✅ FINAL_REPORT.md скопирован в корень проекта"

# ========================================
# ЭТАП 8: Обновление README.md (ручное редактирование)
# ========================================
echo "=== ЭТАП 8: Обновление README.md ==="
echo "Откройте README.md и добавьте:"
echo "  1. Ссылку на FINAL_REPORT.md"
echo "  2. Встраиваемые графики из results/reports/"
echo "  3. Инструкции запуска (см. ниже)"

# Открыть в редакторе (опционально)
# nano README.md
# или
# code README.md

echo "=== ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ ==="
```

---

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

| Этап | Ожидаемый результат | Время |
|-------|-------------------|--------|
| 1 | Директория results/reports создана | <1 сек |
| 2 | Таблица model_comparison.csv и .json созданы | ~30 сек |
| 3 | Два графика созданы (PNG, 300 DPI) | ~10 сек |
| 4 | 3 демо-видео созданы (по 5 эпизодов) | ~2-3 мин |
| 5 | requirements.txt создан | <1 сек |
| 6 | FINAL_REPORT.md создан | <5 сек |
| 7 | FINAL_REPORT.md скопирован в корень | <1 сек |
| 8 | README.md обновлён | ~5 мин |
| **ИТОГО** | **Все артефакты созданы** | **~3-5 минут** |

---

## 🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ

После завершения всех этапов, проверьте:

### 1. Артефакты отчёта

```bash
# Все файлы созданы?
ls -lh results/reports/

# Таблица сравнения существует?
test -f results/reports/model_comparison.csv && echo "✅ Comparison table OK"
test -f results/reports/model_comparison.json && echo "✅ Comparison JSON OK"

# Графики существуют?
test -f results/reports/reward_vs_timesteps.png && echo "✅ Learning curve OK"
test -f results/reports/agent_comparison.png && echo "✅ Comparison chart OK"

# Видео существуют?
test -f results/reports/demo_best_model.mp4 && echo "✅ Best video OK"
test -f results/reports/demo_second_best.mp4 && echo "✅ Second video OK"
test -f results/reports/demo_third_best.mp4 && echo "✅ Third video OK"

# Отчёт существует?
test -f results/reports/FINAL_REPORT.md && echo "✅ Report OK"
test -f FINAL_REPORT.md && echo "✅ Report in root OK"

# Зависимости сохранены?
test -f results/reports/requirements.txt && echo "✅ Requirements OK"
```

### 2. Содержание таблицы сравнения

```bash
# Проверить топ модели
echo "=== Топ 3 модели ==="
head -5 results/reports/model_comparison.csv

# Ожидается (пример):
# experiment_id,best_eval_reward,...
# ppo_seed42_500K,243.45,...
# ppo_seed42_400K,235.24,...
# ppo_seed999,195.09,...
```

### 3. Размеры файлов

```bash
# Графики должны быть > 50KB (300 DPI)
du -h results/reports/*.png

# Видео должны быть > 2MB
du -h results/reports/demo_*.mp4
```

### 4. Проверка отчёта

```bash
# Все секции присутствуют?
grep -E "(Краткое описание задачи|Код обучения|Графики|Анализ)" FINAL_REPORT.md

# Seed задокументирован?
grep "seed=42" FINAL_REPORT.md

# Графики встраиваются?
grep -E "reward_vs_timesteps.png|agent_comparison.png" FINAL_REPORT.md
```

---

## 🆘 ПРОБЛЕМЫ И РЕШЕНИЯ

### Ошибка: ModuleNotFoundError при импорте

**Ошибка**: `ModuleNotFoundError: No module named 'src.reporting'`

**Решение**:
```bash
# Создать __init__.py
touch src/reporting/__init__.py

# Или запустить как модуль
python -m src.reporting.analyze_models  # ✅ Правильно
python src/reporting/analyze_models.py  # ❌ Не работает
```

---

### Ошибка: Нет найденных экспериментов

**Ошибка**: `ValueError: No valid experiments found`

**Решение**:
```bash
# Проверить путь к экспериментам
ls results/experiments/

# Указать правильный путь
python -m src.reporting.analyze_models \
    --experiments-dir results/experiments \
    --verbose
```

---

### Ошибка: metrics.csv не найден

**Ошибка**: `FileNotFoundError: metrics.csv not found`

**Решение**:
```bash
# Проверить наличие metrics.csv
find results/experiments/ -name "metrics.csv" -type f

# Если файлы отсутствуют, нужно обучить модели заново
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42
```

---

### Ошибка: Нет ffmpeg для создания видео

**Ошибка**: `imageio.core.fetching._NeedDownloadError: Need ffmpeg exe.`

**Решение**:
```bash
# Установить ffmpeg
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS

# Или imageio автоматически скачает ffmpeg
pip install imageio-ffmpeg
```

---

### Ошибка: Видео не воспроизводится

**Ошибка**: Видео не открывается в плеере

**Решение**:
```bash
# Проверить кодек (должен быть H.264)
ffprobe results/reports/demo_best_model.mp4 | grep codec_name

# Если кодек неверный, перегенерировать с libx264
python -m src.reporting.generate_videos single \
    --model results/experiments/ppo_seed42/best_model.zip \
    --output results/reports/demo_best_model.mp4 \
    --episodes 5
```

---

## 📚 ДОПОЛНИТЕЛЬНАЯ ДОКУМЕНТАЦИЯ

Для полной документации см. папку `/docs/`:

- 📄 [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- 📄 [PROJECT_COMPLETION_REPORT.md](../../docs/PROJECT_COMPLETION_REPORT.md) - Финальный отчет о проекте
- 📄 [CPU_vs_GPU_Comparison.md](../../docs/CPU_vs_GPU_Comparison.md) - CPU vs GPU
- 📄 [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - Решение проблем
- 📄 [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт

**Планы**:
- 📋 [specs/005-final-report/spec.md](./spec.md) - Спецификация фичи
- 📋 [specs/005-final-report/plan.md](./plan.md) - План реализации
- 📋 [specs/005-final-report/data-model.md](./data-model.md) - Модель данных
- 📋 [specs/005-final-report/contracts/api_contracts.md](./contracts/api_contracts.md) - API контракты
- 📋 [specs/005-final-report/research.md](./research.md) - Исследования

---

## ✅ КРИТЕРИИ УСПЕХА

Все этапы завершены успешно, если:

- ✅ Этап 1: Директория results/reports создана
- ✅ Этап 2: Таблица model_comparison.csv и .json созданы, все модели проанализированы
- ✅ Этап 3: Два графика созданы (reward_vs_timesteps.png, agent_comparison.png), подписи осей на русском
- ✅ Этап 4: 3 демо-видео созданы (по 5 эпизодов), воспроизводятся в плеере
- ✅ Этап 5: requirements.txt создан через pip freeze
- ✅ Этап 6: FINAL_REPORT.md содержит все секции из требований преподавателя
- ✅ Этап 7: FINAL_REPORT.md скопирован в корень проекта
- ✅ Этап 8: README.md обновлён с инструкциями запуска

---

## 📝 ШАБЛОН ДЛЯ README.md

После создания всех артефактов, обновите README.md:

```markdown
# RL Agent Training: LunarLander-v3

Обучение агента с подкреплением для управления посадочным модулем LunarLander.

## Финальный отчёт

Полный отчёт о проекте: [FINAL_REPORT.md](FINAL_REPORT.md)

### Краткие результаты

| Модель | Алгоритм | Seed | Награда | Статус |
|--------|----------|------|---------|--------|
| ppo_seed42_500K | PPO | 42 | 243.45 ± 22.85 | ✅ Converged |
| ppo_seed42_400K | PPO | 42 | 235.24 ± 25.52 | ✅ Converged |
| ppo_seed999 | PPO | 999 | 195.09 ± 30.52 | ❌ Not Converged |

### Графики

#### Кривая обучения
![Кривая обучения](results/reports/reward_vs_timesteps.png)

#### Сравнение агентов
![Сравнение агентов](results/reports/agent_comparison.png)

### Демо-видео

- [Лучший агент (243.45 награды)](results/reports/demo_best_model.mp4)
- [Второй агент (235.24 награды)](results/reports/demo_second_best.mp4)
- [Третий агент (195.09 награды)](results/reports/demo_third_best.mp4)

## Быстрый старт

### 1. Установка зависимостей

```bash
conda activate rocm
pip install -r results/reports/requirements.txt
```

### 2. Загрузка лучшей модели

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Загрузка модели
model = PPO.load("results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip")

# Создание окружения
env = gym.make("LunarLander-v3")

# Запуск
obs, _ = env.reset(seed=42)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 3. Генерация отчётов

См. [specs/005-final-report/quickstart.md](specs/005-final-report/quickstart.md)

## Параметры обучения

```yaml
algorithm: PPO
environment: LunarLander-v3
seed: 42
timesteps: 500000

hyperparameters:
  gamma: 0.999
  ent_coef: 0.01
  gae_lambda: 0.98
  n_steps: 1024
  n_epochs: 4
  batch_size: 64
  learning_rate: 0.0003
```

## Лицензия

MIT
```

---

**Создано**: 2026-02-05 | **Feature**: 005-final-report | **Статус**: Готов к выполнению
