# 📋 Краткий справочник команд для LunarLander-v3 PPO

## 🚀 Быстрый старт: Запуск лучшей модели

### Обучение лучшей модели (269.31 ± 12.90)

```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 500000 \
    --seed 42 \
    --gamma 0.999 \
    --learning-rate 5e-4 \
    --ent-coef 0.01 \
    --gae-lambda 0.98 \
    --n-steps 2048 \
    --n-epochs 10 \
    --batch-size 64 \
    --device cpu
```

**Ожидаемый результат:**
- Финальная награда: 269.31 ± 12.90
- Время обучения: ~207 секунд
- Имя эксперимента: `ppo_seed42_500K_lr5e4`

Ниже пример выполнения обучения модели 

![Пример выполнения обучения модели ](assets/image_1.png)

---

## 📊 Три контролируемых эксперимента

### Эксперимент 1: Влияние gamma (0.99 vs 0.999)

```bash
# Gamma = 0.99
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.99 --device cpu

# Gamma = 0.999 (лучший результат)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --device cpu
```

**Результат:** Gamma=0.999 значительно лучше (все модели >200 при seed=42)

---

### Эксперимент 2: Влияние timesteps (400K vs 500K vs 1M vs 5M)

```bash
# 400K шагов
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 400000 --seed 42 --gamma 0.999 --device cpu

# 500K шагов
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --device cpu

# 1M шагов (оптимум)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 1000000 --seed 42 --gamma 0.999 --device cpu

# 5M шагов
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 5000000 --seed 42 --gamma 0.999 --device cpu
```

**Результат:** 1M шагов — оптимум (268.10 ± 12.26)

---

### Эксперимент 3: Влияние learning_rate (1e-4 vs 3e-4 vs 5e-4)

```bash
# Learning rate = 1e-4 (низкий)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --learning-rate 1e-4 --device cpu

# Learning rate = 3e-4 (средний)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --learning-rate 3e-4 --device cpu

# Learning rate = 5e-4 (высокий, лучший!)
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --learning-rate 5e-4 --device cpu
```

**Результат:** lr5e-4 — лучший (269.31 ± 12.90)

---

## 📈 Генерация графиков

```bash
# График обучения для конкретного эксперимента
python -m src.reporting.generate_plots reward-vs-timestep

# Сравнительный график
python -m src.reporting.generate_plots comparison

# Dashboard со всеми графиками
python -m src.reporting.generate_plots dashboard
```

**Графики создаются в:** `results/reports/` и `results/comparison/`

---

## 🎬 Генерация видео

```bash
# Создание видео из обученной модели
python -c "
import gymnasium as gym
from stable_baselines3 import PPO

# Загрузка модели
model = PPO.load('results/experiments/ppo_seed42_500K_lr5e4/ppo_seed42_500K_lr5e4_model.zip')

# Создание среды с записью видео
env = gym.make('LunarLander-v3', render_mode='rgb_array')

# Запуск эпизода
obs, _ = env.reset()
done = False
frames = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frames.append(env.render())

# Сохранение видео (требуется pip install imageio[ffmpeg])
import imageio
imageio.mimsave('results/videos/demo.mp4', frames, fps=30)

print('Видео сохранено: results/videos/demo.mp4')
"
```

---

## 📋 Генерация отчётов

```bash
# Проверка гипотез
python -m src.reporting.analyze_models --check-hypotheses

# Генерация всех графиков
python -m src.reporting.generate_plots dashboard

# Полный отчёт
python -m src.reporting.generate_report --check-completeness
```

**Отчёты создаются в:** `results/reports/`

---

## 📊 Лучшие результаты

| Эксперимент | Команда | Награда |
|-------------|----------|---------|
| Лучший | lr5e4, gamma=0.999, seed=42, 500K | **269.31 ± 12.90** |
| Второй | lr3e4, gamma=0.999, seed=42, 1M | 268.10 ± 12.26 |
| Третий | lr3e4, gamma=0.999, seed=42, 5M | 246.70 ± 61.87 |

---

## 🎯 Оптимальные параметры для LunarLander-v3

| Параметр | Значение | Описание |
|-----------|----------|----------|
| `--algo` | `ppo` | Алгоритм PPO |
| `--gamma` | `0.999` | Дисконт-фактор |
| `--learning-rate` | `5e-4` | Скорость обучения |
| `--ent-coef` | `0.01` | Коэффициент энтропии |
| `--gae-lambda` | `0.98` | GAE lambda |
| `--n-steps` | `2048` | Шагов на окружение |
| `--n-epochs` | `10` | Эпох оптимизации |
| `--batch-size` | `64` | Размер батча |
| `--seed` | `42` | Seed для воспроизводимости |
| `--device` | `cpu` | Устройство (cpu/gpu) |

---

## ⚙️ Установка зависимостей

```bash
# Установка всех зависимостей
pip install -r requirements.txt

# Или через conda
conda env create -f environment.yml
conda activate rocm
```

---

## 🔧 Проверка качества кода

```bash
# Проверка стиля
ruff check .

# Автоисправление
ruff check . --fix

# Форматирование
ruff format .

# Проверка типов
mypy src/ --strict
```

---

## 📁 Выходные файлы после обучения

```
results/experiments/ppo_seed42_500K_lr5e4/
├── ppo_seed42_500K_lr5e4_model.zip  # Обученная модель
├── config.json                      # Конфигурация
├── metrics.csv                      # Метрики обучения
├── eval_log.csv                     # Логи оценки
└── checkpoints/
    ├── checkpoint_50000.zip
    ├── checkpoint_100000.zip
    └── ...
```

---

## ⚠️ Частые ошибки

### Ошибка: Модель не достигает сходимости (<200)

**Решение:**
```bash
# Используйте seed=42 и gamma=0.999
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 --device cpu
```

### Ошибка: Видео не сохраняется

**Решение:**
```bash
# Установите ffmpeg
pip install imageio[ffmpeg]

# Или используйте RecordVideo wrapper из gymnasium
```

---

## 📊 Сводка проекта

| Параметр | Значение |
|----------|---------|
| Задача | LunarLander-v3 |
| Алгоритм | PPO (Stable-Baselines3) |
| Лучший результат | 269.31 ± 12.90 |
| Оптимальный seed | 42 |
| Оптимальный gamma | 0.999 |
| Оптимальный learning_rate | 5e-4 |
| Оптимальные timesteps | 500K (или 1M) |
| Время обучения | ~3.5 минут |

---

## 🚀 Полный workflow для воспроизводимости

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Обучение лучшей модели
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42 --gamma 0.999 \
    --learning-rate 5e-4 --ent-coef 0.01 --gae-lambda 0.98 \
    --n-steps 2048 --n-epochs 10 --batch-size 64 --device cpu

# 3. Генерация графиков
python -m src.reporting.generate_plots dashboard

# 4. Генерация отчёта
python -m src.reporting.generate_report --check-completeness

# 5. Проверка качества
ruff check . --fix && ruff format .
```

---

**Проект готов к использованию!** ✅

**Примечание:** Полный справочник команд смотрите в `КОМАНДЫ_FULL.md` (для справки). Этот файл содержит краткий минимум для быстрого старта.
