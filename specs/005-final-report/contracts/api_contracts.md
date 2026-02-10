# API Contracts: Final Report Scripts

**Feature**: 005-final-report | **Date**: 2026-02-05
**Project Type**: Machine Learning (Reinforcement Learning) | **Phase**: 1 (Design & Contracts)

---

## 📋 NOTE: ML Project Architecture

**Это ML проект (Reinforcement Learning), NOT traditional web application.**

**Ключевые отличия**:
- ❌ **Нет REST API** - Скрипты используются напрямую через Python CLI
- ❌ **Нет GraphQL API** - Нет запросов от клиентов
- ❌ **Нет базы данных** - Данные хранятся в файлах (CSV, JSON, Markdown)
- ✅ **Есть Python API** - Функции для анализа, графиков, видео
- ✅ **Есть CLI** - Командная строка `python -m src.reporting.*`
- ✅ **Есть конфигурации** - Аргументы командной строки и JSON файлы

**Архитектура скриптов отчётности**:
```
User (Developer/Student)
    │
    │ CLI (Command Line Interface)
    │ python -m src.reporting.analyze_models
    │ python -m src.reporting.generate_plots
    │ python -m src.reporting.generate_videos
    │
    ▼
Python API (Функции для импорта)
    │
    ├── analyze_models.py (сбор метрик)
    ├── generate_plots.py (создание графиков)
    ├── generate_videos.py (создание видео)
    └── generate_report.py (создание отчёта)
    │
    ▼
File System
    │
    ├── results/reports/ (результаты)
    │   ├── model_comparison.csv
    │   ├── reward_vs_timesteps.png
    │   ├── agent_comparison.png
    │   ├── demo_*.mp4
    │   └── FINAL_REPORT.md
    └── results/experiments/ (входные данные)
        └── */config.json, metrics.csv, *.zip
```

**Полную документацию см. в папке `/docs/`**:
- [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт

---

## Python API Contracts

### 1. analyze_models.py - Сбор и анализ метрик

**Модуль**: `src.reporting.analyze_models`

**Основная функция**:

```python
def analyze_all_models(
    experiments_dir: Path = Path("results/experiments"),
    output_dir: Path = Path("results/reports"),
    csv_output: Path = Path("results/reports/model_comparison.csv"),
    json_output: Path = Path("results/reports/model_comparison.json"),
) -> ComparisonTable:
    """
    Анализирует все обученные модели и создаёт таблицу сравнения.

    Сканирует директорию experiments_dir рекурсивно, находит все
    эксперименты с config.json, metrics.csv, eval_log.csv.

    Args:
        experiments_dir: Директория с экспериментами (по умолчанию: results/experiments)
        output_dir: Директория для сохранения результатов (по умолчанию: results/reports)
        csv_output: Путь для сохранения CSV таблицы (по умолчанию: results/reports/model_comparison.csv)
        json_output: Путь для сохранения JSON таблицы (по умолчанию: results/reports/model_comparison.json)

    Returns:
        ComparisonTable: Объект с таблицей сравнения моделей

    Raises:
        FileNotFoundError: Если experiments_dir не существует
        ValueError: Если не найдено ни одного валидного эксперимента

    Example:
        >>> table = analyze_all_models()
        >>> print(f"Total models: {table.total_models}")
        >>> print(f"Converged: {table.count_converged()}")
    """
```

**CLI интерфейс**:
```bash
python -m src.reporting.analyze_models \
    --experiments-dir results/experiments \
    --output-dir results/reports \
    --csv-output results/reports/model_comparison.csv \
    --json-output results/reports/model_comparison.json
```

**Аргументы CLI**:
- `--experiments-dir`: Директория с экспериментами (default: results/experiments)
- `--output-dir`: Директория для результатов (default: results/reports)
- `--csv-output`: Путь к CSV файлу (default: results/reports/model_comparison.csv)
- `--json-output`: Путь к JSON файлу (default: results/reports/model_comparison.json)
- `--verbose`, `-v`: Выводить подробную информацию

**Вспомогательные функции**:

```python
def read_experiment_config(
    config_path: Path,
) -> dict:
    """
    Читает config.json эксперимента.

    Args:
        config_path: Путь к config.json

    Returns:
        dict: Конфигурация эксперимента

    Raises:
        JSONDecodeError: Если JSON не валиден
    """

def extract_metrics_from_csv(
    metrics_path: Path,
) -> tuple[float, float]:
    """
    Извлекает финальные метрики из metrics.csv.

    Args:
        metrics_path: Путь к metrics.csv

    Returns:
        tuple[float, float]: (mean_reward, std_reward) из последней строки

    Raises:
        ValueError: Если CSV пуст или некорректен
    """

def extract_eval_metrics(
    eval_log_path: Path,
) -> tuple[float, float, float, float]:
    """
    Извлекает метрики оценки из eval_log.csv.

    Args:
        eval_log_path: Путь к eval_log.csv

    Returns:
        tuple[float, float, float, float]: (best_reward, best_std, final_reward, final_std)
    """
```

---

### 2. generate_plots.py - Генерация графиков

**Модуль**: `src.reporting.generate_plots`

**Основные функции**:

```python
def generate_learning_curve(
    metrics_path: Path,
    output_path: Path = Path("results/reports/reward_vs_timesteps.png"),
    title: str = "Кривая обучения PPO (seed=42, gamma=0.999)",
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 300,
    show_std: bool = True,
) -> None:
    """
    Генерирует график обучения (награда vs шаги) с полосами стандартного отклонения.

    Args:
        metrics_path: Путь к metrics.csv
        output_path: Путь для сохранения графика (default: results/reports/reward_vs_timesteps.png)
        title: Заголовок графика
        figsize: Размер графика в дюймах (ширина, высота)
        dpi: Разрешение в DPI
        show_std: Показывать ли полосы стандартного отклонения

    Raises:
        FileNotFoundError: Если metrics_path не существует
        ValueError: Если CSV пуст или некорректен

    Example:
        >>> generate_learning_curve(
        ...     Path("results/experiments/ppo_seed42/ppo_seed42_500K/metrics.csv"),
        ...     title="Кривая обучения PPO"
        ... )
    """
```

```python
def generate_comparison_chart(
    comparison_csv: Path,
    output_path: Path = Path("results/reports/agent_comparison.png"),
    title: str = "Сравнение итоговых наград агентов",
    figsize: tuple[int, int] = (14, 6),
    dpi: int = 300,
    show_error_bars: bool = True,
    top_n: int | None = None,
) -> None:
    """
    Генерирует столбчатую диаграмму сравнения моделей с error bars.

    Args:
        comparison_csv: Путь к model_comparison.csv (вывод analyze_models.py)
        output_path: Путь для сохранения диаграммы (default: results/reports/agent_comparison.png)
        title: Заголовок диаграммы
        figsize: Размер графика в дюймах (ширина, высота)
        dpi: Разрешение в DPI
        show_error_bars: Показывать ли error bars (стандартное отклонение)
        top_n: Показывать только топ-N моделей (None = все)

    Raises:
        FileNotFoundError: Если comparison_csv не существует
        ValueError: Если CSV пуст или некорректен

    Example:
        >>> generate_comparison_chart(
        ...     Path("results/reports/model_comparison.csv"),
        ...     top_n=5  # Только топ-5 моделей
        ... )
    """
```

**CLI интерфейс**:
```bash
# Генерация кривой обучения
python -m src.reporting.generate_plots learning-curve \
    --metrics results/experiments/ppo_seed42/ppo_seed42_500K/metrics.csv \
    --output results/reports/reward_vs_timesteps.png \
    --title "Кривая обучения PPO" \
    --dpi 300

# Генерация сравнительной диаграммы
python -m src.reporting.generate_plots comparison \
    --comparison results/reports/model_comparison.csv \
    --output results/reports/agent_comparison.png \
    --title "Сравнение итоговых наград" \
    --top-n 5
```

**Аргументы CLI**:
- Подкоманда: `learning-curve` или `comparison`
- `--metrics`: Путь к metrics.csv (для learning-curve)
- `--comparison`: Путь к model_comparison.csv (для comparison)
- `--output`: Путь для сохранения графика
- `--title`: Заголовок графика
- `--figsize`: Размер графика (WxH, default: 12x6 или 14x6)
- `--dpi`: Разрешение (default: 300)
- `--top-n`: Топ-N моделей (только для comparison)
- `--no-error-bars`, `--no-std`: Отключить error bars / std полосы

---

### 3. generate_videos.py - Генерация демо-видео

**Модуль**: `src.reporting.generate_videos`

**Основная функция**:

```python
def generate_demo_video(
    model_path: Path,
    output_path: Path,
    env_name: str = "LunarLander-v3",
    num_episodes: int = 5,
    fps: int = 30,
    seed: int = 0,
    deterministic: bool = True,
    render_mode: str = "rgb_array",
) -> None:
    """
    Генерирует демо-видео работы обученной модели.

    Загружает модель, запускает её в среде, записывает
    эпизоды и сохраняет в MP4 формате.

    Args:
        model_path: Путь к модели (.zip файл)
        output_path: Путь для сохранения видео
        env_name: Имя окружения (default: LunarLander-v3)
        num_episodes: Количество эпизодов для записи (default: 5)
        fps: Кадров в секунду (default: 30)
        seed: Seed для окружения (default: 0)
        deterministic: Использовать детерминированные действия (default: True)
        render_mode: Режим рендеринга (default: rgb_array)

    Raises:
        FileNotFoundError: Если model_path не существует
        ValueError: Если модель не может быть загружена

    Example:
        >>> generate_demo_video(
        ...     Path("results/experiments/ppo_seed42/best_model.zip"),
        ...     Path("results/reports/demo_best_model.mp4"),
        ...     num_episodes=5,
        ...     fps=30
        ... )
    """
```

```python
def generate_top_n_videos(
    comparison_csv: Path,
    output_dir: Path = Path("results/reports"),
    top_n: int = 3,
    num_episodes: int = 5,
    fps: int = 30,
    seed: int = 0,
) -> list[Path]:
    """
    Генерирует демо-видео для топ-N лучших моделей.

    Автоматически считывает model_comparison.csv, определяет
    топ-N моделей по best_eval_reward и генерирует видео.

    Args:
        comparison_csv: Путь к model_comparison.csv
        output_dir: Директория для сохранения видео
        top_n: Количество лучших моделей (default: 3)
        num_episodes: Количество эпизодов для записи (default: 5)
        fps: Кадров в секунду (default: 30)
        seed: Seed для окружения (default: 0)

    Returns:
        list[Path]: Список путей к созданным видео

    Example:
        >>> videos = generate_top_n_videos(
        ...     Path("results/reports/model_comparison.csv"),
        ...     top_n=3
        ... )
        >>> print(f"Created {len(videos)} videos")
    """
```

**CLI интерфейс**:
```bash
# Генерация видео для одной модели
python -m src.reporting.generate_videos single \
    --model results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip \
    --output results/reports/demo_best_model.mp4 \
    --episodes 5 \
    --fps 30

# Генерация видео для топ-N моделей
python -m src.reporting.generate_videos top-n \
    --comparison results/reports/model_comparison.csv \
    --output-dir results/reports \
    --top-n 3 \
    --episodes 5
```

**Аргументы CLI**:
- Подкоманда: `single` или `top-n`
- `--model`: Путь к модели (для single)
- `--comparison`: Путь к model_comparison.csv (для top-n)
- `--output`: Путь к видео (для single)
- `--output-dir`: Директория для видео (для top-n)
- `--episodes`, `--num-episodes`: Количество эпизодов (default: 5)
- `--fps`: Кадров в секунду (default: 30)
- `--seed`: Seed для окружения (default: 0)
- `--stochastic`, `--non-deterministic`: Недетерминированные действия

---

### 4. generate_report.py - Генерация финального отчёта

**Модуль**: `src.reporting.generate_report`

**Основная функция**:

```python
def generate_final_report(
    comparison_csv: Path,
    learning_curve_path: Path,
    comparison_chart_path: Path,
    video_paths: list[Path],
    output_path: Path = Path("results/reports/FINAL_REPORT.md"),
    seed: int = 42,
    best_model_info: dict | None = None,
) -> None:
    """
    Генерирует финальный отчёт в формате Markdown.

    Структура отчёта:
    1. Краткое описание задачи и среды
    2. Код обучения и параметры
    3. Графики (learning curve + comparison)
    4. Краткий анализ (3-6 предложений)

    Args:
        comparison_csv: Путь к model_comparison.csv
        learning_curve_path: Путь к reward_vs_timesteps.png
        comparison_chart_path: Путь к agent_comparison.png
        video_paths: Список путей к демо-видео
        output_path: Путь для сохранения отчёта (default: results/reports/FINAL_REPORT.md)
        seed: Фиксированный seed для документации (default: 42)
        best_model_info: Информация о лучшей модели (опционально)

    Raises:
        FileNotFoundError: Если какой-либо файл не существует
        ValueError: Если данные некорректны

    Example:
        >>> generate_final_report(
        ...     Path("results/reports/model_comparison.csv"),
        ...     Path("results/reports/reward_vs_timesteps.png"),
        ...     Path("results/reports/agent_comparison.png"),
        ...     [Path("results/reports/demo_best_model.mp4")],
        ...     seed=42
        ... )
    """
```

**CLI интерфейс**:
```bash
python -m src.reporting.generate_report \
    --comparison results/reports/model_comparison.csv \
    --learning-curve results/reports/reward_vs_timesteps.png \
    --comparison-chart results/reports/agent_comparison.png \
    --videos results/reports/demo_best_model.mp4 results/reports/demo_second_best.mp4 \
    --output results/reports/FINAL_REPORT.md \
    --seed 42
```

**Аргументы CLI**:
- `--comparison`: Путь к model_comparison.csv
- `--learning-curve`: Путь к reward_vs_timesteps.png
- `--comparison-chart`: Путь к agent_comparison.png
- `--videos`, `--video`: Пути к демо-видео (можно указать несколько)
- `--output`: Путь для сохранения отчёта (default: results/reports/FINAL_REPORT.md)
- `--seed`: Фиксированный seed (default: 42)
- `--best-model`: Путь к лучшей модели (опционально)

---

## Шаблон финального отчёта (Markdown)

```markdown
# Финальный отчёт: Обучение RL агента для LunarLander-v3

## Краткое описание задачи и среды

### Задача

Обучить агент с подкреплением (Reinforcement Learning) для управления посадочным модулем LunarLander в среде Gymnasium. Цель агента - безопасно посадить модуль на посадочную площадку без опрокидывания, используя правильное управление двигателями.

### Среда (LunarLander-v3)

- **Наблюдение (Observation space)**: 8 непрерывных значений
  - Координаты X, Y
  - Скорости X, Y
  - Угол и угловая скорость
  - Состояние левой и правой опоры (0 = свободна, 1 = коснулась)

- **Действия (Action space)**: 4 дискретных действия
  - 0: Ничего не делать
  - 1: Основной двигатель (влево)
  - 2: Основной двигатель (вправо)
  - 3: Побочные двигатели

- **Награда**:
  - Приземление в центре: от +100 до +140
  - Касание площадки ногой: +10 за каждую ногу
  - Опрокидывание или выход за пределы: -100
  - Использование двигателя: -0.3 за каждый шаг

- **Условие завершения**: Опрокидывание, выход за пределы или успешная посадка

## Код обучения и параметры

### Алгоритм

PPO (Proximal Policy Optimization) - state-of-the-art алгоритм для RL с гарантиями улучшения политики.

### Параметры обучения

```yaml
algorithm: PPO
environment: LunarLander-v3
seed: 42
timesteps: 500000

hyperparameters:
  gamma: 0.999              # Дисконтирующий фактор (дальновидность)
  ent_coef: 0.01            # Коэффициент энтропии (исследование)
  gae_lambda: 0.98          # Коэффициент для GAE
  n_steps: 1024             # Шагов на обновление
  n_epochs: 4               # Эпох оптимизации
  batch_size: 64            # Размер батча
  learning_rate: 0.0003      # Скорость обучения
  max_grad_norm: 0.5        # Клипирование градиентов
```

### Код обучения

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Создание окружения
env = gym.make("LunarLander-v3")

# Создание агента PPO
model = PPO(
    "MlpPolicy",
    env,
    seed=42,
    gamma=0.999,
    ent_coef=0.01,
    gae_lambda=0.98,
    n_steps=1024,
    n_epochs=4,
    batch_size=64,
    learning_rate=0.0003,
    verbose=1
)

# Обучение
model.learn(total_timesteps=500000)

# Сохранение
model.save("ppo_lunarlander")
```

## Графики

### Кривая обучения

![Кривая обучения](reward_vs_timesteps.png)

График показывает сходимость агента от ~-500 до ~240 награды за 500K шагов обучения. Полосы стандартного отклонения (~±20) указывают на стабильную сходимость.

### Сравнение итоговых наград

![Сравнение агентов](agent_comparison.png)

Сравнение различных конфигураций (seed 42 vs 999, gamma 0.999 vs 0.99/0.90). Лучшая модель (seed 42, gamma 0.999) достигла 243.45 ± 22.85 награды.

## Демо-видео

### Лучший агент (seed 42, 243.45 награды)

<video src="demo_best_model.mp4" controls width="600"></video>

### Второй лучший агент (seed 999, 195.09 награды)

<video src="demo_second_best.mp4" controls width="600"></video>

## Краткий анализ

Лучшая конфигурация (PPO с seed=42, gamma=0.999, ent_coef=0.01) достигла награды 243.45, что на 23% выше минимального требования (200). Seed 42 обеспечивает лучшую инициализацию весов нейросети, а gamma=0.999 позволяет агенту планировать на более долгий горизонт. Коэффициент энтропии 0.01 балансирует Exploration и Exploitation, preventing premature convergence. Основные направления улучшения: использование advantage-normalization, tweaking learning rate schedule, and implementing early stopping based on validation rewards.

---

**Зависимости**: `requirements.txt`
**Воспроизводимость**: Seed=42, все параметры задокументированы
**Дата**: 2026-02-05
```

---

## Ссылки на документацию

- 📄 [data-model.md](./data-model.md) - Модель данных сущностей
- 📄 [quickstart.md](./quickstart.md) - Быстрый старт
- 📄 [plan.md](./plan.md) - План реализации
- 📄 [spec.md](./spec.md) - Спецификация фичи
