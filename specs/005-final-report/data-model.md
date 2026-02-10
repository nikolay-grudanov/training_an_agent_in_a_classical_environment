# Data Model: Final Report Artifacts

**Feature**: 005-final-report | **Date**: 2026-02-05
**Project Type**: Machine Learning (Reinforcement Learning) | **Phase**: 1 (Design & Contracts)

---

## 📋 NOTE: ML Project Architecture

**Это ML проект (Reinforcement Learning), NOT traditional web application.**

**Отличия от традиционных проектов:**
- ❌ **Нет REST API** - Модели используются напрямую через Python API
- ❌ **Нет базы данных** - Данные хранятся в файлах (CSV, JSON, ZIP, Markdown)
- ✅ **Есть ML модели** - Обученные агенты (PPO, A2C)
- ✅ **Есть отчёты** - Markdown документы с графиками и видео
- ✅ **Есть артефакты** - Графики (PNG), видео (MP4), таблицы (CSV)

**Полную документацию см. в папке `/docs/`**:
- [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт

---

## Сущности проекта финального отчёта

### 1. ModelMetrics (Метрики модели)

**Описание**: Сводная информация об обученной модели для сравнения

**Поля**:
```python
{
    "experiment_id": "ppo_seed42_500K",
    "algorithm": "PPO",
    "environment": "LunarLander-v3",
    "seed": 42,
    "timesteps": 500000,
    "gamma": 0.999,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "model_path": "results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip",
    "final_train_reward": 224.11,
    "final_train_std": 30.52,
    "best_eval_reward": 243.45,
    "best_eval_std": 22.85,
    "final_eval_reward": 224.11,
    "final_eval_std": 30.52,
    "total_training_time": 190.0,
    "convergence_status": "CONVERGED"
}
```

**Типы данных**:
- `experiment_id`, `algorithm`, `environment`: string
- `seed`, `timesteps`: int | None
- `gamma`, `ent_coef`, `learning_rate`: float | None
- `model_path`: Path
- `*_reward`, `*_std`: float
- `total_training_time`: float (секунды)
- `convergence_status`: "CONVERGED" | "NOT_CONVERGED" | "UNKNOWN"

**Валидация**:
- `experiment_id` должен быть уникальным
- `best_eval_reward` > 200 для CONVERGED
- `model_path` должен существовать и быть валидным .zip файлом

**Отношения**:
- ModelMetrics → Model (относится к конкретной модели)

---

### 2. ComparisonTable (Таблица сравнения)

**Описание**: Агрегированные данные для сравнения всех моделей

**Поля**:
```python
{
    "total_models": 10,
    "converged_models": 4,
    "top_models": [
        # Модель 1: лучшая
        {"experiment_id": "ppo_seed42_500K", "best_eval_reward": 243.45},
        # Модель 2: вторая
        {"experiment_id": "ppo_seed42_400K", "best_eval_reward": 235.24},
        # Модель 3: третья
        {"experiment_id": "ppo_seed999", "best_eval_reward": 195.09}
    ],
    "generated_at": "2026-02-05T22:00:00Z"
}
```

**Типы данных**:
- `total_models`, `converged_models`: int
- `top_models`: list[dict] (словари с ключами experiment_id, best_eval_reward)
- `generated_at`: datetime (ISO 8601)

**Валидация**:
- `converged_models` ≤ `total_models`
- `top_models` отсортированы по `best_eval_reward` по убыванию
- Все `experiment_id` в `top_models` существуют в `total_models`

**Отношения**:
- ComparisonTable содержит множество ModelMetrics

---

### 3. LearningCurvePlot (График обучения)

**Описание**: График награды vs количество шагов

**Файл**: `results/reports/reward_vs_timesteps.png`

**Поля**:
```python
{
    "plot_name": "reward_vs_timesteps",
    "type": "line_plot_with_error_bands",
    "file_path": "results/reports/reward_vs_timesteps.png",
    "file_size": "85KB",
    "format": "PNG",
    "resolution": "1200x800",
    "dpi": 300,
    "x_axis": "Timesteps",
    "y_axis": "Mean Reward",
    "title": "Кривая обучения PPO (seed=42, gamma=0.999)",
    "data_source": "results/experiments/ppo_seed42/ppo_seed42_500K/metrics.csv",
    "created_at": "2026-02-05T22:05:00Z"
}
```

**Типы данных**:
- `plot_name`, `type`, `file_path`, `x_axis`, `y_axis`, `title`: string
- `file_size`: string (например, "85KB")
- `format`: string ("PNG")
- `resolution`: string ("1200x800")
- `dpi`: int
- `data_source`: Path
- `created_at`: datetime

**Валидация**:
- PNG файл должен быть валидным изображением
- График должен показывать сходимость (награда увеличивается)
- Ось X: Timesteps, ось Y: Mean Reward
- Должны быть error bands для стандартного отклонения

**Отношения**:
- LearningCurvePlot → ModelMetrics (визуализирует метрики одной модели)

---

### 4. ComparisonChart (Сравнительная диаграмма)

**Описание**: Столбчатая диаграмма сравнения итоговых наград всех моделей

**Файл**: `results/reports/agent_comparison.png`

**Поля**:
```python
{
    "chart_name": "agent_comparison",
    "type": "bar_chart_with_error_bars",
    "file_path": "results/reports/agent_comparison.png",
    "file_size": "92KB",
    "format": "PNG",
    "resolution": "1400x800",
    "dpi": 300,
    "x_axis": "Model",
    "y_axis": "Mean Reward",
    "title": "Сравнение итоговых наград агентов",
    "data_source": "results/reports/model_comparison.csv",
    "created_at": "2026-02-05T22:10:00Z"
}
```

**Типы данных**: аналогично LearningCurvePlot

**Валидация**:
- PNG файл должен быть валидным
- Каждая модель имеет error bar (стандартное отклонение)
- Модели отсортированы по средней награде
- Ось X: Model, ось Y: Mean Reward
- Названия моделей читаемы (не слишком длинные)

**Отношения**:
- ComparisonChart → ComparisonTable (визуализирует сравнение)

---

### 5. DemoVideo (Демо-видео)

**Описание**: Видео демонстрации работы обученного агента

**Файл**: `results/reports/demo_best_model.mp4`

**Поля**:
```python
{
    "video_name": "demo_best_model",
    "model_source": "results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip",
    "file_path": "results/reports/demo_best_model.mp4",
    "file_size": "4.2MB",
    "format": "MP4",
    "codec": "H.264",
    "fps": 30,
    "duration": 60.5,
    "num_episodes": 5,
    "avg_episode_length": 181,
    "environment": "LunarLander-v3",
    "seed": 42,
    "created_at": "2026-02-05T22:15:00Z"
}
```

**Типы данных**:
- `video_name`, `model_source`, `file_path`, `format`, `codec`, `environment`: string
- `file_size`: string
- `fps`: int (кадров в секунду)
- `duration`: float (секунды)
- `num_episodes`, `avg_episode_length`, `seed`: int
- `created_at`: datetime

**Валидация**:
- MP4 файл должен воспроизводиться
- Кодек: H.264 (совместимый с большинством плееров)
- Должны быть видны успешные посадки (LunarLander не разбился)
- FPS ≥ 24 для плавного воспроизведения

**Отношения**:
- DemoVideo → Model (демонстрирует работу одной модели)

---

### 6. FinalReport (Финальный отчёт)

**Описание**: Markdown документ с анализом и результатами

**Файл**: `results/reports/FINAL_REPORT.md`

**Поля**:
```python
{
    "report_name": "final_report",
    "file_path": "results/reports/FINAL_REPORT.md",
    "file_size": "15KB",
    "format": "Markdown",
    "language": "ru",
    "sections": [
        "Краткое описание задачи и среды",
        "Код обучения и параметры",
        "Графики",
        "Краткий анализ"
    ],
    "embedded_images": [
        "reward_vs_timesteps.png",
        "agent_comparison.png"
    ],
    "embedded_videos": [
        "demo_best_model.mp4",
        "demo_second_best.mp4",
        "demo_third_best.mp4"
    ],
    "model_info": {
        "algorithm": "PPO",
        "environment": "LunarLander-v3",
        "seed": 42,
        "best_reward": 243.45
    },
    "created_at": "2026-02-05T22:20:00Z"
}
```

**Типы данных**:
- `report_name`, `file_path`, `format`, `language`: string
- `file_size`: string
- `sections`: list[string]
- `embedded_images`, `embedded_videos`: list[string]
- `model_info`: dict (ключи: algorithm, environment, seed, best_reward)
- `created_at`: datetime

**Валидация**:
- Markdown файл валидный
- Все секции из требований преподавателя присутствуют
- Все изображения и видео существуют
- Анализ содержит 3-6 предложений
- Seed задокументирован
- Параметры обучения указаны полностью

**Отношения**:
- FinalReport → LearningCurvePlot (встраивает график)
- FinalReport → ComparisonChart (встраивает диаграмму)
- FinalReport → DemoVideo (встраивает видео)
- FinalReport → ModelMetrics (использует метрики)

---

### 7. Requirements (Зависимости)

**Описание**: Полный список пакетов Python для воспроизводимости

**Файл**: `results/reports/requirements.txt`

**Поля** (формат файла):
```
# Python 3.10.14
stable-baselines3==2.7.1
gymnasium==1.2.3
torch==2.5.1+rocm6.2
numpy==1.26.4
matplotlib==3.9.4
pandas==2.2.2
imageio==2.35.1
...
```

**Типы данных**: текстовый файл с требованиями в формате `package==version`

**Валидация**:
- Формат совместим с `pip install -r`
- Все пакеты установлены и версионированы
- Seed задокументирован (внутри файла или отдельно)

**Отношения**:
- Requirements обеспечивает воспроизводимость всех экспериментов

---

## Структура хранилища (Storage Layout)

```
results/
├── experiments/                 # Обученные модели (существует)
│   ├── ppo_seed42/ppo_seed42_500K/      # Лучшая модель
│   │   ├── best_model.zip               # Model
│   │   ├── config.json                 # Configuration
│   │   ├── metrics.csv                 # TrainingMetrics
│   │   └── eval_log.csv                # EvaluationMetrics
│   ├── ppo_seed999/
│   ├── gamma_999/
│   └── a2c_seed42/
│
└── reports/                    # Артефакты финального отчёта (НОВАЯ ДИРЕКТОРИЯ)
    ├── model_comparison.csv          # ComparisonTable (CSV)
    ├── model_comparison.json         # ComparisonTable (JSON)
    ├── reward_vs_timesteps.png      # LearningCurvePlot
    ├── agent_comparison.png          # ComparisonChart
    ├── demo_best_model.mp4           # DemoVideo (лучшая модель)
    ├── demo_second_best.mp4          # DemoVideo (вторая модель)
    ├── demo_third_best.mp4           # DemoVideo (третья модель)
    ├── FINAL_REPORT.md               # FinalReport
    ├── requirements.txt              # Requirements
    └── generated_at.txt             # Timestamp генерации отчёта
```

---

## Схема отношений (Entity Relationship)

```
┌─────────────────┐         ┌──────────────────────┐
│ ModelMetrics    │─────────▶│  Model (.zip)       │
│  (агрегирован) │         │  (обученный агент)  │
└─────────────────┘         └──────────────────────┘
         │                            │
         │ 1                          │ 1
         │                            │
         ▼                            │
┌─────────────────┐                 │
│ ComparisonTable│                 │
│  (все модели)  │                 │
└─────────────────┘                 │
         │                            │
         │                            │
         │ 1                          │
         ▼                            │
┌─────────────────┐                 │
│LearningCurvePlot│◀────────────────│
│ (*.png)        │                 │
└─────────────────┘                 │
         │                            │
         │                            │
┌─────────────────┐                 │
│ComparisonChart │◀────────────────│
│ (*.png)        │                 │
└─────────────────┘                 │
         │                            │
         │                            │
┌─────────────────┐                 │
│  DemoVideo     │◀────────────────│
│  (*.mp4)       │                 │
└─────────────────┘                 │
         │                            │
         │                            │
         ▼                            │
┌─────────────────┐                 │
│  FinalReport   │◀────────────────│
│  (*.md)        │                 │
└─────────────────┘                 │
         │                            │
         │                            │
         ▼                            │
┌─────────────────┐                 │
│ Requirements   │                 │
│  (txt)         │                 │
└─────────────────┘                 │
                                    │
                             (использует)
```

---

## Валидационные правила

### ModelMetrics Validation
```python
def validate_model_metrics(metrics: dict) -> bool:
    """Валидация метрик модели"""
    # 1. Обязательные поля
    required_fields = [
        "experiment_id", "algorithm", "best_eval_reward",
        "model_path", "convergence_status"
    ]
    for field in required_fields:
        assert field in metrics, f"Missing field: {field}"

    # 2. Награды валидные
    assert metrics["best_eval_reward"] > -1000, "Best reward too low"
    assert metrics["best_eval_std"] >= 0, "Std must be non-negative"

    # 3. Статус сходимости
    if metrics["convergence_status"] == "CONVERGED":
        assert metrics["best_eval_reward"] >= 200, "CONVERGED but reward < 200"

    # 4. Путь к модели существует
    assert os.path.exists(metrics["model_path"]), f"Model not found: {metrics['model_path']}"

    return True
```

### FinalReport Validation
```python
def validate_final_report(report_path: str) -> bool:
    """Валидация финального отчёта"""
    # 1. Файл существует
    assert os.path.exists(report_path), f"Report not found: {report_path}"

    # 2. Прочитать Markdown
    with open(report_path) as f:
        content = f.read()

    # 3. Обязательные секции
    required_sections = [
        "Краткое описание задачи",
        "Код обучения",
        "Параметры",
        "Графики",
        "Анализ"
    ]
    for section in required_sections:
        assert section in content, f"Missing section: {section}"

    # 4. Встраиваемые изображения существуют
    images = re.findall(r'!\[.*\]\((.*?\.png)\)', content)
    for img in images:
        img_path = os.path.join(os.path.dirname(report_path), img)
        assert os.path.exists(img_path), f"Image not found: {img}"

    # 5. Seed задокументирован
    assert "seed=42" in content or "seed : 42" in content, "Seed not documented"

    # 6. Анализ 3-6 предложений
    analysis_section = extract_section(content, "Анализ")
    sentences = len([s for s in analysis_section.split('.') if s.strip()])
    assert 3 <= sentences <= 6, f"Analysis must be 3-6 sentences, got {sentences}"

    return True
```

---

## Migration Notes

**Нет миграций** - Это ML проект, данные хранятся в файлах, не в базе данных.

**Артефакты версионируются** через git:
- Код отчётности: Версионирован в Git (src/reporting/)
- Отчёты и результаты: Не версионированы (results/reports/ в .gitignore)
- Финальный отчёт: Версионирован в Git (FINAL_REPORT.md в корне)

---

## Ссылки на документацию

**Полная документация проекта**:

- 📄 [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- 📄 [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт
- 📄 [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - Решение проблем

**Планы**:
- 📋 [specs/005-final-report/plan.md](./plan.md) - План реализации
- 📋 [specs/005-final-report/spec.md](./spec.md) - Спецификация фичи
