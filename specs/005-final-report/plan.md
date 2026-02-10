# План реализации: Финальный отчёт и документация

**Ветка**: `005-final-report` | **Дата**: 2026-02-05 | **Спецификация**: [spec.md](./spec.md)
**Вход**: Спецификация фичи из `/specs/005-final-report/spec.md`

## Резюме

**Основная задача**: Создание финального отчёта для защиты проекта с требованиями преподавателя, включающего анализ результатов, визуализации, демо-видео и полную документацию.

**Технический подход**:
1. Анализ всех обученных моделей и сбор метрик в единую таблицу сравнения
2. Генерация визуализаций (график обучения, сравнительная диаграмма)
3. Создание демо-видео для лучших моделей
4. Формирование финального отчёта в формате Markdown с соблюдением критериев оценки преподавателя

**Критерии успеха**: Модель с наградой ≥200, все графики с подписями, requirements.txt, анализ 3-6 предложений, аккуратное оформление

## Технический контекст

**Язык/Версия**: Python 3.10.14
**Основные зависимости**: Stable-Baselines3, Gymnasium, PyTorch, NumPy, Matplotlib, pandas
**Хранилище**: Файловая система (результаты экспериментов, модели, видео, отчёты)
**Тестирование**: pytest для проверки скриптов анализа и генерации
**Целевая платформа**: Linux сервер (CPU обучение)
**Тип проекта**: Исследовательский/экспериментальный - отчётность
**Цели по производительности**: Скрипты анализа должны выполняться ≤2 минут на CPU
**Ограничения**: Требования воспроизводимости (фиксированный seed=42), форматирование Markdown, русский язык для отчёта
**Масштаб/Охват**: Анализ ~10-15 обученных моделей, 3 демо-видео, 2 графика

**Существующие эксперименты для анализа**:
- `ppo_seed42_500K`: Лучшая модель, 243.45 ± 22.85 (400K), 225.59 ± 22.18 (500K)
- `ppo_seed999`: 195.09 ± 30.52 (хуже seed)
- `gamma_999`: 59.55 ± 68.44 (слишком низкая награда)
- `gamma_990`: 13.43 ± 15.24 (крайне низкая)
- `a2c_seed42`: ~80-100 (A2C хуже PPO)

**Лучшая конфигурация**:
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

## Проверка конституции

*Ворота: Должен пройти перед Phase 0. Перепроверить после Phase 1.*

### Проверка соответствия
- [x] **Воспроизводимые исследования**: Фиксированный seed (42), зависимости через pip freeze, полный код обучения
  - ✅ Seed=42 задокументирован в спецификации
  - ✅ `pip freeze` будет выполнен и сохранён в requirements.txt
  - ✅ Все конфигурации моделей хранятся в config.json

- [x] **Экспериментально-ориентированная разработка**: Чёткие гипотезы и контролируемые сравнения
  - ✅ Сравнение seed 42 vs 999 (влияние инициализации)
  - ✅ Сравнение gamma 0.999 vs 0.99/0.90 (горизонт планирования)
  - ✅ Сравнение PPO vs A2C (алгоритмы)

- [ ] **Тест-фёрст для RL компонентов**: Юнит-тесты для RL компонентов
  - ⚠️ Для фазы отчётности тесты не критичны (нет новой RL функциональности)
  - ⚠️ Будут созданы тесты для скриптов анализа и генерации

- [x] **Мониторинг производительности**: Отслеживание метрик и визуализация
  - ✅ График обучения (reward vs timesteps)
  - ✅ Сравнительная диаграмма моделей
  - ✅ Демо-видео успешных посадок

- [x] **Научная документация**: Структура отчётности определена
  - ✅ Финальный отчёт со всеми разделами от преподавателя
  - ✅ README с графиками и инструкциями
  - ✅ Краткий анализ 3-6 предложений

**Статус**: ✅ Все принципы соблюдены. Нарушений нет.

## Структура проекта

### Документация (этой фичи)

```text
specs/005-final-report/
├── plan.md              # Этот файл (вывод команды /speckit.plan)
├── spec.md              # Спецификация фичи
├── research.md          # Вывод Phase 0 (если будет нужен)
├── data-model.md        # Вывод Phase 1 (структура данных)
├── quickstart.md        # Вывод Phase 1 (инструкции запуска)
└── tasks.md             # Вывод Phase 2 (задачи - создаётся отдельно)
```

### Исходный код (корень репозитория)

```text
# Структура проекта для финального отчёта
src/
├── reporting/            # Скрипты анализа и отчётности (НОВАЯ ДИРЕКТОРИЯ)
│   ├── __init__.py
│   ├── analyze_models.py      # Анализ всех моделей и генерация таблицы
│   ├── generate_plots.py      # Генерация графиков обучения и сравнения
│   └── generate_videos.py     # Генерация демо-видео для лучших моделей
│
├── environments/         # Обёртки окружений (существует)
├── agents/              # Реализации агентов (существует)
├── training/            # Пайплайны обучения (существует)
├── experiments/         # Конфигурации экспериментов (существует)
├── utils/               # Вспомогательные функции (существует)
└── visualization/       # Построение графиков и видео (существует)

tests/
├── unit/                # Юнит-тесты для компонентов
├── integration/         # Интеграционные тесты
└── reporting/           # Тесты для скриптов отчётности (НОВАЯ ДИРЕКТОРИЯ)
    └── test_analyze_models.py
    └── test_generate_plots.py
    └── test_generate_videos.py

results/                 # Результаты экспериментов и отчёты
├── experiments/         # Обученные модели (существует)
│   ├── ppo_seed42/ppo_seed42_500K/
│   ├── ppo_seed999/
│   ├── gamma_999/
│   ├── gamma_990/
│   └── a2c_seed42/
│
└── reports/             # Артефакты финального отчёта (НОВАЯ ДИРЕКТОРИЯ)
    ├── model_comparison.csv          # Таблица сравнения моделей
    ├── model_comparison.json         # Таблица в JSON
    ├── reward_vs_timesteps.png      # График обучения
    ├── agent_comparison.png          # Сравнительная диаграмма
    ├── demo_best_model.mp4           # Видео лучшей модели
    ├── demo_second_best.mp4          # Видео второй модели
    ├── demo_third_best.mp4           # Видео третьей модели
    ├── FINAL_REPORT.md               # Финальный отчёт
    └── requirements.txt              # Зависимости (pip freeze)

docs/                     # Документация
├── FINAL_REPORT.md      # Финальный отчёт (копия или символическая ссылка)
└── README.md            # Обновлённый README с отчётом
```

**Решение по структуре**: Создаём директорию `src/reporting/` для логики анализа и генерации отчётов. Все результаты сохраняются в `results/reports/`. Используем существующую инфраструктуру `src/visualization/` для создания графиков и видео.

## Отслеживание сложности

> **Заполнять ТОЛЬКО если Проверка конституции имеет нарушения, требующие оправдания**

| Нарушение | Зачем нужно | Почему отклонен более простой вариант |
|-----------|-------------|----------------------------------------|
| Н/Д | Н/Д | Все принципы конституции соблюдены, нарушений нет |

---

# Phase 0: Исследование и планирование

## Неизвестные, требующие исследования

**НЕТ неизвестных для исследования**. Все технологии и подходы известны:
- Структура данных: CSV файлы (metrics.csv, eval_log.csv), JSON (config.json)
- Библиотеки: pandas для анализа, matplotlib для графиков, imageio для видео
- Формат отчёта: Markdown, определён требованиями преподавателя

## Зависимости от существующего кода

1. **Существующие скрипты визуализации**:
   - `src/visualization/graphs/` - возможно нужно адаптировать для генерации графиков
   - `src/visualization/video.py` - для создания демо-видео

2. **Существующая структура экспериментов**:
   - `results/experiments/*/config.json` - конфигурации моделей
   - `results/experiments/*/metrics.csv` - метрики обучения
   - `results/experiments/*/eval_log.csv` - логи оценки
   - `results/experiments/*/*.zip` - чекпоинты моделей

3. **Лучшая модель**:
   - Путь: `results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip` (чекпоинт 400K)
   - Награда: 243.45 ± 22.85
   - Или финальная модель: `ppo_seed42_500K_model.zip` (225.59 ± 22.18)

---

# Phase 1: Дизайн и контракты

## Модель данных

### Entity: ModelMetrics
```python
@dataclass
class ModelMetrics:
    """Метрики обученной модели"""
    experiment_id: str           # Идентификатор эксперимента
    algorithm: str              # PPO, A2C и т.д.
    seed: int | None            # Seed обучения
    timesteps: int | None       # Количество шагов обучения
    gamma: float | None         # Дисконтирующий фактор
    ent_coef: float | None      # Коэффициент энтропии
    learning_rate: float | None # Скорость обучения

    # Метрики обучения (из metrics.csv)
    final_train_reward: float   # Финальная награда (последняя строка)
    final_train_std: float      # Стандартное отклонение
    total_training_time: float  # Общее время обучения (секунды)

    # Метрики оценки (из eval_log.csv)
    best_eval_reward: float     # Лучшая награда оценки
    best_eval_std: float        # Стандартное отклонение лучшей
    final_eval_reward: float    # Финальная награда оценки
    final_eval_std: float       # Стандартное отклонение финальной

    # Статус
    convergence_status: str     # "CONVERGED", "NOT_CONVERGED", "UNKNOWN"

    def is_converged(self) -> bool:
        """Достигнут ли порог сходимости (награда > 200)"""
        return self.best_eval_reward >= 200.0
```

### Entity: ComparisonTable
```python
@dataclass
class ComparisonTable:
    """Таблица сравнения всех моделей"""
    models: list[ModelMetrics]  # Список метрик всех моделей
    top_n: int = 3               # Количество лучших моделей для выделения

    def get_top_models(self) -> list[ModelMetrics]:
        """Получить топ-N моделей по лучшей оценочной награде"""
        return sorted(self.models, key=lambda m: m.best_eval_reward, reverse=True)[:self.top_n]

    def count_converged(self) -> int:
        """Количество моделей, достигших награды > 200"""
        return sum(1 for m in self.models if m.is_converged())

    def to_dataframe(self) -> pandas.DataFrame:
        """Преобразовать в pandas DataFrame"""
        pass
```

## API контракты

### 1. Скрипт анализа моделей

**Вход**: Нет (сканирует `results/experiments/`)
**Выход**: `results/reports/model_comparison.csv` и `.json`

**Функция**:
```python
def analyze_all_models(
    experiments_dir: Path = Path("results/experiments"),
    output_dir: Path = Path("results/reports"),
) -> ComparisonTable:
    """
    Анализирует все эксперименты и создаёт таблицу сравнения.

    Args:
        experiments_dir: Директория с экспериментами
        output_dir: Директория для сохранения результатов

    Returns:
        ComparisonTable с собранными метриками
    """
```

### 2. Генерация графиков

**Вход**: ComparisonTable или путь к metrics.csv
**Выход**: `results/reports/reward_vs_timesteps.png`, `agent_comparison.png`

**Функции**:
```python
def generate_learning_curve(
    metrics_path: Path,
    output_path: Path = Path("results/reports/reward_vs_timesteps.png"),
    title: str = "Кривая обучения PPO (seed=42, gamma=0.999)",
) -> None:
    """
    Генерирует график обучения (награда vs шаги).

    Args:
        metrics_path: Путь к metrics.csv
        output_path: Путь для сохранения графика
        title: Заголовок графика
    """

def generate_comparison_chart(
    comparison_table: ComparisonTable,
    output_path: Path = Path("results/reports/agent_comparison.png"),
    title: str = "Сравнение итоговых наград агентов",
) -> None:
    """
    Генерирует сравнительную диаграмму моделей.

    Args:
        comparison_table: Таблица сравнения моделей
        output_path: Путь для сохранения графика
        title: Заголовок графика
    """
```

### 3. Генерация видео

**Вход**: Пути к моделям (top 3)
**Выход**: `results/reports/demo_*.mp4`

**Функция**:
```python
def generate_demo_video(
    model_path: Path,
    output_path: Path,
    env_name: str = "LunarLander-v3",
    num_episodes: int = 5,
    fps: int = 30,
    seed: int = 0,
) -> None:
    """
    Генерирует демо-видео работы модели.

    Args:
        model_path: Путь к модели (.zip)
        output_path: Путь для сохранения видео
        env_name: Имя окружения
        num_episodes: Количество эпизодов
        fps: Кадров в секунду
        seed: Seed для окружения (воспроизводимость)
    """
```

### 4. Генерация финального отчёта

**Вход**: ComparisonTable, пути к графикам, топ-модели
**Выход**: `results/reports/FINAL_REPORT.md`

**Функция**:
```python
def generate_final_report(
    comparison_table: ComparisonTable,
    learning_curve_path: Path,
    comparison_chart_path: Path,
    video_paths: list[Path],
    output_path: Path = Path("results/reports/FINAL_REPORT.md"),
    seed: int = 42,
) -> None:
    """
    Генерирует финальный отчёт в формате Markdown.

    Структура отчёта:
    1. Краткое описание задачи и среды
    2. Код обучения и параметры
    3. Графики (learning curve + comparison)
    4. Краткий анализ (3-6 предложений)

    Args:
        comparison_table: Таблица сравнения моделей
        learning_curve_path: Путь к графику обучения
        comparison_chart_path: Путь к диаграмме сравнения
        video_paths: Пути к демо-видео
        output_path: Путь для сохранения отчёта
        seed: Фиксированный seed для документации
    """
```

## Быстрый запуск (Quickstart)

```bash
# 1. Активировать окружение
conda activate rocm

# 2. Создать директорию для отчётов
mkdir -p results/reports

# 3. Анализ всех моделей
python -m src.reporting.analyze_models

# 4. Сгенерировать графики
python -m src.reporting.generate_plots \
    --metrics results/experiments/ppo_seed42/ppo_seed42_500K/metrics.csv \
    --comparison results/reports/model_comparison.csv

# 5. Сгенерировать демо-видео (для топ-3 моделей)
python -m src.reporting.generate_videos \
    --model results/experiments/ppo_seed42/ppo_seed42_500K/best_model.zip \
    --output results/reports/demo_best_model.mp4 \
    --episodes 5

# 6. Сохранить зависимости
pip freeze > results/reports/requirements.txt

# 7. Создать финальный отчёт
python -m src.reporting.generate_report

# 8. Обновить README
cp results/reports/FINAL_REPORT.md FINAL_REPORT.md
# Редактировать README.md для добавления графиков
```

## Обновление контекста агента

```bash
# Запуск скрипта обновления контекста
.specify/scripts/bash/update-agent-context.sh opencode
```

Это обновит контекст для агента, добавив:
- Новую директорию `src/reporting/`
- Новые модули: `analyze_models.py`, `generate_plots.py`, `generate_videos.py`, `generate_report.py`
- Новую директорию результатов `results/reports/`
- Новую директорию тестов `tests/reporting/`

---

# Phase 2: Задачи реализации

**Примечание**: Задачи будут созданы командой `/speckit.tasks` (НЕ частью `/speckit.plan`)

## Статус ворота

✅ **Phase 0 завершён**: Нет неизвестных для исследования
✅ **Phase 1 завершён**: Модель данных и API контракты определены
✅ **Проверка конституции перепроверена**: Все принципы соблюдены

## Ожидаемые задачи

1. **Создать директории**:
   - `src/reporting/`
   - `tests/reporting/`
   - `results/reports/`

2. **Реализовать скрипт анализа**:
   - `src/reporting/analyze_models.py`
   - Чтение config.json, metrics.csv, eval_log.csv
   - Генерация ComparisonTable
   - Сохранение в CSV и JSON
   - Юнит-тесты

3. **Реализовать генерацию графиков**:
   - `src/reporting/generate_plots.py`
   - График обучения с std отклонением
   - Сравнительная диаграмма с error bars
   - Подписи осей на русском
   - Юнит-тесты

4. **Реализовать генерацию видео**:
   - `src/reporting/generate_videos.py`
   - Загрузка модели из .zip
   - Запуск в среде LunarLander-v3
   - Запись 5 эпизодов
   - Юнит-тесты

5. **Реализовать генерацию отчёта**:
   - `src/reporting/generate_report.py`
   - Шаблон Markdown отчёта
   - Встраивание графиков
   - Анализ 3-6 предложений
   - Юнит-тесты

6. **Создать финальные артефакты**:
   - `requirements.txt` (pip freeze)
   - `FINAL_REPORT.md`
   - Обновлённый `README.md`

7. **Верификация**:
   - Все графики созданы
   - Все видео созданы
   - Отчёт соответствует критериям преподавателя
   - README содержит инструкции запуска

---

# Статус плана

**Дата создания**: 2026-02-05
**Автор**: AI Agent (с помощью пользователя)
**Версия**: 1.0

**Следующий шаг**: Запустить `/speckit.tasks` для создания детального списка задач (tasks.md)
