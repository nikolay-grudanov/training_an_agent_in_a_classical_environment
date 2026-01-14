# Система проверки воспроизводимости RL экспериментов

## Обзор

Модуль `reproducibility_checker.py` предоставляет комплексную систему для проверки и обеспечения воспроизводимости экспериментов машинного обучения с подкреплением (RL). Система интегрируется с существующей архитектурой проекта и предоставляет инструменты для автоматической проверки детерминированности, статистического анализа результатов и генерации отчетов.

## Основные возможности

### 🔍 Проверка воспроизводимости
- **Точное совпадение результатов** при повторных запусках с одинаковыми сидами
- **Статистическая эквивалентность** результатов между запусками
- **Анализ трендов обучения** для выявления консистентности процесса
- **Валидация конфигураций** на предмет настроек воспроизводимости

### 📊 Статистический анализ
- Тесты нормальности распределения (Shapiro-Wilk)
- Тесты равенства средних (ANOVA, Kruskal-Wallis)
- Анализ корреляции трендов обучения
- Вычисление коэффициентов вариации

### 🛠️ Диагностика проблем
- **Автоматическое обнаружение** проблем с сидами
- **Анализ зависимостей** и конфликтов библиотек
- **Проверка настроек среды** выполнения
- **Валидация детерминизма** функций и алгоритмов

### 📈 Уровни строгости
- **MINIMAL**: Базовые проверки точного совпадения
- **STANDARD**: Стандартные проверки + статистический анализ
- **STRICT**: Строгие проверки с детальной диагностикой
- **PARANOID**: Максимально строгие проверки

## Быстрый старт

### Базовое использование

```python
from src.utils.reproducibility_checker import (
    ReproducibilityChecker,
    StrictnessLevel,
    quick_reproducibility_check
)

# Быстрая проверка системы
is_reproducible = quick_reproducibility_check(
    experiment_id="test",
    num_runs=3,
    seed=42
)
print(f"Система воспроизводима: {is_reproducible}")
```

### Детальная проверка эксперимента

```python
from src.utils.config import RLConfig

# Создаем проверщик
checker = ReproducibilityChecker(
    strictness_level=StrictnessLevel.STANDARD
)

# Регистрируем запуски эксперимента
config = RLConfig(experiment_name="my_experiment", seed=42)

for i in range(3):
    # Ваш код обучения агента
    results = train_agent(seed=42)
    
    checker.register_experiment_run(
        experiment_id="my_experiment",
        config=config,
        results=results['final_metrics'],
        metrics=results['training_metrics']
    )

# Проверяем воспроизводимость
report = checker.check_reproducibility("my_experiment")

print(f"Воспроизводимо: {report.is_reproducible}")
print(f"Уверенность: {report.confidence_score:.2f}")
print(f"Проблем найдено: {len(report.issues)}")
```

### Автоматическое тестирование

```python
def my_training_function(seed):
    """Ваша функция обучения агента."""
    set_seed(seed)
    # ... код обучения ...
    return {
        'final_reward': final_reward,
        'metrics': {'rewards': episode_rewards}
    }

# Автоматический тест воспроизводимости
report = checker.run_reproducibility_test(
    test_function=my_training_function,
    experiment_id="auto_test",
    seeds=[42, 42, 42],  # Одинаковые сиды
    config=config
)
```

### Валидация конфигурации

```python
from src.utils.reproducibility_checker import validate_experiment_reproducibility

config = RLConfig(
    experiment_name="validation_test",
    seed=42,
    algorithm=AlgorithmConfig(name="PPO", seed=42)
)

is_valid = validate_experiment_reproducibility(
    config=config,
    num_validation_runs=3
)
```

## Архитектура

### Основные классы

#### `ReproducibilityChecker`
Главный класс для проверки воспроизводимости экспериментов.

**Ключевые методы:**
- `register_experiment_run()` - регистрация запуска эксперимента
- `check_reproducibility()` - проверка воспроизводимости
- `validate_determinism()` - валидация детерминизма функций
- `diagnose_reproducibility_issues()` - диагностика проблем
- `generate_reproducibility_guide()` - генерация руководства

#### `ExperimentRun`
Представляет данные одного запуска эксперимента.

**Поля:**
- `run_id` - уникальный идентификатор
- `seed` - использованный сид
- `config_hash` - хеш конфигурации
- `environment_hash` - хеш среды выполнения
- `results` - результаты эксперимента
- `metrics` - метрики обучения

#### `ReproducibilityReport`
Содержит результаты проверки воспроизводимости.

**Поля:**
- `is_reproducible` - результат проверки
- `confidence_score` - оценка уверенности (0.0-1.0)
- `issues` - список найденных проблем
- `recommendations` - рекомендации по улучшению
- `statistics` - статистический анализ

### Типы проблем

```python
class ReproducibilityIssueType(Enum):
    SEED_MISMATCH = "seed_mismatch"
    ENVIRONMENT_DIFFERENCE = "environment_difference"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    ALGORITHM_NONDETERMINISM = "algorithm_nondeterminism"
    STATISTICAL_DIFFERENCE = "statistical_difference"
    TREND_DEVIATION = "trend_deviation"
    CONFIGURATION_MISMATCH = "configuration_mismatch"
```

## Интеграция с проектом

### Зависимости
Модуль интегрируется с существующими компонентами:

- `src.utils.dependency_tracker` - отслеживание зависимостей
- `src.utils.seeding` - управление сидами
- `src.utils.config` - конфигурация экспериментов
- `src.agents.base.Agent` - базовый класс агентов

### Структура файлов

```
results/
├── reproducibility/
│   ├── runs/                    # Данные запусков
│   │   ├── experiment_1_*.json
│   │   └── experiment_2_*.json
│   └── reports/                 # Отчеты
│       ├── report_exp1_*.json
│       └── report_exp2_*.json
└── dependencies/                # Снимки зависимостей
    └── snapshot_*.json
```

## Примеры использования

### Проверка простого эксперимента

```python
import numpy as np
from src.utils.reproducibility_checker import ReproducibilityChecker

def simple_experiment(seed):
    np.random.seed(seed)
    return {
        'value': np.random.random(),
        'metrics': {'values': np.random.random(10).tolist()}
    }

checker = ReproducibilityChecker()

# Регистрируем несколько запусков
for i in range(3):
    results = simple_experiment(42)
    checker.register_experiment_run(
        experiment_id="simple_test",
        config=RLConfig(seed=42),
        results={'value': results['value']},
        metrics=results['metrics']
    )

# Проверяем воспроизводимость
report = checker.check_reproducibility("simple_test")
print(f"Результат: {report.is_reproducible}")
```

### Диагностика проблем

```python
# Диагностика проблем с воспроизводимостью
diagnosis = checker.diagnose_reproducibility_issues(
    experiment_id="problematic_experiment",
    deep_analysis=True
)

print("Найденные проблемы:")
for issue in diagnosis['issues_found']:
    print(f"- {issue['description']}")

print("\nРекомендации:")
for rec in diagnosis['recommendations']:
    print(f"- {rec}")
```

### Генерация руководства

```python
# Создание руководства по воспроизводимости
guide = checker.generate_reproducibility_guide(
    experiment_id="my_experiment",
    output_path="reproducibility_guide.md"
)

print(f"Руководство создано: {len(guide)} символов")
```

## Настройка уровней строгости

### Конфигурация проверок

```python
# Минимальный уровень - только базовые проверки
checker_minimal = ReproducibilityChecker(
    strictness_level=StrictnessLevel.MINIMAL
)

# Стандартный уровень - проверки + статистика
checker_standard = ReproducibilityChecker(
    strictness_level=StrictnessLevel.STANDARD
)

# Строгий уровень - детальная диагностика
checker_strict = ReproducibilityChecker(
    strictness_level=StrictnessLevel.STRICT
)

# Параноидальный уровень - максимальная строгость
checker_paranoid = ReproducibilityChecker(
    strictness_level=StrictnessLevel.PARANOID
)
```

### Настройка параметров

Каждый уровень строгости имеет свои параметры:

- `tolerance_rtol/atol` - допуски для сравнения чисел
- `statistical_alpha` - уровень значимости для тестов
- `min_runs_for_stats` - минимум запусков для статистики
- `trend_window_size` - размер окна для анализа трендов

## Рекомендации по использованию

### Для разработки
- Используйте `StrictnessLevel.MINIMAL` для быстрых проверок
- Регулярно запускайте `quick_reproducibility_check()`
- Валидируйте конфигурации перед экспериментами

### Для исследований
- Используйте `StrictnessLevel.STANDARD` или `StrictnessLevel.STRICT`
- Создавайте снимки зависимостей перед экспериментами
- Документируйте все найденные проблемы

### Для публикаций
- Используйте `StrictnessLevel.PARANOID` для финальной проверки
- Генерируйте подробные отчеты о воспроизводимости
- Включайте руководства по воспроизведению результатов

## Устранение проблем

### Частые проблемы

1. **Несоответствие сидов**
   ```python
   # Проблема: разные сиды в конфигурации
   config.algorithm.seed = 123  # != config.seed = 42
   
   # Решение: синхронизация сидов
   config.enforce_seed_consistency()
   ```

2. **Недетерминистические операции**
   ```python
   # Проблема: использование SDE
   config.algorithm.use_sde = True
   
   # Решение: отключение SDE
   config.algorithm.use_sde = False
   ```

3. **Различия в среде**
   ```python
   # Создание снимка зависимостей
   from src.utils.dependency_tracker import create_experiment_snapshot
   
   snapshot = create_experiment_snapshot("my_experiment")
   ```

### Диагностические команды

```bash
# Проверка зависимостей
python -c "from src.utils.dependency_tracker import DependencyTracker; DependencyTracker().detect_dependency_conflicts()"

# Быстрая проверка воспроизводимости
python -c "from src.utils.reproducibility_checker import quick_reproducibility_check; print(quick_reproducibility_check())"

# Валидация детерминизма PyTorch
python -c "import torch; print(f'Deterministic: {torch.backends.cudnn.deterministic}, Benchmark: {torch.backends.cudnn.benchmark}')"
```

## Расширение функциональности

### Добавление новых проверок

```python
class CustomReproducibilityChecker(ReproducibilityChecker):
    def _check_custom_metric(self, report, reference_run, comparison_runs):
        """Добавить пользовательскую проверку."""
        # Ваша логика проверки
        pass
    
    def check_reproducibility(self, experiment_id, **kwargs):
        report = super().check_reproducibility(experiment_id, **kwargs)
        self._check_custom_metric(report, ...)
        return report
```

### Интеграция с CI/CD

```yaml
# .github/workflows/reproducibility.yml
name: Reproducibility Check
on: [push, pull_request]

jobs:
  reproducibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Check reproducibility
        run: python -c "from src.utils.reproducibility_checker import quick_reproducibility_check; assert quick_reproducibility_check()"
```

## Заключение

Система проверки воспроизводимости предоставляет комплексные инструменты для обеспечения надежности и воспроизводимости RL экспериментов. Регулярное использование этих инструментов поможет:

- **Повысить качество исследований** за счет надежных результатов
- **Ускорить отладку** проблем с детерминированностью
- **Улучшить документирование** экспериментов
- **Обеспечить соответствие** стандартам воспроизводимости

Для получения дополнительной информации см. примеры в `examples/reproducibility_example.py` и тесты в `tests/unit/utils/test_reproducibility_checker.py`.