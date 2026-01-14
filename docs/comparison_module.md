# Модуль сравнения экспериментов

Комплексные утилиты для статистического сравнения и анализа результатов RL экспериментов с поддержкой строгих статистических методов, визуализации и генерации отчетов.

## 🚀 Основные возможности

### Статистическое сравнение
- **Множественные статистические тесты**: t-test, Mann-Whitney U, Wilcoxon, Bootstrap
- **Размер эффекта**: Cohen's d, Glass's delta, Hedges' g
- **Доверительные интервалы**: Bootstrap и параметрические методы
- **Коррекция множественных сравнений**: Bonferroni, FDR (Benjamini-Hochberg), Holm

### Анализ производительности
- **Метрики сходимости**: Анализ времени достижения пороговых значений
- **Эффективность выборки**: Сравнение скорости обучения
- **Стабильность обучения**: Анализ вариативности и надежности
- **Финальная производительность**: Сравнение итоговых результатов
- **Пиковая производительность**: Поиск и сравнение максимальных достижений

### Продвинутый анализ
- **Анализ чувствительности к гиперпараметрам**: Корреляционный анализ
- **Обобщение между средами**: Анализ переносимости алгоритмов
- **Ранжирование алгоритмов**: Многокритериальное сравнение
- **Анализ Парето**: Многоцелевая оптимизация

### Визуализация
- **Кривые обучения**: Сравнительные графики прогресса
- **Распределения**: Гистограммы, box plots, violin plots
- **Статистические графики**: Heatmaps корреляций, radar charts
- **Интерактивные отчеты**: HTML с встроенными графиками

### Генерация отчетов
- **Множественные форматы**: HTML, Markdown, JSON
- **Экспорт данных**: CSV, Excel, JSON
- **Автоматические рекомендации**: На основе статистического анализа
- **CLI интерфейс**: Удобное использование из командной строки

## 📦 Установка и зависимости

```bash
# Основные зависимости
pip install numpy pandas scipy matplotlib seaborn scikit-learn

# Дополнительные для экспорта в Excel
pip install openpyxl

# Для работы с YAML конфигурациями
pip install pyyaml
```

## 🎯 Быстрый старт

### Базовое сравнение

```python
from src.experiments.comparison import ExperimentComparator
from src.experiments.experiment import Experiment

# Загружаем эксперименты
exp1 = Experiment.load("results/experiment_1.json")
exp2 = Experiment.load("results/experiment_2.json")
exp3 = Experiment.load("results/experiment_3.json")

# Создаем компаратор
comparator = ExperimentComparator()

# Выполняем сравнение
result = comparator.compare_experiments(
    experiments=[exp1, exp2, exp3],
    metrics=['mean_reward', 'stability_score', 'sample_efficiency']
)

# Генерируем отчет
report_path = comparator.generate_comparison_report(
    result, 
    include_plots=True, 
    output_format='html'
)

print(f"Отчет сохранен: {report_path}")
```

### Использование CLI

```bash
# Сравнить конкретные эксперименты
python scripts/compare_experiments.py exp1.json exp2.json exp3.json

# Сравнить все эксперименты в директории
python scripts/compare_experiments.py --dir results/experiments/

# Использовать кастомную конфигурацию
python scripts/compare_experiments.py --config comparison_config.yaml exp1.json exp2.json

# Анализировать конкретные метрики
python scripts/compare_experiments.py --metrics mean_reward stability_score exp1.json exp2.json
```

## 🔧 Конфигурация

### Файл конфигурации (YAML)

```yaml
# configs/comparison_config.yaml
significance_level: 0.05
confidence_level: 0.95
bootstrap_samples: 10000
multiple_comparison_method: "fdr_bh"  # bonferroni, fdr_bh, holm, none
effect_size_method: "cohens_d"        # cohens_d, glass_delta, hedges_g
convergence_threshold: null           # null = автоматически
convergence_window: 100
stability_window: 50
min_sample_size: 10
```

### Программная конфигурация

```python
from src.experiments.comparison import ComparisonConfig, MultipleComparisonMethod

config = ComparisonConfig(
    significance_level=0.01,  # Более строгий уровень значимости
    multiple_comparison_method=MultipleComparisonMethod.BONFERRONI,
    bootstrap_samples=20000   # Больше bootstrap выборок
)

comparator = ExperimentComparator(config)
```

## 📊 Статистические методы

### Выбор статистического теста

```python
# Автоматический выбор на основе нормальности распределения
result = comparator.statistical_significance(data1, data2)

# Принудительное использование конкретного теста
from src.experiments.comparison import StatisticalTest

result = comparator.statistical_significance(
    data1, data2, 
    test=StatisticalTest.MANN_WHITNEY
)
```

### Размер эффекта

```python
from src.experiments.comparison import EffectSizeMethod

# Cohen's d (по умолчанию)
effect_size = comparator.effect_size(data1, data2)

# Glass's delta
effect_size = comparator.effect_size(
    data1, data2, 
    method=EffectSizeMethod.GLASS_DELTA
)

# Интерпретация размера эффекта
if effect_size < 0.2:
    print("Малый эффект")
elif effect_size < 0.5:
    print("Средний эффект")
elif effect_size < 0.8:
    print("Большой эффект")
else:
    print("Очень большой эффект")
```

### Доверительные интервалы

```python
# 95% доверительный интервал
ci_lower, ci_upper = comparator.confidence_intervals(data, 0.95)
print(f"95% ДИ: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 99% доверительный интервал
ci_lower, ci_upper = comparator.confidence_intervals(data, 0.99)
print(f"99% ДИ: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

## 📈 Анализ производительности

### Анализ сходимости

```python
# Анализ сходимости с автоматическим порогом
convergence_info = comparator.convergence_analysis(experiment)

# Анализ с кастомным порогом
convergence_info = comparator.convergence_analysis(
    experiment, 
    metric='episode_reward',
    threshold=200.0,
    window=50
)

print(f"Сошелся: {convergence_info['converged']}")
print(f"Шаги до сходимости: {convergence_info['convergence_timestep']}")
```

### Эффективность обучения

```python
# Сравнение эффективности достижения порога
efficiency = comparator.learning_efficiency(
    experiments, 
    threshold=180.0
)

for exp_id, results in efficiency.items():
    print(f"{exp_id}: {results['steps_to_threshold']} шагов")
```

### Анализ стабильности

```python
# Анализ стабильности обучения
stability = comparator.stability_analysis(experiments)

for exp_id, results in stability.items():
    print(f"{exp_id}: стабильность {results['stability_score']:.3f}")
```

## 🎨 Визуализация

### Кривые обучения

```python
# Сравнение кривых обучения
plot_path = comparator.learning_curves_comparison(
    experiments,
    metric='episode_reward',
    smoothing_window=50
)
```

### Статистические графики

```python
# Box plots для сравнения распределений
box_plot_path = comparator.box_plots(
    experiments,
    metrics=['episode_reward', 'episode_length']
)

# Heatmap корреляций между метриками
heatmap_path = comparator.heatmap_comparison(comparison_result)
```

### Комплексные графики

```python
# Автоматическая генерация всех графиков
plots = comparator.generate_comparison_plots(comparison_result)

for plot_type, plot_path in plots.items():
    print(f"{plot_type}: {plot_path}")
```

## 🔍 Продвинутый анализ

### Анализ чувствительности к гиперпараметрам

```python
# Анализ влияния learning_rate на производительность
sensitivity = comparator.hyperparameter_sensitivity(
    experiments,
    hyperparameter='learning_rate',
    metric='mean_reward'
)

print(f"Корреляция: {sensitivity['correlation']:.3f}")
print(f"R²: {sensitivity['r_squared']:.3f}")
```

### Ранжирование алгоритмов

```python
# Группировка экспериментов по алгоритмам
experiments_by_algorithm = {
    'PPO': [exp1, exp2],
    'A2C': [exp3, exp4],
    'SAC': [exp5, exp6]
}

# Многокритериальное ранжирование
ranking = comparator.algorithm_ranking(
    experiments_by_algorithm,
    metrics=['mean_reward', 'stability_score', 'sample_efficiency'],
    weights=[0.5, 0.3, 0.2]  # Веса важности метрик
)

print("Рейтинг алгоритмов:", ranking['ranking'])
```

### Анализ Парето

```python
# Многоцелевая оптимизация: производительность vs эффективность
pareto_result = comparator.pareto_analysis(
    experiments,
    objective1='mean_reward',      # Максимизируем награду
    objective2='sample_efficiency' # Максимизируем эффективность
)

print("Эксперименты на фронте Парето:")
for exp_id in pareto_result['pareto_front_experiments']:
    print(f"- {exp_id}")
```

## 📋 Генерация отчетов

### HTML отчет

```python
# Полный HTML отчет с графиками
html_report = comparator.generate_comparison_report(
    comparison_result,
    include_plots=True,
    output_format='html'
)
```

### Markdown отчет

```python
# Markdown отчет для документации
md_report = comparator.generate_comparison_report(
    comparison_result,
    include_plots=False,
    output_format='markdown'
)
```

### Экспорт данных

```python
# Экспорт в различные форматы
exported_files = comparator.export_results(
    comparison_result,
    formats=['csv', 'json', 'excel']
)

for format_type, file_path in exported_files.items():
    print(f"{format_type.upper()}: {file_path}")
```

### Форматированные результаты

```python
# Таблица статистических тестов
table = comparator.hypothesis_test_results(
    comparison_result,
    format_type='table'
)
print(table)

# Краткая сводка
summary = comparator.hypothesis_test_results(
    comparison_result,
    format_type='summary'
)
print(summary)
```

## 🎯 Примеры использования

### Сравнение алгоритмов

```python
# Сравнение PPO vs A2C vs SAC
experiments = [ppo_exp, a2c_exp, sac_exp]

result = comparator.compare_experiments(
    experiments,
    metrics=['mean_reward', 'convergence_timesteps', 'stability_score']
)

# Автоматические рекомендации
for recommendation in result.recommendations:
    print(f"💡 {recommendation}")
```

### Анализ влияния гиперпараметров

```python
# Эксперименты с разными learning_rate
lr_experiments = [
    exp_lr_1e4,  # lr=1e-4
    exp_lr_3e4,  # lr=3e-4
    exp_lr_1e3,  # lr=1e-3
    exp_lr_3e3   # lr=3e-3
]

# Анализ чувствительности
sensitivity = comparator.hyperparameter_sensitivity(
    lr_experiments,
    'learning_rate',
    'mean_reward'
)

if sensitivity['correlation'] > 0.7:
    print("Сильная положительная корреляция с learning_rate")
elif sensitivity['correlation'] < -0.7:
    print("Сильная отрицательная корреляция с learning_rate")
else:
    print("Слабая корреляция с learning_rate")
```

### Анализ стабильности

```python
# Множественные запуски одного алгоритма
multiple_runs = [run1, run2, run3, run4, run5]

stability = comparator.stability_analysis(multiple_runs)

# Находим самый стабильный запуск
most_stable = max(
    stability.items(),
    key=lambda x: x[1]['stability_score']
)

print(f"Самый стабильный: {most_stable[0]}")
print(f"Оценка стабильности: {most_stable[1]['stability_score']:.3f}")
```

## ⚙️ Настройка и оптимизация

### Производительность

```python
# Для больших экспериментов уменьшите количество bootstrap выборок
config = ComparisonConfig(
    bootstrap_samples=5000,  # Вместо 10000 по умолчанию
    min_sample_size=5        # Минимальный размер выборки
)

# Отключите создание графиков для ускорения
comparator.generate_comparison_report(
    result,
    include_plots=False
)
```

### Память

```python
# Для экономии памяти при больших экспериментах
# используйте только необходимые метрики
essential_metrics = ['mean_reward', 'final_reward']

result = comparator.compare_experiments(
    experiments,
    metrics=essential_metrics
)
```

### Кастомизация графиков

```python
# Настройка стилей matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# Настройка цветовой палитры seaborn
import seaborn as sns
sns.set_palette("husl")
```

## 🐛 Отладка и диагностика

### Логирование

```python
import logging

# Включить подробное логирование
logging.getLogger('src.experiments.comparison').setLevel(logging.DEBUG)

# Отключить предупреждения scipy
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

### Валидация данных

```python
# Проверка качества данных перед анализом
for exp in experiments:
    if not exp.results or 'baseline' not in exp.results:
        print(f"⚠️ Эксперимент {exp.experiment_id} не содержит результатов")
        continue
    
    metrics_history = exp.results['baseline'].get('metrics_history', [])
    if len(metrics_history) < 10:
        print(f"⚠️ Эксперимент {exp.experiment_id} содержит мало данных")
```

### Обработка ошибок

```python
try:
    result = comparator.compare_experiments(experiments)
except ValueError as e:
    print(f"Ошибка валидации: {e}")
except Exception as e:
    print(f"Неожиданная ошибка: {e}")
    import traceback
    traceback.print_exc()
```

## 📚 Дополнительные ресурсы

### Статистические методы
- [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
- [Benjamini-Hochberg procedure](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure)
- [Bootstrap methods](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

### Визуализация
- [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html)

### Примеры
- `examples/comparison_example.py` - Полная демонстрация возможностей
- `scripts/compare_experiments.py` - CLI интерфейс
- `tests/test_comparison.py` - Примеры использования в тестах

## 🤝 Вклад в развитие

При добавлении новых статистических методов:

1. Добавьте новый тип в `StatisticalTest` enum
2. Реализуйте метод в `_perform_statistical_test()`
3. Добавьте тесты в `tests/test_comparison.py`
4. Обновите документацию

При добавлении новых метрик:

1. Добавьте поле в `PerformanceMetrics` dataclass
2. Реализуйте вычисление в `_extract_performance_metrics()`
3. Добавьте в список метрик по умолчанию
4. Добавьте тесты и документацию