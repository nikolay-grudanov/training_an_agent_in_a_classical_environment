# Performance Plots - Система визуализации производительности RL агентов

## Обзор

Модуль `performance_plots.py` предоставляет комплексную систему для создания графиков производительности обучения агентов подкрепляющего обучения (RL). Поддерживает как статические (matplotlib), так и интерактивные (Plotly) визуализации.

## Основные возможности

### 📊 Типы графиков
- **График вознаграждения** - кривая обучения по временным шагам/эпизодам
- **График длины эпизодов** - динамика продолжительности эпизодов
- **График функций потерь** - мониторинг процесса обучения
- **Сравнительные графики** - анализ нескольких агентов
- **Дашборды** - комплексные панели с множественными метриками

### 🎨 Визуализация
- **Статические графики** (matplotlib) - для публикаций и отчетов
- **Интерактивные графики** (Plotly) - для исследовательского анализа
- **Настраиваемые стили** - seaborn, custom themes
- **Сглаживание данных** - скользящие средние, доверительные интервалы

### 📁 Источники данных
- **CSV файлы** - стандартный формат метрик
- **JSON/JSONL** - структурированные логи
- **Pandas DataFrame** - прямая работа с данными
- **MetricsTracker** - интеграция с системой метрик
- **Stable-Baselines3 логи** - автоматическая конвертация

### 💾 Экспорт
- **Множественные форматы** - PNG, PDF, SVG, HTML
- **Высокое качество** - 300 DPI для печати
- **Пакетный экспорт** - автоматическое создание отчетов

## Быстрый старт

### Базовое использование

```python
from src.visualization.performance_plots import PerformancePlotter
import pandas as pd

# Создание плоттера
plotter = PerformancePlotter(style='seaborn-v0_8')

# Загрузка данных
data = pd.read_csv('training_metrics.csv')

# График вознаграждения
plotter.plot_reward_curve(
    data,
    x_col='timestep',
    y_col='episode_reward',
    save_path='reward_curve.png',
    title='Кривая обучения PPO агента'
)
```

### Интерактивные графики

```python
from src.visualization.performance_plots import InteractivePlotter

# Интерактивный плоттер
interactive_plotter = InteractivePlotter(theme='plotly_white')

# Интерактивный график
interactive_plotter.plot_interactive_reward_curve(
    data,
    save_path='interactive_reward.html',
    title='Интерактивная кривая обучения'
)
```

### Сравнение агентов

```python
# Данные нескольких агентов
agents_data = {
    "PPO": ppo_data,
    "SAC": sac_data,
    "A2C": a2c_data
}

# Сравнительный график
plotter.plot_multiple_agents(
    agents_data,
    metric='episode_reward',
    save_path='agents_comparison.png',
    title='Сравнение RL агентов'
)
```

### Быстрые функции

```python
from src.visualization.performance_plots import quick_reward_plot, quick_comparison_plot

# Быстрый график из файла
quick_reward_plot('metrics.csv', save_path='quick_plot.png')

# Быстрое сравнение
agents_files = {
    "Agent1": "agent1_metrics.csv",
    "Agent2": "agent2_metrics.csv"
}
quick_comparison_plot(agents_files, save_path='comparison.png')
```

## Классы и API

### PlotStyle
Управление стилями и цветовыми схемами.

```python
from src.visualization.performance_plots import PlotStyle

# Настройка стиля
PlotStyle.setup_matplotlib_style('seaborn-v0_8')

# Доступные цветовые палитры
colors = PlotStyle.COLORS_PRIMARY
figsize = PlotStyle.FIGSIZE_LARGE
```

### DataLoader
Загрузка данных из различных источников.

```python
from src.visualization.performance_plots import DataLoader

# Из CSV
data = DataLoader.load_from_csv('metrics.csv')

# Из JSON
data = DataLoader.load_from_json('experiment.json')

# Из MetricsTracker
from src.utils.metrics import MetricsTracker
tracker = MetricsTracker("experiment_1")
data = DataLoader.load_from_metrics_tracker(tracker)

# Из логов SB3
data = DataLoader.convert_sb3_logs('logs_directory/')
```

### PerformancePlotter
Основной класс для статических графиков.

#### Методы:
- `plot_reward_curve()` - график вознаграждения
- `plot_episode_lengths()` - график длины эпизодов  
- `plot_loss_curves()` - графики функций потерь
- `plot_multiple_agents()` - сравнение агентов
- `create_dashboard()` - дашборд метрик

```python
plotter = PerformancePlotter(
    style='seaborn-v0_8',
    color_palette='husl',
    figsize=(12, 8)
)

# График с настройками
plotter.plot_reward_curve(
    data,
    x_col='timestep',
    y_col='reward',
    smooth_window=100,
    show_raw=True,
    confidence_interval=True,
    save_path='reward.png',
    title='Кривая обучения'
)
```

### InteractivePlotter
Класс для интерактивных графиков с Plotly.

#### Методы:
- `plot_interactive_reward_curve()` - интерактивный график вознаграждения
- `plot_interactive_comparison()` - интерактивное сравнение агентов
- `create_interactive_dashboard()` - интерактивный дашборд

```python
interactive_plotter = InteractivePlotter(theme='plotly_white')

# Интерактивный дашборд
interactive_plotter.create_interactive_dashboard(
    data,
    metrics=['episode_reward', 'episode_length', 'training_loss'],
    save_path='dashboard.html',
    title='Дашборд метрик обучения'
)
```

## Утилиты

### Экспорт в множественные форматы

```python
from src.visualization.performance_plots import export_plots_to_formats

# Экспорт в несколько форматов
saved_files = export_plots_to_formats(
    plotter.plot_reward_curve,
    save_dir='plots/',
    formats=['png', 'pdf', 'svg'],
    data=data,
    title='Reward Curve'
)
```

### Создание полного отчета

```python
from src.visualization.performance_plots import create_performance_report

# Полный отчет о производительности
report_path = create_performance_report(
    data,  # или MetricsTracker
    output_dir='performance_report/',
    include_interactive=True,
    include_static=True
)
```

## Интеграция с MetricsTracker

```python
from src.utils.metrics import MetricsTracker
from src.visualization.performance_plots import create_performance_report

# Создание трекера
tracker = MetricsTracker("experiment_ppo")

# Добавление метрик в процессе обучения
for episode in range(1000):
    # ... обучение агента ...
    
    tracker.add_episode_metrics(
        episode=episode,
        timestep=episode * 100,
        reward=episode_reward,
        length=episode_length
    )
    
    if episode % 10 == 0:
        tracker.add_training_metrics(
            timestep=episode * 100,
            loss=training_loss,
            learning_rate=current_lr
        )

# Создание отчета
create_performance_report(tracker, output_dir='results/ppo_report/')
```

## Настройка стилей

### Matplotlib стили

```python
# Доступные стили
styles = ['seaborn-v0_8', 'ggplot', 'bmh', 'classic']

plotter = PerformancePlotter(style='seaborn-v0_8')

# Кастомные настройки
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3
})
```

### Plotly темы

```python
# Доступные темы
themes = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn']

interactive_plotter = InteractivePlotter(theme='plotly_white')
```

### Цветовые палитры

```python
# Использование предустановленных палитр
colors_primary = PlotStyle.COLORS_PRIMARY
colors_gradient = PlotStyle.COLORS_GRADIENT

# Кастомная палитра
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
```

## Примеры использования

### Анализ сходимости

```python
# График с анализом сходимости
plotter.plot_reward_curve(
    data,
    smooth_window=200,
    confidence_interval=True,
    title='Анализ сходимости PPO агента'
)

# Добавление линии целевого значения
import matplotlib.pyplot as plt
plt.axhline(y=target_reward, color='red', linestyle='--', 
           label='Целевое значение')
plt.legend()
```

### Сравнение гиперпараметров

```python
# Данные экспериментов с разными гиперпараметрами
experiments = {
    "lr=0.001": load_experiment_data("exp_lr_001"),
    "lr=0.0001": load_experiment_data("exp_lr_0001"),
    "lr=0.01": load_experiment_data("exp_lr_01")
}

# Сравнительный график
plotter.plot_multiple_agents(
    experiments,
    metric='episode_reward',
    title='Влияние скорости обучения на производительность'
)
```

### Мониторинг в реальном времени

```python
# Обновляемый график для мониторинга
import time

tracker = MetricsTracker("live_experiment")

for episode in range(1000):
    # ... обучение ...
    
    tracker.add_episode_metrics(episode, timestep, reward, length)
    
    # Обновление графика каждые 50 эпизодов
    if episode % 50 == 0:
        data = DataLoader.load_from_metrics_tracker(tracker)
        plotter.plot_reward_curve(
            data['episode_reward'],
            save_path=f'live_plot_ep_{episode}.png',
            title=f'Обучение - Эпизод {episode}'
        )
```

## Обработка ошибок

```python
try:
    plotter.plot_reward_curve(data, save_path='plot.png')
except ValueError as e:
    print(f"Ошибка в данных: {e}")
except FileNotFoundError as e:
    print(f"Файл не найден: {e}")
except Exception as e:
    print(f"Неожиданная ошибка: {e}")
```

## Производительность

### Рекомендации:
- Используйте сглаживание для больших датасетов (>10000 точек)
- Ограничивайте количество точек для интерактивных графиков
- Кэшируйте обработанные данные для повторного использования

```python
# Оптимизация для больших данных
large_data = data.sample(n=5000)  # Сэмплирование
plotter.plot_reward_curve(large_data, smooth_window=500)
```

## Зависимости

```python
# Основные зависимости
matplotlib >= 3.5.0
plotly >= 5.0.0
pandas >= 1.3.0
numpy >= 1.21.0
seaborn >= 0.11.0

# Опциональные
stable-baselines3  # для интеграции с SB3
```

## Лицензия

Часть проекта обучения RL агентов МИФИ.