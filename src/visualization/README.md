# Модуль визуализации RL агентов

Комплексные инструменты для создания демонстрационных видео и визуализации обученных RL агентов.

## Обзор

Этот модуль предоставляет полный набор инструментов визуализации для анализа прогресса обучения RL, метрик производительности и создания демонстрационных видео. Поддерживает как статические графики (Matplotlib), так и интерактивные видео демонстрации агентов.

## Структура модуля

```
src/visualization/
├── __init__.py                 # Инициализация модуля
├── video_generator.py          # Базовая генерация видео
├── agent_demo.py              # Высокоуровневые функции демонстрации
├── performance_plots.py       # Графики производительности
└── README.md                  # Документация модуля
```

## Основные компоненты

### 1. video_generator.py
Базовая функциональность для создания видео:
- `VideoConfig` - конфигурация параметров видео
- `record_agent_episode()` - запись одного эпизода
- `record_multiple_episodes()` - запись нескольких эпизодов
- `create_training_montage()` - монтаж прогресса обучения
- `generate_comparison_video()` - сравнительное видео
- `compress_video()` - сжатие видео

### 2. agent_demo.py
Высокоуровневые функции для демонстрации агентов:
- `create_best_episode_demo()` - демо лучшего эпизода
- `create_average_behavior_demo()` - демо среднего поведения
- `create_before_after_demo()` - сравнение до/после обучения
- `create_training_progress_demo()` - прогресс обучения
- `create_multi_agent_comparison()` - сравнение агентов
- `create_batch_demos()` - пакетное создание демо
- `auto_demo_from_training_results()` - автоматическое создание

### 3. performance_plots.py
Создание графиков производительности:
- Кривые обучения
- Сравнительные графики
- Статистика наград
- Интерактивные визуализации

### Возможности видео демонстраций
- **Демо лучшего эпизода**: Поиск и запись лучшего выступления агента
- **Среднее поведение**: Демонстрация типичного поведения агента
- **Сравнение до/после**: Визуализация прогресса обучения
- **Прогресс обучения**: Монтаж эволюции агента через чекпоинты
- **Сравнение агентов**: Одновременное тестирование нескольких агентов
- **Пакетная обработка**: Автоматическое создание демо для множества агентов
- **Настраиваемые параметры**: Качество, разрешение, сжатие, оверлеи

## Quick Start

### Basic Usage

```python
from src.visualization.plots import plot_learning_curve, PlotConfig

# Simple learning curve
fig = plot_learning_curve(
    timesteps=timesteps,
    rewards=rewards,
    title="PPO Training Progress",
    smooth=True,
    save_path="results/learning_curve.png"
)
```

### Custom Configuration

```python
# Publication-ready styling
config = PlotConfig(
    figure_size=(8, 5),
    dpi=600,
    font_size=14,
    color_palette="publication"
)

fig = plot_learning_curve(
    timesteps=timesteps,
    rewards=rewards,
    config=config,
    confidence_interval=True
)
```

### Algorithm Comparison

```python
from src.visualization.plots import plot_multiple_runs

runs_data = {
    "PPO": {"timesteps": ppo_timesteps, "reward": ppo_rewards},
    "A2C": {"timesteps": a2c_timesteps, "reward": a2c_rewards},
    "SAC": {"timesteps": sac_timesteps, "reward": sac_rewards},
}

fig = plot_multiple_runs(
    runs_data=runs_data,
    metric="reward",
    title="Algorithm Comparison",
    confidence_interval=True
)
```

### Comprehensive Reports

```python
from src.visualization.generate_all import VisualizationGenerator

# Automated report generation
generator = VisualizationGenerator(
    output_dir="results/plots",
    formats=["png", "svg", "pdf"]
)

# Generate complete experiment report
plots = generator.generate_experiment_report(
    experiment_data=experiment_data,
    experiment_name="ppo_lunarlander"
)
```

## Available Functions

### Core Plotting Functions

| Function | Description | Key Features |
|----------|-------------|--------------|
| `plot_learning_curve()` | Learning progress visualization | Smoothing, confidence intervals, dual backend |
| `plot_episode_lengths()` | Episode duration analysis | Trend detection, statistical overlays |
| `plot_loss_curves()` | Multi-component loss tracking | Log scale, multiple metrics, smoothing |
| `plot_reward_distribution()` | Statistical distribution analysis | Histograms, normal fit, summary statistics |
| `plot_convergence_analysis()` | Convergence detection and visualization | Rolling statistics, plateau identification |
| `plot_multiple_runs()` | Multi-run comparison | Algorithm comparison, confidence bands |
| `plot_confidence_intervals()` | Statistical uncertainty visualization | Multiple confidence levels, error bands |

### Utility Functions

| Function | Description |
|----------|-------------|
| `setup_matplotlib_style()` | Configure matplotlib defaults |
| `create_figure_grid()` | Create subplot layouts |
| `save_plot()` | Unified save function with format detection |
| `apply_smoothing()` | Data smoothing utilities |
| `detect_convergence()` | Convergence detection algorithms |

### Configuration and Automation

| Class/Function | Description |
|----------------|-------------|
| `PlotConfig` | Centralized plot configuration management |
| `VisualizationGenerator` | Automated report generation |
| `load_training_data()` | Load data from various log formats |

## Configuration

### Plot Configuration

Use `PlotConfig` for consistent styling:

```python
config = PlotConfig(
    figure_size=(12, 8),
    dpi=300,
    font_size=12,
    color_palette="colorblind",
    line_width=2,
    alpha=0.7
)
```

### Available Color Schemes

- **default**: Standard matplotlib colors
- **colorblind**: Colorblind-friendly palette
- **publication**: High-contrast publication colors
- **dark**: Dark theme compatible colors

### YAML Configuration

```yaml
# configs/visualization_config.yaml
figure_size: [12, 8]
dpi: 300
color_palette: "publication"
smoothing:
  enabled: true
  method: "moving_average"
  window: 50
formats: ["png", "svg", "pdf"]
```

## Data Formats

### Expected Data Structure

```python
# Single run data
data = {
    "timesteps": np.array([0, 100, 200, ...]),
    "rewards": np.array([-200, -150, -100, ...]),
    "episode_lengths": np.array([1000, 800, 600, ...]),
    "policy_loss": np.array([0.1, 0.08, 0.06, ...]),
    "value_loss": np.array([0.5, 0.4, 0.3, ...]),
}

# Multiple runs data
experiment_data = {
    "runs": {
        "ppo_seed_42": data1,
        "ppo_seed_123": data2,
        "a2c_seed_42": data3,
    }
}
```

### Supported Input Formats

- **NumPy arrays**: Direct numerical data
- **Pandas Series**: Time series data with indices
- **Python lists**: Simple numerical sequences
- **CSV files**: Training logs from stable-baselines3
- **JSON files**: Structured experiment data

## Command Line Usage

Generate visualizations from command line:

```bash
# Generate plots from training logs
python -m src.visualization.generate_all \
    --log-dir results/logs/ppo_experiment \
    --output-dir results/plots \
    --experiment-name "PPO LunarLander" \
    --formats png svg pdf

# With custom configuration
python -m src.visualization.generate_all \
    --log-dir results/logs \
    --output-dir results/plots \
    --config-file configs/visualization_config.yaml
```

## Integration with Training Pipeline

### With Trainer

```python
from src.training.trainer import Trainer
from src.visualization.generate_all import VisualizationGenerator

# After training
trainer = Trainer(config)
results = trainer.train()

# Generate visualizations
generator = VisualizationGenerator("results/plots")
plots = generator.generate_single_run_report(
    data=results.metrics,
    run_name="ppo_final"
)
```

### With Experiment Runner

```python
from src.experiments.runner import ExperimentRunner
from src.visualization.generate_all import load_training_data

# After experiment
runner = ExperimentRunner()
runner.run_experiment(config)

# Load and visualize results
data = load_training_data("results/logs/experiment_1")
generator = VisualizationGenerator("results/plots")
plots = generator.generate_experiment_report(data, "experiment_1")
```

## Performance Considerations

### Large Datasets

- **Lazy Loading**: Use `read_repomix_output` for incremental data loading
- **Smoothing**: Apply smoothing to reduce noise and improve readability
- **Sampling**: Downsample very large datasets for visualization
- **Batch Processing**: Use `VisualizationGenerator` for automated processing

### Memory Optimization

```python
# For large datasets, use sampling
if len(timesteps) > 10000:
    indices = np.linspace(0, len(timesteps)-1, 5000, dtype=int)
    timesteps = timesteps[indices]
    rewards = rewards[indices]

# Close figures to free memory
plt.close('all')
```

## Examples

See `examples/visualization_example.py` for comprehensive usage examples:

- Single plot creation
- Algorithm comparison
- Convergence analysis
- Comprehensive report generation
- Custom styling
- Real-world workflow

## Testing

Run tests to verify functionality:

```bash
# Run all visualization tests
pytest tests/unit/visualization/ -v

# Run specific test categories
pytest tests/unit/visualization/test_plots.py::TestPlottingFunctions -v
pytest tests/unit/visualization/test_plots.py::TestSmoothingFunctions -v
```

## Dependencies

### Required
- matplotlib >= 3.5.0
- plotly >= 5.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0

### Optional
- kaleido (for Plotly static image export)
- jupyter (for notebook integration)

## Best Practices

### Code Organization
1. Use `PlotConfig` for consistent styling across all plots
2. Leverage `VisualizationGenerator` for automated report creation
3. Apply appropriate smoothing for noisy training data
4. Include confidence intervals for multi-run experiments

### Publication Quality
1. Use high DPI (600+) for publication figures
2. Choose colorblind-friendly palettes
3. Include proper axis labels and titles
4. Save in vector formats (SVG, PDF) when possible

### Performance
1. Apply smoothing to reduce visual noise
2. Use sampling for very large datasets
3. Close figures explicitly to prevent memory leaks
4. Consider using Plotly for interactive exploration

## Troubleshooting

### Common Issues

1. **Memory warnings**: Close figures with `plt.close('all')`
2. **Slow rendering**: Reduce data size or apply smoothing
3. **Missing dependencies**: Install optional packages for full functionality
4. **Style issues**: Verify matplotlib style availability

### Error Handling

The module includes comprehensive error handling:
- Graceful degradation for missing data
- Automatic fallbacks for unsupported features
- Detailed logging for debugging
- Input validation with clear error messages

## Contributing

When adding new visualization functions:

1. Follow the existing API patterns
2. Include comprehensive docstrings
3. Add corresponding tests
4. Update this README
5. Ensure compatibility with both backends
6. Include example usage in docstrings