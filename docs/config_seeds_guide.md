# Руководство по управлению сидами в конфигурации RL

## Обзор

Модуль `src.utils.config` был расширен для обеспечения консистентности сидов и воспроизводимости в системе обучения RL агентов. Новая функциональность включает:

- ✅ **Принудительную синхронизацию сидов** между всеми компонентами
- ✅ **Валидацию консистентности сидов** в конфигурации
- ✅ **Автоматическое распространение** основного сида на все подкомпоненты
- ✅ **Предотвращение конфликтов сидов** в различных частях системы
- ✅ **Логирование использования сидов** для отладки
- ✅ **Интеграцию с src.utils.seeding** для применения сидов
- ✅ **Проверку детерминированности** настроек
- ✅ **Предупреждения о потенциальных проблемах** с воспроизводимостью

## Новые возможности ReproducibilityConfig

```python
@dataclass
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    use_cuda: bool = False
    # Новые параметры:
    enforce_seed_consistency: bool = True      # Принудительная синхронизация
    validate_determinism: bool = True          # Валидация детерминизма
    warn_on_seed_conflicts: bool = True        # Предупреждения о конфликтах
    auto_propagate_seeds: bool = True          # Автоматическое распространение
```

## Новые методы RLConfig

### enforce_seed_consistency()
Принудительная синхронизация сидов между всеми компонентами:

```python
config = RLConfig(
    seed=42,
    algorithm=AlgorithmConfig(name="PPO", seed=123),  # Конфликт!
    reproducibility=ReproducibilityConfig(seed=456)   # Конфликт!
)

config.enforce_seed_consistency()
# Теперь все сиды = 42
```

### validate_reproducibility()
Проверка настроек воспроизводимости:

```python
is_valid, warnings = config.validate_reproducibility()

if warnings:
    for warning in warnings:
        print(f"⚠️ {warning}")
```

### apply_seeds()
Применение сидов ко всем компонентам системы:

```python
config.apply_seeds()  # Устанавливает глобальные сиды через src.utils.seeding
```

### get_reproducibility_report()
Получение детального отчета о воспроизводимости:

```python
report = config.get_reproducibility_report()
print(f"Статус: {'✓' if report['is_valid'] else '✗'}")
print(f"Сиды консистентны: {report['seeds']['seeds_consistent']}")
```

## Новые методы ConfigLoader

### load_config_with_seed_validation()
Загрузка конфигурации с расширенной валидацией:

```python
loader = ConfigLoader()
config = loader.load_config_with_seed_validation(
    config_path="config.yaml",
    apply_seeds=True,           # Применить сиды сразу
    validate_reproducibility=True  # Валидировать настройки
)
```

### create_reproducibility_report()
Создание и сохранение отчета:

```python
report = loader.create_reproducibility_report(
    config, 
    output_path="reports/reproducibility.json"
)
```

### validate_seed_consistency_across_configs()
Проверка консистентности между конфигурациями:

```python
config_files = ["config1.yaml", "config2.yaml", "config3.yaml"]
report = loader.validate_seed_consistency_across_configs(config_files)

if report["consistency_issues"]:
    print("Обнаружены конфликты сидов!")
```

## Удобные функции

### load_config_with_seeds()
Быстрая загрузка с применением сидов:

```python
from src.utils.config import load_config_with_seeds

config = load_config_with_seeds(
    config_path="my_config.yaml",
    apply_seeds=True
)
```

### enforce_global_seed_consistency()
Синхронизация нескольких конфигураций:

```python
from src.utils.config import enforce_global_seed_consistency

configs = [config1, config2, config3]
synced_configs = enforce_global_seed_consistency(configs, master_seed=42)
```

### create_reproducibility_report()
Быстрое создание отчета:

```python
from src.utils.config import create_reproducibility_report

report = create_reproducibility_report(config, "report.json")
```

## Автоматическая синхронизация

При `auto_propagate_seeds=True` (по умолчанию), сиды автоматически синхронизируются при создании конфигурации:

```python
config = RLConfig(
    seed=42,
    algorithm=AlgorithmConfig(name="PPO", seed=123),  # Будет изменен на 42
    reproducibility=ReproducibilityConfig(
        seed=456,  # Будет изменен на 42
        auto_propagate_seeds=True  # Включено по умолчанию
    )
)
# config.algorithm.seed == 42
# config.reproducibility.seed == 42
```

## Валидация и предупреждения

Система автоматически обнаруживает:

- **Конфликты сидов** между компонентами
- **Противоречивые настройки** (deterministic=True + benchmark=True)
- **Проблемы с SDE** (может снизить воспроизводимость)
- **Неопределенные device** (device="auto")
- **Конфликты CUDA** настроек

## Отчет о воспроизводимости

Генерируемый отчет включает:

```json
{
  "timestamp": "2026-01-14T20:00:22.057076",
  "experiment_name": "my_experiment",
  "is_valid": true,
  "warnings": [...],
  "seeds": {
    "main_seed": 42,
    "seeds_consistent": true
  },
  "determinism": {
    "deterministic_mode": true,
    "benchmark_mode": false
  },
  "algorithm": {
    "algorithm_name": "PPO",
    "use_sde": false
  },
  "system": {
    "python_version": "3.12.7",
    "torch_version": "2.9.1",
    "cuda_available": true
  },
  "recommendations": [...]
}
```

## Примеры использования

### Базовое использование
```python
from src.utils.config import load_config_with_seeds

# Загружаем конфигурацию с автоматическим применением сидов
config = load_config_with_seeds("experiments/ppo_lunar.yaml")

# Проверяем воспроизводимость
is_valid, warnings = config.validate_reproducibility()
if warnings:
    print("⚠️ Обнаружены проблемы с воспроизводимостью")

# Генерируем отчет
report = config.get_reproducibility_report()
```

### Продвинутое использование
```python
from src.utils.config import ConfigLoader, enforce_global_seed_consistency

loader = ConfigLoader()

# Загружаем несколько конфигураций
configs = [
    loader.load_config_with_seed_validation("config1.yaml"),
    loader.load_config_with_seed_validation("config2.yaml"),
    loader.load_config_with_seed_validation("config3.yaml")
]

# Синхронизируем сиды
synced_configs = enforce_global_seed_consistency(configs, master_seed=42)

# Проверяем консистентность
consistency_report = loader.validate_seed_consistency_across_configs([
    "config1.yaml", "config2.yaml", "config3.yaml"
])

# Создаем отчеты для каждой конфигурации
for i, config in enumerate(synced_configs):
    loader.create_reproducibility_report(
        config, 
        f"reports/config_{i}_reproducibility.json"
    )
```

## Интеграция с существующим кодом

Новая функциональность полностью обратно совместима. Существующий код продолжит работать без изменений, но получит дополнительные возможности:

```python
# Старый код (продолжает работать)
config = load_config("my_config.yaml")

# Новые возможности
config.enforce_seed_consistency()  # Синхронизация сидов
config.apply_seeds()              # Применение к системе
report = config.get_reproducibility_report()  # Отчет
```

## Рекомендации

1. **Используйте auto_propagate_seeds=True** для автоматической синхронизации
2. **Явно указывайте device** вместо "auto" для лучшей воспроизводимости
3. **Отключайте SDE** если нужна полная детерминированность
4. **Генерируйте отчеты** для документирования настроек воспроизводимости
5. **Проверяйте консистентность** при работе с несколькими конфигурациями

## Тестирование

Запуск тестов новой функциональности:

```bash
pytest tests/unit/test_config_seeds.py -v
```

Демонстрация возможностей:

```bash
python examples/config_seeds_example.py
```