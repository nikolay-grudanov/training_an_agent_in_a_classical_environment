"""CLI интерфейс для системы обучения RL агентов.

Предоставляет удобный командный интерфейс для:
- Обучения агентов с различными конфигурациями
- Восстановления прерванных сессий
- Оценки обученных моделей
- Сравнения алгоритмов
- Управления экспериментами
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from src.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingMode,
    create_trainer_from_config,
)
from src.utils.logging import setup_logging
from src.utils.config import (
    get_config_loader,
    create_default_configs,
    validate_config_directory,
)

# Инициализация Typer приложения
app = typer.Typer(
    name="rl-trainer",
    help="🎮 Система обучения RL агентов",
    add_completion=False,
)

# Rich console для красивого вывода
console = Console()


@app.command()
def train(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Путь к файлу конфигурации"
    ),
    config_name: Optional[str] = typer.Option(
        None, "--config-name", "-n", help="Имя конфигурации из директории configs/"
    ),
    algorithm: Optional[str] = typer.Option(
        None, "--algorithm", "-a", help="Алгоритм обучения (PPO, A2C, SAC, TD3)"
    ),
    env: Optional[str] = typer.Option(
        None, "--env", "-e", help="Название среды Gymnasium"
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Количество шагов обучения"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Seed для воспроизводимости"),
    experiment_name: Optional[str] = typer.Option(
        None, "--experiment", "-x", help="Имя эксперимента"
    ),
    output_dir: str = typer.Option(
        "results", "--output", "-o", help="Директория для результатов"
    ),
    verbose: int = typer.Option(1, "--verbose", "-v", help="Уровень детализации (0-2)"),
    eval_freq: Optional[int] = typer.Option(
        None, "--eval-freq", help="Частота оценки (в шагах)"
    ),
    save_freq: Optional[int] = typer.Option(
        None, "--save-freq", help="Частота сохранения (в шагах)"
    ),
    early_stopping: bool = typer.Option(
        False, "--early-stopping", help="Включить раннее остановка"
    ),
    patience: int = typer.Option(
        5, "--patience", help="Терпение для раннего остановки"
    ),
    override: List[str] = typer.Option(
        [], "--override", help="Переопределения конфигурации (key=value)"
    ),
) -> None:
    """🚀 Обучить RL агента."""
    
    console.print(Panel.fit("🎮 Запуск обучения RL агента", style="bold blue"))
    
    try:
        # Настройка логирования
        log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbose, 2)]
        setup_logging(level=log_level)
        
        # Подготовка переопределений
        overrides = list(override)
        
        if algorithm:
            overrides.append(f"algorithm.name={algorithm}")
        if env:
            overrides.append(f"environment.name={env}")
        if timesteps:
            overrides.append(f"training.total_timesteps={timesteps}")
        if experiment_name:
            overrides.append(f"experiment_name={experiment_name}")
        if eval_freq:
            overrides.append(f"training.eval_freq={eval_freq}")
        if save_freq:
            overrides.append(f"training.save_freq={save_freq}")
        
        overrides.append(f"seed={seed}")
        overrides.append(f"output_dir={output_dir}")
        overrides.append(f"training.early_stopping={early_stopping}")
        overrides.append(f"training.patience={patience}")
        
        # Создание тренера
        trainer = create_trainer_from_config(
            config_path=config,
            config_name=config_name,
            overrides=overrides,
        )
        
        # Отображение конфигурации
        _display_config(trainer.config)
        
        # Обучение с прогресс-баром
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Обучение...", total=None)
            
            with trainer:
                result = trainer.train()
        
        # Отображение результатов
        _display_results(result)
        
        if result.success:
            console.print("✅ [bold green]Обучение завершено успешно![/bold green]")
            sys.exit(0)
        else:
            console.print(f"❌ [bold red]Ошибка обучения: {result.error_message}[/bold red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n⏹️  [yellow]Обучение прервано пользователем[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"❌ [bold red]Критическая ошибка: {e}[/bold red]")
        if verbose >= 2:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def resume(
    checkpoint: str = typer.Argument(..., help="Путь к чекпоинту для восстановления"),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Общее количество шагов обучения"
    ),
    output_dir: str = typer.Option(
        "results", "--output", "-o", help="Директория для результатов"
    ),
    verbose: int = typer.Option(1, "--verbose", "-v", help="Уровень детализации"),
) -> None:
    """🔄 Восстановить обучение из чекпоинта."""
    
    console.print(Panel.fit("🔄 Восстановление обучения", style="bold yellow"))
    
    try:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            console.print(f"❌ [red]Чекпоинт не найден: {checkpoint}[/red]")
            sys.exit(1)
        
        # Настройка логирования
        log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbose, 2)]
        setup_logging(level=log_level)
        
        # Создание конфигурации для восстановления
        config = TrainerConfig(
            mode=TrainingMode.RESUME,
            resume_from_checkpoint=str(checkpoint_path),
            output_dir=output_dir,
            verbose=verbose,
        )
        
        if timesteps:
            config.total_timesteps = timesteps
        
        console.print(f"📂 Загрузка чекпоинта: {checkpoint}")
        
        with Trainer(config) as trainer:
            result = trainer.train()
        
        _display_results(result)
        
        if result.success:
            console.print("✅ [bold green]Восстановление завершено успешно![/bold green]")
        else:
            console.print(f"❌ [bold red]Ошибка восстановления: {result.error_message}[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"❌ [bold red]Ошибка: {e}[/bold red]")
        sys.exit(1)


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Путь к обученной модели"),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Количество эпизодов"),
    render: bool = typer.Option(False, "--render", "-r", help="Отображать среду"),
    deterministic: bool = typer.Option(True, "--deterministic", "-d", help="Детерминистическая политика"),
    verbose: int = typer.Option(1, "--verbose", "-v", help="Уровень детализации"),
) -> None:
    """🔍 Оценить обученную модель."""
    
    console.print(Panel.fit("🔍 Оценка модели", style="bold green"))
    
    try:
        model_path = Path(model)
        if not model_path.exists():
            console.print(f"❌ [red]Модель не найдена: {model}[/red]")
            sys.exit(1)
        
        # Настройка логирования
        log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbose, 2)]
        setup_logging(level=log_level)
        
        # Создание конфигурации для оценки
        config = TrainerConfig(
            mode=TrainingMode.EVALUATE,
            n_eval_episodes=episodes,
            eval_deterministic=deterministic,
            verbose=verbose,
        )
        
        console.print(f"📂 Загрузка модели: {model}")
        console.print(f"🎯 Эпизодов: {episodes}, Детерминистическая: {deterministic}")
        
        with Trainer(config) as trainer:
            # Загрузка модели
            # TODO: Реализовать загрузку модели в тренере
            
            eval_result = trainer.evaluate(
                n_episodes=episodes,
                deterministic=deterministic,
                render=render,
            )
        
        # Отображение результатов оценки
        _display_evaluation_results(eval_result)
        
    except Exception as e:
        console.print(f"❌ [bold red]Ошибка оценки: {e}[/bold red]")
        sys.exit(1)


@app.command()
def compare(
    algorithms: List[str] = typer.Option(
        ["PPO", "A2C"], "--algorithm", "-a", help="Алгоритмы для сравнения"
    ),
    env: str = typer.Option("LunarLander-v3", "--env", "-e", help="Среда для тестирования"),
    timesteps: int = typer.Option(50_000, "--timesteps", "-t", help="Шаги обучения"),
    runs: int = typer.Option(1, "--runs", "-r", help="Количество запусков"),
    seeds: List[int] = typer.Option([42], "--seed", "-s", help="Seeds для экспериментов"),
    output_dir: str = typer.Option("results/comparison", "--output", "-o", help="Директория результатов"),
) -> None:
    """📊 Сравнить алгоритмы обучения."""
    
    console.print(Panel.fit("📊 Сравнение алгоритмов", style="bold magenta"))
    
    try:
        setup_logging(level=logging.INFO)
        
        results = {}
        
        # Расширение seeds если нужно
        if len(seeds) < runs:
            seeds = seeds * runs
        seeds = seeds[:runs]
        
        total_experiments = len(algorithms) * runs
        
        with Progress(console=console) as progress:
            task = progress.add_task("Сравнение алгоритмов...", total=total_experiments)
            
            for algorithm in algorithms:
                algorithm_results = []
                
                for run, seed in enumerate(seeds):
                    console.print(f"🎯 {algorithm} - запуск {run + 1}/{runs} (seed={seed})")
                    
                    config = TrainerConfig(
                        experiment_name=f"comparison_{algorithm.lower()}_run{run + 1}",
                        algorithm=algorithm,
                        environment_name=env,
                        total_timesteps=timesteps,
                        seed=seed,
                        output_dir=output_dir,
                        verbose=0,
                        eval_freq=timesteps // 4,  # 4 оценки за обучение
                    )
                    
                    with Trainer(config) as trainer:
                        result = trainer.train()
                        algorithm_results.append(result)
                    
                    progress.update(task, advance=1)
                
                results[algorithm] = algorithm_results
        
        # Отображение сравнения
        _display_comparison_results(results)
        
    except Exception as e:
        console.print(f"❌ [bold red]Ошибка сравнения: {e}[/bold red]")
        sys.exit(1)


@app.command()
def config(
    action: str = typer.Argument(..., help="Действие: create, validate, list"),
    config_dir: str = typer.Option("configs", "--dir", "-d", help="Директория конфигураций"),
) -> None:
    """⚙️  Управление конфигурациями."""
    
    config_path = Path(config_dir)
    
    if action == "create":
        console.print("📝 Создание конфигураций по умолчанию...")
        create_default_configs(config_path)
        console.print(f"✅ Конфигурации созданы в {config_path}")
        
    elif action == "validate":
        console.print("🔍 Валидация конфигураций...")
        is_valid = validate_config_directory(config_path)
        if is_valid:
            console.print("✅ [green]Все конфигурации валидны[/green]")
        else:
            console.print("❌ [red]Найдены ошибки в конфигурациях[/red]")
            sys.exit(1)
            
    elif action == "list":
        console.print("📋 Доступные конфигурации:")
        loader = get_config_loader(config_path)
        configs = loader.get_available_configs()
        
        if configs:
            table = Table(title="Конфигурации")
            table.add_column("Имя", style="cyan")
            table.add_column("Путь", style="magenta")
            
            for config_name in configs:
                config_file = config_path / f"{config_name}.yaml"
                table.add_row(config_name, str(config_file))
            
            console.print(table)
        else:
            console.print("📭 Конфигурации не найдены")
    
    else:
        console.print(f"❌ [red]Неизвестное действие: {action}[/red]")
        console.print("Доступные действия: create, validate, list")
        sys.exit(1)


def _display_config(config: TrainerConfig) -> None:
    """Отобразить конфигурацию обучения."""
    table = Table(title="Конфигурация обучения")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="magenta")
    
    table.add_row("Эксперимент", config.experiment_name)
    table.add_row("Алгоритм", config.algorithm)
    table.add_row("Среда", config.environment_name)
    table.add_row("Шаги", f"{config.total_timesteps:,}")
    table.add_row("Seed", str(config.seed))
    table.add_row("Режим", config.mode.value)
    table.add_row("Выходная директория", config.output_dir)
    
    console.print(table)


def _display_results(result) -> None:
    """Отобразить результаты обучения."""
    if result.success:
        table = Table(title="Результаты обучения")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Финальная награда", f"{result.final_mean_reward:.2f} ± {result.final_std_reward:.2f}")
        table.add_row("Лучшая награда", f"{result.best_mean_reward:.2f}")
        table.add_row("Время обучения", f"{result.training_time:.1f} сек")
        table.add_row("Общие шаги", f"{result.total_timesteps:,}")
        
        if result.early_stopped:
            table.add_row("Раннее остановка", "Да")
        
        if result.model_path:
            table.add_row("Модель", result.model_path)
        
        console.print(table)
    else:
        console.print(f"❌ [red]Обучение неуспешно: {result.error_message}[/red]")


def _display_evaluation_results(eval_result: dict) -> None:
    """Отобразить результаты оценки."""
    table = Table(title="Результаты оценки")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")
    
    table.add_row("Средняя награда", f"{eval_result['mean_reward']:.2f}")
    table.add_row("Стд. отклонение", f"{eval_result['std_reward']:.2f}")
    table.add_row("Мин. награда", f"{eval_result['min_reward']:.2f}")
    table.add_row("Макс. награда", f"{eval_result['max_reward']:.2f}")
    table.add_row("Средняя длина", f"{eval_result['mean_length']:.1f}")
    
    console.print(table)


def _display_comparison_results(results: dict) -> None:
    """Отобразить результаты сравнения."""
    table = Table(title="Сравнение алгоритмов")
    table.add_column("Алгоритм", style="cyan")
    table.add_column("Средняя награда", style="green")
    table.add_column("Стд. отклонение", style="yellow")
    table.add_column("Успешных", style="blue")
    
    for algorithm, algorithm_results in results.items():
        successful_results = [r for r in algorithm_results if r.success]
        
        if successful_results:
            rewards = [r.final_mean_reward for r in successful_results]
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
            
            table.add_row(
                algorithm,
                f"{mean_reward:.2f}",
                f"{std_reward:.2f}",
                f"{len(successful_results)}/{len(algorithm_results)}"
            )
        else:
            table.add_row(algorithm, "ОШИБКА", "-", f"0/{len(algorithm_results)}")
    
    console.print(table)


def main() -> None:
    """Главная функция CLI."""
    app()


if __name__ == "__main__":
    main()