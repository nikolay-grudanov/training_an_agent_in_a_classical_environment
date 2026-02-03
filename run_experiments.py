#!/usr/bin/env python3
"""
TUI (Text User Interface) для управления экспериментами RL.

Позволяет запускать полный workflow или отдельные элементы через интерактивное меню.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}\n")


def print_option(number: int, text: str) -> None:
    """Print numbered option."""
    print(f"  {Colors.OKGREEN}{number}){Colors.ENDC} {text}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def run_command(cmd: List[str], description: str) -> bool:
    """Run command and handle errors.

    Args:
        cmd: Command to run as list
        description: Description of what command does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{Colors.OKBLUE}→{Colors.ENDC} {description}")
    print(f"{Colors.BOLD}Команда:{Colors.ENDC} {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
        )

        if result.returncode == 0:
            print_success("Успешно завершено!")
            if result.stdout:
                print(result.stdout[-500:])  # Show last 500 chars
        else:
            print_success("Успешно завершено (без вывода)")
        return True
    except Exception as e:
        print_error(f"Ошибка: {e}")
        return False


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    dependencies = ["python", "pytest", "ruff"]
    missing = []

    for dep in dependencies:
        result = subprocess.run(
            ["which", dep],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(dep)

    if missing:
        print_warning(f"Отсутствуют зависимости: {', '.join(missing)}")
    else:
        print_success("Все зависимости установлены")


# ============================================================================
# Menu Options
# ============================================================================


def menu_main() -> None:
    """Main menu - top level."""
    print_header("📋 RL Experiments - Главное меню")

    print_option(1, "🎓 ОБУЧЕНИЕ МОДЕЛЕЙ")
    print_option(2, "📊 ОЦЕНКА МОДЕЛЕЙ")
    print_option(3, "📈 ГЕНЕРАЦИЯ ГРАФИКОВ")
    print_option(4, "🎬 ГЕНЕРАЦИЯ ВИДЕО")
    print_option(5, "📋 ГЕНЕРАЦИЯ ОТЧЁТОВ")
    print_option(6, "🧪 ТЕСТИРОВАНИЕ")
    print_option(7, "✅ ПРОВЕРКА КАЧЕСТВА КОДА")
    print_option(8, "🚀 ПОЛНЫЙ WORKFLOW")
    print_option(0, "🚪 ВЫХОД")


def menu_training() -> None:
    """Training menu."""
    print_header("🎓 ОБУЧЕНИЕ МОДЕЛЕЙ")

    print_option(1, "PPO - 200K шагов (стандартное)")
    print_option(2, "PPO - 50K шагов (быстрое)")
    print_option(3, "PPO - 300K шагов (расширенное)")
    print_option(4, "PPO - 500K шагов (для сходимости >200)")
    print_option(5, "A2C - 200K шагов (стандартное)")
    print_option(6, "A2C - 50K шагов (быстрое)")
    print_option(7, "A2C - 300K шагов (расширенное)")
    print_option(8, "Пользовательский ввод (алгоритм, шаги, seed, gamma)")
    print_option(0, "← Назад")


def menu_evaluation() -> None:
    """Evaluation menu."""
    print_header("📊 ОЦЕНКА МОДЕЛЕЙ")

    print_option(1, "Оценить PPO модель")
    print_option(2, "Оценить A2C модель")
    print_option(3, "Оценить обе модели")
    print_option(0, "← Назад")


def menu_graphs() -> None:
    """Graph generation menu."""
    print_header("📈 ГЕНЕРАЦИЯ ГРАФИКОВ")

    print_option(1, "Кривая обучения PPO")
    print_option(2, "Кривая обучения A2C")
    print_option(3, "Сравнение PPO vs A2C")
    print_option(4, "Сравнение Gamma гиперпараметров")
    print_option(0, "← Назад")


def menu_video() -> None:
    """Video generation menu."""
    print_header("🎬 ГЕНЕРАЦИЯ ВИДЕО")

    print_option(1, "Видео PPO агента")
    print_option(2, "Видео A2C агента")
    print_option(3, "Видео с оценкой очков")
    print_option(4, "Пользовательский ввод (модель, путь, эпизоды)")
    print_option(0, "← Назад")


def menu_reports() -> None:
    """Report generation menu."""
    print_header("📋 ГЕНЕРАЦИЯ ОТЧЁТОВ")

    print_option(1, "Отчёт по базовым моделям")
    print_option(2, "Полный отчёт (с Gamma)")
    print_option(3, "Краткий отчёт (без медиа)")
    print_option(0, "← Назад")


def menu_testing() -> None:
    """Testing menu."""
    print_header("🧪 ТЕСТИРОВАНИЕ")

    print_option(1, "Все unit-тесты")
    print_option(2, "Тесты callbacks")
    print_option(3, "Тесты оценки")
    print_option(4, "Тесты графиков")
    print_option(5, "Тесты статистики")
    print_option(6, "Интеграционные тесты")
    print_option(7, "Тесты с покрытием")
    print_option(0, "← Назад")


def menu_quality() -> None:
    """Code quality check menu."""
    print_header("✅ ПРОВЕРКА КАЧЕСТВА КОДА")

    print_option(1, "Проверить стиль (ruff check)")
    print_option(2, "Исправить ошибки (ruff check --fix)")
    print_option(3, "Форматировать код (ruff format)")
    print_option(4, "Проверить типы (mypy)")
    print_option(5, "Полная проверка (ruff + mypy)")
    print_option(0, "← Назад")


# ============================================================================
# Action Functions
# ============================================================================


def action_train_ppo(timesteps: int = 200000, seed: int = 42, gamma: float = 0.99) -> bool:
    """Train PPO model."""
    cmd = [
        "python", "-m", "src.experiments.completion.baseline_training",
        "--algo", "ppo",
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--gamma", str(gamma),
    ]
    desc = f"Обучение PPO: {timesteps/1000}K шагов, seed={seed}, gamma={gamma}"
    return run_command(cmd, desc)


def action_train_a2c(timesteps: int = 200000, seed: int = 42, gamma: float = 0.99) -> bool:
    """Train A2C model."""
    cmd = [
        "python", "-m", "src.experiments.completion.baseline_training",
        "--algo", "a2c",
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--gamma", str(gamma),
    ]
    desc = f"Обучение A2C: {timesteps/1000}K шагов, seed={seed}, gamma={gamma}"
    return run_command(cmd, desc)


def action_train_custom() -> bool:
    """Train with custom parameters."""
    print_header("Настройка обучения")

    try:
        algo = input("Алгоритм (a2c/ppo) [ppo]: ").strip() or "ppo"
        timesteps = int(input("Количество шагов [200000]: ").strip() or "200000")
        seed = int(input("Seed [42]: ").strip() or "42")
        gamma = float(input("Gamma [0.99]: ").strip() or "0.99")

        if algo.lower() in ["a2c", "ppo"]:
            if algo.lower() == "a2c":
                return action_train_a2c(timesteps, seed, gamma)
            else:
                return action_train_ppo(timesteps, seed, gamma)
        else:
            print_error("Некорректный алгоритм!")
            return False
    except ValueError as e:
        print_error(f"Некорректный ввод: {e}")
        return False


def action_evaluate_ppo() -> bool:
    """Evaluate PPO model."""
    model_path = "results/experiments/ppo_seed42/ppo_seed42_model.zip"

    if not Path(model_path).exists():
        print_warning(f"Модель не найдена: {model_path}")
        print_option(1, "Обучить PPO (200K шагов)")
        print_option(2, "← Назад")
        choice = input("Выбор: ")
        if choice == "1":
            return action_train_ppo()
        return True

    cmd = [
        "python", "-c",
        f"""
from src.training.evaluation import evaluate_agent
result = evaluate_agent(
    model_path='{model_path}',
    env_id='LunarLander-v3',
    n_eval_episodes=10,
)
print(f'Средняя награда: {{result["mean_reward"]:.2f}} ± {{result["std_reward"]:.2f}}')
print(f'Сходимость: {{"ДА" if result["convergence_achieved"] else "НЕТ"}}')
""",
    ]
    return run_command(cmd, "Оценка PPO модели")


def action_evaluate_a2c() -> bool:
    """Evaluate A2C model."""
    model_path = "results/experiments/a2c_seed42/a2c_seed42_model.zip"

    if not Path(model_path).exists():
        print_warning(f"Модель не найдена: {model_path}")
        print_option(1, "Обучить A2C (200K шагов)")
        print_option(2, "← Назад")
        choice = input("Выбор: ")
        if choice == "1":
            return action_train_a2c()
        return True

    cmd = [
        "python", "-c",
        f"""
from src.training.evaluation import evaluate_agent
result = evaluate_agent(
    model_path='{model_path}',
    env_id='LunarLander-v3',
    n_eval_episodes=10,
)
print(f'Средняя награда: {{result["mean_reward"]:.2f}} ± {{result["std_reward"]:.2f}}')
print(f'Сходимость: {{"ДА" if result["convergence_achieved"] else "НЕТ"}}')
""",
    ]
    return run_command(cmd, "Оценка A2C модели")


def action_evaluate_both() -> bool:
    """Evaluate both models."""
    print("Оценка обеих моделей...\n")

    ppo_ok = action_evaluate_ppo()
    print()
    a2c_ok = action_evaluate_a2c()

    return ppo_ok and a2c_ok


def action_graph_ppo() -> bool:
    """Generate PPO learning curve."""
    cmd = [
        "python", "-m", "src.visualization.graphs",
        "--experiment", "ppo_seed42",
        "--type", "learning_curve",
        "--output", "results/experiments/ppo_seed42/reward_curve.png",
        "--title", "Кривая обучения PPO (Seed=42)",
    ]
    return run_command(cmd, "Генерация кривой обучения PPO")


def action_graph_a2c() -> bool:
    """Generate A2C learning curve."""
    cmd = [
        "python", "-m", "src.visualization.graphs",
        "--experiment", "a2c_seed42",
        "--type", "learning_curve",
        "--output", "results/experiments/a2c_seed42/reward_curve.png",
        "--title", "Кривая обучения A2C (Seed=42)",
    ]
    return run_command(cmd, "Генерация кривой обучения A2C")


def action_graph_comparison() -> bool:
    """Generate comparison graph."""
    cmd = [
        "python", "-m", "src.visualization.graphs",
        "--experiment", "a2c_seed42,ppo_seed42",
        "--type", "comparison",
        "--output", "results/comparison/a2c_vs_ppo.png",
        "--title", "Сравнение алгоритмов: A2C vs PPO",
    ]
    return run_command(cmd, "Генерация сравнительного графика")


def action_graph_gamma() -> bool:
    """Generate gamma comparison graph."""
    cmd = [
        "python", "-m", "src.visualization.graphs",
        "--experiment", "gamma_090,gamma_099,gamma_0999",
        "--type", "gamma_comparison",
        "--output", "results/comparison/gamma_comparison.png",
        "--title", "Сравнение гиперпараметра gamma",
    ]
    return run_command(cmd, "Генерация графика gamma сравнения")


def action_video_ppo() -> bool:
    """Generate PPO video."""
    model_path = "results/experiments/ppo_seed42/ppo_seed42_model.zip"

    if not Path(model_path).exists():
        print_warning(f"Модель не найдена: {model_path}")
        print_option(1, "Обучить PPO (200K шагов)")
        print_option(2, "← Назад")
        choice = input("Выбор: ")
        if choice == "1":
            return action_train_ppo()
        return True

    cmd = [
        "python", "-m", "src.visualization.video",
        "--model", model_path,
        "--output", "results/experiments/ppo_seed42/video.mp4",
        "--episodes", "5",
    ]
    return run_command(cmd, "Генерация видео PPO агента")


def action_video_a2c() -> bool:
    """Generate A2C video."""
    model_path = "results/experiments/a2c_seed42/a2c_seed42_model.zip"

    if not Path(model_path).exists():
        print_warning(f"Модель не найдена: {model_path}")
        print_option(1, "Обучить A2C (200K шагов)")
        print_option(2, "← Назад")
        choice = input("Выбор: ")
        if choice == "1":
            return action_train_a2c()
        return True

    cmd = [
        "python", "-m", "src.visualization.video",
        "--model", model_path,
        "--output", "results/experiments/a2c_seed42/video.mp4",
        "--episodes", "5",
    ]
    return run_command(cmd, "Генерация видео A2C агента")


def action_video_custom() -> bool:
    """Generate video with custom parameters."""
    print_header("Настройка видео")

    try:
        model_path = input("Путь к модели: ").strip()
        output_path = input("Путь к видео (default: video.mp4): ").strip() or "video.mp4"
        episodes = int(input("Количество эпизодов [5]: ").strip() or "5")

        if not Path(model_path).exists():
            print_error(f"Модель не найдена: {model_path}")
            return False

        cmd = [
            "python", "-m", "src.visualization.video",
            "--model", model_path,
            "--output", output_path,
            "--episodes", str(episodes),
        ]
        return run_command(cmd, f"Генерация видео: {episodes} эпизодов")
    except ValueError as e:
        print_error(f"Некорректный ввод: {e}")
        return False


def action_report_baseline() -> bool:
    """Generate baseline report."""
    cmd = [
        "python", "-m", "src.reporting.report_generator",
        "--output", "results/reports/experiment_report.md",
        "--experiments", "a2c_seed42", "ppo_seed42",
        "--include-graphs",
        "--include-videos",
    ]
    return run_command(cmd, "Генерация отчёта по базовым моделям")


def action_report_full() -> bool:
    """Generate full report with gamma."""
    cmd = [
        "python", "-m", "src.reporting.report_generator",
        "--output", "results/reports/full_report.md",
        "--experiments", "a2c_seed42", "ppo_seed42", "gamma_090", "gamma_099", "gamma_0999",
        "--include-graphs",
        "--include-videos",
    ]
    return run_command(cmd, "Генерация полного отчёта с Gamma экспериментами")


def action_report_quick() -> bool:
    """Generate quick report without media."""
    cmd = [
        "python", "-m", "src.reporting.report_generator",
        "--output", "results/reports/quick_report.md",
        "--experiments", "a2c_seed42", "ppo_seed42",
        "--no-include-graphs",
        "--no-include-videos",
    ]
    return run_command(cmd, "Генерация краткого отчёта (без медиа)")


def action_test_unit() -> bool:
    """Run all unit tests."""
    cmd = ["pytest", "tests/unit/", "-v"]
    return run_command(cmd, "Запуск всех unit-тестов")


def action_test_integration() -> bool:
    """Run integration tests."""
    cmd = ["pytest", "tests/integration/test_full_workflow.py::TestFullWorkflow", "-v"]
    return run_command(cmd, "Запуск интеграционных тестов")


def action_test_coverage() -> bool:
    """Run tests with coverage."""
    cmd = ["pytest", "tests/", "-v", "--cov=src/", "--cov-report=html"]
    return run_command(cmd, "Запуск тестов с покрытием (HTML отчет)")


def action_quality_check() -> bool:
    """Run code quality check."""
    cmd = ["ruff", "check", "."]
    return run_command(cmd, "Проверка стиля кода (ruff check)")


def action_quality_fix() -> bool:
    """Auto-fix code quality issues."""
    cmd = ["ruff", "check", ".", "--fix"]
    return run_command(cmd, "Автоисправление ошибок (ruff check --fix)")


def action_quality_format() -> bool:
    """Format code."""
    cmd = ["ruff", "format", "."]
    return run_command(cmd, "Форматирование кода (ruff format)")


def action_quality_type() -> bool:
    """Type checking."""
    cmd = ["mypy", "src/", "--strict"]
    return run_command(cmd, "Проверка типов (mypy strict)")


def action_quality_all() -> bool:
    """Run all quality checks."""
    print("Полная проверка качества кода...\n")

    check_ok = action_quality_check()
    format_ok = action_quality_format()
    type_ok = action_quality_type()

    return check_ok and format_ok and type_ok


def action_full_workflow() -> bool:
    """Run complete workflow."""
    print_header("🚀 ЗАПУСК ПОЛНОГО WORKFLOW")

    steps: List[tuple[str, Callable[[], bool]]] = [
        ("Обучение PPO (200K)", lambda: action_train_ppo()),
        ("Обучение A2C (200K)", lambda: action_train_a2c()),
        ("Генерация графиков", lambda: action_graph_comparison()),
        ("Генерация видео PPO", lambda: action_video_ppo()),
        ("Генерация отчёта", lambda: action_report_baseline()),
    ]

    success_count = 0
    for i, (desc, action) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"Шаг {i}/{len(steps)}: {desc}")
        print(f"{'='*60}")

        if action():
            success_count += 1
            print_success(f"Шаг {i} выполнен")
        else:
            print_error(f"Шаг {i} провален")
            print_option(1, "Продолжить")
            print_option(2, "Прервать")
            choice = input("Выбор: ")
            if choice == "1":
                continue
            return False

    print_header(f"РЕЗУЛЬТАТ: {success_count}/{len(steps)} шагов выполнено")

    # Generate final report
    action_report_baseline()

    return success_count == len(steps)


# ============================================================================
# Main Loop
# ============================================================================


def main() -> None:
    """Main TUI loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="TUI для RL экспериментов")
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Проверить зависимости и выйти",
    )
    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        return

    # Print welcome
    print(f"""
{Colors.HEADER}{'='*60}
{Colors.BOLD}      RL EXPERIMENTS MANAGER      {Colors.ENDC}
{Colors.HEADER}{'='*60}{Colors.ENDC}

TUI интерфейс для управления экспериментами RL агентов.
Проект: RL Agent Training System
Среда: LunarLander-v3 (Gymnasium)

{Colors.OKGREEN}Совет:{Colors.ENDC} Запустите с флагом --check-deps для проверки зависимостей
""")

    # Main menu loop
    while True:
        menu_main()
        choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

        if choice == "0":
            print("\n👋 До свидания!")
            break
        elif choice == "1":
            # Training menu
            while True:
                menu_training()
                t_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if t_choice == "0":
                    break
                elif t_choice == "1":
                    action_train_ppo(200000)
                elif t_choice == "2":
                    action_train_ppo(50000)
                elif t_choice == "3":
                    action_train_ppo(300000)
                elif t_choice == "4":
                    action_train_ppo(500000)
                elif t_choice == "5":
                    action_train_a2c(200000)
                elif t_choice == "6":
                    action_train_a2c(50000)
                elif t_choice == "7":
                    action_train_a2c(300000)
                elif t_choice == "8":
                    action_train_custom()
                else:
                    print_error("Некорректный выбор")

        elif choice == "2":
            # Evaluation menu
            while True:
                menu_evaluation()
                e_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if e_choice == "0":
                    break
                elif e_choice == "1":
                    action_evaluate_ppo()
                elif e_choice == "2":
                    action_evaluate_a2c()
                elif e_choice == "3":
                    action_evaluate_both()
                else:
                    print_error("Некорректный выбор")

        elif choice == "3":
            # Graphs menu
            while True:
                menu_graphs()
                g_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if g_choice == "0":
                    break
                elif g_choice == "1":
                    action_graph_ppo()
                elif g_choice == "2":
                    action_graph_a2c()
                elif g_choice == "3":
                    action_graph_comparison()
                elif g_choice == "4":
                    action_graph_gamma()
                else:
                    print_error("Некорректный выбор")

        elif choice == "4":
            # Video menu
            while True:
                menu_video()
                v_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if v_choice == "0":
                    break
                elif v_choice == "1":
                    action_video_ppo()
                elif v_choice == "2":
                    action_video_a2c()
                elif v_choice == "3":
                    action_video_ppo()  # TODO: implement show_scores
                elif v_choice == "4":
                    action_video_custom()
                else:
                    print_error("Некорректный выбор")

        elif choice == "5":
            # Reports menu
            while True:
                menu_reports()
                r_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if r_choice == "0":
                    break
                elif r_choice == "1":
                    action_report_baseline()
                elif r_choice == "2":
                    action_report_full()
                elif r_choice == "3":
                    action_report_quick()
                else:
                    print_error("Некорректный выбор")

        elif choice == "6":
            # Testing menu
            while True:
                menu_testing()
                t_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if t_choice == "0":
                    break
                elif t_choice == "1":
                    action_test_unit()
                elif t_choice == "2":
                    # TODO: test callbacks
                    print_warning("Функция в разработке")
                elif t_choice == "3":
                    # TODO: test evaluation
                    print_warning("Функция в разработке")
                elif t_choice == "4":
                    # TODO: test graphs
                    print_warning("Функция в разработке")
                elif t_choice == "5":
                    # TODO: test statistics
                    print_warning("Функция в разработке")
                elif t_choice == "6":
                    action_test_integration()
                elif t_choice == "7":
                    action_test_coverage()
                else:
                    print_error("Некорректный выбор")

        elif choice == "7":
            # Quality menu
            while True:
                menu_quality()
                q_choice = input(f"\n{Colors.BOLD}Выбор:{Colors.ENDC} ").strip()

                if q_choice == "0":
                    break
                elif q_choice == "1":
                    action_quality_check()
                elif q_choice == "2":
                    action_quality_fix()
                elif q_choice == "3":
                    action_quality_format()
                elif q_choice == "4":
                    action_quality_type()
                elif q_choice == "5":
                    action_quality_all()
                else:
                    print_error("Некорректный выбор")

        elif choice == "8":
            # Full workflow
            action_full_workflow()
            print("\nНажмите Enter для продолжения...")
            input()

        else:
            print_error("Некорректный выбор")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Прервано пользователем{Colors.ENDC}")
        sys.exit(0)
