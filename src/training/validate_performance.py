"""Валидация производительности обучения RL агентов.

Этот скрипт проверяет соответствие критериям успеха:
- SC-001: Audit завершается < 10 мин
- SC-003: Обучение < 30 мин каждый
- SC-006: Reward > 200 для обоих алгоритмов

Использование:
    python -m src.training.validate_performance

Результаты сохраняются в: results/performance_benchmarks.json
"""

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.rl_logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmark:
    """Результаты бенчмаркинга производительности."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_steps: int = 1000

    # Времена выполнения (в секундах)
    audit_time: float = 0.0
    ppo_time: float = 0.0
    a2c_time: float = 0.0

    # Награды
    ppo_reward: float = 0.0
    a2c_reward: float = 0.0

    # Статусы валидации
    audit_passed: bool = False
    ppo_time_passed: bool = False
    a2c_time_passed: bool = False
    ppo_reward_passed: bool = False
    a2c_reward_passed: bool = False

    # Общий статус
    validation_status: str = "UNKNOWN"  # PASSED, FAILED, PARTIAL

    # Пороги (в секундах)
    audit_threshold: float = 600.0  # 10 минут
    training_threshold: float = 1800.0  # 30 минут
    reward_threshold: float = 200.0

    # Детальная информация
    details: Dict[str, Any] = field(default_factory=dict)


def time_audit_operation(
    skip_smoke_tests: bool = True,
    scope: str = "src/",
) -> float:
    """Измерить время выполнения аудита.

    Args:
        skip_smoke_tests: Пропустить smoke тесты для ускорения
        scope: Директория для аудита

    Returns:
        Время выполнения в секундах

    Raises:
        RuntimeError: Если аудит завершился с ошибкой
    """
    logger.info(f"Запуск аудита: scope={scope}, skip_smoke_tests={skip_smoke_tests}")

    # Формируем команду
    cmd = [
        sys.executable,
        "-m",
        "src.audit.run",
        "--scope",
        scope,
        "--format",
        "json",
    ]

    if skip_smoke_tests:
        cmd.append("--skip-smoke-tests")

    # Запускаем процесс
    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 минут таймаут
            check=True,
        )

        elapsed_time = time.time() - start_time

        logger.info(f"Аудит завершен за {elapsed_time:.2f} сек")

        # Проверяем наличие файла отчета
        audit_file = Path("audit_report.json")
        if not audit_file.exists():
            logger.warning(
                "Файл audit_report.json не найден, но команда завершилась успешно"
            )

        return elapsed_time

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"Аудит превысил таймаут ({elapsed_time:.2f} сек)")
        raise RuntimeError(f"Аудит превысил таймаут: {elapsed_time:.2f} сек")

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Аудит завершился с ошибкой (exit code {e.returncode})")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise RuntimeError(f"Аудит завершился с ошибкой: {e}")


def time_training_operation(
    algo: str = "ppo",
    seed: int = 42,
    steps: int = 1000,
) -> tuple[float, float]:
    """Измерить время выполнения обучения и получить награду.

    Args:
        algo: Алгоритм (ppo или a2c)
        seed: Seed для воспроизводимости
        steps: Количество шагов обучения

    Returns:
        Кортеж (время в секундах, финальная награда)

    Raises:
        RuntimeError: Если обучение завершилось с ошибкой
    """
    logger.info(f"Запуск обучения: algo={algo}, seed={seed}, steps={steps}")

    # Формируем команду
    cmd = [
        sys.executable,
        "-m",
        "src.training.train",
        "--algo",
        algo,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
    ]

    # Запускаем процесс
    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 минут таймаут
            check=True,
        )

        elapsed_time = time.time() - start_time

        logger.info(f"Обучение завершено за {elapsed_time:.2f} сек")

        # Читаем результаты из JSON файла
        results_file = Path(
            f"results/experiments/{algo}_seed{seed}/{algo}_seed{seed}_results.json"
        )

        if not results_file.exists():
            raise RuntimeError(f"Файл результатов не найден: {results_file}")

        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        # Извлекаем финальную награду
        exp_results = results_data.get("experiment_results", {})
        metrics = exp_results.get("metrics", {})

        final_reward = metrics.get("final_mean_reward", 0.0)

        logger.info(f"Финальная награда: {final_reward:.2f}")

        return elapsed_time, final_reward

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"Обучение превысило таймаут ({elapsed_time:.2f} сек)")
        raise RuntimeError(f"Обучение превысило таймаут: {elapsed_time:.2f} сек")

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Обучение завершилось с ошибкой (exit code {e.returncode})")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise RuntimeError(f"Обучение завершилось с ошибкой: {e}")


def run_performance_benchmark(
    validation_steps: int = 1000,
    skip_smoke_tests: bool = True,
) -> PerformanceBenchmark:
    """Запустить бенчмаркинг производительности.

    Args:
        validation_steps: Количество шагов для валидации
        skip_smoke_tests: Пропустить smoke тесты в аудите

    Returns:
        Результаты бенчмаркинга
    """
    logger.info("=" * 60)
    logger.info("ВАЛИДАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    logger.info("=" * 60)
    logger.info(f"  Шагов валидации: {validation_steps}")
    logger.info("  Порог SC-001 (Audit): < 10 мин (600 сек)")
    logger.info("  Порог SC-003 (Training): < 30 мин (1800 сек)")
    logger.info("  Порог SC-006 (Reward): > 200")
    logger.info("=" * 60)

    benchmark = PerformanceBenchmark(
        validation_steps=validation_steps,
        audit_threshold=600.0,
        training_threshold=1800.0,
        reward_threshold=200.0,
    )

    # 1. Тестируем аудита (SC-001)
    logger.info("\n--- Тест SC-001: Audit Performance ---")
    try:
        benchmark.audit_time = time_audit_operation(
            skip_smoke_tests=skip_smoke_tests, scope="src/"
        )
        benchmark.audit_passed = benchmark.audit_time < benchmark.audit_threshold

        status = "✓ PASSED" if benchmark.audit_passed else "✗ FAILED"
        logger.info(
            f"  Время: {benchmark.audit_time:.2f} сек / {benchmark.audit_threshold:.2f} сек"
        )
        logger.info(f"  Статус: {status}")

    except Exception as e:
        logger.error(f"Ошибка при выполнении аудита: {e}")
        benchmark.audit_passed = False
        benchmark.details["audit_error"] = str(e)

    # 2. Тестируем PPO обучение (SC-003, SC-006)
    logger.info("\n--- Тест SC-003/SC-006: PPO Performance ---")
    try:
        benchmark.ppo_time, benchmark.ppo_reward = time_training_operation(
            algo="ppo", seed=42, steps=validation_steps
        )

        benchmark.ppo_time_passed = benchmark.ppo_time < benchmark.training_threshold
        benchmark.ppo_reward_passed = benchmark.ppo_reward > benchmark.reward_threshold

        time_status = "✓ PASSED" if benchmark.ppo_time_passed else "✗ FAILED"
        reward_status = "✓ PASSED" if benchmark.ppo_reward_passed else "✗ FAILED"

        logger.info(
            f"  Время: {benchmark.ppo_time:.2f} сек / {benchmark.training_threshold:.2f} сек"
        )
        logger.info(
            f"  Награда: {benchmark.ppo_reward:.2f} / {benchmark.reward_threshold:.2f}"
        )
        logger.info(f"  Статус времени: {time_status}")
        logger.info(f"  Статус награды: {reward_status}")

    except Exception as e:
        logger.error(f"Ошибка при обучении PPO: {e}")
        benchmark.ppo_time_passed = False
        benchmark.ppo_reward_passed = False
        benchmark.details["ppo_error"] = str(e)

    # 3. Тестируем A2C обучение (SC-003, SC-006)
    logger.info("\n--- Тест SC-003/SC-006: A2C Performance ---")
    try:
        benchmark.a2c_time, benchmark.a2c_reward = time_training_operation(
            algo="a2c", seed=42, steps=validation_steps
        )

        benchmark.a2c_time_passed = benchmark.a2c_time < benchmark.training_threshold
        benchmark.a2c_reward_passed = benchmark.a2c_reward > benchmark.reward_threshold

        time_status = "✓ PASSED" if benchmark.a2c_time_passed else "✗ FAILED"
        reward_status = "✓ PASSED" if benchmark.a2c_reward_passed else "✗ FAILED"

        logger.info(
            f"  Время: {benchmark.a2c_time:.2f} сек / {benchmark.training_threshold:.2f} сек"
        )
        logger.info(
            f"  Награда: {benchmark.a2c_reward:.2f} / {benchmark.reward_threshold:.2f}"
        )
        logger.info(f"  Статус времени: {time_status}")
        logger.info(f"  Статус награды: {reward_status}")

    except Exception as e:
        logger.error(f"Ошибка при обучении A2C: {e}")
        benchmark.a2c_time_passed = False
        benchmark.a2c_reward_passed = False
        benchmark.details["a2c_error"] = str(e)

    # Вычисляем общий статус
    all_passed = all(
        [
            benchmark.audit_passed,
            benchmark.ppo_time_passed,
            benchmark.a2c_time_passed,
            benchmark.ppo_reward_passed,
            benchmark.a2c_reward_passed,
        ]
    )

    some_passed = any(
        [
            benchmark.audit_passed,
            benchmark.ppo_time_passed,
            benchmark.a2c_time_passed,
            benchmark.ppo_reward_passed,
            benchmark.a2c_reward_passed,
        ]
    )

    if all_passed:
        benchmark.validation_status = "PASSED"
    elif some_passed:
        benchmark.validation_status = "PARTIAL"
    else:
        benchmark.validation_status = "FAILED"

    # Добавляем детальную информацию
    benchmark.details.update(
        {
            "time_per_step_ppo": benchmark.ppo_time / validation_steps
            if validation_steps > 0
            else 0,
            "time_per_step_a2c": benchmark.a2c_time / validation_steps
            if validation_steps > 0
            else 0,
            "estimated_full_ppo_time": benchmark.ppo_time * (50000 / validation_steps)
            if validation_steps > 0
            else 0,
            "estimated_full_a2c_time": benchmark.a2c_time * (50000 / validation_steps)
            if validation_steps > 0
            else 0,
        }
    )

    # Выводим итоговый результат
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    logger.info("=" * 60)
    logger.info(
        f"  SC-001 (Audit < 10 мин): {'✓ PASSED' if benchmark.audit_passed else '✗ FAILED'}"
    )
    logger.info(
        f"  SC-003 (PPO < 30 мин): {'✓ PASSED' if benchmark.ppo_time_passed else '✗ FAILED'}"
    )
    logger.info(
        f"  SC-003 (A2C < 30 мин): {'✓ PASSED' if benchmark.a2c_time_passed else '✗ FAILED'}"
    )
    logger.info(
        f"  SC-006 (PPO > 200): {'✓ PASSED' if benchmark.ppo_reward_passed else '✗ FAILED'}"
    )
    logger.info(
        f"  SC-006 (A2C > 200): {'✓ PASSED' if benchmark.a2c_reward_passed else '✗ FAILED'}"
    )
    logger.info("=" * 60)
    logger.info(f"  Общий статус: {benchmark.validation_status}")
    logger.info("=" * 60)

    return benchmark


def save_benchmark_result(
    benchmark: PerformanceBenchmark,
    output_path: Path = Path("results/performance_benchmarks.json"),
) -> None:
    """Сохранить результат бенчмаркинга в JSON файл.

    Args:
        benchmark: Результат бенчмаркинга
        output_path: Путь для сохранения
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Преобразуем в словарь
    result_dict = {
        "timestamp": benchmark.timestamp,
        "validation_steps": benchmark.validation_steps,
        "times": {
            "audit_seconds": benchmark.audit_time,
            "audit_threshold_seconds": benchmark.audit_threshold,
            "ppo_seconds": benchmark.ppo_time,
            "a2c_seconds": benchmark.a2c_time,
            "training_threshold_seconds": benchmark.training_threshold,
        },
        "rewards": {
            "ppo_reward": benchmark.ppo_reward,
            "a2c_reward": benchmark.a2c_reward,
            "reward_threshold": benchmark.reward_threshold,
        },
        "validation": {
            "sc001_audit_passed": benchmark.audit_passed,
            "sc003_ppo_time_passed": benchmark.ppo_time_passed,
            "sc003_a2c_time_passed": benchmark.a2c_time_passed,
            "sc006_ppo_reward_passed": benchmark.ppo_reward_passed,
            "sc006_a2c_reward_passed": benchmark.a2c_reward_passed,
            "overall_status": benchmark.validation_status,
        },
        "details": benchmark.details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Результат сохранен: {output_path}")


def main() -> int:
    """Главная функция."""
    # Настройка логирования
    setup_logging(log_level=logging.INFO, console_output=True)

    try:
        # Запускаем бенчмаркинг
        benchmark = run_performance_benchmark(
            validation_steps=1000, skip_smoke_tests=True
        )

        # Сохраняем результат
        save_benchmark_result(benchmark)

        # Возвращаем код выхода
        if benchmark.validation_status == "PASSED":
            logger.info("\n✅ Все тесты пройдены!")
            return 0
        elif benchmark.validation_status == "PARTIAL":
            logger.warning("\n⚠️  Некоторые тесты не пройдены")
            return 1
        else:
            logger.error("\n❌ Все тесты не пройдены")
            return 2

    except Exception as e:
        logger.error(f"\n❌ Ошибка валидации: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())
