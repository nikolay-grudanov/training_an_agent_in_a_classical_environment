"""Валидация воспроизводимости обучения RL агентов.

Этот скрипт запускает обучение PPO агента дважды с одинаковым seed=42
и проверяет, что результаты идентичны (std deviation < 0.01).

Использование:
    python -m src.training.validate_reproducibility

Результаты сохраняются в: results/reproducibility_validation.json
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
class ReproducibilityResult:
    """Результат проверки воспроизводимости."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: int = 42
    algorithm: str = "PPO"
    total_timesteps: int = 1000
    runs: list[Dict[str, Any]] = field(default_factory=list)
    reward_mean: float = 0.0
    reward_std: float = 0.0
    status: str = "UNKNOWN"  # PERFECT, GOOD, POOR
    is_reproducible: bool = False
    validation_threshold: float = 0.01


def run_training_command(
    algo: str = "ppo",
    seed: int = 42,
    steps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Запустить обучение через CLI команду.

    Args:
        algo: Алгоритм (ppo или a2c)
        seed: Seed для воспроизводимости
        steps: Количество шагов обучения
        verbose: Включить подробное логирование

    Returns:
        Словарь с результатами обучения

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

    if verbose:
        cmd.append("--verbose")

    # Запускаем процесс
    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 минут таймаут
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
        final_std = metrics.get("final_std_reward", 0.0)

        return {
            "run_id": f"{algo}_seed{seed}_{datetime.now().strftime('%H%M%S')}",
            "algorithm": algo.upper(),
            "seed": seed,
            "total_timesteps": steps,
            "final_reward_mean": final_reward,
            "final_reward_std": final_std,
            "elapsed_time": elapsed_time,
            "results_file": str(results_file),
            "success": True,
        }

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


def validate_reproducibility(
    seed: int = 42,
    num_runs: int = 2,
    total_timesteps: int = 1000,
    validation_threshold: float = 0.01,
) -> ReproducibilityResult:
    """Проверить воспроизводимость обучения.

    Args:
        seed: Seed для воспроизводимости
        num_runs: Количество запусков
        total_timesteps: Количество шагов обучения
        validation_threshold: Порог std deviation для PERFECT статуса

    Returns:
        Результат проверки воспроизводимости
    """
    logger.info("=" * 60)
    logger.info("ВАЛИДАЦИЯ ВОСПРОИЗВОДИМОСТИ")
    logger.info("=" * 60)
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Количество запусков: {num_runs}")
    logger.info(f"  Шагов обучения: {total_timesteps}")
    logger.info(f"  Порог PERFECT: std < {validation_threshold}")
    logger.info("=" * 60)

    result = ReproducibilityResult(
        seed=seed,
        algorithm="PPO",
        total_timesteps=total_timesteps,
        validation_threshold=validation_threshold,
    )

    # Запускаем обучение несколько раз
    rewards = []

    for run_idx in range(num_runs):
        logger.info(f"\n--- Запуск {run_idx + 1}/{num_runs} ---")

        try:
            run_result = run_training_command(
                algo="ppo", seed=seed, steps=total_timesteps, verbose=False
            )

            result.runs.append(run_result)
            rewards.append(run_result["final_reward_mean"])

            logger.info(
                f"Финальная награда: {run_result['final_reward_mean']:.4f} "
                f"± {run_result['final_reward_std']:.4f}"
            )

        except Exception as e:
            logger.error(f"Ошибка в запуске {run_idx + 1}: {e}")
            result.runs.append(
                {
                    "run_id": f"failed_run_{run_idx + 1}",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

    # Вычисляем статистику
    if rewards:
        result.reward_mean = sum(rewards) / len(rewards)
        result.reward_std = (
            sum((r - result.reward_mean) ** 2 for r in rewards) / len(rewards)
        ) ** 0.5

        # Определяем статус
        if result.reward_std < validation_threshold:
            result.status = "PERFECT"
            result.is_reproducible = True
        elif result.reward_std < validation_threshold * 10:
            result.status = "GOOD"
            result.is_reproducible = True
        else:
            result.status = "POOR"
            result.is_reproducible = False

    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    logger.info("=" * 60)
    logger.info(f"  Средняя награда: {result.reward_mean:.4f}")
    logger.info(f"  Std deviation: {result.reward_std:.6f}")
    logger.info(f"  Статус: {result.status}")
    logger.info(f"  Воспроизводимость: {'✓ ДА' if result.is_reproducible else '✗ НЕТ'}")
    logger.info(f"  Порог PERFECT: < {validation_threshold}")
    logger.info("=" * 60)

    return result


def save_validation_result(
    result: ReproducibilityResult,
    output_path: Path = Path("results/reproducibility_validation.json"),
) -> None:
    """Сохранить результат валидации в JSON файл.

    Args:
        result: Результат валидации
        output_path: Путь для сохранения
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Преобразуем в словарь
    result_dict = {
        "timestamp": result.timestamp,
        "seed": result.seed,
        "algorithm": result.algorithm,
        "total_timesteps": result.total_timesteps,
        "runs": result.runs,
        "reward_mean": result.reward_mean,
        "reward_std": result.reward_std,
        "status": result.status,
        "is_reproducible": result.is_reproducible,
        "validation_threshold": result.validation_threshold,
        "success_criteria": {
            "perfect": f"std < {result.validation_threshold}",
            "good": f"std < {result.validation_threshold * 10}",
            "poor": f"std >= {result.validation_threshold * 10}",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Результат сохранен: {output_path}")


def main() -> int:
    """Главная функция."""
    # Настройка логирования
    setup_logging(log_level=logging.INFO, console_output=True)

    try:
        # Проверяем воспроизводимость
        result = validate_reproducibility(
            seed=42, num_runs=2, total_timesteps=1000, validation_threshold=0.01
        )

        # Сохраняем результат
        save_validation_result(result)

        # Возвращаем код выхода
        if result.is_reproducible:
            logger.info("\n✅ Воспроизводимость подтверждена!")
            return 0
        else:
            logger.warning("\n⚠️  Воспроизводимость НЕ подтверждена")
            return 1

    except Exception as e:
        logger.error(f"\n❌ Ошибка валидации: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
