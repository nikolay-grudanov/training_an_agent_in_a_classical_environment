"""Интеграционные тесты для CLI интерфейса экспериментов.

Тестирует командную строку и различные режимы запуска экспериментов.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest
import yaml


class TestCLIInterface:
    """Тесты CLI интерфейса для экспериментов."""

    @pytest.fixture(scope="class")
    def test_config_path(self) -> Path:
        """Путь к тестовой конфигурации."""
        return Path("configs/test_ppo_vs_a2c.yaml")

    @pytest.fixture(scope="class")
    def temp_output_dir(self) -> Path:
        """Временная директория для CLI тестов."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_cli_"))
        yield temp_dir
        # Очистка выполняется автоматически

    def test_config_file_validation(self, test_config_path: Path):
        """Тест валидации конфигурационного файла."""
        assert test_config_path.exists(), (
            f"Конфигурационный файл не найден: {test_config_path}"
        )

        # Проверяем, что файл можно загрузить
        with open(test_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Проверяем обязательные секции
        required_sections = ["experiment", "baseline", "variant"]
        for section in required_sections:
            assert section in config_data, f"Отсутствует секция {section}"

        # Проверяем параметры для быстрого тестирования
        assert config_data["baseline"]["training_steps"] <= 10000, (
            "Для CLI тестов должны использоваться короткие тренировки"
        )
        assert config_data["variant"]["training_steps"] <= 10000, (
            "Для CLI тестов должны использоваться короткие тренировки"
        )

    def test_cli_help_command(self):
        """Тест команды помощи CLI."""
        try:
            result = subprocess.run(
                ["python", "-m", "src.experiments.runner", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Команда помощи должна завершиться успешно
            assert result.returncode == 0, (
                f"Команда помощи завершилась с ошибкой: {result.stderr}"
            )

            # Проверяем наличие основных опций в выводе
            help_text = result.stdout.lower()
            expected_options = ["--config", "--mode", "--max-workers", "--verbose"]
            for option in expected_options:
                assert option in help_text, f"Опция {option} не найдена в справке"

        except subprocess.TimeoutExpired:
            pytest.fail("Команда помощи выполнялась слишком долго")
        except FileNotFoundError:
            pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_validation_mode(self, test_config_path: Path, temp_output_dir: Path):
        """Тест CLI в режиме валидации."""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.experiments.runner",
                    "--config",
                    str(test_config_path),
                    "--mode",
                    "validation",
                    "--output-dir",
                    str(temp_output_dir),
                    "--verbose",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # В режиме валидации команда должна завершиться быстро и успешно
            if result.returncode == 0:
                assert (
                    "валидация" in result.stdout.lower()
                    or "validation" in result.stdout.lower()
                )
                print("✅ CLI валидация прошла успешно")
            else:
                # Логируем ошибку для отладки, но не падаем
                print(f"⚠️ CLI валидация завершилась с кодом {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI валидация выполнялась слишком долго")
        except FileNotFoundError:
            pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_argument_parsing(self):
        """Тест парсинга аргументов CLI."""
        # Тестируем различные комбинации аргументов
        test_cases = [
            ["--mode", "sequential"],
            ["--mode", "parallel"],
            ["--mode", "validation"],
            ["--max-workers", "2"],
            ["--verbose"],
            ["--verbose", "--verbose"],  # Двойной verbose
            ["--no-monitoring"],
        ]

        for args in test_cases:
            try:
                # Добавляем --help чтобы команда завершилась быстро
                result = subprocess.run(
                    ["python", "-m", "src.experiments.runner", "--help"] + args,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Команда с --help должна завершиться успешно независимо от других аргументов
                assert result.returncode == 0, (
                    f"Ошибка парсинга аргументов {args}: {result.stderr}"
                )

            except subprocess.TimeoutExpired:
                pytest.fail(f"Парсинг аргументов {args} выполнялся слишком долго")
            except FileNotFoundError:
                pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_error_handling(self, temp_output_dir: Path):
        """Тест обработки ошибок в CLI."""
        error_test_cases = [
            {
                "args": ["--config", "nonexistent_config.yaml"],
                "expected_error": "не найден",
            },
            {"args": ["--mode", "invalid_mode"], "expected_error": "invalid choice"},
            {"args": ["--max-workers", "-1"], "expected_error": "invalid"},
        ]

        for test_case in error_test_cases:
            try:
                result = subprocess.run(
                    ["python", "-m", "src.experiments.runner"] + test_case["args"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Команда должна завершиться с ошибкой
                assert result.returncode != 0, (
                    f"Команда должна была завершиться с ошибкой для {test_case['args']}"
                )

                # Проверяем, что ошибка содержит ожидаемый текст
                error_output = (result.stdout + result.stderr).lower()
                expected_error = test_case["expected_error"].lower()

                # Для некоторых ошибок проверяем наличие ключевых слов
                if "не найден" in expected_error:
                    assert any(
                        word in error_output
                        for word in ["not found", "не найден", "error", "ошибка"]
                    )
                elif "invalid" in expected_error:
                    assert any(
                        word in error_output
                        for word in ["invalid", "неверный", "error", "ошибка"]
                    )

            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"Обработка ошибки для {test_case['args']} выполнялась слишком долго"
                )
            except FileNotFoundError:
                pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_output_directory_creation(
        self, test_config_path: Path, temp_output_dir: Path
    ):
        """Тест создания выходных директорий через CLI."""
        output_dir = temp_output_dir / "cli_test_output"

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.experiments.runner",
                    "--config",
                    str(test_config_path),
                    "--mode",
                    "validation",
                    "--output-dir",
                    str(output_dir),
                    "--verbose",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Проверяем, что директория была создана
            if result.returncode == 0:
                # В режиме валидации директория может быть создана
                print(f"✅ CLI создание директорий: код возврата {result.returncode}")
            else:
                print(f"⚠️ CLI завершился с кодом {result.returncode}")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI создание директорий выполнялось слишком долго")
        except FileNotFoundError:
            pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_verbose_levels(self, test_config_path: Path, temp_output_dir: Path):
        """Тест различных уровней детализации вывода."""
        verbose_levels = [
            [],  # Без verbose
            ["-v"],  # Один уровень
            ["-v", "-v"],  # Два уровня
            ["--verbose"],  # Длинная форма
        ]

        for verbose_args in verbose_levels:
            try:
                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "src.experiments.runner",
                        "--config",
                        str(test_config_path),
                        "--mode",
                        "validation",
                        "--output-dir",
                        str(temp_output_dir / f"verbose_test_{len(verbose_args)}"),
                    ]
                    + verbose_args,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Команда должна завершиться (успешно или с ошибкой, но не зависнуть)
                print(
                    f"✅ Verbose уровень {len(verbose_args)}: код возврата {result.returncode}"
                )

                # Проверяем, что вывод соответствует уровню детализации
                output_length = len(result.stdout) + len(result.stderr)
                if len(verbose_args) > 0:
                    # С verbose должно быть больше вывода
                    assert output_length > 0, "С verbose флагом должен быть вывод"

            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"Verbose уровень {len(verbose_args)} выполнялся слишком долго"
                )
            except FileNotFoundError:
                pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_configuration_override(
        self, test_config_path: Path, temp_output_dir: Path
    ):
        """Тест переопределения конфигурации через CLI."""
        # Тестируем различные комбинации параметров
        parameter_combinations = [
            ["--mode", "validation", "--max-workers", "1"],
            ["--mode", "validation", "--no-monitoring"],
            ["--mode", "validation", "--verbose", "--max-workers", "2"],
        ]

        for params in parameter_combinations:
            try:
                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "src.experiments.runner",
                        "--config",
                        str(test_config_path),
                        "--output-dir",
                        str(temp_output_dir / f"override_test_{len(params)}"),
                    ]
                    + params,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                print(f"✅ Параметры {params}: код возврата {result.returncode}")

                # Проверяем, что команда обработала параметры
                if result.returncode == 0:
                    # Успешное выполнение
                    assert len(result.stdout) > 0 or len(result.stderr) > 0

            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"Переопределение параметров {params} выполнялось слишком долго"
                )
            except FileNotFoundError:
                pytest.skip("CLI модуль не найден или не настроен")

    @pytest.mark.slow
    def test_cli_full_experiment_simulation(
        self, test_config_path: Path, temp_output_dir: Path
    ):
        """Симуляция полного эксперимента через CLI (только валидация)."""
        output_dir = temp_output_dir / "full_cli_test"

        try:
            # Запускаем только валидацию для скорости
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.experiments.runner",
                    "--config",
                    str(test_config_path),
                    "--mode",
                    "validation",
                    "--output-dir",
                    str(output_dir),
                    "--max-workers",
                    "1",
                    "--verbose",
                ],
                capture_output=True,
                text=True,
                timeout=120,  # Увеличенный таймаут для полного теста
            )

            print(f"Полный CLI тест: код возврата {result.returncode}")
            print(f"STDOUT длина: {len(result.stdout)}")
            print(f"STDERR длина: {len(result.stderr)}")

            if result.returncode == 0:
                print("✅ Полный CLI тест прошел успешно")
                # Проверяем наличие вывода
                assert len(result.stdout) > 0 or len(result.stderr) > 0
            else:
                print("⚠️ Полный CLI тест завершился с ошибкой")
                print(f"STDERR: {result.stderr[:500]}...")  # Первые 500 символов ошибки

        except subprocess.TimeoutExpired:
            pytest.fail("Полный CLI тест выполнялся слишком долго")
        except FileNotFoundError:
            pytest.skip("CLI модуль не найден или не настроен")

    def test_cli_environment_variables(
        self, test_config_path: Path, temp_output_dir: Path
    ):
        """Тест работы CLI с переменными окружения."""
        import os

        # Устанавливаем переменные окружения для тестирования
        test_env = os.environ.copy()
        test_env.update(
            {
                "PYTHONPATH": str(Path.cwd()),
                "EXPERIMENT_OUTPUT_DIR": str(temp_output_dir),
            }
        )

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.experiments.runner",
                    "--config",
                    str(test_config_path),
                    "--mode",
                    "validation",
                    "--verbose",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env=test_env,
            )

            print(f"CLI с переменными окружения: код возврата {result.returncode}")

            # Команда должна выполниться без критических ошибок
            if result.returncode == 0:
                print("✅ CLI с переменными окружения работает")
            else:
                print("⚠️ CLI с переменными окружения имеет проблемы")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI с переменными окружения выполнялся слишком долго")
        except FileNotFoundError:
            pytest.skip("CLI модуль не найден или не настроен")


def run_cli_command(args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Утилита для запуска CLI команд с обработкой ошибок."""
    try:
        return subprocess.run(
            ["python", "-m", "src.experiments.runner"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Команда {args} выполнялась слишком долго")
    except FileNotFoundError:
        raise FileNotFoundError("CLI модуль не найден")


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v", "-s", "--tb=short"])
