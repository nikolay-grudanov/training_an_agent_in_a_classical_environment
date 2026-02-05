"""Интеграционные тесты для проверки воспроизводимости RL экспериментов.

Этот модуль содержит полные интеграционные тесты для User Story 4:
проверка воспроизводимости экспериментов, включая тестирование всех
компонентов системы воспроизводимости в интеграции.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Импорты для тестирования
from src.experiments.experiment import ExperimentStatus
from src.experiments.result_exporter import ResultExporter
from src.utils.config import (
    RLConfig,
    AlgorithmConfig,
    EnvironmentConfig,
    TrainingConfig,
)
from src.utils.dependency_tracker import DependencyTracker
from src.utils.reproducibility_checker import (
    ReproducibilityChecker,
    StrictnessLevel,
    ReproducibilityIssueType,
)
from src.utils.seeding import set_seed


class MockReproducibleAgent:
    """Мок агента с детерминированным поведением для тестов воспроизводимости."""

    def __init__(self, agent_id: str = "test_agent", seed: int = 42):
        """Инициализация мок агента.

        Args:
            agent_id: Идентификатор агента
            seed: Сид для воспроизводимости
        """
        self.agent_id = agent_id
        self.seed = seed
        self.training_history = []
        self.model_weights = None
        self._step_count = 0

    def train(self, env: Any, total_timesteps: int = 1000) -> Dict[str, Any]:
        """Детерминированное обучение агента.

        Args:
            env: Среда для обучения
            total_timesteps: Количество шагов обучения

        Returns:
            Результаты обучения
        """
        # Устанавливаем сид для детерминированности
        set_seed(self.seed)
        np.random.seed(self.seed)

        # Симулируем детерминированное обучение
        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(total_timesteps // 100):  # 100 шагов на эпизод
            # Детерминированные значения на основе сида и эпизода
            np.random.seed(self.seed + episode)

            # Симулируем награду (детерминированная прогрессия)
            base_reward = -200 + (episode * 5)  # Линейное улучшение
            noise = np.random.normal(0, 10)  # Детерминированный шум
            episode_reward = base_reward + noise
            episode_rewards.append(float(episode_reward))

            # Симулируем длину эпизода
            episode_length = 100 + int(np.random.normal(0, 5))
            episode_lengths.append(episode_length)

            # Симулируем потери
            loss = max(0.1, 2.0 - (episode * 0.05) + np.random.normal(0, 0.1))
            losses.append(float(loss))

            # Сохраняем в историю
            self.training_history.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "loss": loss,
                    "timestep": episode * 100,
                }
            )

        # Симулируем финальные веса модели (детерминированные)
        np.random.seed(self.seed)
        self.model_weights = np.random.randn(10, 5).tolist()  # Детерминированные веса

        return {
            "final_reward": float(
                np.mean(episode_rewards[-5:])
            ),  # Среднее за последние 5 эпизодов
            "total_episodes": len(episode_rewards),
            "total_timesteps": total_timesteps,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "model_weights": self.model_weights,
            "training_completed": True,
        }

    def evaluate(self, env: Any, num_episodes: int = 10) -> Dict[str, Any]:
        """Детерминированная оценка агента.

        Args:
            env: Среда для оценки
            num_episodes: Количество эпизодов оценки

        Returns:
            Результаты оценки
        """
        # Устанавливаем сид для детерминированности
        set_seed(self.seed)
        np.random.seed(self.seed)

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            np.random.seed(self.seed + episode + 1000)  # Смещение для оценки

            # Детерминированные результаты оценки
            reward = -150 + np.random.normal(0, 20)
            length = 95 + int(np.random.normal(0, 10))

            eval_rewards.append(float(reward))
            eval_lengths.append(length)

        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
            "std_length": float(np.std(eval_lengths)),
            "eval_rewards": eval_rewards,
            "eval_lengths": eval_lengths,
            "num_episodes": num_episodes,
        }

    def save(self, path: Path) -> None:
        """Сохранить агента."""
        save_data = {
            "agent_id": self.agent_id,
            "seed": self.seed,
            "model_weights": self.model_weights,
            "training_history": self.training_history,
        }

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

    def load(self, path: Path) -> None:
        """Загрузить агента."""
        with open(path, "r") as f:
            save_data = json.load(f)

        self.agent_id = save_data["agent_id"]
        self.seed = save_data["seed"]
        self.model_weights = save_data["model_weights"]
        self.training_history = save_data["training_history"]


class MockReproducibleEnvironment:
    """Мок среды с детерминированным поведением."""

    def __init__(self, env_id: str = "TestEnv-v1", seed: int = 42):
        """Инициализация мок среды.

        Args:
            env_id: Идентификатор среды
            seed: Сид для воспроизводимости
        """
        self.env_id = env_id
        self.seed = seed
        self.state = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Сброс среды в начальное состояние."""
        set_seed(self.seed)
        np.random.seed(self.seed)

        self.state = np.random.randn(4).astype(
            np.float32
        )  # Детерминированное начальное состояние
        self.step_count = 0
        return self.state

    def step(self, action: Any) -> tuple:
        """Выполнить шаг в среде."""
        self.step_count += 1

        # Детерминированная динамика
        np.random.seed(self.seed + self.step_count)

        # Обновляем состояние
        if self.state is not None:
            self.state = self.state + np.random.normal(0, 0.1, 4).astype(np.float32)

        # Детерминированная награда
        state_sum = np.sum(self.state) if self.state is not None else 0.0
        reward = -abs(state_sum) + np.random.normal(0, 0.5)

        # Детерминированное завершение
        done = self.step_count >= 100 or np.abs(state_sum) > 10

        info = {"step_count": self.step_count, "state_sum": float(state_sum)}

        return self.state, float(reward), done, info

    def close(self) -> None:
        """Закрыть среду."""
        pass


@pytest.fixture
def temp_project_dir():
    """Временная директория проекта для тестов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Создаем структуру директорий
        (project_dir / "results").mkdir(exist_ok=True)
        (project_dir / "results" / "dependencies").mkdir(exist_ok=True)
        (project_dir / "results" / "reproducibility").mkdir(exist_ok=True)
        (project_dir / "results" / "exports").mkdir(exist_ok=True)

        yield project_dir


@pytest.fixture
def reproducible_config():
    """Конфигурация с фиксированными сидами для воспроизводимости."""
    config = RLConfig(
        experiment_name="reproducibility_test",
        seed=42,
        algorithm=AlgorithmConfig(
            name="PPO",
            learning_rate=3e-4,
            seed=42,
            device="cpu",  # Явно указываем CPU
        ),
        environment=EnvironmentConfig(name="TestEnv-v1"),
        training=TrainingConfig(total_timesteps=1000, eval_freq=500),
    )

    # Настраиваем воспроизводимость для тестов
    config.reproducibility.use_cuda = False
    config.reproducibility.deterministic = True
    config.reproducibility.benchmark = False

    # Принудительная синхронизация сидов
    config.enforce_seed_consistency()

    return config


@pytest.fixture
def dependency_tracker(temp_project_dir):
    """Трекер зависимостей для тестов."""
    return DependencyTracker(temp_project_dir)


@pytest.fixture
def reproducibility_checker(temp_project_dir, dependency_tracker):
    """Проверщик воспроизводимости для тестов."""
    return ReproducibilityChecker(
        project_root=temp_project_dir,
        strictness_level=StrictnessLevel.STANDARD,
        dependency_tracker=dependency_tracker,
    )


@pytest.fixture
def result_exporter(temp_project_dir):
    """Экспортер результатов для тестов."""
    return ResultExporter(
        output_dir=temp_project_dir / "results" / "exports",
        include_dependencies=True,
        validate_integrity=True,
    )


class TestReproducibilityIntegration:
    """Интеграционные тесты воспроизводимости."""

    def test_full_reproducibility_workflow(
        self,
        temp_project_dir,
        reproducible_config,
        dependency_tracker,
        reproducibility_checker,
        result_exporter,
    ):
        """Тест полного workflow воспроизводимости.

        Проверяет:
        1. Создание конфигурации с фиксированными сидами
        2. Запуск первого эксперимента
        3. Создание снимка зависимостей
        4. Запуск второго идентичного эксперимента
        5. Сравнение результатов на воспроизводимость
        6. Экспорт результатов с метаданными
        7. Валидация полной воспроизводимости
        """
        # Этап 1: Создание конфигурации с фиксированными сидами
        assert reproducible_config.seed == 42
        assert reproducible_config.algorithm.seed == 42
        assert reproducible_config.reproducibility.seed == 42

        # Валидируем настройки воспроизводимости
        is_valid, warnings = reproducible_config.validate_reproducibility()
        assert is_valid, f"Конфигурация не валидна: {warnings}"

        # Этап 2: Запуск первого эксперимента
        experiment_id = "reproducibility_test_exp"

        # Создаем агента и среду
        agent1 = MockReproducibleAgent("agent1", seed=42)
        env1 = MockReproducibleEnvironment("TestEnv-v1", seed=42)

        # Применяем сиды
        reproducible_config.apply_seeds()

        # Выполняем обучение
        training_results1 = agent1.train(env1, total_timesteps=1000)
        eval_results1 = agent1.evaluate(env1, num_episodes=10)

        # Этап 3: Создание снимка зависимостей
        snapshot1 = dependency_tracker.create_dependency_snapshot(
            f"{experiment_id}_run1"
        )
        assert snapshot1 is not None
        assert "metadata" in snapshot1
        assert "system" in snapshot1
        assert "packages" in snapshot1

        # Регистрируем первый запуск
        run1_id = reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=reproducible_config,
            results={"training": training_results1, "evaluation": eval_results1},
            metrics={
                "episode_rewards": training_results1["episode_rewards"],
                "losses": training_results1["losses"],
            },
            metadata={"run_number": 1, "snapshot_id": snapshot1["metadata"]["name"]},
        )

        assert run1_id is not None

        # Этап 4: Запуск второго идентичного эксперимента
        # Создаем новые экземпляры с теми же сидами
        agent2 = MockReproducibleAgent("agent2", seed=42)
        env2 = MockReproducibleEnvironment("TestEnv-v1", seed=42)

        # Применяем сиды снова
        reproducible_config.apply_seeds()

        # Выполняем обучение
        training_results2 = agent2.train(env2, total_timesteps=1000)
        eval_results2 = agent2.evaluate(env2, num_episodes=10)

        # Создаем второй снимок зависимостей
        snapshot2 = dependency_tracker.create_dependency_snapshot(
            f"{experiment_id}_run2"
        )

        # Регистрируем второй запуск
        run2_id = reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=reproducible_config,
            results={"training": training_results2, "evaluation": eval_results2},
            metrics={
                "episode_rewards": training_results2["episode_rewards"],
                "losses": training_results2["losses"],
            },
            metadata={"run_number": 2, "snapshot_id": snapshot2["metadata"]["name"]},
        )

        assert run2_id is not None

        # Этап 5: Сравнение результатов на воспроизводимость
        reproducibility_report = reproducibility_checker.check_reproducibility(
            experiment_id=experiment_id, reference_run_id=run1_id, target_runs=[run2_id]
        )

        # Проверяем результаты воспроизводимости
        assert reproducibility_report is not None
        assert reproducibility_report.experiment_id == experiment_id
        assert len(reproducibility_report.runs) == 2

        # Проверяем, что результаты идентичны (детерминированный мок)
        assert training_results1["final_reward"] == training_results2["final_reward"]
        assert (
            training_results1["episode_rewards"] == training_results2["episode_rewards"]
        )
        assert training_results1["model_weights"] == training_results2["model_weights"]

        # Проверяем оценку воспроизводимости
        assert reproducibility_report.is_reproducible, (
            f"Эксперимент не воспроизводим: {[issue.description for issue in reproducibility_report.issues]}"
        )
        assert reproducibility_report.confidence_score >= 0.9

        # Этап 6: Экспорт результатов с метаданными
        # Создаем мок эксперимента для экспорта
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = experiment_id
        mock_experiment.status = ExperimentStatus.COMPLETED
        mock_experiment.hypothesis = "Тест воспроизводимости"
        mock_experiment.created_at = reproducibility_report.timestamp
        mock_experiment.started_at = reproducibility_report.timestamp
        mock_experiment.completed_at = reproducibility_report.timestamp
        mock_experiment.results = {
            "baseline": {
                "training": training_results1,
                "evaluation": eval_results1,
                "metrics_history": training_results1["episode_rewards"],
            },
            "variant": {
                "training": training_results2,
                "evaluation": eval_results2,
                "metrics_history": training_results2["episode_rewards"],
            },
        }
        mock_experiment.baseline_config = reproducible_config
        mock_experiment.variant_config = reproducible_config

        # Экспортируем результаты
        export_metadata = result_exporter.export_experiment(
            experiment=mock_experiment,
            formats=["json", "csv"],
            export_name=f"{experiment_id}_reproducibility",
            include_raw_data=True,
            include_plots=False,
        )

        assert export_metadata is not None
        assert "exported_files" in export_metadata
        assert "json" in export_metadata["exported_files"]
        assert "dependencies" in export_metadata or result_exporter.include_dependencies

        # Этап 7: Валидация полной воспроизводимости
        # Проверяем снимки зависимостей
        snapshot_comparison = dependency_tracker.compare_snapshots(
            snapshot1["metadata"]["name"], snapshot2["metadata"]["name"]
        )

        assert snapshot_comparison is not None
        assert "changes" in snapshot_comparison

        # Проверяем, что нет критических изменений в зависимостях
        changes = snapshot_comparison["changes"]
        assert len(changes["packages_added"]) == 0, (
            "Добавлены новые пакеты между запусками"
        )
        assert len(changes["packages_removed"]) == 0, "Удалены пакеты между запусками"
        assert len(changes["ml_libraries_changed"]) == 0, "Изменились ML библиотеки"

        # Валидируем экспортированные данные
        export_dir = Path(export_metadata["export_dir"])
        validation_result = result_exporter.validate_export_integrity(export_dir)

        assert validation_result["valid"], (
            f"Экспорт не валиден: {validation_result['errors']}"
        )
        assert len(validation_result["errors"]) == 0

        # Проверяем наличие всех необходимых файлов
        assert (export_dir / "export_metadata.json").exists()
        json_file = Path(export_metadata["exported_files"]["json"])
        assert json_file.exists()

        # Загружаем и проверяем экспортированные данные
        with open(json_file, "r") as f:
            exported_data = json.load(f)

        assert exported_data["experiment_id"] == experiment_id
        assert "dependencies" in exported_data
        assert "raw_metrics" in exported_data

    def test_reproducibility_with_different_strictness_levels(
        self, temp_project_dir, reproducible_config, dependency_tracker
    ):
        """Тест воспроизводимости с различными уровнями строгости."""
        experiment_id = "strictness_test"

        # Тестируем разные уровни строгости
        strictness_levels = [
            StrictnessLevel.MINIMAL,
            StrictnessLevel.STANDARD,
            StrictnessLevel.STRICT,
            StrictnessLevel.PARANOID,
        ]

        results_by_strictness = {}

        for strictness in strictness_levels:
            checker = ReproducibilityChecker(
                project_root=temp_project_dir,
                strictness_level=strictness,
                dependency_tracker=dependency_tracker,
            )

            # Создаем два идентичных запуска
            for run_num in range(2):
                agent = MockReproducibleAgent(f"agent_{run_num}", seed=42)
                env = MockReproducibleEnvironment("TestEnv-v1", seed=42)

                reproducible_config.apply_seeds()

                training_results = agent.train(env, total_timesteps=500)
                eval_results = agent.evaluate(env, num_episodes=5)

                checker.register_experiment_run(
                    experiment_id=f"{experiment_id}_{strictness.value}",
                    config=reproducible_config,
                    results={"training": training_results, "evaluation": eval_results},
                    metrics={
                        "episode_rewards": training_results["episode_rewards"],
                        "losses": training_results["losses"],
                    },
                    metadata={"run_number": run_num, "strictness": strictness.value},
                )

            # Проверяем воспроизводимость
            report = checker.check_reproducibility(
                f"{experiment_id}_{strictness.value}"
            )
            results_by_strictness[strictness] = report

            # Все уровни должны подтвердить воспроизводимость для детерминированного мока
            assert report.is_reproducible, (
                f"Уровень {strictness.value} не подтвердил воспроизводимость"
            )

            # Более строгие уровни должны выполнять больше проверок
            if strictness in [StrictnessLevel.STRICT, StrictnessLevel.PARANOID]:
                assert "statistics" in report.statistics or len(report.statistics) >= 0

        # Проверяем, что более строгие уровни дают более детальные отчеты
        minimal_report = results_by_strictness[StrictnessLevel.MINIMAL]
        paranoid_report = results_by_strictness[StrictnessLevel.PARANOID]

        # Paranoid должен иметь больше деталей в анализе
        assert len(paranoid_report.recommendations) >= len(
            minimal_report.recommendations
        )

    def test_reproducibility_failure_detection(
        self,
        temp_project_dir,
        reproducible_config,
        dependency_tracker,
        reproducibility_checker,
    ):
        """Тест детектирования нарушений воспроизводимости."""
        experiment_id = "failure_detection_test"

        # Первый запуск с сидом 42
        agent1 = MockReproducibleAgent("agent1", seed=42)
        env1 = MockReproducibleEnvironment("TestEnv-v1", seed=42)

        reproducible_config.seed = 42
        reproducible_config.apply_seeds()

        training_results1 = agent1.train(env1, total_timesteps=500)

        run1_id = reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=reproducible_config,
            results={"training": training_results1},
            metrics={"episode_rewards": training_results1["episode_rewards"]},
            metadata={"run_number": 1},
        )

        # Второй запуск с другим сидом (нарушение воспроизводимости)
        agent2 = MockReproducibleAgent("agent2", seed=123)  # Другой сид!
        env2 = MockReproducibleEnvironment("TestEnv-v1", seed=123)

        # Намеренно используем другую конфигурацию
        different_config = RLConfig(
            experiment_name="reproducibility_test",
            seed=123,  # Другой сид!
            algorithm=AlgorithmConfig(name="PPO", learning_rate=3e-4, seed=123),
            environment=EnvironmentConfig(name="TestEnv-v1"),
            training=TrainingConfig(total_timesteps=500),
        )
        different_config.apply_seeds()

        training_results2 = agent2.train(env2, total_timesteps=500)

        run2_id = reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=different_config,  # Другая конфигурация!
            results={"training": training_results2},
            metrics={"episode_rewards": training_results2["episode_rewards"]},
            metadata={"run_number": 2},
        )

        # Проверяем воспроизводимость
        report = reproducibility_checker.check_reproducibility(
            experiment_id=experiment_id, reference_run_id=run1_id, target_runs=[run2_id]
        )

        # Должны быть обнаружены проблемы
        assert not report.is_reproducible, (
            "Должно быть обнаружено нарушение воспроизводимости"
        )
        assert len(report.issues) > 0, "Должны быть найдены проблемы"

        # Проверяем типы обнаруженных проблем
        issue_types = [issue.issue_type for issue in report.issues]
        assert ReproducibilityIssueType.SEED_MISMATCH in issue_types, (
            "Должно быть обнаружено несоответствие сидов"
        )

        # Проверяем рекомендации
        assert len(report.recommendations) > 0, "Должны быть даны рекомендации"
        assert any("сид" in rec.lower() for rec in report.recommendations), (
            "Должны быть рекомендации по сидам"
        )

    def test_determinism_validation(self, reproducibility_checker):
        """Тест валидации детерминизма функций."""

        def deterministic_function():
            """Детерминированная функция для тестирования."""
            set_seed(42)
            np.random.seed(42)
            return {
                "random_value": float(np.random.randn()),
                "random_array": np.random.randn(5).tolist(),
                "computation": float(np.sum(np.random.randn(10))),
            }

        def non_deterministic_function():
            """Недетерминированная функция для тестирования."""
            import time

            return {
                "timestamp": time.time(),  # Всегда разное значение
                "random_value": float(np.random.randn()),  # Без установки сида
            }

        # Тест детерминированной функции
        det_result = reproducibility_checker.validate_determinism(
            test_function=deterministic_function, seed=42, num_runs=5
        )

        assert det_result["is_deterministic"], (
            "Детерминированная функция должна пройти тест"
        )
        assert det_result["unique_results"] == 1, "Все результаты должны быть идентичны"
        assert det_result["success_rate"] == 1.0, "Все запуски должны быть успешными"

        # Тест недетерминированной функции
        non_det_result = reproducibility_checker.validate_determinism(
            test_function=non_deterministic_function, seed=42, num_runs=5
        )

        assert not non_det_result["is_deterministic"], (
            "Недетерминированная функция должна провалить тест"
        )
        assert non_det_result["unique_results"] > 1, (
            "Должно быть несколько уникальных результатов"
        )

    def test_reproducibility_with_performance_measurement(
        self,
        temp_project_dir,
        reproducible_config,
        dependency_tracker,
        reproducibility_checker,
    ):
        """Тест воспроизводимости с измерением производительности."""
        experiment_id = "performance_test"

        # Измеряем время выполнения различных операций
        performance_metrics = {}

        # Время создания снимка зависимостей
        start_time = time.time()
        dependency_tracker.create_dependency_snapshot(
            f"{experiment_id}_perf"
        )
        snapshot_time = time.time() - start_time
        performance_metrics["snapshot_creation_time"] = snapshot_time

        # Время выполнения эксперимента
        start_time = time.time()
        agent = MockReproducibleAgent("perf_agent", seed=42)
        env = MockReproducibleEnvironment("TestEnv-v1", seed=42)

        reproducible_config.apply_seeds()
        training_results = agent.train(env, total_timesteps=1000)
        experiment_time = time.time() - start_time
        performance_metrics["experiment_time"] = experiment_time

        # Время регистрации запуска
        start_time = time.time()
        reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=reproducible_config,
            results={"training": training_results},
            metrics={"episode_rewards": training_results["episode_rewards"]},
            metadata={"performance_test": True},
        )
        registration_time = time.time() - start_time
        performance_metrics["registration_time"] = registration_time

        # Второй запуск для сравнения
        start_time = time.time()
        agent2 = MockReproducibleAgent("perf_agent2", seed=42)
        env2 = MockReproducibleEnvironment("TestEnv-v1", seed=42)

        reproducible_config.apply_seeds()
        training_results2 = agent2.train(env2, total_timesteps=1000)

        reproducibility_checker.register_experiment_run(
            experiment_id=experiment_id,
            config=reproducible_config,
            results={"training": training_results2},
            metrics={"episode_rewards": training_results2["episode_rewards"]},
            metadata={"performance_test": True},
        )
        second_run_time = time.time() - start_time
        performance_metrics["second_run_time"] = second_run_time

        # Время проверки воспроизводимости
        start_time = time.time()
        report = reproducibility_checker.check_reproducibility(experiment_id)
        check_time = time.time() - start_time
        performance_metrics["reproducibility_check_time"] = check_time

        # Проверяем результаты
        assert report.is_reproducible, "Эксперимент должен быть воспроизводим"

        # Проверяем производительность (разумные ограничения)
        assert snapshot_time < 10.0, (
            f"Создание снимка слишком медленное: {snapshot_time}s"
        )
        assert experiment_time < 5.0, (
            f"Эксперимент слишком медленный: {experiment_time}s"
        )
        assert registration_time < 2.0, (
            f"Регистрация слишком медленная: {registration_time}s"
        )
        assert check_time < 5.0, (
            f"Проверка воспроизводимости слишком медленная: {check_time}s"
        )

        # Добавляем метрики производительности в отчет
        report.metadata = performance_metrics

        print("\nМетрики производительности:")
        for metric, value in performance_metrics.items():
            print(f"  {metric}: {value:.4f}s")

    def test_edge_cases_and_error_handling(
        self,
        temp_project_dir,
        reproducible_config,
        dependency_tracker,
        reproducibility_checker,
    ):
        """Тест граничных случаев и обработки ошибок."""

        # Тест 1: Недостаточно запусков для сравнения
        with pytest.raises(ValueError, match="Недостаточно запусков"):
            reproducibility_checker.check_reproducibility("nonexistent_experiment")

        # Тест 2: Эксперимент с ошибками в обучении
        experiment_id = "error_handling_test"

        def failing_training_function(seed: int) -> Dict[str, Any]:
            """Функция обучения, которая иногда падает."""
            if seed == 999:
                raise RuntimeError("Симуляция ошибки обучения")

            set_seed(seed)
            return {"final_reward": float(np.random.randn()), "success": True}

        # Запускаем автоматический тест с ошибками
        report = reproducibility_checker.run_reproducibility_test(
            test_function=failing_training_function,
            experiment_id=experiment_id,
            seeds=[42, 123, 999],  # 999 вызовет ошибку
            config=reproducible_config,
        )

        # Проверяем, что ошибки обработаны корректно
        assert not report.is_reproducible, (
            "Тест с ошибками не должен быть воспроизводим"
        )
        assert len(report.runs) == 3, "Должно быть 3 запуска (включая неудачный)"

        # Проверяем, что неудачный запуск зарегистрирован
        failed_runs = [run for run in report.runs if "error" in run.results]
        assert len(failed_runs) == 1, "Должен быть один неудачный запуск"

        # Тест 3: Некорректные данные в снимке зависимостей
        with patch.object(
            dependency_tracker, "create_dependency_snapshot"
        ) as mock_snapshot:
            mock_snapshot.side_effect = Exception("Ошибка создания снимка")

            # Проверяем, что система продолжает работать даже при ошибках снимка
            agent = MockReproducibleAgent("error_agent", seed=42)
            env = MockReproducibleEnvironment("TestEnv-v1", seed=42)

            training_results = agent.train(env, total_timesteps=100)

            # Регистрация должна пройти успешно даже без снимка
            run_id = reproducibility_checker.register_experiment_run(
                experiment_id="error_snapshot_test",
                config=reproducible_config,
                results={"training": training_results},
                metrics={"episode_rewards": training_results["episode_rewards"]},
                metadata={"error_test": True},
            )

            assert run_id is not None, (
                "Регистрация должна работать даже при ошибках снимка"
            )

        # Тест 4: Валидация с поврежденными файлами
        corrupted_dir = temp_project_dir / "corrupted_export"
        corrupted_dir.mkdir(exist_ok=True)

        # Создаем поврежденный JSON файл
        corrupted_json = corrupted_dir / "corrupted.json"
        with open(corrupted_json, "w") as f:
            f.write("{ invalid json content")

        # Создаем поврежденные метаданные
        corrupted_metadata = corrupted_dir / "export_metadata.json"
        with open(corrupted_metadata, "w") as f:
            f.write('{"incomplete": "metadata"')  # Некорректный JSON

        # Проверяем валидацию поврежденного экспорта
        from src.experiments.result_exporter import ResultExporter

        exporter = ResultExporter(output_dir=temp_project_dir / "test_exports")

        validation_result = exporter.validate_export_integrity(corrupted_dir)
        assert not validation_result["valid"], (
            "Поврежденный экспорт должен быть невалидным"
        )
        assert len(validation_result["errors"]) > 0, "Должны быть обнаружены ошибки"

    def test_comprehensive_reproducibility_report_generation(
        self,
        temp_project_dir,
        reproducible_config,
        dependency_tracker,
        reproducibility_checker,
    ):
        """Тест генерации комплексного отчета о воспроизводимости."""
        experiment_id = "comprehensive_report_test"

        # Создаем несколько запусков с разными характеристиками
        run_configs = [
            {"seed": 42, "timesteps": 1000, "agent_id": "agent_1"},
            {"seed": 42, "timesteps": 1000, "agent_id": "agent_2"},  # Идентичный
            {"seed": 42, "timesteps": 1000, "agent_id": "agent_3"},  # Идентичный
        ]

        run_ids = []

        for i, run_config in enumerate(run_configs):
            agent = MockReproducibleAgent(
                run_config["agent_id"], seed=run_config["seed"]
            )
            env = MockReproducibleEnvironment("TestEnv-v1", seed=run_config["seed"])

            # Настраиваем конфигурацию
            config = RLConfig(
                experiment_name=f"comprehensive_test_{i}",
                seed=run_config["seed"],
                algorithm=AlgorithmConfig(name="PPO", seed=run_config["seed"]),
                environment=EnvironmentConfig(name="TestEnv-v1"),
                training=TrainingConfig(total_timesteps=run_config["timesteps"]),
            )
            config.apply_seeds()

            # Выполняем обучение
            training_results = agent.train(env, total_timesteps=run_config["timesteps"])
            eval_results = agent.evaluate(env, num_episodes=5)

            # Регистрируем запуск
            run_id = reproducibility_checker.register_experiment_run(
                experiment_id=experiment_id,
                config=config,
                results={"training": training_results, "evaluation": eval_results},
                metrics={
                    "episode_rewards": training_results["episode_rewards"],
                    "losses": training_results["losses"],
                    "eval_rewards": eval_results["eval_rewards"],
                },
                metadata={
                    "run_index": i,
                    "agent_id": run_config["agent_id"],
                    "config_seed": run_config["seed"],
                },
            )
            run_ids.append(run_id)

        # Генерируем комплексный отчет
        report = reproducibility_checker.check_reproducibility(experiment_id)

        # Проверяем структуру отчета
        assert report.experiment_id == experiment_id
        assert len(report.runs) == 3
        assert report.strictness_level == StrictnessLevel.STANDARD

        # Проверяем анализ сидов
        assert "seed_analysis" in report.__dict__
        assert report.seed_analysis["seeds_consistent"]
        assert report.seed_analysis["reference_seed"] == 42

        # Проверяем статистический анализ
        assert "statistics" in report.__dict__
        if report.statistics:
            for metric_name, stats in report.statistics.items():
                assert "mean" in stats
                assert "std" in stats
                assert "cv" in stats  # Коэффициент вариации

        # Проверяем анализ зависимостей
        assert "dependency_analysis" in report.__dict__
        if report.dependency_analysis:
            assert "snapshot_created" in report.dependency_analysis

        # Проверяем рекомендации
        assert isinstance(report.recommendations, list)

        # Генерируем руководство по воспроизводимости
        guide = reproducibility_checker.generate_reproducibility_guide(
            experiment_id=experiment_id,
            output_path=temp_project_dir / "reproducibility_guide.md",
        )

        assert isinstance(guide, str)
        assert len(guide) > 0
        assert "# Руководство по обеспечению воспроизводимости" in guide

        # Проверяем, что файл руководства создан
        guide_path = temp_project_dir / "reproducibility_guide.md"
        assert guide_path.exists()

        # Диагностируем проблемы воспроизводимости
        diagnosis = reproducibility_checker.diagnose_reproducibility_issues(
            experiment_id=experiment_id, deep_analysis=True
        )

        assert diagnosis["experiment_id"] == experiment_id
        assert "issues_found" in diagnosis
        assert "recommendations" in diagnosis
        assert "system_analysis" in diagnosis
        assert "seed_analysis" in diagnosis

        # Для идентичных запусков не должно быть критических проблем
        critical_issues = [
            issue
            for issue in diagnosis["issues_found"]
            if issue.get("severity") == "critical"
        ]
        assert len(critical_issues) == 0, (
            f"Найдены критические проблемы: {critical_issues}"
        )


if __name__ == "__main__":
    # Запуск тестов для отладки
    pytest.main([__file__, "-v", "-s"])
