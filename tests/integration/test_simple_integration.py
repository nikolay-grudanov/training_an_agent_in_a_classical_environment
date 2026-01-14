"""Упрощенный интеграционный тест для демонстрации функциональности.

Этот тест проверяет основные компоненты системы без полного запуска обучения.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict

import pytest
import yaml

from src.experiments.experiment import Experiment, ExperimentStatus
from src.utils.config import ConfigLoader, RLConfig


class TestSimpleIntegration:
    """Упрощенные интеграционные тесты."""

    @pytest.fixture(scope="class")
    def test_output_dir(self) -> Path:
        """Создать временную директорию для тестов."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_simple_integration_"))
        yield temp_dir
        # Очистка после тестов
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def config_loader(self) -> ConfigLoader:
        """Загрузчик конфигураций."""
        return ConfigLoader()

    @pytest.fixture(scope="class")
    def test_configs(self, config_loader: ConfigLoader) -> Dict[str, RLConfig]:
        """Создать тестовые конфигурации PPO и A2C."""
        # Создаем baseline конфигурацию (PPO)
        baseline_config = config_loader._create_config_object({
            "algorithm": {
                "name": "PPO",
                "learning_rate": 0.0003,
                "n_steps": 512,
                "batch_size": 32,
                "gamma": 0.99
            },
            "environment": {
                "name": "LunarLander-v2"
            },
            "training": {
                "total_timesteps": 5000,
                "eval_freq": 1000
            },
            "seed": 42
        })

        # Создаем variant конфигурацию (A2C)
        variant_config = config_loader._create_config_object({
            "algorithm": {
                "name": "A2C",
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99
            },
            "environment": {
                "name": "LunarLander-v2"
            },
            "training": {
                "total_timesteps": 5000,
                "eval_freq": 1000
            },
            "seed": 42
        })

        return {
            "baseline": baseline_config,
            "variant": variant_config
        }

    def test_config_creation_and_validation(self, test_configs: Dict[str, RLConfig]):
        """Тест создания и валидации конфигураций."""
        baseline = test_configs["baseline"]
        variant = test_configs["variant"]

        # Проверяем, что конфигурации созданы правильно
        assert baseline.algorithm.name == "PPO"
        assert variant.algorithm.name == "A2C"
        assert baseline.environment.name == variant.environment.name == "LunarLander-v2"
        assert baseline.seed == variant.seed == 42

        # Проверяем различия между конфигурациями
        assert baseline.algorithm.name != variant.algorithm.name
        assert baseline.algorithm.learning_rate != variant.algorithm.learning_rate

        print(f"✅ Конфигурации созданы: {baseline.algorithm.name} vs {variant.algorithm.name}")

    def test_experiment_creation_and_lifecycle(
        self, 
        test_configs: Dict[str, RLConfig],
        test_output_dir: Path
    ):
        """Тест создания эксперимента и управления жизненным циклом."""
        # Создаем эксперимент
        experiment = Experiment(
            baseline_config=test_configs["baseline"],
            variant_config=test_configs["variant"],
            hypothesis="PPO покажет лучшую стабильность обучения чем A2C",
            output_dir=test_output_dir
        )

        # Проверяем начальное состояние
        assert experiment.status == ExperimentStatus.CREATED
        assert experiment.experiment_id is not None
        assert experiment.experiment_dir.exists()
        assert experiment.hypothesis == "PPO покажет лучшую стабильность обучения чем A2C"

        # Тестируем жизненный цикл
        experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.started_at is not None

        experiment.pause()
        assert experiment.status == ExperimentStatus.PAUSED
        assert experiment.paused_at is not None

        experiment.resume()
        assert experiment.status == ExperimentStatus.RUNNING

        experiment.stop(failed=False)
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.completed_at is not None

        print(f"✅ Эксперимент {experiment.experiment_id} прошел полный жизненный цикл")

    def test_experiment_results_simulation(
        self, 
        test_configs: Dict[str, RLConfig],
        test_output_dir: Path
    ):
        """Тест добавления результатов и сравнения."""
        # Создаем эксперимент
        experiment = Experiment(
            baseline_config=test_configs["baseline"],
            variant_config=test_configs["variant"],
            hypothesis="Тест симуляции результатов",
            output_dir=test_output_dir / "results_test"
        )

        # Симулируем результаты обучения
        baseline_results = {
            "mean_reward": 150.5,
            "final_reward": 180.2,
            "episode_length": 250,
            "convergence_timesteps": 3000,
            "training_time": 120.5,
            "success": True
        }

        variant_results = {
            "mean_reward": 140.8,
            "final_reward": 165.3,
            "episode_length": 280,
            "convergence_timesteps": 3500,
            "training_time": 110.2,
            "success": True
        }

        # Добавляем результаты
        experiment.add_result("baseline", baseline_results)
        experiment.add_result("variant", variant_results)

        # Проверяем, что результаты добавлены
        assert experiment.results["baseline"]["mean_reward"] == 150.5
        assert experiment.results["variant"]["mean_reward"] == 140.8
        assert experiment._baseline_completed
        assert experiment._variant_completed

        # Проверяем автоматическое сравнение
        comparison = experiment.compare_results()
        assert "performance_metrics" in comparison
        assert "mean_reward" in comparison["performance_metrics"]
        
        # Проверяем расчет улучшения
        improvement = comparison["performance_metrics"]["mean_reward"]["improvement"]
        expected_improvement = 140.8 - 150.5  # -9.7
        assert abs(improvement - expected_improvement) < 0.1

        print(f"✅ Результаты добавлены и сравнены: улучшение {improvement:.1f}")

    def test_experiment_serialization(
        self, 
        test_configs: Dict[str, RLConfig],
        test_output_dir: Path
    ):
        """Тест сериализации и десериализации эксперимента."""
        # Создаем эксперимент с результатами
        experiment = Experiment(
            baseline_config=test_configs["baseline"],
            variant_config=test_configs["variant"],
            hypothesis="Тест сериализации",
            output_dir=test_output_dir / "serialization_test"
        )

        # Добавляем некоторые результаты
        experiment.add_result("baseline", {"mean_reward": 100.0, "success": True})
        experiment.add_result("variant", {"mean_reward": 95.0, "success": True})

        # Сохраняем эксперимент
        saved_path = experiment.save(format_type="json")
        assert saved_path.exists()
        assert saved_path.suffix == ".json"

        # Проверяем содержимое файла
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert "experiment_id" in saved_data
        assert "hypothesis" in saved_data
        assert "baseline_config" in saved_data
        assert "variant_config" in saved_data
        assert saved_data["experiment_id"] == experiment.experiment_id

        # Загружаем эксперимент
        loaded_experiment = Experiment.load(saved_path)
        assert loaded_experiment.experiment_id == experiment.experiment_id
        assert loaded_experiment.hypothesis == experiment.hypothesis
        assert loaded_experiment.results["baseline"]["mean_reward"] == 100.0

        print(f"✅ Эксперимент сериализован и загружен: {saved_path}")

    def test_experiment_status_and_summary(
        self, 
        test_configs: Dict[str, RLConfig],
        test_output_dir: Path
    ):
        """Тест получения статуса и сводки эксперимента."""
        experiment = Experiment(
            baseline_config=test_configs["baseline"],
            variant_config=test_configs["variant"],
            hypothesis="Тест статуса и сводки",
            output_dir=test_output_dir / "status_test"
        )

        # Получаем статус
        status = experiment.get_status()
        required_fields = [
            "experiment_id", "status", "hypothesis", "created_at",
            "baseline_completed", "variant_completed", "results_available",
            "output_dir"
        ]

        for field in required_fields:
            assert field in status, f"Поле {field} должно присутствовать в статусе"

        assert status["experiment_id"] == experiment.experiment_id
        assert status["status"] == "created"
        assert not status["baseline_completed"]
        assert not status["variant_completed"]
        assert not status["results_available"]

        # Получаем сводку
        summary = experiment.get_summary()
        assert "experiment_id" in summary
        assert "configurations" in summary
        assert "baseline" in summary["configurations"]
        assert "variant" in summary["configurations"]

        # Проверяем конфигурации в сводке
        baseline_config = summary["configurations"]["baseline"]
        variant_config = summary["configurations"]["variant"]
        
        assert baseline_config["algorithm"] == "PPO"
        assert variant_config["algorithm"] == "A2C"
        assert baseline_config["environment"] == "LunarLander-v2"
        assert variant_config["environment"] == "LunarLander-v2"

        print("✅ Статус и сводка эксперимента работают корректно")

    def test_configuration_error_handling(self, config_loader: ConfigLoader):
        """Тест обработки ошибок в конфигурации."""
        # Создаем валидную конфигурацию
        valid_config = config_loader._create_config_object({
            "algorithm": {"name": "PPO", "learning_rate": 0.0003},
            "environment": {"name": "LunarLander-v2"},
            "training": {"total_timesteps": 1000},
            "seed": 42
        })

        # Тест с идентичными конфигурациями (должен вызвать ошибку)
        with pytest.raises(Exception):  # ConfigurationError
            Experiment(
                baseline_config=valid_config,
                variant_config=valid_config,  # Идентичная конфигурация
                hypothesis="Невалидная гипотеза"
            )

        # Тест с пустой гипотезой
        variant_config = config_loader._create_config_object({
            "algorithm": {"name": "A2C", "learning_rate": 0.0007},
            "environment": {"name": "LunarLander-v2"},
            "training": {"total_timesteps": 1000},
            "seed": 42
        })

        with pytest.raises(Exception):  # ConfigurationError
            Experiment(
                baseline_config=valid_config,
                variant_config=variant_config,
                hypothesis=""  # Пустая гипотеза
            )

        print("✅ Обработка ошибок конфигурации работает корректно")

    def test_yaml_config_loading(self):
        """Тест загрузки конфигурации из YAML файла."""
        config_path = Path("configs/test_ppo_vs_a2c.yaml")
        
        # Проверяем, что файл существует
        assert config_path.exists(), f"Конфигурационный файл не найден: {config_path}"

        # Загружаем и проверяем структуру
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Проверяем обязательные секции
        required_sections = ["experiment", "baseline", "variant", "evaluation", "comparison"]
        for section in required_sections:
            assert section in config_data, f"Секция {section} отсутствует в конфигурации"

        # Проверяем параметры эксперимента
        exp_config = config_data["experiment"]
        assert "name" in exp_config
        assert "description" in exp_config
        assert "hypothesis" in exp_config

        # Проверяем конфигурации алгоритмов
        baseline_config = config_data["baseline"]
        variant_config = config_data["variant"]
        
        assert baseline_config["algorithm"] != variant_config["algorithm"]
        assert baseline_config["environment"] == variant_config["environment"]
        assert baseline_config["training_steps"] <= 10000  # Для быстрого тестирования
        assert variant_config["training_steps"] <= 10000

        print(f"✅ YAML конфигурация загружена и валидна: {config_path}")

    @pytest.mark.integration
    def test_full_integration_pipeline(
        self,
        test_configs: Dict[str, RLConfig],
        test_output_dir: Path
    ):
        """Полный интеграционный тест пайплайна (без реального обучения)."""
        print("\n🚀 Запуск полного интеграционного теста пайплайна...")
        
        # 1. Создание эксперимента
        experiment = Experiment(
            baseline_config=test_configs["baseline"],
            variant_config=test_configs["variant"],
            hypothesis="Полный интеграционный тест PPO vs A2C (симуляция)",
            output_dir=test_output_dir / "full_pipeline"
        )
        print("✅ Эксперимент создан")

        # 2. Запуск эксперимента
        experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        print("✅ Эксперимент запущен")

        # 3. Симуляция результатов обучения
        import numpy as np
        
        # Симулируем более реалистичные результаты
        np.random.seed(42)
        
        # PPO обычно более стабильный
        baseline_rewards = np.random.normal(145, 15, 10)  # Среднее 145, стандартное отклонение 15
        baseline_results = {
            "mean_reward": float(np.mean(baseline_rewards)),
            "std_reward": float(np.std(baseline_rewards)),
            "final_reward": float(baseline_rewards[-1]),
            "max_reward": float(np.max(baseline_rewards)),
            "min_reward": float(np.min(baseline_rewards)),
            "episode_length": 180,
            "convergence_timesteps": 3200,
            "training_time": 150.0,
            "success": True
        }

        # A2C может быть менее стабильным
        variant_rewards = np.random.normal(138, 20, 10)  # Среднее 138, большее отклонение
        variant_results = {
            "mean_reward": float(np.mean(variant_rewards)),
            "std_reward": float(np.std(variant_rewards)),
            "final_reward": float(variant_rewards[-1]),
            "max_reward": float(np.max(variant_rewards)),
            "min_reward": float(np.min(variant_rewards)),
            "episode_length": 200,
            "convergence_timesteps": 3800,
            "training_time": 140.0,
            "success": True
        }

        experiment.add_result("baseline", baseline_results)
        experiment.add_result("variant", variant_results)
        print("✅ Результаты добавлены")

        # 4. Статистическое сравнение
        comparison_result = experiment.compare_results()
        assert "performance_metrics" in comparison_result
        assert "summary" in comparison_result
        print("✅ Статистическое сравнение выполнено")

        # 5. Анализ результатов
        mean_reward_comparison = comparison_result["performance_metrics"]["mean_reward"]
        improvement = mean_reward_comparison["improvement"]
        better_algorithm = mean_reward_comparison["better"]
        
        print(f"📊 PPO средняя награда: {baseline_results['mean_reward']:.2f}")
        print(f"📊 A2C средняя награда: {variant_results['mean_reward']:.2f}")
        print(f"📊 Улучшение: {improvement:+.2f}")
        print(f"📊 Лучший алгоритм: {better_algorithm}")

        # 6. Сохранение результатов
        saved_path = experiment.save()
        assert saved_path.exists()
        print(f"✅ Результаты сохранены: {saved_path}")

        # 7. Завершение эксперимента
        experiment.stop(failed=False)
        assert experiment.status == ExperimentStatus.COMPLETED
        print("✅ Эксперимент завершен успешно")

        # 8. Валидация финального состояния
        final_summary = experiment.get_summary()
        assert "results" in final_summary
        
        final_status = experiment.get_status()
        assert final_status["results_available"]
        assert final_status["baseline_completed"]
        assert final_status["variant_completed"]
        print("✅ Финальное состояние валидировано")

        # 9. Проверка выходных файлов
        assert experiment.experiment_dir.exists()
        assert any(experiment.experiment_dir.iterdir())  # Директория не пустая
        print("✅ Выходные файлы созданы")

        print("🎉 Полный интеграционный тест завершен успешно!")

        # Возвращаем результаты для дополнительных проверок
        return {
            "experiment": experiment,
            "baseline_results": baseline_results,
            "variant_results": variant_results,
            "comparison": comparison_result
        }


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v", "-s", "--tb=short"])