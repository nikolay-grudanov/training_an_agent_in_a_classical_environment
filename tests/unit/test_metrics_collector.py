"""Тесты для MetricsCollector.

Этот модуль содержит модульные тесты для класса MetricsCollector,
проверяющие сбор метрик, расчет статистики и экспорт/импорт в JSON.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.training.metrics_collector import (
    AggregatedStatistics,
    MetricsCollector,
    MetricsCollectorConfig,
    TrainingMetrics,
)


class TestMetricsCollectorConfig:
    """Тесты для MetricsCollectorConfig."""

    def test_valid_config(self) -> None:
        """Тест создания валидной конфигурации."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
            recording_interval=100,
        )

        assert config.experiment_id == "exp_001"
        assert config.algorithm == "PPO"
        assert config.environment == "LunarLander-v3"
        assert config.seed == 42
        assert config.recording_interval == 100

    def test_default_recording_interval(self) -> None:
        """Тест значения recording_interval по умолчанию."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )

        assert config.recording_interval == 100

    def test_invalid_recording_interval(self) -> None:
        """Тест невалидного значения recording_interval."""
        with pytest.raises(ValueError, match="recording_interval должен быть > 0"):
            MetricsCollectorConfig(
                experiment_id="exp_001",
                algorithm="PPO",
                environment="LunarLander-v3",
                seed=42,
                recording_interval=0,
            )

    def test_invalid_seed(self) -> None:
        """Тест невалидного значения seed."""
        with pytest.raises(ValueError, match="seed должен быть >= 0"):
            MetricsCollectorConfig(
                experiment_id="exp_001",
                algorithm="PPO",
                environment="LunarLander-v3",
                seed=-1,
            )


class TestTrainingMetrics:
    """Тесты для TrainingMetrics."""

    def test_create_metrics(self) -> None:
        """Тест создания метрик."""
        metrics = TrainingMetrics(
            timestep=100,
            episode=1,
            reward=10.5,
            episode_length=200,
            loss=0.25,
        )

        assert metrics.timestep == 100
        assert metrics.episode == 1
        assert metrics.reward == 10.5
        assert metrics.episode_length == 200
        assert metrics.loss == 0.25
        assert metrics.timestamp is not None

    def test_metrics_without_loss(self) -> None:
        """Тест создания метрик без loss."""
        metrics = TrainingMetrics(
            timestep=100,
            episode=1,
            reward=10.5,
            episode_length=200,
        )

        assert metrics.loss is None

    def test_metrics_to_dict(self) -> None:
        """Тест преобразования метрик в словарь."""
        metrics = TrainingMetrics(
            timestep=100,
            episode=1,
            reward=10.5,
            episode_length=200,
            loss=0.25,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["timestep"] == 100
        assert metrics_dict["episode"] == 1
        assert metrics_dict["reward"] == 10.5
        assert metrics_dict["episode_length"] == 200
        assert metrics_dict["loss"] == 0.25
        assert "timestamp" in metrics_dict


class TestAggregatedStatistics:
    """Тесты для AggregatedStatistics."""

    def test_create_statistics(self) -> None:
        """Тест создания статистики."""
        stats = AggregatedStatistics(
            reward_mean=10.5,
            reward_std=2.3,
            reward_min=5.0,
            reward_max=15.0,
            episode_length_mean=200.0,
            total_timesteps=1000,
            total_episodes=5,
            loss_mean=0.25,
        )

        assert stats.reward_mean == 10.5
        assert stats.reward_std == 2.3
        assert stats.reward_min == 5.0
        assert stats.reward_max == 15.0
        assert stats.episode_length_mean == 200.0
        assert stats.total_timesteps == 1000
        assert stats.total_episodes == 5
        assert stats.loss_mean == 0.25

    def test_statistics_without_loss(self) -> None:
        """Тест создания статистики без loss."""
        stats = AggregatedStatistics(
            reward_mean=10.5,
            reward_std=2.3,
            reward_min=5.0,
            reward_max=15.0,
            episode_length_mean=200.0,
            total_timesteps=1000,
            total_episodes=5,
        )

        assert stats.loss_mean is None

    def test_statistics_to_dict(self) -> None:
        """Тест преобразования статистики в словарь."""
        stats = AggregatedStatistics(
            reward_mean=10.5,
            reward_std=2.3,
            reward_min=5.0,
            reward_max=15.0,
            episode_length_mean=200.0,
            total_timesteps=1000,
            total_episodes=5,
        )

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["reward_mean"] == 10.5
        assert stats_dict["reward_std"] == 2.3
        assert stats_dict["reward_min"] == 5.0
        assert stats_dict["reward_max"] == 15.0
        assert stats_dict["episode_length_mean"] == 200.0
        assert stats_dict["total_timesteps"] == 1000
        assert stats_dict["total_episodes"] == 5


class TestMetricsCollector:
    """Тесты для MetricsCollector."""

    def test_initialization(self) -> None:
        """Тест инициализации коллектора."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        assert collector.config == config
        assert collector.get_metrics_count() == 0

    def test_record_metrics(self) -> None:
        """Тест записи метрик."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
            recording_interval=100,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)

        assert collector.get_metrics_count() == 2

    def test_record_with_interval(self) -> None:
        """Тест записи с учетом интервала."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
            recording_interval=100,
        )
        collector = MetricsCollector(config)

        # Первая запись должна пройти
        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        assert collector.get_metrics_count() == 1

        # Вторая запись должна пройти (интервал 100)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)
        assert collector.get_metrics_count() == 2

        # Третья запись не должна пройти (интервал < 100)
        collector.record(timestep=250, episode=3, reward=12.0, episode_length=190)
        assert collector.get_metrics_count() == 2

        # Четвертая запись должна пройти (интервал >= 100)
        collector.record(timestep=300, episode=4, reward=18.5, episode_length=170)
        assert collector.get_metrics_count() == 3

    def test_record_with_loss(self) -> None:
        """Тест записи метрик с loss."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(
            timestep=100, episode=1, reward=10.5, episode_length=200, loss=0.25
        )

        metrics = collector.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].loss == 0.25

    def test_record_invalid_timestep(self) -> None:
        """Тест записи с невалидным timestep."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        with pytest.raises(ValueError, match="timestep должен быть >= 0"):
            collector.record(timestep=-1, episode=1, reward=10.5, episode_length=200)

    def test_record_invalid_episode(self) -> None:
        """Тест записи с невалидным episode."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        with pytest.raises(ValueError, match="episode должен быть >= 0"):
            collector.record(timestep=100, episode=-1, reward=10.5, episode_length=200)

    def test_calculate_statistics(self) -> None:
        """Тест расчета статистики."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)
        collector.record(timestep=300, episode=3, reward=12.8, episode_length=190)

        stats = collector.calculate_statistics()

        assert stats.reward_mean == pytest.approx(12.83, rel=1e-2)
        assert stats.reward_min == 10.5
        assert stats.reward_max == 15.2
        assert stats.episode_length_mean == pytest.approx(190.0, rel=1e-2)
        assert stats.total_timesteps == 300
        assert stats.total_episodes == 3

    def test_calculate_statistics_no_metrics(self) -> None:
        """Тест расчета статистики без метрик."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        with pytest.raises(RuntimeError, match="Нет собранных метрик"):
            collector.calculate_statistics()

    def test_calculate_statistics_with_loss(self) -> None:
        """Тест расчета статистики с loss."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(
            timestep=100, episode=1, reward=10.5, episode_length=200, loss=0.25
        )
        collector.record(
            timestep=200, episode=2, reward=15.2, episode_length=180, loss=0.30
        )
        collector.record(
            timestep=300, episode=3, reward=12.8, episode_length=190, loss=0.20
        )

        stats = collector.calculate_statistics()

        assert stats.loss_mean == pytest.approx(0.25, rel=1e-2)

    def test_get_metrics(self) -> None:
        """Тест получения метрик."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)

        metrics = collector.get_metrics()

        assert len(metrics) == 2
        assert metrics[0].timestep == 100
        assert metrics[1].timestep == 200

    def test_get_metrics_returns_copy(self) -> None:
        """Тест что get_metrics возвращает копию."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)

        metrics = collector.get_metrics()
        metrics.clear()

        # Оригинальный список не должен измениться
        assert collector.get_metrics_count() == 1

    def test_clear_metrics(self) -> None:
        """Тест очистки метрик."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)

        assert collector.get_metrics_count() == 2

        collector.clear_metrics()

        assert collector.get_metrics_count() == 0

    def test_export_to_json(self) -> None:
        """Тест экспорта в JSON."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            result_path = collector.export_to_json(filepath)

            assert result_path == filepath
            assert Path(filepath).exists()

            # Проверяем содержимое файла
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "metrics" in data
            assert "statistics" in data
            assert "export_timestamp" in data

            assert data["metadata"]["experiment_id"] == "exp_001"
            assert data["metadata"]["algorithm"] == "PPO"
            assert data["metadata"]["environment"] == "LunarLander-v3"
            assert data["metadata"]["seed"] == 42

            assert len(data["metrics"]) == 2

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_to_json_no_metrics(self) -> None:
        """Тест экспорта без метрик."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(RuntimeError, match="Нет собранных метрик"):
                collector.export_to_json(filepath)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_from_json(self) -> None:
        """Тест загрузки из JSON."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Экспортируем
            collector.export_to_json(filepath)

            # Загружаем
            loaded_collector = MetricsCollector.load_from_json(filepath)

            assert loaded_collector.config.experiment_id == "exp_001"
            assert loaded_collector.config.algorithm == "PPO"
            assert loaded_collector.config.environment == "LunarLander-v3"
            assert loaded_collector.config.seed == 42
            assert loaded_collector.get_metrics_count() == 2

            metrics = loaded_collector.get_metrics()
            assert metrics[0].timestep == 100
            assert metrics[0].reward == 10.5
            assert metrics[1].timestep == 200
            assert metrics[1].reward == 15.2

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_from_json_file_not_found(self) -> None:
        """Тест загрузки из несуществующего файла."""
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            MetricsCollector.load_from_json("/nonexistent/file.json")

    def test_load_from_json_invalid_format(self) -> None:
        """Тест загрузки из файла с невалидным JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
            f.write("invalid json")

        try:
            with pytest.raises(ValueError, match="Некорректный JSON формат"):
                MetricsCollector.load_from_json(filepath)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_from_json_missing_fields(self) -> None:
        """Тест загрузки из файла с отсутствующими полями."""
        data = {"metadata": {"experiment_id": "exp_001"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
            json.dump(data, f)

        try:
            with pytest.raises(ValueError, match="Отсутствует обязательное поле"):
                MetricsCollector.load_from_json(filepath)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_len_operator(self) -> None:
        """Тест оператора len."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        assert len(collector) == 0

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        assert len(collector) == 1

        collector.record(timestep=200, episode=2, reward=15.2, episode_length=180)
        assert len(collector) == 2

    def test_repr(self) -> None:
        """Тест строкового представления."""
        config = MetricsCollectorConfig(
            experiment_id="exp_001",
            algorithm="PPO",
            environment="LunarLander-v3",
            seed=42,
        )
        collector = MetricsCollector(config)

        repr_str = repr(collector)

        assert "MetricsCollector" in repr_str
        assert "exp_001" in repr_str
        assert "PPO" in repr_str
        assert "LunarLander-v3" in repr_str
        assert "metrics_count=0" in repr_str

        collector.record(timestep=100, episode=1, reward=10.5, episode_length=200)
        repr_str = repr(collector)
        assert "metrics_count=1" in repr_str
