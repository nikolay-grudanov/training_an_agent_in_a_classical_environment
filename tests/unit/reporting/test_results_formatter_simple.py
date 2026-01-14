"""Упрощенные тесты для форматировщика результатов экспериментов."""

import tempfile
from pathlib import Path

import pytest

from src.evaluation.evaluator import EvaluationMetrics
from src.reporting.results_formatter import ReportConfig, ResultsFormatter


class TestResultsFormatterSimple:
    """Упрощенные тесты для форматировщика результатов."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Временная директория для тестов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def formatter(self, temp_dir: Path) -> ResultsFormatter:
        """Форматировщик для тестов."""
        templates_dir = temp_dir / "templates"
        output_dir = temp_dir / "output"
        
        return ResultsFormatter(
            templates_dir=templates_dir,
            output_dir=output_dir,
        )
    
    @pytest.fixture
    def sample_evaluation_metrics(self) -> EvaluationMetrics:
        """Примерные метрики оценки."""
        return EvaluationMetrics(
            mean_reward=150.5,
            std_reward=25.3,
            min_reward=100.0,
            max_reward=200.0,
            mean_episode_length=200.0,
            std_episode_length=30.5,
            min_episode_length=150,
            max_episode_length=250,
            reward_ci_lower=140.0,
            reward_ci_upper=161.0,
            success_rate=0.85,
            total_episodes=100,
            total_timesteps=20000,
            evaluation_time=120.0,
        )
    
    def test_initialization(self, temp_dir: Path) -> None:
        """Тест инициализации форматировщика."""
        templates_dir = temp_dir / "templates"
        output_dir = temp_dir / "output"
        config = ReportConfig(language="en")
        
        formatter = ResultsFormatter(
            templates_dir=templates_dir,
            output_dir=output_dir,
            config=config,
        )
        
        assert formatter.templates_dir == templates_dir
        assert formatter.output_dir == output_dir
        assert formatter.config.language == "en"
        assert templates_dir.exists()
        assert output_dir.exists()
    
    def test_single_agent_report_html(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_metrics: EvaluationMetrics,
    ) -> None:
        """Тест создания HTML отчета по одному агенту."""
        output_path = formatter.format_single_agent_report(
            agent_name="PPO_Agent",
            evaluation_results=sample_evaluation_metrics,
            output_format="html",
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".html"
        
        # Проверяем содержимое файла
        content = output_path.read_text(encoding="utf-8")
        assert "PPO_Agent" in content
        assert "150.5" in content  # mean_reward
    
    def test_export_to_csv_single_agent(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_metrics: EvaluationMetrics,
    ) -> None:
        """Тест экспорта результатов одного агента в CSV."""
        output_path = formatter.export_to_csv(
            data=sample_evaluation_metrics,
            filename="single_agent_results",
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".csv"
    
    def test_export_to_json(self, formatter: ResultsFormatter) -> None:
        """Тест экспорта данных в JSON."""
        test_data = {
            "experiment": "test_experiment",
            "agents": ["PPO", "DQN"],
            "results": {"best_reward": 200.0},
        }
        
        output_path = formatter.export_to_json(
            data=test_data,
            filename="test_export",
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
    
    def test_evaluation_results_to_dict(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_metrics: EvaluationMetrics,
    ) -> None:
        """Тест преобразования результатов оценки в словарь."""
        result_dict = formatter._evaluation_results_to_dict(sample_evaluation_metrics)
        
        assert result_dict["mean_reward"] == 150.5
        assert result_dict["std_reward"] == 25.3
        assert result_dict["mean_episode_length"] == 200.0
        assert result_dict["std_episode_length"] == 30.5
        assert result_dict["success_rate"] == 0.85