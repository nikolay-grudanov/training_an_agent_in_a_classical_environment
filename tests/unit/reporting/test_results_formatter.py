"""Тесты для форматировщика результатов экспериментов."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.evaluation.evaluator import EvaluationMetrics
from src.reporting.results_formatter import ReportConfig, ResultsFormatter


class TestReportConfig:
    """Тесты для конфигурации отчетов."""

    def test_default_config(self) -> None:
        """Тест создания конфигурации по умолчанию."""
        config = ReportConfig()

        assert config.language == "ru"
        assert config.theme == "default"
        assert config.include_plots is True
        assert config.include_statistics is True
        assert config.decimal_places == 4
        assert config.date_format == "%Y-%m-%d %H:%M:%S"

    def test_custom_config(self) -> None:
        """Тест создания пользовательской конфигурации."""
        config = ReportConfig(
            language="en",
            theme="dark",
            include_plots=False,
            decimal_places=2,
        )

        assert config.language == "en"
        assert config.theme == "dark"
        assert config.include_plots is False
        assert config.decimal_places == 2


class TestResultsFormatter:
    """Тесты для форматировщика результатов."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Временная директория для тестов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def formatter(self, temp_dir: Path) -> ResultsFormatter:
        """Форматировщик для тестов."""
        output_dir = temp_dir / "output"

        return ResultsFormatter(
            output_dir=output_dir,
        )

    @pytest.fixture
    def sample_evaluation_results(self) -> EvaluationMetrics:
        """Примерные результаты оценки."""
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

    @pytest.fixture
    def sample_quantitative_results(self) -> Mock:
        """Примерные количественные результаты."""
        mock_results = Mock()
        mock_results.rewards = [100.0, 150.0, 200.0, 175.0, 125.0]
        mock_results.episode_lengths = [180, 200, 220, 190, 210]
        return mock_results

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

    def test_jinja_filters(self, formatter: ResultsFormatter) -> None:
        """Тест пользовательских фильтров Jinja2."""
        # Тестируем фильтр format_number
        template_content = "{{ value | format_number(2) }}"
        template = formatter.jinja_env.from_string(template_content)
        result = template.render(value=3.14159)
        assert result == "3.14"

        # Тестируем фильтр format_percentage
        template_content = "{{ value | format_percentage }}"
        template = formatter.jinja_env.from_string(template_content)
        result = template.render(value=0.85)
        assert result == "85.00%"

        # Тестируем фильтр translate
        template_content = "{{ 'agent' | translate }}"
        template = formatter.jinja_env.from_string(template_content)
        result = template.render()
        assert result == "Агент"

    def test_single_agent_report_html(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
        sample_quantitative_results: Mock,
    ) -> None:
        """Тест создания HTML отчета по одному агенту."""
        output_path = formatter.format_single_agent_report(
            agent_name="PPO_Agent",
            evaluation_results=sample_evaluation_results,
            quantitative_results=sample_quantitative_results,
            output_format="html",
        )

        assert output_path.exists()
        assert output_path.suffix == ".html"

        # Проверяем содержимое файла
        content = output_path.read_text(encoding="utf-8")
        assert "PPO_Agent" in content
        assert "150.5" in content  # mean_reward
        assert "Отчет по результатам" in content

    def test_single_agent_report_markdown(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест создания Markdown отчета по одному агенту."""
        output_path = formatter.format_single_agent_report(
            agent_name="DQN_Agent",
            evaluation_results=sample_evaluation_results,
            output_format="markdown",
        )

        assert output_path.exists()
        assert output_path.suffix == ".markdown"

        content = output_path.read_text(encoding="utf-8")
        assert "# Отчет по результатам" in content
        assert "DQN_Agent" in content
        assert "150.5" in content

    def test_comparison_report(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест создания сравнительного отчета."""
        agents_results = {
            "PPO_Agent": sample_evaluation_results,
            "DQN_Agent": EvaluationMetrics(
                mean_reward=120.0,
                std_reward=20.0,
                min_reward=80.0,
                max_reward=160.0,
                mean_episode_length=180.0,
                std_episode_length=25.0,
                min_episode_length=150,
                max_episode_length=210,
                reward_ci_lower=110.0,
                reward_ci_upper=130.0,
                success_rate=0.75,
                total_episodes=100,
                total_timesteps=20000,
                evaluation_time=100.0,
            ),
        }

        output_path = formatter.format_comparison_report(
            agents_results=agents_results,
            output_format="html",
        )

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "PPO_Agent" in content
        assert "DQN_Agent" in content

    def test_experiment_report(self, formatter: ResultsFormatter) -> None:
        """Тест создания отчета по эксперименту."""
        experiment_data = {
            "name": "LunarLander_Experiment",
            "environment": "LunarLander-v2",
            "agents": ["PPO", "DQN", "A2C"],
            "duration": "2 hours",
            "results": {"best_agent": "PPO", "best_reward": 200.0},
        }

        output_path = formatter.format_experiment_report(
            experiment_name="LunarLander_Experiment",
            experiment_data=experiment_data,
            output_format="html",
        )

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "LunarLander_Experiment" in content

    def test_summary_report(self, formatter: ResultsFormatter) -> None:
        """Тест создания сводного отчета."""
        experiments_data = {
            "Experiment_1": {
                "agents": ["PPO", "DQN"],
                "best_reward": 180.0,
            },
            "Experiment_2": {
                "agents": ["A2C", "SAC"],
                "best_reward": 220.0,
            },
        }

        output_path = formatter.format_summary_report(
            experiments_data=experiments_data,
            output_format="html",
        )

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Experiment_1" in content
        assert "Experiment_2" in content

    def test_export_to_csv_single_agent(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест экспорта результатов одного агента в CSV."""
        output_path = formatter.export_to_csv(
            data=sample_evaluation_results,
            filename="single_agent_results",
        )

        assert output_path.exists()
        assert output_path.suffix == ".csv"

        # Проверяем содержимое CSV
        df = pd.read_csv(output_path)
        assert "mean_reward" in df.columns
        assert df.iloc[0]["mean_reward"] == 150.5

    def test_export_to_csv_multiple_agents(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест экспорта результатов нескольких агентов в CSV."""
        agents_results = {
            "PPO_Agent": sample_evaluation_results,
            "DQN_Agent": EvaluationMetrics(
                mean_reward=120.0,
                std_reward=20.0,
                min_reward=80.0,
                max_reward=160.0,
                mean_episode_length=180.0,
                std_episode_length=25.0,
                min_episode_length=150,
                max_episode_length=210,
                reward_ci_lower=110.0,
                reward_ci_upper=130.0,
                success_rate=0.75,
                total_episodes=100,
                total_timesteps=20000,
                evaluation_time=100.0,
            ),
        }

        output_path = formatter.export_to_csv(
            data=agents_results,
            filename="multiple_agents_results",
        )

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "agent" in df.columns
        assert "PPO_Agent" in df["agent"].values
        assert "DQN_Agent" in df["agent"].values

    def test_export_to_json(self, formatter: ResultsFormatter) -> None:
        """Тест экспорта данных в JSON."""
        test_data = {
            "experiment": "test_experiment",
            "agents": ["PPO", "DQN"],
            "results": {"best_reward": 200.0},
            "timestamp": datetime.now(),
        }

        output_path = formatter.export_to_json(
            data=test_data,
            filename="test_export",
        )

        assert output_path.exists()
        assert output_path.suffix == ".json"

        # Проверяем содержимое JSON
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["experiment"] == "test_experiment"
        assert loaded_data["agents"] == ["PPO", "DQN"]

    def test_calculate_statistics(
        self,
        formatter: ResultsFormatter,
        sample_quantitative_results: Mock,
    ) -> None:
        """Тест расчета статистики."""
        stats = formatter._calculate_statistics(sample_quantitative_results)

        assert "reward" in stats
        reward_stats = stats["reward"]
        assert "mean" in reward_stats
        assert "std" in reward_stats
        assert "min" in reward_stats
        assert "max" in reward_stats
        assert "median" in reward_stats

        # Проверяем значения
        assert reward_stats["mean"] == 150.0  # среднее [100, 150, 200, 175, 125]
        assert reward_stats["min"] == 100.0
        assert reward_stats["max"] == 200.0

    def test_prepare_comparison_data(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест подготовки данных для сравнения."""
        agents_results = {
            "PPO_Agent": sample_evaluation_results,
            "DQN_Agent": EvaluationMetrics(
                mean_reward=120.0,
                std_reward=20.0,
                min_reward=80.0,
                max_reward=160.0,
                mean_episode_length=180.0,
                std_episode_length=25.0,
                min_episode_length=150,
                max_episode_length=210,
                reward_ci_lower=110.0,
                reward_ci_upper=130.0,
                success_rate=0.75,
                total_episodes=100,
                total_timesteps=20000,
                evaluation_time=100.0,
            ),
        }

        comparison_data = formatter._prepare_comparison_data(agents_results)

        assert "agents" in comparison_data
        assert "metrics" in comparison_data
        assert len(comparison_data["agents"]) == 2
        assert "PPO_Agent" in comparison_data["agents"]
        assert "DQN_Agent" in comparison_data["agents"]

        assert "mean_reward" in comparison_data["metrics"]
        assert comparison_data["metrics"]["mean_reward"]["PPO_Agent"] == 150.5
        assert comparison_data["metrics"]["mean_reward"]["DQN_Agent"] == 120.0

    def test_evaluation_results_to_dict(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест преобразования результатов оценки в словарь."""
        result_dict = formatter._evaluation_results_to_dict(sample_evaluation_results)

        assert result_dict["mean_reward"] == 150.5
        assert result_dict["std_reward"] == 25.3
        assert result_dict["mean_episode_length"] == 200.0
        assert result_dict["std_episode_length"] == 30.5
        assert result_dict["success_rate"] == 0.85

    def test_language_switching(self, temp_dir: Path) -> None:
        """Тест переключения языка."""
        # Тест русского языка
        config_ru = ReportConfig(language="ru")
        formatter_ru = ResultsFormatter(
            templates_dir=temp_dir / "templates",
            output_dir=temp_dir / "output",
            config=config_ru,
        )

        template_content = "{{ 'agent' | translate }}"
        template = formatter_ru.jinja_env.from_string(template_content)
        result = template.render()
        assert result == "Агент"

        # Тест английского языка
        config_en = ReportConfig(language="en")
        formatter_en = ResultsFormatter(
            templates_dir=temp_dir / "templates",
            output_dir=temp_dir / "output",
            config=config_en,
        )

        template = formatter_en.jinja_env.from_string(template_content)
        result = template.render()
        assert result == "Agent"

    def test_custom_filename(
        self,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест использования пользовательского имени файла."""
        custom_filename = "custom_report_name"

        output_path = formatter.format_single_agent_report(
            agent_name="Test_Agent",
            evaluation_results=sample_evaluation_results,
            output_format="html",
            filename=custom_filename,
        )

        assert output_path.name == f"{custom_filename}.html"

    @patch("src.reporting.results_formatter.logger")
    def test_logging(
        self,
        mock_logger: Mock,
        formatter: ResultsFormatter,
        sample_evaluation_results: EvaluationMetrics,
    ) -> None:
        """Тест логирования."""
        formatter.format_single_agent_report(
            agent_name="Test_Agent",
            evaluation_results=sample_evaluation_results,
            output_format="html",
        )

        # Проверяем, что логирование происходило
        mock_logger.info.assert_called()

        # Проверяем конкретные сообщения
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Создание отчета по агенту" in msg for msg in log_calls)
        assert any("Отчет сохранен" in msg for msg in log_calls)
