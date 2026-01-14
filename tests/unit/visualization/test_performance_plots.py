"""Тесты для модуля performance_plots.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import MetricsTracker
from src.visualization.performance_plots import (
    DataLoader,
    InteractivePlotter,
    PerformancePlotter,
    PlotStyle,
    create_performance_report,
    export_plots_to_formats,
    quick_comparison_plot,
    quick_reward_plot,
)


class TestPlotStyle:
    """Тесты для класса PlotStyle."""
    
    def test_setup_matplotlib_style(self):
        """Тест настройки стиля matplotlib."""
        # Тестируем с валидным стилем
        PlotStyle.setup_matplotlib_style('default')
        
        # Тестируем с невалидным стилем (должен использовать fallback)
        PlotStyle.setup_matplotlib_style('nonexistent_style')
    
    def test_color_palettes(self):
        """Тест цветовых палитр."""
        assert len(PlotStyle.COLORS_PRIMARY) == 5
        assert len(PlotStyle.COLORS_SECONDARY) == 5
        assert len(PlotStyle.COLORS_GRADIENT) == 4
        
        # Проверяем, что цвета в правильном формате
        for color in PlotStyle.COLORS_PRIMARY:
            assert isinstance(color, str)
            assert color.startswith('#')
    
    def test_figsize_constants(self):
        """Тест констант размеров фигур."""
        assert PlotStyle.FIGSIZE_SMALL == (8, 6)
        assert PlotStyle.FIGSIZE_MEDIUM == (12, 8)
        assert PlotStyle.FIGSIZE_LARGE == (16, 10)
        assert PlotStyle.FIGSIZE_WIDE == (16, 6)


class TestDataLoader:
    """Тесты для класса DataLoader."""
    
    def test_load_from_csv(self):
        """Тест загрузки данных из CSV."""
        # Создаем временный CSV файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestep,episode,value\n")
            f.write("1,1,10.5\n")
            f.write("2,1,12.3\n")
            f.write("3,2,15.7\n")
            csv_path = f.name
        
        try:
            df = DataLoader.load_from_csv(csv_path)
            assert len(df) == 3
            assert 'timestep' in df.columns
            assert 'episode' in df.columns
            assert 'value' in df.columns
            assert df['value'].iloc[0] == 10.5
        finally:
            Path(csv_path).unlink()
    
    def test_load_from_csv_file_not_found(self):
        """Тест загрузки несуществующего CSV файла."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_from_csv("nonexistent.csv")
    
    def test_load_from_json(self):
        """Тест загрузки данных из JSON."""
        test_data = {
            "experiment_id": "test_exp",
            "metrics": {
                "episode_reward": [
                    {"timestep": 1, "value": 10.5},
                    {"timestep": 2, "value": 12.3}
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name
        
        try:
            data = DataLoader.load_from_json(json_path)
            assert data["experiment_id"] == "test_exp"
            assert "metrics" in data
            assert "episode_reward" in data["metrics"]
        finally:
            Path(json_path).unlink()
    
    def test_load_from_json_invalid_format(self):
        """Тест загрузки невалидного JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            json_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Ошибка при парсинге JSON файла"):
                DataLoader.load_from_json(json_path)
        finally:
            Path(json_path).unlink()
    
    def test_load_from_jsonl(self):
        """Тест загрузки данных из JSONL."""
        test_lines = [
            {"timestep": 1, "episode": 1, "reward": 10.5},
            {"timestep": 2, "episode": 1, "reward": 12.3},
            {"timestep": 3, "episode": 2, "reward": 15.7}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for line in test_lines:
                f.write(json.dumps(line) + '\n')
            jsonl_path = f.name
        
        try:
            data = DataLoader.load_from_jsonl(jsonl_path)
            assert len(data) == 3
            assert data[0]["timestep"] == 1
            assert data[0]["reward"] == 10.5
        finally:
            Path(jsonl_path).unlink()
    
    def test_load_from_metrics_tracker(self):
        """Тест загрузки данных из MetricsTracker."""
        tracker = MetricsTracker("test_exp")
        
        # Добавляем тестовые метрики
        tracker.add_metric("episode_reward", 10.5, timestep=1, episode=1)
        tracker.add_metric("episode_reward", 12.3, timestep=2, episode=1)
        tracker.add_metric("episode_length", 100, timestep=1, episode=1)
        
        data = DataLoader.load_from_metrics_tracker(tracker)
        
        assert "episode_reward" in data
        assert "episode_length" in data
        assert len(data["episode_reward"]) == 2
        assert len(data["episode_length"]) == 1
        
        reward_df = data["episode_reward"]
        assert "timestep" in reward_df.columns
        assert "value" in reward_df.columns
        assert reward_df["value"].iloc[0] == 10.5
    
    @patch('src.visualization.performance_plots.Path.glob')
    def test_convert_sb3_logs_no_files(self, mock_glob):
        """Тест конвертации логов SB3 когда файлы не найдены."""
        mock_glob.return_value = []
        
        with pytest.raises(FileNotFoundError, match="Файлы логов SB3 не найдены"):
            DataLoader.convert_sb3_logs("/fake/path")


class TestPerformancePlotter:
    """Тесты для класса PerformancePlotter."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        return pd.DataFrame({
            'timestep': range(1, 101),
            'episode': [i // 10 + 1 for i in range(100)],
            'value': np.random.normal(10, 2, 100)
        })
    
    @pytest.fixture
    def plotter(self):
        """Создает экземпляр PerformancePlotter."""
        return PerformancePlotter()
    
    def test_init(self, plotter):
        """Тест инициализации плоттера."""
        assert plotter.style == 'seaborn-v0_8'
        assert plotter.color_palette == 'husl'
        assert plotter.figsize == PlotStyle.FIGSIZE_MEDIUM
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_reward_curve(self, mock_close, mock_savefig, plotter, sample_data):
        """Тест построения графика вознаграждения."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_reward.png"
            
            result_path = plotter.plot_reward_curve(
                sample_data,
                y_col='value',
                save_path=save_path,
                smooth_window=10
            )
            
            assert result_path == str(save_path)
            mock_savefig.assert_called_once()
            # Не проверяем количество вызовов close, так как может быть вызван несколько раз
    
    def test_plot_reward_curve_empty_data(self, plotter):
        """Тест построения графика с пустыми данными."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Данные для построения графика пустые"):
            plotter.plot_reward_curve(empty_df)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_episode_lengths(self, mock_close, mock_savefig, plotter, sample_data):
        """Тест построения графика длины эпизодов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_lengths.png"
            
            result_path = plotter.plot_episode_lengths(
                sample_data,
                x_col='episode',
                y_col='value',
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_multiple_agents(self, mock_close, mock_savefig, plotter, sample_data):
        """Тест сравнительного графика агентов."""
        # Создаем данные для нескольких агентов
        agents_data = {
            "agent1": sample_data.copy(),
            "agent2": sample_data.copy()
        }
        agents_data["agent2"]["value"] = agents_data["agent2"]["value"] + 5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_comparison.png"
            
            result_path = plotter.plot_multiple_agents(
                agents_data,
                metric='value',
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_dashboard(self, mock_close, mock_savefig, plotter):
        """Тест создания дашборда."""
        # Создаем данные в формате MetricsTracker
        data = {
            "episode_reward": pd.DataFrame({
                'timestep': range(1, 51),
                'value': np.random.normal(10, 2, 50)
            }),
            "episode_length": pd.DataFrame({
                'timestep': range(1, 51),
                'value': np.random.normal(100, 10, 50)
            })
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_dashboard.png"
            
            result_path = plotter.create_dashboard(
                data,
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_create_dashboard_no_metrics(self, plotter):
        """Тест создания дашборда без метрик."""
        empty_data = {}
        
        with pytest.raises(ValueError, match="Нет доступных метрик для отображения"):
            plotter.create_dashboard(empty_data)


class TestInteractivePlotter:
    """Тесты для класса InteractivePlotter."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        return pd.DataFrame({
            'timestep': range(1, 101),
            'value': np.random.normal(10, 2, 100)
        })
    
    @pytest.fixture
    def interactive_plotter(self):
        """Создает экземпляр InteractivePlotter."""
        return InteractivePlotter()
    
    def test_init(self, interactive_plotter):
        """Тест инициализации интерактивного плоттера."""
        assert interactive_plotter.theme == 'plotly_white'
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_plot_interactive_reward_curve(self, mock_write_html, interactive_plotter, sample_data):
        """Тест создания интерактивного графика вознаграждения."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_interactive.html"
            
            result_path = interactive_plotter.plot_interactive_reward_curve(
                sample_data,
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_write_html.assert_called_once_with(str(save_path))
    
    def test_plot_interactive_reward_curve_empty_data(self, interactive_plotter):
        """Тест создания интерактивного графика с пустыми данными."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Данные для построения графика пустые"):
            interactive_plotter.plot_interactive_reward_curve(empty_df)
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_plot_interactive_comparison(self, mock_write_html, interactive_plotter, sample_data):
        """Тест создания интерактивного сравнительного графика."""
        agents_data = {
            "agent1": sample_data.copy(),
            "agent2": sample_data.copy()
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_comparison.html"
            
            result_path = interactive_plotter.plot_interactive_comparison(
                agents_data,
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_write_html.assert_called_once_with(str(save_path))
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_dashboard(self, mock_write_html, interactive_plotter):
        """Тест создания интерактивного дашборда."""
        data = {
            "episode_reward": pd.DataFrame({
                'timestep': range(1, 51),
                'value': np.random.normal(10, 2, 50)
            }),
            "episode_length": pd.DataFrame({
                'timestep': range(1, 51),
                'value': np.random.normal(100, 10, 50)
            })
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_dashboard.html"
            
            result_path = interactive_plotter.create_interactive_dashboard(
                data,
                save_path=save_path
            )
            
            assert result_path == str(save_path)
            mock_write_html.assert_called_once_with(str(save_path))


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""
    
    @patch('src.visualization.performance_plots.PerformancePlotter.plot_reward_curve')
    def test_export_plots_to_formats(self, mock_plot_function):
        """Тест экспорта графиков в разные форматы."""
        mock_plot_function.return_value = "/fake/path/plot.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = export_plots_to_formats(
                mock_plot_function,
                temp_dir,
                formats=['png', 'pdf'],
                title='Test Plot'
            )
            
            assert 'png' in saved_files
            assert 'pdf' in saved_files
            assert mock_plot_function.call_count == 2
    
    @patch('src.visualization.performance_plots.PerformancePlotter')
    @patch('src.visualization.performance_plots.InteractivePlotter')
    def test_create_performance_report(self, mock_interactive, mock_static):
        """Тест создания отчета о производительности."""
        # Настраиваем моки
        mock_static_instance = Mock()
        mock_interactive_instance = Mock()
        mock_static.return_value = mock_static_instance
        mock_interactive.return_value = mock_interactive_instance
        
        # Тестовые данные
        data = {
            "episode_reward": pd.DataFrame({
                'timestep': range(1, 51),
                'value': np.random.normal(10, 2, 50)
            })
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = create_performance_report(
                data,
                output_dir=temp_dir
            )
            
            assert result_path == temp_dir
            assert Path(temp_dir).exists()
            
            # Проверяем, что методы плоттеров были вызваны
            mock_static_instance.plot_reward_curve.assert_called()
            mock_interactive_instance.plot_interactive_reward_curve.assert_called()
    
    @patch('src.visualization.performance_plots.DataLoader.load_from_csv')
    @patch('src.visualization.performance_plots.PerformancePlotter.plot_reward_curve')
    def test_quick_reward_plot_from_csv(self, mock_plot, mock_load):
        """Тест быстрого создания графика из CSV."""
        mock_load.return_value = pd.DataFrame({
            'timestep': range(1, 11),
            'value': range(1, 11)
        })
        mock_plot.return_value = "/fake/path/plot.png"
        
        result = quick_reward_plot("test.csv")
        
        assert result == "/fake/path/plot.png"
        mock_load.assert_called_once_with(Path("test.csv"))
        mock_plot.assert_called_once()
    
    @patch('src.visualization.performance_plots.DataLoader.load_from_csv')
    @patch('src.visualization.performance_plots.PerformancePlotter.plot_multiple_agents')
    def test_quick_comparison_plot(self, mock_plot, mock_load):
        """Тест быстрого создания сравнительного графика."""
        mock_load.return_value = pd.DataFrame({
            'timestep': range(1, 11),
            'value': range(1, 11)
        })
        mock_plot.return_value = "/fake/path/comparison.png"
        
        agents_data = {
            "agent1": "agent1.csv",
            "agent2": "agent2.csv"
        }
        
        result = quick_comparison_plot(agents_data)
        
        assert result == "/fake/path/comparison.png"
        assert mock_load.call_count == 2
        mock_plot.assert_called_once()
    
    def test_quick_reward_plot_unsupported_format(self):
        """Тест быстрого создания графика с неподдерживаемым форматом."""
        with pytest.raises(ValueError, match="Неподдерживаемый формат файла"):
            quick_reward_plot("test.txt")


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_workflow_with_metrics_tracker(self):
        """Тест полного рабочего процесса с MetricsTracker."""
        # Создаем трекер и добавляем данные
        tracker = MetricsTracker("integration_test")
        
        for i in range(100):
            tracker.add_metric("episode_reward", np.random.normal(10, 2), timestep=i, episode=i//10)
            tracker.add_metric("episode_length", np.random.randint(50, 150), timestep=i, episode=i//10)
        
        # Загружаем данные
        data = DataLoader.load_from_metrics_tracker(tracker)
        
        # Создаем плоттер и строим графики
        plotter = PerformancePlotter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # График вознаграждения - используем правильную структуру данных
            reward_path = plotter.plot_reward_curve(
                data['episode_reward'],  # Передаем конкретный DataFrame
                y_col='value',
                save_path=Path(temp_dir) / "reward.png"
            )
            assert Path(reward_path).exists()
            
            # Дашборд
            dashboard_path = plotter.create_dashboard(
                data,
                save_path=Path(temp_dir) / "dashboard.png"
            )
            assert Path(dashboard_path).exists()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_performance_report_creation(self, mock_write_html, mock_close, mock_savefig):
        """Тест создания полного отчета о производительности."""
        # Создаем тестовые данные
        tracker = MetricsTracker("report_test")
        
        for i in range(50):
            tracker.add_metric("episode_reward", np.random.normal(10, 2), timestep=i)
            tracker.add_metric("episode_length", np.random.randint(50, 150), timestep=i)
            tracker.add_metric("training_loss", np.random.exponential(0.1), timestep=i)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = create_performance_report(
                tracker,
                output_dir=temp_dir,
                include_interactive=True,
                include_static=True
            )
            
            assert report_path == temp_dir
            
            # Проверяем, что директории созданы
            static_dir = Path(temp_dir) / 'static'
            interactive_dir = Path(temp_dir) / 'interactive'
            
            assert static_dir.exists()
            assert interactive_dir.exists()
            
            # Проверяем, что функции построения графиков были вызваны
            assert mock_savefig.call_count >= 3  # Минимум 3 статических графика
            assert mock_write_html.call_count >= 2  # Минимум 2 интерактивных графика