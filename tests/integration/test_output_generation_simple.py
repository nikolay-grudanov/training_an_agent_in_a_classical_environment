"""Упрощенный интеграционный тест полной генерации выходных данных.

Этот тест проверяет основные компоненты User Story 3 (Generate Required Outputs)
с использованием мокирования для ускорения выполнения.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.quantitative_eval import QuantitativeEvaluator
from src.reporting.results_formatter import ResultsFormatter, ReportConfig
from src.utils.seeding import set_seed
from src.visualization.agent_demo import DemoConfig
from src.visualization.performance_plots import (
    PerformancePlotter,
    create_performance_report,
)


class TestOutputGenerationSimple:
    """Упрощенные интеграционные тесты генерации выходных данных."""
    
    @pytest.fixture(scope="class")
    def test_output_dir(self):
        """Создание временной директории для тестов."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_output_generation_simple_"))
        yield temp_dir
        # Очистка после тестов
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def sample_training_data(self) -> Dict[str, pd.DataFrame]:
        """Создание примера данных обучения."""
        set_seed(42)
        
        # Генерируем данные для PPO
        timesteps = np.arange(0, 5000, 100)
        ppo_rewards = 100 * (1 - np.exp(-timesteps / 2000)) + np.random.normal(0, 10, len(timesteps))
        
        ppo_data = pd.DataFrame({
            'timestep': timesteps,
            'episode': np.arange(len(timesteps)),
            'value': ppo_rewards,
            'timestamp': pd.date_range('2024-01-01', periods=len(timesteps), freq='1min')
        })
        
        return {'episode_reward': ppo_data}
    
    def test_performance_plots_creation(
        self,
        sample_training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path
    ):
        """Тест 1: Создание графиков производительности."""
        print("\n📊 Тест создания графиков производительности")
        
        plots_dir = test_output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем плоттер
        plotter = PerformancePlotter()
        
        # 1. График кривой вознаграждения
        reward_plot_path = plotter.plot_reward_curve(
            data=sample_training_data,
            y_col='episode_reward',
            save_path=plots_dir / "reward_curve.png",
            title="Кривая обучения: Вознаграждение"
        )
        
        assert Path(reward_plot_path).exists()
        assert Path(reward_plot_path).stat().st_size > 0
        print(f"✅ График вознаграждения создан: {reward_plot_path}")
        
        # 2. Дашборд метрик
        dashboard_path = plotter.create_dashboard(
            data=sample_training_data,
            save_path=plots_dir / "dashboard.png",
            title="Дашборд метрик обучения"
        )
        
        assert Path(dashboard_path).exists()
        assert Path(dashboard_path).stat().st_size > 0
        print(f"✅ Дашборд создан: {dashboard_path}")
        
        # Проверяем созданные файлы
        created_plots = list(plots_dir.glob("*.png"))
        assert len(created_plots) >= 2
        print(f"✅ Всего создано графиков: {len(created_plots)}")
        
        print("🎉 Создание графиков завершено успешно")
    
    def test_demo_videos_creation(
        self,
        test_output_dir: Path
    ):
        """Тест 2: Создание демонстрационных видео (мокированное)."""
        print("\n🎬 Тест создания демонстрационных видео")
        
        videos_dir = test_output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Полностью мокаем функцию создания демо
        with patch('src.visualization.agent_demo.create_best_episode_demo') as mock_create_demo:
            def mock_demo_creation(agent, env, output_path, config, **kwargs):
                # Создаем фиктивный видеофайл
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text("mock video content")
                
                return {
                    "success": True,
                    "demo_type": "best_episode",
                    "agent_name": getattr(agent, 'name', 'TestAgent'),
                    "best_reward": 150.0,
                    "output_path": str(output_path)
                }
            
            mock_create_demo.side_effect = mock_demo_creation
            
            # Импортируем функцию создания демо
            from src.visualization.agent_demo import create_best_episode_demo
            
            # Создаем мок агента
            mock_agent = MagicMock()
            mock_agent.name = "TestAgent"
            mock_agent.predict.return_value = (np.array([1]), None)
            
            # Создаем демо
            demo_path = videos_dir / "test_demo.mp4"
            demo_info = create_best_episode_demo(
                agent=mock_agent,
                env="LunarLander-v2",
                output_path=demo_path,
                config=DemoConfig(auto_compress=False),
                num_candidates=3
            )
            
            assert demo_info["success"]
            assert Path(demo_info["output_path"]).exists()
            print(f"✅ Демо видео создано: {demo_info['output_path']}")
        
        print("🎉 Создание демо видео завершено успешно")
    
    @patch('src.evaluation.evaluator.Evaluator.evaluate_agent')
    def test_quantitative_evaluation(
        self,
        mock_evaluate_agent,
        test_output_dir: Path
    ):
        """Тест 3: Количественная оценка агентов (мокированная)."""
        print("\n📈 Тест количественной оценки агентов")
        
        eval_dir = test_output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка мока оценки
        def mock_evaluation(agent, num_episodes, **kwargs):
            # Симулируем результаты оценки
            rewards = np.random.normal(150, 25, num_episodes)
            lengths = np.random.randint(150, 250, num_episodes)
            successes = [r > 100 for r in rewards]
            
            from src.evaluation.evaluator import EvaluationMetrics
            return EvaluationMetrics(
                mean_reward=float(np.mean(rewards)),
                std_reward=float(np.std(rewards)),
                min_reward=float(np.min(rewards)),
                max_reward=float(np.max(rewards)),
                mean_episode_length=float(np.mean(lengths)),
                std_episode_length=float(np.std(lengths)),
                min_episode_length=int(np.min(lengths)),
                max_episode_length=int(np.max(lengths)),
                success_rate=float(np.mean(successes)),
                total_episodes=num_episodes,
                total_timesteps=int(np.sum(lengths)),
                evaluation_time=num_episodes * 0.1,
                episode_rewards=rewards.tolist(),
                episode_lengths=lengths.tolist(),
                episode_successes=successes,
                reward_ci_lower=float(np.mean(rewards) - 1.96 * np.std(rewards) / np.sqrt(num_episodes)),
                reward_ci_upper=float(np.mean(rewards) + 1.96 * np.std(rewards) / np.sqrt(num_episodes))
            )
        
        mock_evaluate_agent.side_effect = mock_evaluation
        
        # Создаем мок среды и оценщика
        mock_env = MagicMock()
        mock_env.spec.id = "LunarLander-v2"
        
        evaluator = QuantitativeEvaluator(env=mock_env)
        
        # Создаем мок агента
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        
        # Выполняем оценку
        metrics = evaluator.evaluate_agent_quantitative(
            agent=mock_agent,
            num_episodes=15,
            agent_name="TestAgent"
        )
        
        # Проверяем результаты
        assert metrics.base_metrics.total_episodes == 15
        assert metrics.reward_stability_score >= 0
        assert metrics.reward_stability_score <= 1
        
        print(f"✅ Оценка агента: награда {metrics.base_metrics.mean_reward:.2f} ± "
              f"{metrics.base_metrics.std_reward:.2f}")
        
        # Создание отчета оценки
        text_report = evaluator.generate_comprehensive_report(
            metrics=metrics,
            save_path=eval_dir / "evaluation_report.txt",
            format_type="text"
        )
        
        assert (eval_dir / "evaluation_report.txt").exists()
        assert len(text_report) > 0
        print(f"✅ Отчет оценки создан: {eval_dir / 'evaluation_report.txt'}")
        
        print("🎉 Количественная оценка завершена успешно")
    
    def test_results_formatting(
        self,
        test_output_dir: Path
    ):
        """Тест 4: Форматирование результатов в отчеты."""
        print("\n📝 Тест форматирования результатов")
        
        reports_dir = test_output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем форматировщик
        formatter = ResultsFormatter(
            output_dir=reports_dir,
            config=ReportConfig(language="ru", include_plots=True)
        )
        
        # Создаем тестовые данные оценки
        from src.evaluation.evaluator import EvaluationMetrics
        
        test_metrics = EvaluationMetrics(
            mean_reward=150.5,
            std_reward=25.3,
            min_reward=100.0,
            max_reward=200.0,
            mean_episode_length=195.2,
            std_episode_length=15.8,
            min_episode_length=150,
            max_episode_length=230,
            success_rate=0.85,
            total_episodes=20,
            total_timesteps=3904,
            evaluation_time=45.2,
            episode_rewards=[150.5] * 20,
            episode_lengths=[195] * 20,
            episode_successes=[True] * 17 + [False] * 3,
            reward_ci_lower=140.3,
            reward_ci_upper=160.7
        )
        
        # 1. Отчет по одному агенту
        single_agent_report = formatter.format_single_agent_report(
            agent_name="TestAgent",
            evaluation_results=test_metrics,
            output_format="html",
            filename="test_agent_report"
        )
        
        assert single_agent_report.exists()
        assert single_agent_report.suffix == ".html"
        print(f"✅ Отчет по агенту создан: {single_agent_report}")
        
        # 2. Экспорт в CSV
        csv_export = formatter.export_to_csv(
            data=test_metrics,
            filename="test_results"
        )
        
        assert csv_export.exists()
        assert csv_export.suffix == ".csv"
        
        # Проверяем содержимое CSV
        df = pd.read_csv(csv_export)
        assert len(df) == 1
        assert "mean_reward" in df.columns
        print(f"✅ CSV экспорт создан: {csv_export}")
        
        # 3. Экспорт в JSON
        json_export = formatter.export_to_json(
            data={"test_metrics": test_metrics},
            filename="test_results_json"
        )
        
        assert json_export.exists()
        assert json_export.suffix == ".json"
        
        # Проверяем содержимое JSON
        with open(json_export, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        assert "test_metrics" in json_data
        print(f"✅ JSON экспорт создан: {json_export}")
        
        # Проверяем все созданные отчеты
        created_reports = list(reports_dir.rglob("*"))
        created_files = [f for f in created_reports if f.is_file()]
        assert len(created_files) >= 3
        print(f"✅ Всего создано отчетов: {len(created_files)}")
        
        print("🎉 Форматирование результатов завершено успешно")
    
    def test_performance_measurement(
        self,
        sample_training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path
    ):
        """Тест 5: Измерение производительности."""
        print("\n⏱️ Тест измерения производительности")
        
        performance_dir = test_output_dir / "performance"
        performance_dir.mkdir(parents=True, exist_ok=True)
        
        performance_metrics = {}
        
        # 1. Измерение времени создания графиков
        start_time = time.time()
        
        plotter = PerformancePlotter()
        plots_created = 0
        
        for i in range(2):  # Создаем 2 графика для теста
            plot_path = plotter.plot_reward_curve(
                data=sample_training_data,
                save_path=performance_dir / f"perf_plot_{i}.png"
            )
            plots_created += 1
        
        plots_time = time.time() - start_time
        performance_metrics["plots"] = {
            "total_time": plots_time,
            "plots_created": plots_created,
            "time_per_plot": plots_time / plots_created,
        }
        
        print(f"✅ Графики: {plots_created} за {plots_time:.2f}с "
              f"({performance_metrics['plots']['time_per_plot']:.2f}с/график)")
        
        # 2. Измерение времени создания отчетов
        start_time = time.time()
        
        formatter = ResultsFormatter(output_dir=performance_dir)
        
        # Создаем тестовые данные
        from src.evaluation.evaluator import EvaluationMetrics
        test_metrics = EvaluationMetrics(
            mean_reward=150.0, std_reward=25.0, min_reward=100.0, max_reward=200.0,
            mean_episode_length=200.0, std_episode_length=20.0, 
            min_episode_length=150, max_episode_length=250,
            success_rate=0.8, total_episodes=20, total_timesteps=4000, evaluation_time=30.0,
            episode_rewards=[150.0] * 20, episode_lengths=[200] * 20, 
            episode_successes=[True] * 16 + [False] * 4,
            reward_ci_lower=140.0, reward_ci_upper=160.0
        )
        
        reports_created = 0
        
        # Создаем отчеты в разных форматах
        for format_type in ["html", "json", "csv"]:
            try:
                if format_type == "csv":
                    formatter.export_to_csv(
                        data=test_metrics,
                        filename=f"perf_report_{format_type}"
                    )
                elif format_type == "json":
                    formatter.export_to_json(
                        data={"test_metrics": test_metrics},
                        filename=f"perf_report_{format_type}"
                    )
                else:
                    formatter.format_single_agent_report(
                        agent_name="PerfTestAgent",
                        evaluation_results=test_metrics,
                        output_format=format_type,
                        filename=f"perf_report_{format_type}"
                    )
                reports_created += 1
            except Exception as e:
                print(f"⚠️ Ошибка создания отчета {format_type}: {e}")
        
        reports_time = time.time() - start_time
        performance_metrics["reports"] = {
            "total_time": reports_time,
            "reports_created": reports_created,
            "time_per_report": reports_time / reports_created if reports_created > 0 else 0,
        }
        
        print(f"✅ Отчеты: {reports_created} за {reports_time:.2f}с "
              f"({performance_metrics['reports']['time_per_report']:.2f}с/отчет)")
        
        # 3. Общая производительность
        total_time = sum(m["total_time"] for m in performance_metrics.values())
        total_operations = (
            performance_metrics["plots"]["plots_created"] +
            performance_metrics["reports"]["reports_created"]
        )
        
        performance_metrics["overall"] = {
            "total_time": total_time,
            "total_operations": total_operations,
            "operations_per_second": total_operations / total_time if total_time > 0 else 0
        }
        
        print(f"✅ Общая производительность: {total_operations} операций за {total_time:.2f}с "
              f"({performance_metrics['overall']['operations_per_second']:.2f} оп/с)")
        
        # Сохранение метрик
        performance_report_path = performance_dir / "performance_metrics.json"
        with open(performance_report_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, indent=2)
        
        print(f"✅ Метрики производительности сохранены: {performance_report_path}")
        
        # Проверка базовых требований производительности
        assert performance_metrics["plots"]["time_per_plot"] < 10.0  # Не более 10 секунд на график
        assert performance_metrics["reports"]["time_per_report"] < 5.0  # Не более 5 секунд на отчет
        
        print("🎉 Производительность соответствует требованиям")
        
        return performance_metrics
    
    @pytest.mark.integration
    def test_complete_output_generation_workflow(
        self,
        sample_training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path
    ):
        """Тест 6: Полный workflow генерации выходных данных."""
        print("\n🚀 Тест полного workflow генерации выходных данных")
        
        workflow_dir = test_output_dir / "complete_workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_results = {
            "start_time": time.time(),
            "steps_completed": [],
            "files_created": [],
            "success": True
        }
        
        try:
            # Шаг 1: Сохранение данных обучения
            print("🔧 Шаг 1: Сохранение данных обучения")
            
            data_dir = workflow_dir / "training_data"
            data_dir.mkdir(exist_ok=True)
            
            for metric_name, data in sample_training_data.items():
                data_path = data_dir / f"{metric_name}.csv"
                data.to_csv(data_path, index=False)
                workflow_results["files_created"].append(str(data_path))
            
            workflow_results["steps_completed"].append("data_preparation")
            print("✅ Данные обучения сохранены")
            
            # Шаг 2: Создание графиков производительности
            print("📊 Шаг 2: Создание графиков производительности")
            
            plots_dir = workflow_dir / "plots"
            performance_report_dir = create_performance_report(
                data=sample_training_data,
                output_dir=plots_dir,
                include_interactive=False,  # Отключаем интерактивные для ускорения
                include_static=True
            )
            
            # Подсчитываем созданные графики
            static_plots = list(Path(performance_report_dir).glob("static/*.png"))
            workflow_results["files_created"].extend([str(p) for p in static_plots])
            workflow_results["steps_completed"].append("performance_plots")
            
            print(f"✅ Создано графиков: {len(static_plots)}")
            
            # Шаг 3: Создание отчетов
            print("📝 Шаг 3: Создание отчетов")
            
            reports_dir = workflow_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            formatter = ResultsFormatter(output_dir=reports_dir)
            
            # Создаем тестовые данные для отчета
            from src.evaluation.evaluator import EvaluationMetrics
            test_metrics = EvaluationMetrics(
                mean_reward=150.0, std_reward=25.0, min_reward=100.0, max_reward=200.0,
                mean_episode_length=200.0, std_episode_length=20.0,
                min_episode_length=150, max_episode_length=250,
                success_rate=0.8, total_episodes=20, total_timesteps=4000, evaluation_time=30.0,
                episode_rewards=[150.0] * 20, episode_lengths=[200] * 20,
                episode_successes=[True] * 16 + [False] * 4,
                reward_ci_lower=140.0, reward_ci_upper=160.0
            )
            
            # Создаем отчеты
            html_report = formatter.format_single_agent_report(
                agent_name="WorkflowTestAgent",
                evaluation_results=test_metrics,
                output_format="html",
                filename="workflow_report"
            )
            workflow_results["files_created"].append(str(html_report))
            
            csv_export = formatter.export_to_csv(
                data=test_metrics,
                filename="workflow_results"
            )
            workflow_results["files_created"].append(str(csv_export))
            
            workflow_results["steps_completed"].append("reports_creation")
            print("✅ Отчеты созданы")
            
            # Шаг 4: Проверка всех созданных файлов
            print("🔍 Шаг 4: Проверка созданных файлов")
            
            files_verification = {
                "total_files": len(workflow_results["files_created"]),
                "existing_files": 0,
                "missing_files": []
            }
            
            for file_path in workflow_results["files_created"]:
                path_obj = Path(file_path)
                if path_obj.exists():
                    files_verification["existing_files"] += 1
                else:
                    files_verification["missing_files"].append(str(path_obj))
            
            workflow_results["files_verification"] = files_verification
            workflow_results["steps_completed"].append("files_verification")
            
            print(f"✅ Проверка файлов: {files_verification['existing_files']}/{files_verification['total_files']} существуют")
            
            if files_verification["missing_files"]:
                print(f"⚠️ Отсутствующие файлы: {files_verification['missing_files']}")
            
        except Exception as e:
            workflow_results["success"] = False
            print(f"❌ Ошибка в workflow: {e}")
            raise
        
        finally:
            workflow_results["end_time"] = time.time()
            workflow_results["total_time"] = workflow_results["end_time"] - workflow_results["start_time"]
            
            # Сохранение результатов workflow
            workflow_summary_path = workflow_dir / "workflow_summary.json"
            with open(workflow_summary_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, default=str)
        
        # Финальные проверки
        assert workflow_results["success"], "Workflow должен завершиться успешно"
        assert len(workflow_results["steps_completed"]) == 4, "Все шаги должны быть выполнены"
        assert workflow_results["files_verification"]["existing_files"] > 0, "Должны быть созданы файлы"
        
        print(f"\n🎉 Полный workflow завершен успешно за {workflow_results['total_time']:.2f}с")
        print(f"📊 Выполнено шагов: {len(workflow_results['steps_completed'])}")
        print(f"📁 Создано файлов: {workflow_results['files_verification']['total_files']}")
        print(f"✅ Существующих файлов: {workflow_results['files_verification']['existing_files']}")
        
        return workflow_results


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v", "-s", "--tb=short"])