"""Интеграционный тест полной генерации выходных данных с завершенной сессией обучения.

Этот тест проверяет полный пайплайн генерации выходов User Story 3 (Generate Required Outputs):
- Генерация графиков производительности
- Создание демонстрационных видео агентов
- Количественная оценка на 10-20 эпизодах
- Форматирование результатов в отчеты

Тест использует реальные обученные агенты или их моки для проверки всех компонентов
интеграции между модулями с обработкой ошибок и измерением производительности.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.quantitative_eval import (
    QuantitativeEvaluator,
    QuantitativeMetrics,
    evaluate_agent_standard,
)
from src.reporting.results_formatter import ResultsFormatter, ReportConfig
from src.utils.seeding import set_seed
from src.visualization.agent_demo import (
    AgentDemoError,
    DemoConfig,
    create_best_episode_demo,
    create_batch_demos,
    quick_demo,
)
from src.visualization.performance_plots import (
    PerformancePlotter,
    create_performance_report,
    quick_reward_plot,
)


class MockAgent:
    """Мок агента для тестирования без реального обучения."""

    def __init__(
        self,
        name: str = "MockAgent",
        performance_level: str = "good",
        deterministic: bool = True,
    ):
        """Инициализация мок агента.

        Args:
            name: Имя агента
            performance_level: Уровень производительности ('poor', 'good', 'excellent')
            deterministic: Детерминированное поведение
        """
        self.name = name
        self.performance_level = performance_level
        self.deterministic = deterministic
        self._episode_count = 0

        # Настройка базовых параметров производительности
        if performance_level == "poor":
            self._base_reward = -200
            self._reward_variance = 50
            self._success_rate = 0.1
        elif performance_level == "good":
            self._base_reward = 100
            self._reward_variance = 30
            self._success_rate = 0.6
        else:  # excellent
            self._base_reward = 200
            self._reward_variance = 20
            self._success_rate = 0.9

    def predict(
        self, observation, deterministic: bool = True, **kwargs
    ) -> Tuple[np.ndarray, Any]:
        """Предсказание действия агента."""
        # Простая эвристика для LunarLander
        action = np.random.randint(0, 4) if not deterministic else 1
        return np.array([action]), None

    def learn(self, total_timesteps: int, **kwargs) -> "MockAgent":
        """Имитация обучения агента."""
        return self

    def save(self, path: str) -> None:
        """Сохранение модели агента."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем простые метаданные
        metadata = {
            "name": self.name,
            "performance_level": self.performance_level,
            "deterministic": self.deterministic,
            "episode_count": self._episode_count,
        }

        with open(save_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MockAgent":
        """Загрузка модели агента."""
        with open(path, "r") as f:
            metadata = json.load(f)

        agent = cls(
            name=metadata["name"],
            performance_level=metadata["performance_level"],
            deterministic=metadata["deterministic"],
        )
        agent._episode_count = metadata["episode_count"]
        return agent

    def simulate_episode_reward(self) -> float:
        """Симуляция награды за эпизод."""
        self._episode_count += 1

        # Добавляем небольшой тренд улучшения со временем
        trend_bonus = min(self._episode_count * 0.5, 20)

        # Генерируем награду с некоторой случайностью
        if self.deterministic:
            np.random.seed(42 + self._episode_count)

        reward = np.random.normal(
            self._base_reward + trend_bonus, self._reward_variance
        )

        return float(reward)


class MockEnvironment:
    """Мок среды для тестирования."""

    def __init__(self, env_name: str = "LunarLander-v2"):
        self.env_name = env_name
        self.spec = MagicMock()
        self.spec.id = env_name
        self._episode_length = 0
        self._max_episode_length = 200

    def reset(self, seed: Optional[int] = None):
        """Сброс среды."""
        if seed is not None:
            np.random.seed(seed)
        self._episode_length = 0
        observation = np.random.random(
            8
        )  # LunarLander имеет 8-мерное пространство наблюдений
        info = {}
        return observation, info

    def step(self, action):
        """Шаг в среде."""
        self._episode_length += 1

        # Простая симуляция
        observation = np.random.random(8)
        reward = np.random.normal(0, 1)  # Случайная награда
        done = self._episode_length >= self._max_episode_length
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def close(self):
        """Закрытие среды."""
        pass


class TestOutputGeneration:
    """Интеграционные тесты полной генерации выходных данных."""

    @pytest.fixture(scope="class")
    def test_output_dir(self):
        """Создание временной директории для тестов."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_output_generation_"))
        yield temp_dir
        # Очистка после тестов
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def mock_env(self) -> MockEnvironment:
        """Мок среды для тестирования."""
        return MockEnvironment("LunarLander-v2")

    @pytest.fixture(scope="class")
    def trained_agents(self) -> Dict[str, MockAgent]:
        """Набор обученных агентов разного уровня."""
        return {
            "PPO_Excellent": MockAgent("PPO_Excellent", "excellent"),
            "A2C_Good": MockAgent("A2C_Good", "good"),
            "SAC_Poor": MockAgent("SAC_Poor", "poor"),
        }

    @pytest.fixture(scope="class")
    def training_data(self) -> Dict[str, pd.DataFrame]:
        """Симулированные данные обучения для графиков."""
        set_seed(42)

        # Генерируем данные для PPO (хорошая сходимость)
        timesteps = np.arange(0, 10000, 100)
        ppo_rewards = 100 * (1 - np.exp(-timesteps / 3000)) + np.random.normal(
            0, 10, len(timesteps)
        )
        ppo_lengths = (
            200
            - 50 * (1 - np.exp(-timesteps / 2000))
            + np.random.normal(0, 5, len(timesteps))
        )

        ppo_data = pd.DataFrame(
            {
                "timestep": timesteps,
                "episode": np.arange(len(timesteps)),
                "value": ppo_rewards,
                "episode_length": ppo_lengths.astype(int),
                "timestamp": pd.date_range(
                    "2024-01-01", periods=len(timesteps), freq="1min"
                ),
            }
        )

        # Генерируем данные для A2C (более нестабильная сходимость)
        a2c_rewards = 80 * (1 - np.exp(-timesteps / 4000)) + np.random.normal(
            0, 15, len(timesteps)
        )
        a2c_lengths = (
            220
            - 40 * (1 - np.exp(-timesteps / 3000))
            + np.random.normal(0, 8, len(timesteps))
        )

        a2c_data = pd.DataFrame(
            {
                "timestep": timesteps,
                "episode": np.arange(len(timesteps)),
                "value": a2c_rewards,
                "episode_length": a2c_lengths.astype(int),
                "timestamp": pd.date_range(
                    "2024-01-01", periods=len(timesteps), freq="1min"
                ),
            }
        )

        return {
            "episode_reward": ppo_data,
            "episode_reward_a2c": a2c_data,
            "episode_length": ppo_data[
                ["timestep", "episode", "episode_length", "timestamp"]
            ].rename(columns={"episode_length": "value"}),
        }

    def test_data_preparation_and_validation(
        self,
        trained_agents: Dict[str, MockAgent],
        training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path,
    ):
        """Тест 1: Подготовка и валидация данных обучения."""
        print("\n🔧 Тест 1: Подготовка данных обучения")

        # Проверяем агентов
        assert len(trained_agents) == 3
        for name, agent in trained_agents.items():
            assert isinstance(agent, MockAgent)
            assert agent.name == name
            print(f"✅ Агент {name} подготовлен (уровень: {agent.performance_level})")

        # Проверяем данные обучения
        assert len(training_data) >= 2
        for metric_name, data in training_data.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert "timestep" in data.columns
            assert "value" in data.columns
            print(f"✅ Данные {metric_name}: {len(data)} записей")

        # Сохраняем тестовые данные
        data_dir = test_output_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, data in training_data.items():
            data_path = data_dir / f"{metric_name}.csv"
            data.to_csv(data_path, index=False)
            assert data_path.exists()
            print(f"✅ Данные {metric_name} сохранены: {data_path}")

        print("🎉 Подготовка данных завершена успешно")

    def test_performance_plots_generation(
        self, training_data: Dict[str, pd.DataFrame], test_output_dir: Path
    ):
        """Тест 2: Генерация графиков производительности."""
        print("\n📊 Тест 2: Генерация графиков производительности")

        plots_dir = test_output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Инициализируем плоттер
        plotter = PerformancePlotter()

        # 1. График кривой вознаграждения
        reward_plot_path = plotter.plot_reward_curve(
            data=training_data,
            y_col="episode_reward",
            save_path=plots_dir / "reward_curve.png",
            title="Кривая обучения: Вознаграждение",
        )

        assert Path(reward_plot_path).exists()
        print(f"✅ График вознаграждения создан: {reward_plot_path}")

        # 2. График длины эпизодов
        length_plot_path = plotter.plot_episode_lengths(
            data=training_data,
            y_col="episode_length",
            save_path=plots_dir / "episode_lengths.png",
            title="Длина эпизодов",
        )

        assert Path(length_plot_path).exists()
        print(f"✅ График длины эпизодов создан: {length_plot_path}")

        # 3. Сравнительный график агентов
        agents_data = {
            "PPO": training_data["episode_reward"],
            "A2C": training_data["episode_reward_a2c"],
        }

        comparison_plot_path = plotter.plot_multiple_agents(
            agents_data=agents_data,
            metric="episode_reward",
            save_path=plots_dir / "agents_comparison.png",
            title="Сравнение агентов: PPO vs A2C",
        )

        assert Path(comparison_plot_path).exists()
        print(f"✅ Сравнительный график создан: {comparison_plot_path}")

        # 4. Дашборд метрик
        dashboard_path = plotter.create_dashboard(
            data=training_data,
            save_path=plots_dir / "training_dashboard.png",
            title="Дашборд метрик обучения",
        )

        assert Path(dashboard_path).exists()
        print(f"✅ Дашборд создан: {dashboard_path}")

        # 5. Быстрый график через удобную функцию
        quick_plot_path = quick_reward_plot(
            data_source=training_data["episode_reward"],
            save_path=plots_dir / "quick_reward_plot.png",
        )

        assert Path(quick_plot_path).exists()
        print(f"✅ Быстрый график создан: {quick_plot_path}")

        # Проверяем, что все файлы созданы
        created_plots = list(plots_dir.glob("*.png"))
        assert len(created_plots) >= 5
        print(f"✅ Всего создано графиков: {len(created_plots)}")

        print("🎉 Генерация графиков завершена успешно")

    @patch("src.visualization.video_generator.setup_recording_environment")
    @patch("src.visualization.video_generator.record_agent_episode")
    def test_agent_demo_videos_generation(
        self,
        mock_record_episode,
        mock_setup_env,
        trained_agents: Dict[str, MockAgent],
        mock_env: MockEnvironment,
        test_output_dir: Path,
    ):
        """Тест 3: Создание демонстрационных видео агентов."""
        print("\n🎬 Тест 3: Создание демонстрационных видео")

        videos_dir = test_output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Настройка моков
        mock_setup_env.return_value = mock_env
        mock_record_episode.return_value = {
            "success": True,
            "total_reward": 150.0,
            "episode_length": 200,
            "output_path": str(videos_dir / "test_video.mp4"),
        }

        # Создаем фиктивный видеофайл для тестирования
        def create_mock_video(path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("mock video content")

        # 1. Демо лучшего эпизода для одного агента
        best_agent = trained_agents["PPO_Excellent"]
        demo_config = DemoConfig(
            auto_compress=False,  # Отключаем сжатие для тестов
            auto_naming=True,
        )

        best_demo_path = videos_dir / "best_episode_demo.mp4"
        create_mock_video(best_demo_path)

        # Мокаем функцию создания демо
        with patch("src.visualization.agent_demo.record_agent_episode") as mock_record:
            mock_record.return_value = {
                "success": True,
                "total_reward": 180.5,
                "episode_length": 195,
                "output_path": str(best_demo_path),
            }

            demo_info = create_best_episode_demo(
                agent=best_agent,
                env="LunarLander-v2",
                output_path=best_demo_path,
                config=demo_config,
                num_candidates=5,
            )

        assert demo_info["success"]
        assert demo_info["demo_type"] == "best_episode"
        assert demo_info["agent_name"] == "PPO_Excellent"
        print(f"✅ Демо лучшего эпизода создано: {demo_info['output_path']}")

        # 2. Пакетное создание демо для всех агентов
        batch_demos_dir = videos_dir / "batch_demos"

        # Мокаем пакетное создание
        with patch(
            "src.visualization.agent_demo.create_best_episode_demo"
        ) as mock_batch_demo:

            def mock_demo_creation(agent, env, output_path, config):
                create_mock_video(Path(output_path))
                return {
                    "success": True,
                    "demo_type": "best_episode",
                    "agent_name": agent.name,
                    "best_reward": agent.simulate_episode_reward(),
                    "output_path": str(output_path),
                }

            mock_batch_demo.side_effect = mock_demo_creation

            agents_list = [(name, agent) for name, agent in trained_agents.items()]
            batch_result = create_batch_demos(
                agents=agents_list,
                env="LunarLander-v2",
                output_dir=batch_demos_dir,
                demo_types=["best_episode"],
                config=demo_config,
            )

        assert batch_result["success"]
        assert batch_result["demos_created"] == len(trained_agents)
        assert batch_result["demos_failed"] == 0
        print(f"✅ Пакетные демо созданы: {batch_result['demos_created']} видео")

        # 3. Быстрое демо через удобную функцию
        quick_demo_path = videos_dir / "quick_demo.mp4"
        create_mock_video(quick_demo_path)

        with patch(
            "src.visualization.agent_demo.create_best_episode_demo"
        ) as mock_quick:
            mock_quick.return_value = {
                "success": True,
                "output_path": str(quick_demo_path),
                "compressed_path": None,
            }

            quick_result_path = quick_demo(
                agent=best_agent, env="LunarLander-v2", output_path=quick_demo_path
            )

        assert Path(quick_result_path).exists()
        print(f"✅ Быстрое демо создано: {quick_result_path}")

        # Проверяем созданные видеофайлы
        created_videos = list(videos_dir.rglob("*.mp4"))
        assert len(created_videos) >= 3
        print(f"✅ Всего создано видео: {len(created_videos)}")

        print("🎉 Создание демонстрационных видео завершено успешно")

    def test_quantitative_evaluation(
        self,
        trained_agents: Dict[str, MockAgent],
        mock_env: MockEnvironment,
        test_output_dir: Path,
    ):
        """Тест 4: Количественная оценка агентов на 10-20 эпизодах."""
        print("\n📈 Тест 4: Количественная оценка агентов")

        eval_dir = test_output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Инициализируем количественный оценщик
        evaluator = QuantitativeEvaluator(
            env=mock_env, baseline_threshold=100.0, min_effect_size=0.5, random_seed=42
        )

        # Мокаем базовый оценщик для ускорения тестов
        with patch.object(evaluator.evaluator, "evaluate_agent") as mock_eval:

            def mock_evaluation(agent, num_episodes, **kwargs):
                # Симулируем результаты оценки на основе уровня агента
                rewards = [agent.simulate_episode_reward() for _ in range(num_episodes)]
                lengths = [np.random.randint(150, 250) for _ in range(num_episodes)]
                successes = [r > 0 for r in rewards]

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
                    total_timesteps=sum(lengths),
                    evaluation_time=num_episodes * 0.1,
                    episode_rewards=rewards,
                    episode_lengths=lengths,
                    episode_successes=successes,
                    reward_ci_lower=float(
                        np.mean(rewards)
                        - 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                    ),
                    reward_ci_upper=float(
                        np.mean(rewards)
                        + 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                    ),
                )

            mock_eval.side_effect = mock_evaluation

            # 1. Оценка отдельных агентов
            agents_metrics = {}
            for name, agent in trained_agents.items():
                print(f"📊 Оценка агента {name}...")

                metrics = evaluator.evaluate_agent_quantitative(
                    agent=agent,
                    num_episodes=15,  # Стандартное количество для тестов
                    agent_name=name,
                )

                agents_metrics[name] = metrics

                # Проверяем структуру метрик
                assert isinstance(metrics, QuantitativeMetrics)
                assert metrics.base_metrics.total_episodes == 15
                assert metrics.reward_stability_score >= 0
                assert metrics.reward_stability_score <= 1

                print(
                    f"✅ {name}: награда {metrics.base_metrics.mean_reward:.2f} ± "
                    f"{metrics.base_metrics.std_reward:.2f}, "
                    f"стабильность {metrics.reward_stability_score:.3f}"
                )

            # 2. Пакетная оценка всех агентов
            print("📊 Пакетная оценка агентов...")

            batch_result = evaluator.evaluate_multiple_agents_batch(
                agents=trained_agents, num_episodes=12, include_pairwise_comparison=True
            )

            assert len(batch_result.agents_metrics) == len(trained_agents)
            assert batch_result.best_agent in trained_agents.keys()
            assert len(batch_result.ranking) == len(trained_agents)

            # Проверяем, что лучший агент действительно лучший
            best_reward = batch_result.ranking[0][1]
            for _, reward in batch_result.ranking[1:]:
                assert best_reward >= reward

            print(
                f"✅ Лучший агент: {batch_result.best_agent} "
                f"(награда: {best_reward:.2f})"
            )

            # 3. Сравнение с базовой линией
            print("📊 Сравнение с базовой линией...")

            best_agent = trained_agents[batch_result.best_agent]
            baseline_agent = trained_agents[
                "SAC_Poor"
            ]  # Используем слабого агента как базовую линию

            comparison = evaluator.compare_with_baseline(
                agent=best_agent,
                baseline_agent=baseline_agent,
                num_episodes=10,
                agent_name=batch_result.best_agent,
                baseline_name="SAC_Poor",
            )

            assert comparison.agent_name == batch_result.best_agent
            assert comparison.baseline_name == "SAC_Poor"
            assert comparison.is_better  # Лучший агент должен быть лучше слабого

            print(
                f"✅ Сравнение: улучшение {comparison.reward_improvement:.1f}%, "
                f"размер эффекта {comparison.effect_size:.3f}"
            )

        # 4. Генерация отчетов оценки
        print("📊 Генерация отчетов оценки...")

        # Текстовый отчет
        text_report = evaluator.generate_comprehensive_report(
            metrics=batch_result,
            save_path=eval_dir / "evaluation_report.txt",
            format_type="text",
        )

        assert (eval_dir / "evaluation_report.txt").exists()
        assert len(text_report) > 0
        print(f"✅ Текстовый отчет создан: {eval_dir / 'evaluation_report.txt'}")

        # JSON отчет
        json_report = evaluator.generate_comprehensive_report(
            metrics=batch_result,
            save_path=eval_dir / "evaluation_report.json",
            format_type="json",
        )

        assert (eval_dir / "evaluation_report.json").exists()
        json_data = json.loads(json_report)
        assert "agents_metrics" in json_data
        print(f"✅ JSON отчет создан: {eval_dir / 'evaluation_report.json'}")

        # CSV отчет
        csv_report = evaluator.generate_comprehensive_report(
            metrics=batch_result,
            save_path=eval_dir / "evaluation_report.csv",
            format_type="csv",
        )

        assert (eval_dir / "evaluation_report.csv").exists()
        print(f"✅ CSV отчет создан: {eval_dir / 'evaluation_report.csv'}")

        print("🎉 Количественная оценка завершена успешно")

        return {
            "agents_metrics": agents_metrics,
            "batch_result": batch_result,
            "comparison": comparison,
        }

    def test_results_formatting_and_reporting(self, test_output_dir: Path):
        """Тест 5: Форматирование результатов в отчеты."""
        print("\n📝 Тест 5: Форматирование результатов в отчеты")

        reports_dir = test_output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Инициализируем форматировщик результатов
        formatter = ResultsFormatter(
            output_dir=reports_dir,
            config=ReportConfig(
                language="ru", include_plots=True, include_statistics=True
            ),
        )

        # Создаем тестовые данные оценки
        from src.evaluation.evaluator import EvaluationMetrics

        test_metrics = {
            "PPO_Excellent": EvaluationMetrics(
                mean_reward=180.5,
                std_reward=25.3,
                min_reward=120.0,
                max_reward=220.0,
                mean_episode_length=195.2,
                std_episode_length=15.8,
                min_episode_length=150,
                max_episode_length=230,
                success_rate=0.85,
                total_episodes=20,
                total_timesteps=3904,
                evaluation_time=45.2,
                episode_rewards=[180.5] * 20,
                episode_lengths=[195] * 20,
                episode_successes=[True] * 17 + [False] * 3,
                reward_ci_lower=169.3,
                reward_ci_upper=191.7,
            ),
            "A2C_Good": EvaluationMetrics(
                mean_reward=145.8,
                std_reward=32.1,
                min_reward=80.0,
                max_reward=200.0,
                mean_episode_length=210.5,
                std_episode_length=22.4,
                min_episode_length=160,
                max_episode_length=250,
                success_rate=0.65,
                total_episodes=20,
                total_timesteps=4210,
                evaluation_time=48.7,
                episode_rewards=[145.8] * 20,
                episode_lengths=[210] * 20,
                episode_successes=[True] * 13 + [False] * 7,
                reward_ci_lower=131.5,
                reward_ci_upper=160.1,
            ),
        }

        # 1. Отчет по одному агенту
        single_agent_report = formatter.format_single_agent_report(
            agent_name="PPO_Excellent",
            evaluation_results=test_metrics["PPO_Excellent"],
            output_format="html",
            filename="ppo_agent_report",
        )

        assert single_agent_report.exists()
        assert single_agent_report.suffix == ".html"
        print(f"✅ Отчет по агенту создан: {single_agent_report}")

        # 2. Сравнительный отчет
        comparison_report = formatter.format_comparison_report(
            agents_results=test_metrics,
            output_format="html",
            filename="agents_comparison_report",
        )

        assert comparison_report.exists()
        assert comparison_report.suffix == ".html"
        print(f"✅ Сравнительный отчет создан: {comparison_report}")

        # 3. Отчет по эксперименту
        experiment_data = {
            "experiment_name": "PPO vs A2C Comparison",
            "hypothesis": "PPO покажет лучшую стабильность обучения чем A2C",
            "agents_results": test_metrics,
            "conclusion": "Гипотеза подтверждена",
            "statistical_significance": True,
        }

        experiment_report = formatter.format_experiment_report(
            experiment_name="PPO_vs_A2C_Experiment",
            experiment_data=experiment_data,
            output_format="markdown",
            filename="experiment_report",
        )

        assert experiment_report.exists()
        assert experiment_report.suffix == ".markdown"
        print(f"✅ Отчет по эксперименту создан: {experiment_report}")

        # 4. Сводный отчет
        experiments_data = {
            "PPO_vs_A2C": experiment_data,
            "Baseline_Comparison": {
                "experiment_name": "Baseline Comparison",
                "agents_results": {"PPO": test_metrics["PPO_Excellent"]},
                "conclusion": "Превышает базовую линию",
            },
        }

        summary_report = formatter.format_summary_report(
            experiments_data=experiments_data,
            output_format="html",
            filename="experiments_summary",
        )

        assert summary_report.exists()
        assert summary_report.suffix == ".html"
        print(f"✅ Сводный отчет создан: {summary_report}")

        # 5. Экспорт в CSV
        csv_export = formatter.export_to_csv(
            data=test_metrics, filename="agents_results"
        )

        assert csv_export.exists()
        assert csv_export.suffix == ".csv"

        # Проверяем содержимое CSV
        df = pd.read_csv(csv_export)
        assert len(df) == len(test_metrics)
        assert "agent" in df.columns
        assert "mean_reward" in df.columns
        print(f"✅ CSV экспорт создан: {csv_export}")

        # 6. Экспорт в JSON
        json_export = formatter.export_to_json(
            data={"agents_results": test_metrics, "experiment_data": experiment_data},
            filename="complete_results",
        )

        assert json_export.exists()
        assert json_export.suffix == ".json"

        # Проверяем содержимое JSON
        with open(json_export, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        assert "agents_results" in json_data
        assert "experiment_data" in json_data
        print(f"✅ JSON экспорт создан: {json_export}")

        # Проверяем все созданные отчеты
        created_reports = list(reports_dir.rglob("*"))
        created_files = [f for f in created_reports if f.is_file()]
        assert len(created_files) >= 6
        print(f"✅ Всего создано отчетов: {len(created_files)}")

        print("🎉 Форматирование результатов завершено успешно")

    def test_integration_between_modules(
        self,
        trained_agents: Dict[str, MockAgent],
        training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path,
    ):
        """Тест 6: Проверка интеграции между модулями."""
        print("\n🔗 Тест 6: Интеграция между модулями")

        integration_dir = test_output_dir / "integration"
        integration_dir.mkdir(parents=True, exist_ok=True)

        # 1. Интеграция данных обучения и графиков
        print("🔗 Тестирование интеграции: данные → графики")

        # Создаем полный отчет о производительности
        performance_report_dir = create_performance_report(
            data=training_data,
            output_dir=integration_dir / "performance_report",
            include_interactive=True,
            include_static=True,
        )

        assert Path(performance_report_dir).exists()
        static_plots = list(Path(performance_report_dir).glob("static/*.png"))
        interactive_plots = list(
            Path(performance_report_dir).glob("interactive/*.html")
        )

        assert len(static_plots) >= 2
        assert len(interactive_plots) >= 1
        print(
            f"✅ Отчет о производительности: {len(static_plots)} статических, "
            f"{len(interactive_plots)} интерактивных графиков"
        )

        # 2. Интеграция оценки и отчетности
        print("🔗 Тестирование интеграции: оценка → отчеты")

        # Используем стандартную функцию оценки
        mock_env = MockEnvironment()

        with patch("src.evaluation.evaluator.Evaluator.evaluate_agent") as mock_eval:

            def mock_evaluation(agent, num_episodes, **kwargs):
                rewards = [agent.simulate_episode_reward() for _ in range(num_episodes)]
                lengths = [np.random.randint(150, 250) for _ in range(num_episodes)]
                successes = [r > 0 for r in rewards]

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
                    total_timesteps=sum(lengths),
                    evaluation_time=num_episodes * 0.1,
                    episode_rewards=rewards,
                    episode_lengths=lengths,
                    episode_successes=successes,
                    reward_ci_lower=float(
                        np.mean(rewards)
                        - 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                    ),
                    reward_ci_upper=float(
                        np.mean(rewards)
                        + 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                    ),
                )

            mock_eval.side_effect = mock_evaluation

            # Оценка агента через стандартную функцию
            best_agent = trained_agents["PPO_Excellent"]
            evaluation_metrics = evaluate_agent_standard(
                agent=best_agent,
                env=mock_env,
                num_episodes=15,
                agent_name="PPO_Excellent",
            )

            # Создание отчета на основе оценки
            formatter = ResultsFormatter(
                output_dir=integration_dir / "integrated_reports"
            )

            integrated_report = formatter.format_single_agent_report(
                agent_name="PPO_Excellent",
                evaluation_results=evaluation_metrics.base_metrics,
                quantitative_results=evaluation_metrics,
                output_format="html",
            )

            assert integrated_report.exists()
            print(f"✅ Интегрированный отчет создан: {integrated_report}")

        # 3. Интеграция всех компонентов в единый пайплайн
        print("🔗 Тестирование полной интеграции: данные → оценка → графики → отчеты")

        pipeline_dir = integration_dir / "full_pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Симулируем полный пайплайн
        pipeline_results = {
            "training_data": training_data,
            "agents_evaluated": len(trained_agents),
            "plots_created": len(static_plots) + len(interactive_plots),
            "reports_generated": 1,
            "integration_successful": True,
        }

        # Сохраняем результаты пайплайна
        pipeline_summary_path = pipeline_dir / "pipeline_summary.json"
        with open(pipeline_summary_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_results, f, indent=2, default=str)

        assert pipeline_summary_path.exists()
        print(f"✅ Сводка пайплайна сохранена: {pipeline_summary_path}")

        print("🎉 Интеграция между модулями работает корректно")

    def test_error_handling_and_edge_cases(self, test_output_dir: Path):
        """Тест 7: Обработка ошибок и граничных случаев."""
        print("\n⚠️ Тест 7: Обработка ошибок и граничных случаев")

        error_test_dir = test_output_dir / "error_handling"
        error_test_dir.mkdir(parents=True, exist_ok=True)

        # 1. Тест обработки ошибок в графиках
        print("⚠️ Тестирование ошибок в графиках")

        plotter = PerformancePlotter()

        # Пустые данные
        with pytest.raises(ValueError, match="пустые"):
            empty_data = pd.DataFrame()
            plotter.plot_reward_curve(data=empty_data)

        # Отсутствующие колонки
        with pytest.raises(ValueError, match="не найдена"):
            invalid_data = {
                "nonexistent_metric": pd.DataFrame({"x": [1, 2], "y": [1, 2]})
            }
            plotter.plot_reward_curve(data=invalid_data, y_col="missing_column")

        print("✅ Ошибки в графиках обрабатываются корректно")

        # 2. Тест обработки ошибок в демо видео
        print("⚠️ Тестирование ошибок в демо видео")

        # Некорректный агент
        with pytest.raises(AgentDemoError):
            with patch(
                "src.visualization.agent_demo.record_agent_episode"
            ) as mock_record:
                mock_record.side_effect = Exception("Ошибка записи видео")

                create_best_episode_demo(
                    agent=MockAgent("ErrorAgent"),
                    env="InvalidEnv",
                    output_path=error_test_dir / "error_demo.mp4",
                    config=DemoConfig(continue_on_error=False),
                )

        print("✅ Ошибки в демо видео обрабатываются корректно")

        # 3. Тест обработки ошибок в оценке
        print("⚠️ Тестирование ошибок в оценке")

        mock_env = MockEnvironment()
        evaluator = QuantitativeEvaluator(env=mock_env)

        # Слишком мало эпизодов
        with pytest.raises(ValueError, match="должно быть >= 5"):
            evaluator.evaluate_agent_quantitative(
                agent=MockAgent("TestAgent"),
                num_episodes=3,  # Меньше минимума
            )

        # Отсутствующие данные для сравнения
        with pytest.raises(ValueError, match="Необходимо предоставить"):
            evaluator.compare_with_baseline(
                agent=MockAgent("TestAgent"), baseline_agent=None, baseline_metrics=None
            )

        print("✅ Ошибки в оценке обрабатываются корректно")

        # 4. Тест обработки ошибок в отчетах
        print("⚠️ Тестирование ошибок в отчетах")

        formatter = ResultsFormatter(output_dir=error_test_dir)

        # Неподдерживаемый формат
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            formatter.generate_comprehensive_report(
                metrics={}, format_type="unsupported_format"
            )

        print("✅ Ошибки в отчетах обрабатываются корректно")

        # 5. Тест граничных случаев
        print("⚠️ Тестирование граничных случаев")

        # Минимальные данные
        minimal_data = pd.DataFrame(
            {
                "timestep": [0, 1],
                "episode": [0, 1],
                "value": [0.0, 1.0],
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="1min"),
            }
        )

        minimal_plot_path = plotter.plot_reward_curve(
            data={"episode_reward": minimal_data},
            save_path=error_test_dir / "minimal_plot.png",
            smooth_window=1,  # Минимальное окно
        )

        assert Path(minimal_plot_path).exists()
        print("✅ Минимальные данные обрабатываются корректно")

        # Одинаковые значения
        constant_data = pd.DataFrame(
            {
                "timestep": range(10),
                "episode": range(10),
                "value": [100.0] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            }
        )

        constant_plot_path = plotter.plot_reward_curve(
            data={"episode_reward": constant_data},
            save_path=error_test_dir / "constant_plot.png",
        )

        assert Path(constant_plot_path).exists()
        print("✅ Константные данные обрабатываются корректно")

        print("🎉 Обработка ошибок и граничных случаев работает корректно")

    def test_performance_and_timing(
        self,
        trained_agents: Dict[str, MockAgent],
        training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path,
    ):
        """Тест 8: Измерение производительности и времени выполнения."""
        print("\n⏱️ Тест 8: Производительность и время выполнения")

        performance_dir = test_output_dir / "performance"
        performance_dir.mkdir(parents=True, exist_ok=True)

        performance_metrics = {}

        # 1. Измерение времени генерации графиков
        print("⏱️ Измерение производительности графиков")

        start_time = time.time()

        plotter = PerformancePlotter()
        plots_created = 0

        # Создаем несколько графиков
        for i in range(3):
            plot_path = plotter.plot_reward_curve(
                data=training_data, save_path=performance_dir / f"perf_plot_{i}.png"
            )
            plots_created += 1

        plots_time = time.time() - start_time
        performance_metrics["plots"] = {
            "total_time": plots_time,
            "plots_created": plots_created,
            "time_per_plot": plots_time / plots_created,
            "plots_per_second": plots_created / plots_time,
        }

        print(
            f"✅ Графики: {plots_created} за {plots_time:.2f}с "
            f"({performance_metrics['plots']['time_per_plot']:.2f}с/график)"
        )

        # 2. Измерение времени оценки агентов
        print("⏱️ Измерение производительности оценки")

        mock_env = MockEnvironment()
        evaluator = QuantitativeEvaluator(env=mock_env)

        start_time = time.time()
        agents_evaluated = 0

        with patch.object(evaluator.evaluator, "evaluate_agent") as mock_eval:

            def quick_evaluation(agent, num_episodes, **kwargs):
                # Быстрая симуляция для измерения производительности
                rewards = [agent.simulate_episode_reward() for _ in range(num_episodes)]
                lengths = [200] * num_episodes
                successes = [True] * num_episodes

                from src.evaluation.evaluator import EvaluationMetrics

                return EvaluationMetrics(
                    mean_reward=float(np.mean(rewards)),
                    std_reward=float(np.std(rewards)),
                    min_reward=float(np.min(rewards)),
                    max_reward=float(np.max(rewards)),
                    mean_episode_length=200.0,
                    std_episode_length=0.0,
                    min_episode_length=200,
                    max_episode_length=200,
                    success_rate=1.0,
                    total_episodes=num_episodes,
                    total_timesteps=num_episodes * 200,
                    evaluation_time=0.1,
                    episode_rewards=rewards,
                    episode_lengths=lengths,
                    episode_successes=successes,
                    reward_ci_lower=float(np.mean(rewards) - 10),
                    reward_ci_upper=float(np.mean(rewards) + 10),
                )

            mock_eval.side_effect = quick_evaluation

            for name, agent in trained_agents.items():
                evaluator.evaluate_agent_quantitative(
                    agent=agent, num_episodes=10, agent_name=name
                )
                agents_evaluated += 1

        evaluation_time = time.time() - start_time
        performance_metrics["evaluation"] = {
            "total_time": evaluation_time,
            "agents_evaluated": agents_evaluated,
            "time_per_agent": evaluation_time / agents_evaluated,
            "agents_per_second": agents_evaluated / evaluation_time,
        }

        print(
            f"✅ Оценка: {agents_evaluated} агентов за {evaluation_time:.2f}с "
            f"({performance_metrics['evaluation']['time_per_agent']:.2f}с/агент)"
        )

        # 3. Измерение времени создания отчетов
        print("⏱️ Измерение производительности отчетов")

        formatter = ResultsFormatter(output_dir=performance_dir)

        # Создаем тестовые данные
        from src.evaluation.evaluator import EvaluationMetrics

        test_metrics = EvaluationMetrics(
            mean_reward=150.0,
            std_reward=25.0,
            min_reward=100.0,
            max_reward=200.0,
            mean_episode_length=200.0,
            std_episode_length=20.0,
            min_episode_length=150,
            max_episode_length=250,
            success_rate=0.8,
            total_episodes=20,
            total_timesteps=4000,
            evaluation_time=30.0,
            episode_rewards=[150.0] * 20,
            episode_lengths=[200] * 20,
            episode_successes=[True] * 16 + [False] * 4,
            reward_ci_lower=140.0,
            reward_ci_upper=160.0,
        )

        start_time = time.time()
        reports_created = 0

        # Создаем отчеты в разных форматах
        for format_type in ["html", "json", "csv"]:
            try:
                if format_type == "csv":
                    formatter.export_to_csv(
                        data=test_metrics, filename=f"perf_report_{format_type}"
                    )
                elif format_type == "json":
                    formatter.export_to_json(
                        data={"test_metrics": test_metrics},
                        filename=f"perf_report_{format_type}",
                    )
                else:
                    formatter.format_single_agent_report(
                        agent_name="TestAgent",
                        evaluation_results=test_metrics,
                        output_format=format_type,
                        filename=f"perf_report_{format_type}",
                    )
                reports_created += 1
            except Exception as e:
                print(f"⚠️ Ошибка создания отчета {format_type}: {e}")

        reports_time = time.time() - start_time
        performance_metrics["reports"] = {
            "total_time": reports_time,
            "reports_created": reports_created,
            "time_per_report": reports_time / reports_created
            if reports_created > 0
            else 0,
            "reports_per_second": reports_created / reports_time
            if reports_time > 0
            else 0,
        }

        print(
            f"✅ Отчеты: {reports_created} за {reports_time:.2f}с "
            f"({performance_metrics['reports']['time_per_report']:.2f}с/отчет)"
        )

        # 4. Общая производительность
        total_time = sum(m["total_time"] for m in performance_metrics.values())
        total_operations = (
            performance_metrics["plots"]["plots_created"]
            + performance_metrics["evaluation"]["agents_evaluated"]
            + performance_metrics["reports"]["reports_created"]
        )

        performance_metrics["overall"] = {
            "total_time": total_time,
            "total_operations": total_operations,
            "operations_per_second": total_operations / total_time
            if total_time > 0
            else 0,
        }

        print(
            f"✅ Общая производительность: {total_operations} операций за {total_time:.2f}с "
            f"({performance_metrics['overall']['operations_per_second']:.2f} оп/с)"
        )

        # 5. Сохранение метрик производительности
        performance_report_path = performance_dir / "performance_metrics.json"
        with open(performance_report_path, "w", encoding="utf-8") as f:
            json.dump(performance_metrics, f, indent=2)

        print(f"✅ Метрики производительности сохранены: {performance_report_path}")

        # 6. Проверка производительности (базовые требования)
        assert (
            performance_metrics["plots"]["time_per_plot"] < 5.0
        )  # Не более 5 секунд на график
        assert (
            performance_metrics["evaluation"]["time_per_agent"] < 2.0
        )  # Не более 2 секунд на агента
        assert (
            performance_metrics["reports"]["time_per_report"] < 3.0
        )  # Не более 3 секунд на отчет

        print("🎉 Производительность соответствует требованиям")

        return performance_metrics

    @pytest.mark.integration
    def test_full_output_generation_workflow(
        self,
        trained_agents: Dict[str, MockAgent],
        training_data: Dict[str, pd.DataFrame],
        test_output_dir: Path,
    ):
        """Тест 9: Полный workflow генерации выходных данных."""
        print("\n🚀 Тест 9: Полный workflow генерации выходных данных")

        workflow_dir = test_output_dir / "full_workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)

        workflow_results = {
            "start_time": time.time(),
            "steps_completed": [],
            "files_created": [],
            "errors_encountered": [],
            "success": True,
        }

        try:
            # Шаг 1: Подготовка данных обучения
            print("🔧 Шаг 1: Подготовка данных обучения")

            data_dir = workflow_dir / "training_data"
            data_dir.mkdir(exist_ok=True)

            for metric_name, data in training_data.items():
                data_path = data_dir / f"{metric_name}.csv"
                data.to_csv(data_path, index=False)
                workflow_results["files_created"].append(str(data_path))

            workflow_results["steps_completed"].append("data_preparation")
            print("✅ Данные обучения подготовлены")

            # Шаг 2: Генерация графиков производительности
            print("📊 Шаг 2: Генерация графиков производительности")

            plots_dir = workflow_dir / "plots"
            performance_report_dir = create_performance_report(
                data=training_data,
                output_dir=plots_dir,
                include_interactive=True,
                include_static=True,
            )

            # Подсчитываем созданные графики
            static_plots = list(Path(performance_report_dir).glob("static/*.png"))
            interactive_plots = list(
                Path(performance_report_dir).glob("interactive/*.html")
            )

            workflow_results["files_created"].extend([str(p) for p in static_plots])
            workflow_results["files_created"].extend(
                [str(p) for p in interactive_plots]
            )
            workflow_results["steps_completed"].append("performance_plots")

            print(
                f"✅ Создано графиков: {len(static_plots)} статических, {len(interactive_plots)} интерактивных"
            )

            # Шаг 3: Создание демонстрационных видео
            print("🎬 Шаг 3: Создание демонстрационных видео")

            videos_dir = workflow_dir / "videos"
            videos_dir.mkdir(exist_ok=True)

            # Мокаем создание видео для ускорения тестов
            with patch(
                "src.visualization.agent_demo.create_best_episode_demo"
            ) as mock_demo:

                def mock_demo_creation(agent, env, output_path, config, **kwargs):
                    # Создаем фиктивный видеофайл
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_text("mock video content")

                    return {
                        "success": True,
                        "demo_type": "best_episode",
                        "agent_name": agent.name,
                        "best_reward": agent.simulate_episode_reward(),
                        "output_path": str(output_path),
                    }

                mock_demo.side_effect = mock_demo_creation

                agents_list = [(name, agent) for name, agent in trained_agents.items()]
                batch_result = create_batch_demos(
                    agents=agents_list,
                    env="LunarLander-v2",
                    output_dir=videos_dir,
                    demo_types=["best_episode"],
                    config=DemoConfig(auto_compress=False),
                )

            # Подсчитываем созданные видео
            created_videos = list(videos_dir.rglob("*.mp4"))
            workflow_results["files_created"].extend([str(v) for v in created_videos])
            workflow_results["steps_completed"].append("demo_videos")

            print(f"✅ Создано видео: {len(created_videos)}")

            # Шаг 4: Количественная оценка агентов
            print("📈 Шаг 4: Количественная оценка агентов")

            eval_dir = workflow_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)

            mock_env = MockEnvironment()
            evaluator = QuantitativeEvaluator(env=mock_env)

            # Мокаем оценку для ускорения
            with patch.object(evaluator.evaluator, "evaluate_agent") as mock_eval:

                def mock_evaluation(agent, num_episodes, **kwargs):
                    rewards = [
                        agent.simulate_episode_reward() for _ in range(num_episodes)
                    ]
                    lengths = [np.random.randint(150, 250) for _ in range(num_episodes)]
                    successes = [r > 0 for r in rewards]

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
                        total_timesteps=sum(lengths),
                        evaluation_time=num_episodes * 0.1,
                        episode_rewards=rewards,
                        episode_lengths=lengths,
                        episode_successes=successes,
                        reward_ci_lower=float(
                            np.mean(rewards)
                            - 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                        ),
                        reward_ci_upper=float(
                            np.mean(rewards)
                            + 1.96 * np.std(rewards) / np.sqrt(num_episodes)
                        ),
                    )

                mock_eval.side_effect = mock_evaluation

                # Пакетная оценка агентов
                batch_evaluation = evaluator.evaluate_multiple_agents_batch(
                    agents=trained_agents,
                    num_episodes=15,
                    include_pairwise_comparison=True,
                )

                # Создание отчетов оценки
                for format_type in ["text", "json", "csv"]:
                    report_path = eval_dir / f"evaluation_report.{format_type}"
                    evaluator.generate_comprehensive_report(
                        metrics=batch_evaluation,
                        save_path=report_path,
                        format_type=format_type,
                    )
                    workflow_results["files_created"].append(str(report_path))

            workflow_results["steps_completed"].append("quantitative_evaluation")
            print(f"✅ Оценка завершена, лучший агент: {batch_evaluation.best_agent}")

            # Шаг 5: Форматирование итогового отчета
            print("📝 Шаг 5: Форматирование итогового отчета")

            reports_dir = workflow_dir / "reports"
            reports_dir.mkdir(exist_ok=True)

            formatter = ResultsFormatter(output_dir=reports_dir)

            # Создаем комплексный отчет
            experiment_data = {
                "experiment_name": "Full Output Generation Workflow Test",
                "hypothesis": "Полный пайплайн генерации выходов работает корректно",
                "training_data_metrics": {
                    "datasets": len(training_data),
                    "total_records": sum(len(data) for data in training_data.values()),
                },
                "evaluation_results": {
                    "agents_evaluated": len(trained_agents),
                    "best_agent": batch_evaluation.best_agent,
                    "evaluation_time": batch_evaluation.statistical_summary[
                        "evaluation_time"
                    ],
                },
                "output_files": {
                    "plots_created": len(static_plots) + len(interactive_plots),
                    "videos_created": len(created_videos),
                    "reports_created": 3,  # text, json, csv
                },
                "conclusion": "Пайплайн выполнен успешно",
            }

            # Создаем отчеты в разных форматах
            for format_type in ["html", "markdown"]:
                report_path = formatter.format_experiment_report(
                    experiment_name="Full_Workflow_Test",
                    experiment_data=experiment_data,
                    output_format=format_type,
                    filename="full_workflow_report",
                )
                workflow_results["files_created"].append(str(report_path))

            # Экспорт сводных данных
            summary_data = {
                "workflow_results": workflow_results,
                "experiment_data": experiment_data,
                "batch_evaluation": {
                    "best_agent": batch_evaluation.best_agent,
                    "ranking": batch_evaluation.ranking,
                    "statistical_summary": batch_evaluation.statistical_summary,
                },
            }

            json_export = formatter.export_to_json(
                data=summary_data, filename="workflow_summary"
            )
            workflow_results["files_created"].append(str(json_export))

            workflow_results["steps_completed"].append("results_formatting")
            print("✅ Итоговые отчеты созданы")

            # Шаг 6: Проверка всех созданных файлов
            print("🔍 Шаг 6: Проверка созданных файлов")

            files_verification = {
                "total_files": len(workflow_results["files_created"]),
                "existing_files": 0,
                "missing_files": [],
                "file_sizes": {},
            }

            for file_path in workflow_results["files_created"]:
                path_obj = Path(file_path)
                if path_obj.exists():
                    files_verification["existing_files"] += 1
                    files_verification["file_sizes"][str(path_obj)] = (
                        path_obj.stat().st_size
                    )
                else:
                    files_verification["missing_files"].append(str(path_obj))

            workflow_results["files_verification"] = files_verification
            workflow_results["steps_completed"].append("files_verification")

            print(
                f"✅ Проверка файлов: {files_verification['existing_files']}/{files_verification['total_files']} существуют"
            )

            if files_verification["missing_files"]:
                print(f"⚠️ Отсутствующие файлы: {files_verification['missing_files']}")

        except Exception as e:
            workflow_results["success"] = False
            workflow_results["errors_encountered"].append(str(e))
            print(f"❌ Ошибка в workflow: {e}")
            raise

        finally:
            workflow_results["end_time"] = time.time()
            workflow_results["total_time"] = (
                workflow_results["end_time"] - workflow_results["start_time"]
            )

            # Сохранение результатов workflow
            workflow_summary_path = workflow_dir / "workflow_summary.json"
            with open(workflow_summary_path, "w", encoding="utf-8") as f:
                json.dump(workflow_results, f, indent=2, default=str)

        # Финальные проверки
        assert workflow_results["success"], "Workflow должен завершиться успешно"
        assert len(workflow_results["steps_completed"]) == 6, (
            "Все шаги должны быть выполнены"
        )
        assert len(workflow_results["errors_encountered"]) == 0, "Не должно быть ошибок"
        assert workflow_results["files_verification"]["existing_files"] > 0, (
            "Должны быть созданы файлы"
        )

        print(
            f"\n🎉 Полный workflow завершен успешно за {workflow_results['total_time']:.2f}с"
        )
        print(f"📊 Выполнено шагов: {len(workflow_results['steps_completed'])}")
        print(
            f"📁 Создано файлов: {workflow_results['files_verification']['total_files']}"
        )
        print(
            f"✅ Существующих файлов: {workflow_results['files_verification']['existing_files']}"
        )

        return workflow_results


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not integration"])
