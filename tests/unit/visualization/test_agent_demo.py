"""Тесты для модуля демонстрации агентов.

Проверяет функциональность создания демонстрационных видео обученных RL агентов
с различными режимами и настройками.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.agents.base import Agent, AgentConfig
from src.visualization.agent_demo import (
    AgentDemoError,
    DemoConfig,
    auto_demo_from_training_results,
    create_average_behavior_demo,
    create_batch_demos,
    create_before_after_demo,
    create_best_episode_demo,
    create_multi_agent_comparison,
    create_training_progress_demo,
    generate_demo_summary,
    quick_comparison,
    quick_demo,
)
from src.visualization.video_generator import VideoConfig


class MockAgent(Agent):
    """Мок-агент для тестирования."""
    
    def __init__(self, name: str = "MockAgent", reward_range: tuple = (0, 100)):
        """Инициализация мок-агента."""
        config = AgentConfig(algorithm="Mock", env_name="CartPole-v1")
        # Мокаем все зависимости
        with patch('src.agents.base.get_experiment_logger'), \
             patch('src.agents.base.get_metrics_tracker'), \
             patch('gymnasium.make'):
            super().__init__(config)
        
        self.name = name
        self.reward_range = reward_range
        self.is_trained = True
        self.model = Mock()
        
    def _create_model(self):
        """Создать мок-модель."""
        return Mock()
        
    def train(self, total_timesteps=None, callback=None, **kwargs):
        """Мок-обучение."""
        from src.agents.base import TrainingResult
        return TrainingResult(
            total_timesteps=total_timesteps or 1000,
            training_time=1.0,
            final_mean_reward=100.0,
            final_std_reward=10.0,
        )
        
    def predict(self, observation, deterministic=True, **kwargs):
        """Мок-предсказание."""
        # Возвращаем случайное действие как numpy array
        action = np.array([np.random.randint(0, 2)])
        return action, None
        
    @classmethod
    def load(cls, path, env=None, **kwargs):
        """Мок-загрузка."""
        return cls(name=f"Loaded_{Path(path).stem}")


@pytest.fixture
def mock_env():
    """Мок-среда для тестирования."""
    env = Mock()
    env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
    env.step.return_value = (
        np.array([0.2, 0.3, 0.4, 0.5]),  # obs
        1.0,  # reward
        False,  # done
        False,  # truncated
        {}  # info
    )
    env.render.return_value = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    env.close = Mock()
    return env


@pytest.fixture
def demo_config():
    """Конфигурация демонстрации для тестов."""
    return DemoConfig(
        video_config=VideoConfig(fps=10, format="mp4", quality="low"),
        num_episodes=2,
        max_episode_length=10,
        auto_compress=False,  # Отключаем сжатие для тестов
        continue_on_error=True,
    )


@pytest.fixture
def temp_output_dir():
    """Временная директория для выходных файлов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestDemoConfig:
    """Тесты конфигурации демонстрации."""
    
    def test_default_config(self):
        """Тест создания конфигурации по умолчанию."""
        config = DemoConfig()
        
        assert config.demo_type == "best_episode"
        assert config.num_episodes == 5
        assert config.auto_naming is True
        assert config.auto_compress is True
        assert isinstance(config.video_config, VideoConfig)
    
    def test_custom_config(self):
        """Тест создания кастомной конфигурации."""
        video_config = VideoConfig(fps=60, quality="high")
        config = DemoConfig(
            video_config=video_config,
            demo_type="average",
            num_episodes=10,
            auto_compress=False,
        )
        
        assert config.demo_type == "average"
        assert config.num_episodes == 10
        assert config.auto_compress is False
        assert config.video_config.fps == 60


class TestBestEpisodeDemo:
    """Тесты создания демо лучшего эпизода."""
    
    @patch('src.visualization.agent_demo.record_agent_episode')
    @patch('src.visualization.agent_demo.setup_recording_environment')
    def test_create_best_episode_demo_success(
        self, 
        mock_setup_env, 
        mock_record_episode,
        mock_env,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного создания демо лучшего эпизода."""
        # Настройка моков
        mock_setup_env.return_value = mock_env
        mock_record_episode.return_value = {
            "total_reward": 95.0,
            "episode_length": 100,
            "success": True,
            "output_path": str(temp_output_dir / "demo.mp4"),
        }
        
        agent = MockAgent(reward_range=(80, 100))
        output_path = temp_output_dir / "best_demo.mp4"
        
        # Мокаем поиск лучшего эпизода
        with patch('numpy.random.randint', side_effect=[42, 43, 44]), \
             patch.object(agent, 'predict', return_value=(1, None)):
            
            # Мокаем эпизоды с разными наградами
            episode_rewards = [85.0, 95.0, 90.0]  # Лучший - второй
            mock_env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
            
            def step_side_effect(*args):
                reward = episode_rewards[mock_env.step.call_count % len(episode_rewards)]
                return (
                    np.array([0.2, 0.3, 0.4, 0.5]),
                    reward,
                    True,  # done после одного шага для быстроты
                    False,
                    {}
                )
            
            mock_env.step.side_effect = step_side_effect
            
            result = create_best_episode_demo(
                agent=agent,
                env="CartPole-v1",
                output_path=output_path,
                config=demo_config,
                num_candidates=3,
            )
        
        # Проверки
        assert result["success"] is True
        assert result["demo_type"] == "best_episode"
        assert result["best_reward"] == 95.0
        assert result["candidates_tested"] == 3
        assert "reward_statistics" in result
        
        # Проверяем, что record_agent_episode был вызван
        mock_record_episode.assert_called_once()
        call_args = mock_record_episode.call_args
        # Проверяем, что был передан один из сидов
        assert call_args[1]["episode_seed"] in [42, 43, 44]
    
    @patch('src.visualization.agent_demo.setup_recording_environment')
    def test_create_best_episode_demo_no_episodes(
        self, 
        mock_setup_env,
        mock_env,
        demo_config,
        temp_output_dir
    ):
        """Тест обработки случая, когда не удалось найти подходящий эпизод."""
        mock_setup_env.return_value = mock_env
        mock_env.step.side_effect = Exception("Ошибка среды")
        
        agent = MockAgent()
        output_path = temp_output_dir / "best_demo.mp4"
        
        with pytest.raises(AgentDemoError, match="Не удалось создать демо"):
            create_best_episode_demo(
                agent=agent,
                env="CartPole-v1",
                output_path=output_path,
                config=demo_config,
                num_candidates=1,
            )
    
    def test_create_best_episode_demo_auto_naming(
        self, 
        mock_env,
        temp_output_dir
    ):
        """Тест автоматического именования файлов."""
        config = DemoConfig(
            auto_naming=True,
            include_timestamp=False,
            title_prefix="Test_Demo",
            auto_compress=False,  # Отключаем сжатие для теста
        )
        
        agent = MockAgent(name="TestAgent")
        output_path = temp_output_dir / "demo.mp4"
        
        with patch('src.visualization.agent_demo.record_agent_episode') as mock_record, \
             patch('src.visualization.agent_demo.setup_recording_environment') as mock_setup, \
             patch.object(agent, 'predict', return_value=(1, None)):
            
            mock_setup.return_value = mock_env
            mock_record.return_value = {"success": True, "total_reward": 100.0}
            
            # Мокаем успешный эпизод
            mock_env.step.return_value = (
                np.array([0.1, 0.2, 0.3, 0.4]), 100.0, True, False, {}
            )
            
            result = create_best_episode_demo(
                agent=agent,
                env=mock_env,
                output_path=output_path,
                config=config,
                num_candidates=1,
            )
        
        # Проверяем, что имя файла было изменено
        output_path_str = result["output_path"]
        assert "Test_Demo_TestAgent_best" in output_path_str


class TestAverageBehaviorDemo:
    """Тесты создания демо среднего поведения."""
    
    @patch('src.visualization.agent_demo.record_multiple_episodes')
    @patch('shutil.copy2')
    @patch('shutil.rmtree')
    def test_create_average_behavior_demo_success(
        self,
        mock_rmtree,
        mock_copy,
        mock_record_multiple,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного создания демо среднего поведения."""
        # Мокаем результаты записи эпизодов
        episodes_info = [
            {"success": True, "total_reward": 80.0, "output_path": "/tmp/ep1.mp4"},
            {"success": True, "total_reward": 90.0, "output_path": "/tmp/ep2.mp4"},
            {"success": True, "total_reward": 85.0, "output_path": "/tmp/ep3.mp4"},
        ]
        mock_record_multiple.return_value = episodes_info
        
        # Создаем временную директорию для теста
        temp_episodes_dir = temp_output_dir / "temp_episodes"
        temp_episodes_dir.mkdir()
        
        agent = MockAgent()
        output_path = temp_output_dir / "average_demo.mp4"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = create_average_behavior_demo(
                agent=agent,
                env="CartPole-v1",
                output_path=output_path,
                config=demo_config,
            )
        
        # Проверки
        assert result["success"] is True
        assert result["demo_type"] == "average_behavior"
        assert result["selected_reward"] == 85.0  # Ближайший к среднему (85.0)
        assert result["mean_reward"] == 85.0
        assert result["episodes_analyzed"] == 3
        
        # Проверяем, что файл был скопирован
        mock_copy.assert_called_once()
        
        # Проверяем, что временная директория была удалена
        mock_rmtree.assert_called_once()
    
    @patch('src.visualization.agent_demo.record_multiple_episodes')
    def test_create_average_behavior_demo_no_successful_episodes(
        self,
        mock_record_multiple,
        demo_config,
        temp_output_dir
    ):
        """Тест обработки случая без успешных эпизодов."""
        # Все эпизоды неудачные
        episodes_info = [
            {"success": False, "error": "Ошибка 1"},
            {"success": False, "error": "Ошибка 2"},
        ]
        mock_record_multiple.return_value = episodes_info
        
        agent = MockAgent()
        output_path = temp_output_dir / "average_demo.mp4"
        
        with pytest.raises(AgentDemoError, match="Не удалось записать ни одного успешного эпизода"):
            create_average_behavior_demo(
                agent=agent,
                env="CartPole-v1",
                output_path=output_path,
                config=demo_config,
            )


class TestBeforeAfterDemo:
    """Тесты создания демо сравнения до/после обучения."""
    
    @patch('src.visualization.agent_demo.generate_comparison_video')
    def test_create_before_after_demo_success(
        self,
        mock_generate_comparison,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного создания демо сравнения."""
        mock_generate_comparison.return_value = {
            "agents": [
                {"name": "До обучения", "total_reward": 20.0, "steps": 50},
                {"name": "После обучения", "total_reward": 90.0, "steps": 200},
            ],
            "total_frames": 300,
            "success": True,
        }
        
        untrained_agent = MockAgent(name="Untrained")
        trained_agent = MockAgent(name="Trained")
        output_path = temp_output_dir / "before_after.mp4"
        
        result = create_before_after_demo(
            untrained_agent=untrained_agent,
            trained_agent=trained_agent,
            env="CartPole-v1",
            output_path=output_path,
            config=demo_config,
        )
        
        # Проверки
        assert result["success"] is True
        assert result["demo_type"] == "before_after"
        assert result["agent_name"] == "Trained"
        assert "comparison_info" in result
        
        # Проверяем, что generate_comparison_video был вызван с правильными агентами
        mock_generate_comparison.assert_called_once()
        call_args = mock_generate_comparison.call_args[1]
        agents = call_args["agents"]
        assert len(agents) == 2
        assert agents[0][0] == "До обучения"
        assert agents[1][0] == "После обучения"


class TestTrainingProgressDemo:
    """Тесты создания демо прогресса обучения."""
    
    @patch('src.visualization.agent_demo.create_training_montage')
    def test_create_training_progress_demo_success(
        self,
        mock_create_montage,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного создания демо прогресса обучения."""
        mock_create_montage.return_value = {
            "checkpoints": [
                {"name": "Чекпоинт 1", "frames": 100},
                {"name": "Чекпоинт 2", "frames": 120},
            ],
            "total_frames": 220,
            "success": True,
        }
        
        # Создаем временные файлы чекпоинтов
        checkpoint_paths = [
            temp_output_dir / "checkpoint_1.zip",
            temp_output_dir / "checkpoint_2.zip",
        ]
        
        for path in checkpoint_paths:
            path.touch()  # Создаем пустые файлы
        
        with patch.object(MockAgent, 'load', side_effect=[
            MockAgent(name="Agent_1"),
            MockAgent(name="Agent_2"),
        ]):
            result = create_training_progress_demo(
                checkpoint_paths=checkpoint_paths,
                agent_class=MockAgent,
                env="CartPole-v1",
                output_path=temp_output_dir / "progress.mp4",
                config=demo_config,
                checkpoint_names=["Начало", "Конец"],
            )
        
        # Проверки
        assert result["success"] is True
        assert result["demo_type"] == "training_progress"
        assert result["checkpoints_loaded"] == 2
        assert result["checkpoint_names"] == ["Начало", "Конец"]
        assert "montage_info" in result
    
    def test_create_training_progress_demo_load_error(
        self,
        demo_config,
        temp_output_dir
    ):
        """Тест обработки ошибок загрузки чекпоинтов."""
        checkpoint_paths = [temp_output_dir / "nonexistent.zip"]
        
        # Настраиваем config для остановки при ошибке
        demo_config.continue_on_error = False
        
        with patch.object(MockAgent, 'load', side_effect=FileNotFoundError("Файл не найден")):
            with pytest.raises(AgentDemoError):
                create_training_progress_demo(
                    checkpoint_paths=checkpoint_paths,
                    agent_class=MockAgent,
                    env="CartPole-v1",
                    output_path=temp_output_dir / "progress.mp4",
                    config=demo_config,
                )


class TestMultiAgentComparison:
    """Тесты создания сравнения нескольких агентов."""
    
    @patch('src.visualization.agent_demo.generate_comparison_video')
    def test_create_multi_agent_comparison_success(
        self,
        mock_generate_comparison,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного создания сравнения агентов."""
        mock_generate_comparison.return_value = {
            "agents": [
                {"name": "Agent1", "total_reward": 80.0, "steps": 100},
                {"name": "Agent2", "total_reward": 90.0, "steps": 120},
                {"name": "Agent3", "total_reward": 85.0, "steps": 110},
            ],
            "total_frames": 400,
            "success": True,
        }
        
        agents = [
            ("Agent1", MockAgent(name="Agent1")),
            ("Agent2", MockAgent(name="Agent2")),
            ("Agent3", MockAgent(name="Agent3")),
        ]
        
        result = create_multi_agent_comparison(
            agents=agents,
            env="CartPole-v1",
            output_path=temp_output_dir / "comparison.mp4",
            config=demo_config,
        )
        
        # Проверки
        assert result["success"] is True
        assert result["demo_type"] == "multi_agent_comparison"
        assert result["agents_compared"] == 3
        assert result["agent_names"] == ["Agent1", "Agent2", "Agent3"]


class TestBatchDemos:
    """Тесты пакетного создания демонстраций."""
    
    @patch('src.visualization.agent_demo.create_best_episode_demo')
    @patch('src.visualization.agent_demo.create_average_behavior_demo')
    def test_create_batch_demos_success(
        self,
        mock_create_average,
        mock_create_best,
        demo_config,
        temp_output_dir
    ):
        """Тест успешного пакетного создания демо."""
        # Мокаем успешные результаты
        mock_create_best.return_value = {"success": True, "demo_type": "best_episode"}
        mock_create_average.return_value = {"success": True, "demo_type": "average"}
        
        agents = [
            ("Agent1", MockAgent(name="Agent1")),
            ("Agent2", MockAgent(name="Agent2")),
        ]
        
        demo_types = ["best_episode", "average"]
        
        result = create_batch_demos(
            agents=agents,
            env="CartPole-v1",
            output_dir=temp_output_dir,
            demo_types=demo_types,
            config=demo_config,
        )
        
        # Проверки
        assert result["success"] is True
        assert result["agents_processed"] == 2
        assert result["demos_created"] == 4  # 2 агента × 2 типа демо
        assert result["demos_failed"] == 0
        assert len(result["demo_results"]) == 2
        
        # Проверяем, что функции создания демо были вызваны
        assert mock_create_best.call_count == 2
        assert mock_create_average.call_count == 2
    
    @patch('src.visualization.agent_demo.create_best_episode_demo')
    def test_create_batch_demos_with_errors(
        self,
        mock_create_best,
        demo_config,
        temp_output_dir
    ):
        """Тест пакетного создания с ошибками."""
        # Первый агент успешно, второй с ошибкой
        mock_create_best.side_effect = [
            {"success": True, "demo_type": "best_episode"},
            AgentDemoError("Ошибка создания демо"),
        ]
        
        agents = [
            ("Agent1", MockAgent(name="Agent1")),
            ("Agent2", MockAgent(name="Agent2")),
        ]
        
        result = create_batch_demos(
            agents=agents,
            env="CartPole-v1",
            output_dir=temp_output_dir,
            demo_types=["best_episode"],
            config=demo_config,
        )
        
        # Проверки
        assert result["success"] is True  # continue_on_error=True
        assert result["agents_processed"] == 2
        assert result["demos_created"] == 1
        assert result["demos_failed"] == 1


class TestUtilityFunctions:
    """Тесты вспомогательных функций."""
    
    def test_generate_demo_summary(self, temp_output_dir):
        """Тест генерации сводного отчета."""
        demo_results = [
            {"demo_type": "best_episode", "success": True, "agent_name": "Agent1"},
            {"demo_type": "average", "success": True, "agent_name": "Agent2"},
            {"demo_type": "best_episode", "success": False, "error": "Ошибка"},
        ]
        
        summary_path = temp_output_dir / "summary.json"
        
        generate_demo_summary(demo_results, summary_path)
        
        # Проверяем, что файл создан
        assert summary_path.exists()
        
        # Проверяем содержимое
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        assert summary["total_demos"] == 3
        assert summary["successful_demos"] == 2
        assert summary["failed_demos"] == 1
        assert "best_episode" in summary["demo_types"]
        assert "average" in summary["demo_types"]
    
    @patch('src.visualization.agent_demo.create_best_episode_demo')
    def test_auto_demo_from_training_results(
        self,
        mock_create_best,
        demo_config,
        temp_output_dir
    ):
        """Тест автоматического создания демо из результатов обучения."""
        # Создаем структуру директории с моделями
        models_dir = temp_output_dir / "models"
        models_dir.mkdir()
        
        model_files = [
            models_dir / "agent1.zip",
            models_dir / "agent2.zip",
        ]
        
        for model_file in model_files:
            model_file.touch()
        
        mock_create_best.return_value = {"success": True, "demo_type": "best_episode"}
        
        with patch.object(MockAgent, 'load', return_value=MockAgent()):
            result = auto_demo_from_training_results(
                training_results_dir=temp_output_dir,
                agent_class=MockAgent,
                env="CartPole-v1",
                output_dir=temp_output_dir / "demos",
                config=demo_config,
            )
        
        # Проверки
        assert result["success"] is True
        assert result["models_found"] == 2
        assert result["demos_created"] == 2
        assert "summary_path" in result
        
        # Проверяем, что сводный отчет создан
        summary_path = Path(result["summary_path"])
        assert summary_path.exists()
    
    @patch('src.visualization.agent_demo.create_best_episode_demo')
    def test_quick_demo(self, mock_create_best):
        """Тест быстрого создания демо."""
        mock_create_best.return_value = {
            "output_path": "/path/to/demo.mp4",
            "compressed_path": "/path/to/demo_compressed.mp4",
        }
        
        agent = MockAgent()
        result_path = quick_demo(
            agent=agent,
            env="CartPole-v1",
            output_path="/path/to/demo.mp4",
            demo_type="best_episode",
        )
        
        assert result_path == "/path/to/demo_compressed.mp4"
        mock_create_best.assert_called_once()
    
    @patch('src.visualization.agent_demo.create_multi_agent_comparison')
    def test_quick_comparison(self, mock_create_comparison):
        """Тест быстрого создания сравнения."""
        mock_create_comparison.return_value = {
            "output_path": "/path/to/comparison.mp4",
            "compressed_path": "/path/to/comparison_compressed.mp4",
        }
        
        agents = [
            ("Agent1", MockAgent(name="Agent1")),
            ("Agent2", MockAgent(name="Agent2")),
        ]
        
        result_path = quick_comparison(
            agents=agents,
            env="CartPole-v1",
            output_path="/path/to/comparison.mp4",
        )
        
        assert result_path == "/path/to/comparison_compressed.mp4"
        mock_create_comparison.assert_called_once()


class TestErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_invalid_demo_type_quick_demo(self):
        """Тест обработки неверного типа демо в quick_demo."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="Неподдерживаемый тип демо"):
            quick_demo(
                agent=agent,
                env="CartPole-v1",
                output_path="/path/to/demo.mp4",
                demo_type="invalid_type",
            )
    
    @patch('src.visualization.agent_demo.setup_recording_environment')
    def test_environment_error(self, mock_setup_env, demo_config, temp_output_dir):
        """Тест обработки ошибок среды."""
        mock_setup_env.side_effect = Exception("Ошибка создания среды")
        
        agent = MockAgent()
        
        with pytest.raises(AgentDemoError, match="Не удалось создать демо"):
            create_best_episode_demo(
                agent=agent,
                env="InvalidEnv",
                output_path=temp_output_dir / "demo.mp4",
                config=demo_config,
            )


if __name__ == "__main__":
    pytest.main([__file__])