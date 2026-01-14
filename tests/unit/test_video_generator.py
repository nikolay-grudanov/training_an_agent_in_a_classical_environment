"""Тесты для модуля генерации видео RL агентов."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import gymnasium as gym

from src.visualization.video_generator import (
    VideoConfig,
    VideoGenerationError,
    setup_recording_environment,
    add_metrics_overlay,
    record_agent_episode,
    record_multiple_episodes,
    create_training_montage,
    generate_comparison_video,
    compress_video,
    _save_video,
    _create_transition_frame,
    _combine_frames_side_by_side,
)


class MockAgent:
    """Мок агента для тестирования."""
    
    def __init__(self, name: str = "TestAgent", actions: list = None):
        self.name = name
        self.actions = actions or [0, 1, 0, 1]  # Циклические действия
        self.action_index = 0
    
    def predict(self, observation):
        """Предсказание действия."""
        action = self.actions[self.action_index % len(self.actions)]
        self.action_index += 1
        return action


class MockEnv:
    """Мок среды для тестирования."""
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)
        
    def reset(self, seed=None):
        """Сброс среды."""
        self.current_step = 0
        return np.array([0.1, 0.2, 0.3, 0.4]), {}
    
    def step(self, action):
        """Шаг в среде."""
        self.current_step += 1
        obs = np.random.random(4)
        reward = np.random.random() - 0.5
        done = self.current_step >= self.max_steps
        truncated = False
        info = {}
        return obs, reward, done, truncated, info
    
    def render(self):
        """Рендеринг кадра."""
        # Создание простого тестового кадра
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return frame
    
    def close(self):
        """Закрытие среды."""
        pass


@pytest.fixture
def video_config():
    """Базовая конфигурация видео для тестов."""
    return VideoConfig(
        fps=10,
        format="mp4",
        quality="medium",
        show_metrics=True,
        max_episode_length=20,
    )


@pytest.fixture
def mock_agent():
    """Мок агента для тестов."""
    return MockAgent("TestPPO", actions=[0, 1, 0, 1, 1])


@pytest.fixture
def mock_env():
    """Мок среды для тестов."""
    return MockEnv(max_steps=5)


@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestVideoConfig:
    """Тесты конфигурации видео."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = VideoConfig()
        
        assert config.fps == 30
        assert config.format == "mp4"
        assert config.quality == "high"
        assert config.show_metrics is True
        assert config.text_color == (255, 255, 255)
        assert config.max_episode_length == 1000
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = VideoConfig(
            fps=15,
            format="gif",
            quality="low",
            show_metrics=False,
            width=640,
            height=480,
        )
        
        assert config.fps == 15
        assert config.format == "gif"
        assert config.quality == "low"
        assert config.show_metrics is False
        assert config.width == 640
        assert config.height == 480


class TestSetupRecordingEnvironment:
    """Тесты подготовки среды для записи."""
    
    @patch('src.visualization.video_generator.gym.make')
    @patch('src.visualization.video_generator.set_seed')
    def test_setup_environment_success(self, mock_set_seed, mock_gym_make):
        """Тест успешной подготовки среды."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env
        
        env = setup_recording_environment("CartPole-v1", seed=42)
        
        mock_gym_make.assert_called_once_with("CartPole-v1", render_mode="rgb_array")
        mock_set_seed.assert_called_once_with(42)
        mock_env.reset.assert_called_once_with(seed=42)
        assert env == mock_env
    
    @patch('src.visualization.video_generator.gym.make')
    def test_setup_environment_no_seed(self, mock_gym_make):
        """Тест подготовки среды без сида."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env
        
        env = setup_recording_environment("CartPole-v1")
        
        mock_gym_make.assert_called_once_with("CartPole-v1", render_mode="rgb_array")
        assert env == mock_env
    
    @patch('src.visualization.video_generator.gym.make')
    def test_setup_environment_failure(self, mock_gym_make):
        """Тест ошибки при создании среды."""
        mock_gym_make.side_effect = Exception("Environment not found")
        
        with pytest.raises(VideoGenerationError, match="Не удалось создать среду"):
            setup_recording_environment("InvalidEnv-v1")


class TestAddMetricsOverlay:
    """Тесты добавления оверлея с метриками."""
    
    def test_add_metrics_overlay(self, video_config):
        """Тест добавления метрик на кадр."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        metrics = {
            "agent_name": "TestAgent",
            "step": 10,
            "reward": 1.5,
            "total_reward": 15.0,
            "action": 1,
        }
        
        result_frame = add_metrics_overlay(frame, metrics, video_config)
        
        assert result_frame.shape == frame.shape
        assert result_frame.dtype == np.uint8
        # Проверяем, что кадр изменился (добавился текст)
        assert not np.array_equal(frame, result_frame)
    
    def test_no_overlay_when_disabled(self, video_config):
        """Тест отсутствия оверлея при отключенных метриках."""
        video_config.show_metrics = False
        video_config.show_episode_info = False
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        metrics = {"reward": 1.0}
        
        result_frame = add_metrics_overlay(frame, metrics, video_config)
        
        # Кадр не должен измениться
        assert np.array_equal(frame, result_frame)
    
    def test_empty_metrics(self, video_config):
        """Тест с пустыми метриками."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        metrics = {}
        
        result_frame = add_metrics_overlay(frame, metrics, video_config)
        
        assert result_frame.shape == frame.shape
        assert result_frame.dtype == np.uint8


class TestRecordAgentEpisode:
    """Тесты записи эпизода агента."""
    
    def test_record_episode_success(self, mock_agent, mock_env, video_config, temp_dir):
        """Тест успешной записи эпизода."""
        output_path = temp_dir / "test_episode.mp4"
        
        with patch('src.visualization.video_generator._save_video') as mock_save:
            result = record_agent_episode(
                agent=mock_agent,
                env=mock_env,
                output_path=output_path,
                config=video_config,
                episode_seed=42,
            )
        
        assert result["success"] is True
        assert result["episode_length"] == 5  # max_steps в mock_env
        assert "total_reward" in result
        assert "frames_recorded" in result
        assert result["agent_name"] == "TestPPO"
        
        mock_save.assert_called_once()
    
    def test_record_episode_with_string_env(self, mock_agent, video_config, temp_dir):
        """Тест записи с названием среды как строкой."""
        output_path = temp_dir / "test_episode.mp4"
        
        with patch('src.visualization.video_generator.setup_recording_environment') as mock_setup:
            mock_setup.return_value = MockEnv(max_steps=3)
            
            with patch('src.visualization.video_generator._save_video'):
                result = record_agent_episode(
                    agent=mock_agent,
                    env="CartPole-v1",
                    output_path=output_path,
                    config=video_config,
                )
        
        mock_setup.assert_called_once_with("CartPole-v1", seed=None)
        assert result["success"] is True
    
    def test_record_episode_max_steps(self, mock_agent, video_config, temp_dir):
        """Тест ограничения максимального количества шагов."""
        mock_env = MockEnv(max_steps=100)  # Большое количество шагов
        output_path = temp_dir / "test_episode.mp4"
        
        with patch('src.visualization.video_generator._save_video'):
            result = record_agent_episode(
                agent=mock_agent,
                env=mock_env,
                output_path=output_path,
                config=video_config,
                max_steps=3,  # Ограничиваем 3 шагами
            )
        
        assert result["episode_length"] <= 3
        assert result["success"] is True
    
    def test_record_episode_no_frames(self, mock_agent, video_config, temp_dir):
        """Тест ошибки при отсутствии кадров."""
        # Мок среды, которая не возвращает кадры
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, True, False, {})
        mock_env.render.return_value = None  # Нет кадров
        
        output_path = temp_dir / "test_episode.mp4"
        
        with pytest.raises(VideoGenerationError, match="Не удалось захватить кадры"):
            record_agent_episode(
                agent=mock_agent,
                env=mock_env,
                output_path=output_path,
                config=video_config,
            )


class TestRecordMultipleEpisodes:
    """Тесты записи нескольких эпизодов."""
    
    def test_record_multiple_episodes_success(self, mock_agent, mock_env, video_config, temp_dir):
        """Тест успешной записи нескольких эпизодов."""
        with patch('src.visualization.video_generator.record_agent_episode') as mock_record:
            mock_record.return_value = {
                "success": True,
                "total_reward": 10.0,
                "episode_length": 5,
            }
            
            results = record_multiple_episodes(
                agent=mock_agent,
                env=mock_env,
                output_dir=temp_dir,
                num_episodes=3,
                config=video_config,
            )
        
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert all(result["episode_number"] == i + 1 for i, result in enumerate(results))
        assert mock_record.call_count == 3
    
    def test_record_multiple_episodes_with_seeds(self, mock_agent, mock_env, video_config, temp_dir):
        """Тест записи с заданными сидами."""
        seeds = [100, 200, 300]
        
        with patch('src.visualization.video_generator.record_agent_episode') as mock_record:
            mock_record.return_value = {"success": True, "total_reward": 5.0}
            
            record_multiple_episodes(
                agent=mock_agent,
                env=mock_env,
                output_dir=temp_dir,
                num_episodes=3,
                config=video_config,
                seeds=seeds,
            )
        
        # Проверяем, что использовались правильные сиды
        for i, call in enumerate(mock_record.call_args_list):
            assert call[1]["episode_seed"] == seeds[i]
    
    def test_record_multiple_episodes_partial_failure(self, mock_agent, mock_env, video_config, temp_dir):
        """Тест частичной неудачи при записи эпизодов."""
        def side_effect(*args, **kwargs):
            if kwargs.get("episode_seed") == 43:  # Второй эпизод
                raise VideoGenerationError("Test error")
            return {"success": True, "total_reward": 5.0}
        
        with patch('src.visualization.video_generator.record_agent_episode', side_effect=side_effect):
            results = record_multiple_episodes(
                agent=mock_agent,
                env=mock_env,
                output_dir=temp_dir,
                num_episodes=3,
                config=video_config,
            )
        
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "error" in results[1]
        assert results[2]["success"] is True


class TestCreateTrainingMontage:
    """Тесты создания монтажа прогресса обучения."""
    
    def test_create_montage_success(self, video_config, temp_dir):
        """Тест успешного создания монтажа."""
        # Создание мок агентов
        agent1 = MockAgent("Early", actions=[0, 0, 0])
        agent2 = MockAgent("Late", actions=[1, 1, 1])
        checkpoints = [("Early Training", agent1), ("Late Training", agent2)]
        
        output_path = temp_dir / "montage.mp4"
        
        with patch('src.visualization.video_generator.setup_recording_environment') as mock_setup:
            mock_setup.return_value = MockEnv(max_steps=3)
            
            with patch('src.visualization.video_generator._save_video') as mock_save:
                result = create_training_montage(
                    agent_checkpoints=checkpoints,
                    env="CartPole-v1",
                    output_path=output_path,
                    config=video_config,
                    episodes_per_checkpoint=1,
                )
        
        assert result["success"] is True
        assert len(result["checkpoints"]) == 2
        assert result["checkpoints"][0]["name"] == "Early Training"
        assert result["checkpoints"][1]["name"] == "Late Training"
        assert result["total_frames"] > 0
        
        mock_save.assert_called_once()
    
    def test_create_montage_no_frames(self, video_config, temp_dir):
        """Тест ошибки при отсутствии кадров в монтаже."""
        agent = MockAgent("Test")
        checkpoints = [("Test", agent)]
        
        # Мок среды без кадров
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0, 0, 0, 0]), {})
        mock_env.step.return_value = (np.array([0, 0, 0, 0]), 1.0, True, False, {})
        mock_env.render.return_value = None
        
        output_path = temp_dir / "montage.mp4"
        
        with patch('src.visualization.video_generator.setup_recording_environment', return_value=mock_env):
            with pytest.raises(VideoGenerationError, match="Не удалось создать кадры для монтажа"):
                create_training_montage(
                    agent_checkpoints=checkpoints,
                    env="CartPole-v1",
                    output_path=output_path,
                    config=video_config,
                )


class TestGenerateComparisonVideo:
    """Тесты создания сравнительного видео."""
    
    def test_comparison_video_success(self, video_config, temp_dir):
        """Тест успешного создания сравнительного видео."""
        agent1 = MockAgent("PPO", actions=[0, 1, 0])
        agent2 = MockAgent("A2C", actions=[1, 0, 1])
        agents = [("PPO Agent", agent1), ("A2C Agent", agent2)]
        
        output_path = temp_dir / "comparison.mp4"
        
        with patch('src.visualization.video_generator.setup_recording_environment') as mock_setup:
            mock_setup.side_effect = [MockEnv(max_steps=3), MockEnv(max_steps=3)]
            
            with patch('src.visualization.video_generator._save_video') as mock_save:
                with patch('src.visualization.video_generator._combine_frames_side_by_side') as mock_combine:
                    mock_combine.return_value = np.zeros((200, 400, 3), dtype=np.uint8)
                    
                    result = generate_comparison_video(
                        agents=agents,
                        env="CartPole-v1",
                        output_path=output_path,
                        config=video_config,
                    )
        
        assert result["success"] is True
        assert len(result["agents"]) == 2
        assert result["agents"][0]["name"] == "PPO Agent"
        assert result["agents"][1]["name"] == "A2C Agent"
        
        mock_save.assert_called_once()
    
    def test_comparison_video_insufficient_agents(self, video_config, temp_dir):
        """Тест ошибки при недостаточном количестве агентов."""
        agent = MockAgent("Solo")
        agents = [("Solo Agent", agent)]
        
        output_path = temp_dir / "comparison.mp4"
        
        with pytest.raises(VideoGenerationError, match="Для сравнения нужно минимум 2 агента"):
            generate_comparison_video(
                agents=agents,
                env="CartPole-v1",
                output_path=output_path,
                config=video_config,
            )


class TestCompressVideo:
    """Тесты сжатия видео."""
    
    def test_compress_video_file_not_found(self, temp_dir):
        """Тест ошибки при отсутствии исходного файла."""
        input_path = temp_dir / "nonexistent.mp4"
        output_path = temp_dir / "compressed.mp4"
        
        with pytest.raises(VideoGenerationError, match="Исходный файл не найден"):
            compress_video(input_path, output_path)
    
    @patch('src.visualization.video_generator.imageio.get_reader')
    @patch('src.visualization.video_generator.imageio.get_writer')
    def test_compress_video_success(self, mock_writer, mock_reader, temp_dir):
        """Тест успешного сжатия видео."""
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "compressed.mp4"
        
        # Создание файла
        input_path.touch()
        
        # Мок reader
        mock_reader_instance = Mock()
        mock_reader_instance.get_meta_data.return_value = {"fps": 30}
        mock_reader_instance.__iter__ = Mock(return_value=iter([
            np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)
        ]))
        mock_reader.return_value = mock_reader_instance
        
        # Мок writer
        mock_writer_instance = Mock()
        mock_writer.return_value.__enter__ = Mock(return_value=mock_writer_instance)
        mock_writer.return_value.__exit__ = Mock(return_value=None)
        
        # Создание выходного файла для статистики
        output_path.write_bytes(b"compressed_data")
        
        result = compress_video(input_path, output_path, compression_level="medium")
        
        assert result["success"] is True
        assert "original_size_mb" in result
        assert "compressed_size_mb" in result
        assert "compression_ratio" in result


class TestUtilityFunctions:
    """Тесты вспомогательных функций."""
    
    def test_create_transition_frame(self, video_config):
        """Тест создания переходного кадра."""
        base_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        text = "Переход к следующему агенту"
        
        result = _create_transition_frame(base_frame, text, video_config)
        
        assert result.shape == base_frame.shape
        assert result.dtype == np.uint8
        # Кадр должен быть затемнен
        assert np.mean(result) < np.mean(base_frame)
    
    def test_combine_frames_side_by_side_two_frames(self):
        """Тест объединения двух кадров."""
        frame1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
        frame2 = np.ones((50, 50, 3), dtype=np.uint8) * 200
        frames = [frame1, frame2]
        agents = [("Agent1", None), ("Agent2", None)]
        
        result = _combine_frames_side_by_side(frames, agents)
        
        assert result.shape == (50, 100, 3)  # Объединение по горизонтали
        assert result.dtype == np.uint8
    
    def test_combine_frames_side_by_side_multiple_frames(self):
        """Тест объединения нескольких кадров в сетку."""
        frames = [
            np.ones((50, 50, 3), dtype=np.uint8) * i * 50 
            for i in range(4)
        ]
        agents = [(f"Agent{i}", None) for i in range(4)]
        
        result = _combine_frames_side_by_side(frames, agents)
        
        assert result.shape == (100, 100, 3)  # Сетка 2x2
        assert result.dtype == np.uint8
    
    def test_combine_frames_empty_list(self):
        """Тест объединения пустого списка кадров."""
        frames = []
        agents = []
        
        result = _combine_frames_side_by_side(frames, agents)
        
        assert result.shape == (200, 200, 3)  # Дефолтный размер
        assert result.dtype == np.uint8
    
    @patch('src.visualization.video_generator.imageio.mimsave')
    def test_save_video_gif(self, mock_mimsave, temp_dir):
        """Тест сохранения видео в формате GIF."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        output_path = temp_dir / "test.gif"
        config = VideoConfig(format="gif", fps=10)
        
        _save_video(frames, output_path, config)
        
        mock_mimsave.assert_called_once_with(
            output_path,
            frames,
            fps=10,
            loop=0,
        )
    
    @patch('src.visualization.video_generator.imageio.get_writer')
    def test_save_video_mp4(self, mock_get_writer, temp_dir):
        """Тест сохранения видео в формате MP4."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        output_path = temp_dir / "test.mp4"
        config = VideoConfig(format="mp4", fps=30, quality="high")
        
        mock_writer = Mock()
        mock_get_writer.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_get_writer.return_value.__exit__ = Mock(return_value=None)
        
        _save_video(frames, output_path, config)
        
        mock_get_writer.assert_called_once()
        assert mock_writer.append_data.call_count == len(frames)


if __name__ == "__main__":
    pytest.main([__file__])