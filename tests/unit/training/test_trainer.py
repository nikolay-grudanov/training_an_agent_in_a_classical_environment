"""Тесты для модуля обучения RL агентов."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingMode,
    TrainingResult,
    create_trainer_from_config,
)
from src.agents.base import AgentConfig
from src.utils.config import RLConfig, AlgorithmConfig, EnvironmentConfig, TrainingConfig


class TestTrainerConfig:
    """Тесты для TrainerConfig."""
    
    def test_default_config(self):
        """Тест создания конфигурации по умолчанию."""
        config = TrainerConfig()
        
        assert config.experiment_name == "default_experiment"
        assert config.algorithm == "PPO"
        assert config.environment_name == "LunarLander-v3"
        assert config.mode == TrainingMode.TRAIN
        assert config.total_timesteps == 100_000
        assert config.seed == 42
        assert config.agent_config is not None
        assert config.agent_config.algorithm == "PPO"
    
    def test_custom_config(self):
        """Тест создания пользовательской конфигурации."""
        config = TrainerConfig(
            experiment_name="test_experiment",
            algorithm="A2C",
            environment_name="CartPole-v1",
            total_timesteps=50_000,
            seed=123,
        )
        
        assert config.experiment_name == "test_experiment"
        assert config.algorithm == "A2C"
        assert config.environment_name == "CartPole-v1"
        assert config.total_timesteps == 50_000
        assert config.seed == 123
        assert config.agent_config.algorithm == "A2C"
    
    def test_invalid_algorithm(self):
        """Тест валидации неподдерживаемого алгоритма."""
        with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
            TrainerConfig(algorithm="INVALID")
    
    def test_invalid_timesteps(self):
        """Тест валидации некорректного количества шагов."""
        with pytest.raises(ValueError, match="total_timesteps должен быть > 0"):
            TrainerConfig(total_timesteps=0)
        
        with pytest.raises(ValueError, match="total_timesteps должен быть > 0"):
            TrainerConfig(total_timesteps=-1000)
    
    def test_invalid_eval_freq(self):
        """Тест валидации некорректной частоты оценки."""
        with pytest.raises(ValueError, match="eval_freq должен быть > 0"):
            TrainerConfig(eval_freq=0)
    
    def test_path_setup(self):
        """Тест настройки путей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainerConfig(
                experiment_name="test_exp",
                output_dir=temp_dir,
            )
            
            # Проверяем, что пути созданы
            assert config.model_save_path is not None
            assert config.logs_dir is not None
            assert config.tensorboard_log is not None
            
            # Проверяем, что директории существуют
            assert Path(config.logs_dir).exists()
            assert Path(config.tensorboard_log).exists()
            assert Path(config.model_save_path).parent.exists()
    
    def test_from_rl_config(self):
        """Тест создания TrainerConfig из RLConfig."""
        rl_config = RLConfig(
            experiment_name="test_rl_experiment",
            algorithm=AlgorithmConfig(name="SAC", learning_rate=1e-3),
            environment=EnvironmentConfig(name="Pendulum-v1"),
            training=TrainingConfig(total_timesteps=200_000, eval_freq=5000),
            seed=999,
        )
        
        trainer_config = TrainerConfig.from_rl_config(rl_config)
        
        assert trainer_config.experiment_name == "test_rl_experiment"
        assert trainer_config.algorithm == "SAC"
        assert trainer_config.environment_name == "Pendulum-v1"
        assert trainer_config.total_timesteps == 200_000
        assert trainer_config.eval_freq == 5000
        assert trainer_config.seed == 999
        assert trainer_config.agent_config.learning_rate == 1e-3


class TestTrainingResult:
    """Тесты для TrainingResult."""
    
    def test_default_result(self):
        """Тест создания результата по умолчанию."""
        result = TrainingResult(
            success=True,
            total_timesteps=100_000,
            training_time=300.5,
            final_mean_reward=250.0,
            final_std_reward=50.0,
        )
        
        assert result.success is True
        assert result.total_timesteps == 100_000
        assert result.training_time == 300.5
        assert result.final_mean_reward == 250.0
        assert result.final_std_reward == 50.0
        assert result.best_mean_reward == float("-inf")
        assert result.early_stopped is False
    
    def test_to_dict(self):
        """Тест преобразования результата в словарь."""
        result = TrainingResult(
            success=True,
            total_timesteps=50_000,
            training_time=150.0,
            final_mean_reward=100.0,
            final_std_reward=20.0,
            experiment_name="test_exp",
            algorithm="PPO",
            environment_name="LunarLander-v3",
            seed=42,
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["total_timesteps"] == 50_000
        assert result_dict["experiment_name"] == "test_exp"
        assert result_dict["algorithm"] == "PPO"
    
    def test_save_result(self):
        """Тест сохранения результата в файл."""
        result = TrainingResult(
            success=True,
            total_timesteps=25_000,
            training_time=75.0,
            final_mean_reward=150.0,
            final_std_reward=30.0,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "test_result.yaml"
            result.save(result_path)
            
            assert result_path.exists()
            
            # Проверяем, что файл можно прочитать
            import yaml
            with open(result_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data["success"] is True
            assert loaded_data["total_timesteps"] == 25_000


class TestTrainer:
    """Тесты для Trainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Временная директория для тестов."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def basic_config(self, temp_dir):
        """Базовая конфигурация для тестов."""
        return TrainerConfig(
            experiment_name="test_experiment",
            algorithm="PPO",
            environment_name="CartPole-v1",
            total_timesteps=1000,
            output_dir=temp_dir,
            eval_freq=500,
            save_freq=500,
            checkpoint_freq=500,
        )
    
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_trainer_init(self, mock_set_seed, mock_get_metrics, mock_get_logger, basic_config):
        """Тест инициализации тренера."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        
        assert trainer.config == basic_config
        assert trainer.experiment_name == "test_experiment"
        assert trainer.current_timestep == 0
        assert trainer.best_mean_reward == float("-inf")
        
        mock_set_seed.assert_called_once_with(basic_config.seed)
        mock_get_logger.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('src.training.trainer.EnvironmentWrapper')
    @patch('src.training.trainer.PPOAgent')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_setup(self, mock_set_seed, mock_get_metrics, mock_get_logger, 
                   mock_ppo_agent, mock_env_wrapper, basic_config):
        """Тест настройки компонентов."""
        # Настройка моков
        mock_env = Mock()
        mock_env.action_space = Mock()
        mock_env.observation_space = Mock()
        mock_env_wrapper.return_value = mock_env
        
        mock_agent = Mock()
        mock_ppo_agent.return_value = mock_agent
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        trainer.setup()
        
        assert trainer.env is not None
        assert trainer.agent is not None
        mock_env_wrapper.assert_called_once()
        mock_ppo_agent.assert_called_once()
    
    @patch('src.training.trainer.EnvironmentWrapper')
    @patch('src.training.trainer.PPOAgent')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_evaluate_only_mode(self, mock_set_seed, mock_get_metrics, mock_get_logger,
                               mock_ppo_agent, mock_env_wrapper, basic_config):
        """Тест режима только оценки."""
        # Настройка конфигурации для режима оценки
        basic_config.mode = TrainingMode.EVALUATE
        
        # Настройка моков
        mock_env = Mock()
        mock_env_wrapper.return_value = mock_env
        
        mock_agent = Mock()
        mock_agent.is_trained = True
        mock_agent.evaluate.return_value = {
            "mean_reward": 200.0,
            "std_reward": 25.0,
        }
        mock_ppo_agent.return_value = mock_agent
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        trainer.setup()
        
        result = trainer.train()
        
        assert result.success is True
        assert result.final_mean_reward == 200.0
        assert result.final_std_reward == 25.0
        assert result.total_timesteps == 0  # Нет обучения в режиме оценки
        mock_agent.evaluate.assert_called_once()
    
    @patch('src.training.trainer.EnvironmentWrapper')
    @patch('src.training.trainer.PPOAgent')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_evaluate_untrained_agent(self, mock_set_seed, mock_get_metrics, mock_get_logger,
                                     mock_ppo_agent, mock_env_wrapper, basic_config):
        """Тест оценки необученного агента."""
        basic_config.mode = TrainingMode.EVALUATE
        
        # Настройка моков
        mock_env = Mock()
        mock_env_wrapper.return_value = mock_env
        
        mock_agent = Mock()
        mock_agent.is_trained = False  # Агент не обучен
        mock_ppo_agent.return_value = mock_agent
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        trainer.setup()
        
        with pytest.raises(RuntimeError, match="Агент не обучен"):
            trainer.train()
    
    @patch('src.training.trainer.EnvironmentWrapper')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_unsupported_algorithm(self, mock_set_seed, mock_get_metrics, mock_get_logger,
                                  mock_env_wrapper, basic_config):
        """Тест неподдерживаемого алгоритма."""
        # Изменяем алгоритм на неподдерживаемый после создания конфигурации
        basic_config.algorithm = "INVALID"
        
        mock_env = Mock()
        mock_env_wrapper.return_value = mock_env
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        
        with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
            trainer.setup()
    
    @patch('src.training.trainer.EnvironmentWrapper')
    @patch('src.training.trainer.PPOAgent')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_context_manager(self, mock_set_seed, mock_get_metrics, mock_get_logger,
                            mock_ppo_agent, mock_env_wrapper, basic_config):
        """Тест использования тренера как контекстного менеджера."""
        # Настройка моков
        mock_env = Mock()
        mock_env_wrapper.return_value = mock_env
        
        mock_agent = Mock()
        mock_ppo_agent.return_value = mock_agent
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = Trainer(basic_config)
        
        with trainer as t:
            assert t is trainer
            assert trainer.env is not None
            assert trainer.agent is not None
        
        # Проверяем, что cleanup был вызван
        mock_env.close.assert_called_once()
    
    def test_checkpoint_operations(self, basic_config, temp_dir):
        """Тест операций с чекпоинтами."""
        with patch('src.training.trainer.get_experiment_logger'), \
             patch('src.training.trainer.get_metrics_tracker'), \
             patch('src.training.trainer.set_seed'):
            
            trainer = Trainer(basic_config)
            
            # Тест сохранения чекпоинта
            checkpoint_path = trainer.save_checkpoint(1000)
            assert Path(checkpoint_path).exists()
            
            # Тест загрузки чекпоинта
            trainer.load_checkpoint(checkpoint_path)
            assert trainer.current_timestep == 1000


class TestCreateTrainerFromConfig:
    """Тесты для create_trainer_from_config."""
    
    @patch('src.training.trainer.load_config')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_create_from_rl_config(self, mock_set_seed, mock_get_metrics, 
                                  mock_get_logger, mock_load_config):
        """Тест создания тренера из RLConfig."""
        # Настройка мока
        rl_config = RLConfig(
            experiment_name="test_from_config",
            algorithm=AlgorithmConfig(name="A2C"),
            environment=EnvironmentConfig(name="CartPole-v1"),
            training=TrainingConfig(total_timesteps=5000),
        )
        mock_load_config.return_value = rl_config
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = create_trainer_from_config(config_name="test_config")
        
        assert isinstance(trainer, Trainer)
        assert trainer.config.experiment_name == "test_from_config"
        assert trainer.config.algorithm == "A2C"
        assert trainer.config.environment_name == "CartPole-v1"
        assert trainer.config.total_timesteps == 5000
        
        mock_load_config.assert_called_once_with(
            config_name="test_config",
            config_path=None,
            overrides=None,
        )
    
    @patch('src.training.trainer.load_config')
    @patch('src.training.trainer.get_experiment_logger')
    @patch('src.training.trainer.get_metrics_tracker')
    @patch('src.training.trainer.set_seed')
    def test_create_with_overrides(self, mock_set_seed, mock_get_metrics,
                                  mock_get_logger, mock_load_config):
        """Тест создания тренера с переопределениями."""
        rl_config = RLConfig()
        mock_load_config.return_value = rl_config
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_get_metrics.return_value = Mock()
        
        trainer = create_trainer_from_config(
            config_name="test_config",
            overrides=["algorithm.name=SAC", "training.total_timesteps=10000"],
            mode=TrainingMode.RESUME,
        )
        
        assert isinstance(trainer, Trainer)
        assert trainer.config.mode == TrainingMode.RESUME
        
        mock_load_config.assert_called_once_with(
            config_name="test_config",
            config_path=None,
            overrides=["algorithm.name=SAC", "training.total_timesteps=10000"],
        )


@pytest.mark.parametrize("algorithm", ["PPO", "A2C", "SAC", "TD3"])
def test_supported_algorithms(algorithm):
    """Тест поддерживаемых алгоритмов."""
    config = TrainerConfig(algorithm=algorithm)
    assert config.algorithm == algorithm


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_training_modes(mode):
    """Тест всех режимов обучения."""
    config = TrainerConfig(mode=mode)
    assert config.mode == mode


def test_training_mode_enum():
    """Тест enum TrainingMode."""
    assert TrainingMode.TRAIN.value == "train"
    assert TrainingMode.RESUME.value == "resume"
    assert TrainingMode.EVALUATE.value == "evaluate"
    assert TrainingMode.FINETUNE.value == "finetune"
    
    # Тест создания из строки
    assert TrainingMode("train") == TrainingMode.TRAIN
    assert TrainingMode("resume") == TrainingMode.RESUME