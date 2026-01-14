#!/usr/bin/env python3
"""Пример использования базового класса Agent.

Демонстрирует создание конфигурации, инициализацию агента,
и основные операции с базовым классом.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base import Agent, AgentConfig, TrainingResult
from src.utils import set_seed

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOAgent(Agent):
    """Пример реализации агента PPO на основе базового класса."""
    
    def _create_model(self):
        """Создать модель PPO."""
        return PPO(
            policy=self.config.policy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            use_sde=self.config.use_sde,
            sde_sample_freq=self.config.sde_sample_freq,
            target_kl=self.config.target_kl,
            policy_kwargs=self.config.policy_kwargs,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=self.config.seed,
            tensorboard_log=self.config.tensorboard_log,
        )
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Обучить агента PPO."""
        import time
        
        if self.model is None:
            self.model = self._create_model()
        
        timesteps = total_timesteps or self.config.total_timesteps
        start_time = time.time()
        
        try:
            # Обучение модели
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback,
                **kwargs
            )
            
            training_time = time.time() - start_time
            self.is_trained = True
            
            # Оценка производительности
            eval_metrics = self.evaluate(
                n_episodes=self.config.n_eval_episodes,
                deterministic=True,
            )
            
            # Создание результата обучения
            self.training_result = TrainingResult(
                total_timesteps=timesteps,
                training_time=training_time,
                final_mean_reward=eval_metrics["mean_reward"],
                final_std_reward=eval_metrics["std_reward"],
                best_mean_reward=eval_metrics["mean_reward"],
                success=True,
            )
            
            self.logger.info(
                f"Обучение завершено успешно за {training_time:.2f} сек",
                extra={
                    "timesteps": timesteps,
                    "mean_reward": eval_metrics["mean_reward"],
                    "std_reward": eval_metrics["std_reward"],
                }
            )
            
            return self.training_result
            
        except Exception as e:
            error_msg = f"Ошибка обучения: {e}"
            self.logger.error(error_msg)
            
            self.training_result = TrainingResult(
                total_timesteps=timesteps,
                training_time=time.time() - start_time,
                final_mean_reward=float('-inf'),
                final_std_reward=0.0,
                success=False,
                error_message=error_msg,
            )
            
            raise RuntimeError(error_msg) from e
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказать действие."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите train().")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> "PPOAgent":
        """Загрузить агента PPO."""
        import yaml
        
        # Загрузка конфигурации
        config_path = Path(path).with_suffix('.yaml')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            config = AgentConfig(**config_dict)
        else:
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        # Создание агента
        agent = cls(config=config, env=env)
        
        # Загрузка модели
        try:
            agent.model = PPO.load(path, env=agent.env)
            agent.is_trained = True
            
            agent.logger.info(f"Агент загружен из: {path}")
            return agent
            
        except Exception as e:
            error_msg = f"Ошибка загрузки модели из {path}: {e}"
            agent.logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def main():
    """Основная функция демонстрации."""
    print("🚀 Демонстрация базового класса Agent")
    print("=" * 50)
    
    # 1. Создание конфигурации
    print("\n1. Создание конфигурации агента...")
    config = AgentConfig(
        algorithm="PPO",
        env_name="CartPole-v1",
        total_timesteps=10_000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=42,
        verbose=1,
    )
    print(f"✅ Конфигурация создана: {config.algorithm} для {config.env_name}")
    
    # 2. Инициализация агента
    print("\n2. Инициализация агента...")
    agent = PPOAgent(config=config, experiment_name="demo_cartpole")
    print(f"✅ Агент инициализирован: {agent}")
    
    # 3. Получение информации о модели
    print("\n3. Информация о модели...")
    info = agent.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 4. Обучение агента
    print("\n4. Обучение агента...")
    try:
        result = agent.train()
        print(f"✅ Обучение завершено:")
        print(f"   Время обучения: {result.training_time:.2f} сек")
        print(f"   Средняя награда: {result.final_mean_reward:.2f}")
        print(f"   Стандартное отклонение: {result.final_std_reward:.2f}")
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return
    
    # 5. Тестирование предсказаний
    print("\n5. Тестирование предсказаний...")
    obs, _ = agent.env.reset()
    for i in range(5):
        action, _ = agent.predict(obs, deterministic=True)
        print(f"   Шаг {i+1}: наблюдение={obs[:2]}, действие={action}")
        obs, reward, terminated, truncated, _ = agent.env.step(action)
        if terminated or truncated:
            obs, _ = agent.env.reset()
    
    # 6. Оценка производительности
    print("\n6. Оценка производительности...")
    eval_metrics = agent.evaluate(n_episodes=5, deterministic=True)
    print(f"✅ Результаты оценки:")
    for metric, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    # 7. Сохранение модели
    print("\n7. Сохранение модели...")
    save_path = "demo_cartpole_model.zip"
    try:
        agent.save(save_path)
        print(f"✅ Модель сохранена: {save_path}")
        
        # Проверка файлов
        model_file = Path(save_path)
        config_file = model_file.with_suffix('.yaml')
        print(f"   Файл модели: {model_file.exists()}")
        print(f"   Файл конфигурации: {config_file.exists()}")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
    
    # 8. Загрузка модели
    print("\n8. Загрузка модели...")
    try:
        loaded_agent = PPOAgent.load(save_path)
        print(f"✅ Модель загружена: {loaded_agent}")
        
        # Тест загруженной модели
        obs, _ = loaded_agent.env.reset()
        action, _ = loaded_agent.predict(obs)
        print(f"   Тест предсказания: наблюдение={obs[:2]}, действие={action}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
    
    # 9. Сброс модели
    print("\n9. Сброс модели...")
    agent.reset_model()
    info_after_reset = agent.get_model_info()
    print(f"✅ Модель сброшена, обучена: {info_after_reset['is_trained']}")
    
    print("\n🎉 Демонстрация завершена!")
    
    # Очистка файлов
    try:
        Path(save_path).unlink(missing_ok=True)
        Path(save_path).with_suffix('.yaml').unlink(missing_ok=True)
        print("🧹 Временные файлы удалены")
    except Exception:
        pass


if __name__ == "__main__":
    main()