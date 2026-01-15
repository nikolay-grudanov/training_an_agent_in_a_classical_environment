"""Wrapper для Gymnasium сред с поддержкой Stable-Baselines3.

Этот модуль предоставляет универсальный wrapper для сред Gymnasium,
который добавляет удобный интерфейс, обработку ошибок, логирование
и совместимость с Stable-Baselines3.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from src.utils.rl_logging import get_logger

logger = get_logger(__name__)


class EnvironmentWrapper:
    """Универсальный wrapper для сред Gymnasium.
    
    Оборачивает среду Gymnasium, предоставляя единый интерфейс для:
    - Инициализации с конфигурацией
    - Управления seed'ами для воспроизводимости
    - Выполнения шагов и сбросов
    - Извлечения информации о среде
    - Визуализации
    
    Атрибуты:
        env: Базовая среда Gymnasium
        config: Конфигурация среды
        seed: Текущий seed
        episode_count: Счетчик эпизодов
        step_count: Счетчик шагов
        total_reward: Общая награда за эпизод
    """
    
    def __init__(
        self,
        env_id: str,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Инициализировать wrapper для среды.
        
        Args:
            env_id: Идентификатор среды Gymnasium (например, 'LunarLander-v3')
            config: Конфигурация для создания среды
            render_mode: Режим рендеринга (None, 'human', 'rgb_array', 'ansi')
            
        Raises:
            ValueError: Если env_id некорректен или среда не может быть создана
            ImportError: Если требуемые зависимости не установлены
        """
        self.env_id = env_id
        self.config = config or {}
        self.render_mode = render_mode
        
        # Инициализация счетчиков
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # Создание базовой среды
        try:
            env_kwargs = self.config.copy()
            if render_mode:
                env_kwargs['render_mode'] = render_mode
            
            self.env = gym.make(env_id, **env_kwargs)
            logger.info(
                f"Создана среда {env_id}",
                env_id=env_id,
                action_space=self.action_space,
                observation_space=self.observation_space,
            )
        except gym.error.Error as e:
            logger.error(f"Не удалось создать среду {env_id}: {e}")
            raise ValueError(f"Некорректный env_id или конфигурация: {env_id}") from e
        
        # Инициализация seed
        self._seed: Optional[int] = None
        
    @property
    def action_space(self) -> gym.Space:
        """Получить пространство действий.
        
        Returns:
            Пространство действий среды
        """
        return self.env.action_space
    
    @property
    def observation_space(self) -> gym.Space:
        """Получить пространство наблюдений.
        
        Returns:
            Пространство наблюдений среды
        """
        return self.env.observation_space
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Получить метаданные среды.
        
        Returns:
            Метаданные среды
        """
        return self.env.metadata
    
    @property
    def reward_range(self) -> Tuple[float, float]:
        """Получить диапазон наград.
        
        Returns:
            Кортеж (минимальная_награда, максимальная_награда)
        """
        return self.env.reward_range
    
    @property
    def spec(self) -> Optional[Any]:
        """Получить спецификацию среды.
        
        Returns:
            Спецификация среды или None
        """
        return self.env.spec
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Установить seed для воспроизводимости.
        
        Args:
            seed: Значение seed или None для случайного seed
            
        Returns:
            None
        """
        self._seed = seed
        
        # Установка seed для среды Gymnasium
        if seed is not None:
            try:
                self.env.reset(seed=seed)
                logger.debug(f"Установлен seed {seed} для среды {self.env_id}")
            except AttributeError:
                # Некоторые среды не поддерживают seed в reset
                logger.warning(
                    f"Среда {self.env_id} не поддерживает seed в reset",
                    env_id=self.env_id,
                )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сбросить среду к начальному состоянию.
        
        Args:
            seed: Seed для генератора случайных чисел среды
            options: Дополнительные опции для сброса
            
        Returns:
            Кортеж (начальное_наблюдение, информация_о_сбросе)
            
        Raises:
            RuntimeError: Если сброс не удался
        """
        try:
            # Используем переданный seed или сохраненный
            reset_seed = seed if seed is not None else self._seed
            
            observation, info = self.env.reset(seed=reset_seed, options=options)
            
            # Сброс счетчиков эпизода
            self.total_reward = 0.0
            self.episode_count += 1
            
            logger.debug(
                f"Сброс среды {self.env_id}, эпизод {self.episode_count}",
                env_id=self.env_id,
                episode=self.episode_count,
                observation_shape=observation.shape,
            )
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Ошибка при сбросе среды {self.env_id}: {e}")
            raise RuntimeError(f"Не удалось сбросить среду {self.env_id}") from e
    
    def step(
        self,
        action: Union[int, float, np.ndarray, list],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполнить шаг в среде.
        
        Args:
            action: Действие для выполнения
            
        Returns:
            Кортеж (наблюдение, награда, завершено, усечено, информация)
            
        Raises:
            ValueError: Если действие некорректно
            RuntimeError: Если выполнение шага не удалось
        """
        try:
            # Преобразование действия при необходимости
            if isinstance(action, (list, np.ndarray)):
                action = np.array(action, dtype=self.action_space.dtype)
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Обновление счетчиков
            self.step_count += 1
            self.total_reward += reward
            
            # Добавление дополнительной информации
            info['episode'] = self.episode_count
            info['total_reward'] = self.total_reward
            info['step'] = self.step_count
            
            logger.debug(
                f"Шг {self.step_count} в среде {self.env_id}",
                env_id=self.env_id,
                episode=self.episode_count,
                step=self.step_count,
                reward=reward,
                total_reward=self.total_reward,
                terminated=terminated,
                truncated=truncated,
            )
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(
                f"Ошибка при выполнении шага в среде {self.env_id}: {e}",
                env_id=self.env_id,
                action=action,
            )
            raise RuntimeError(f"Не удалось выполнить шаг в среде {self.env_id}") from e
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Визуализировать текущее состояние среды.
        
        Returns:
            В зависимости от render_mode:
            - None: Если рендеринг не поддерживается
            - np.ndarray: RGB массив для 'rgb_array'
            - str: Текстовое представление для 'ansi'
            
        Raises:
            RuntimeError: Если рендеринг не поддерживается
        """
        try:
            return self.env.render()
        except Exception as e:
            logger.warning(f"Ошибка при рендеринге среды {self.env_id}: {e}")
            raise RuntimeError(f"Рендеринг не поддерживается для {self.env_id}") from e
    
    def close(self) -> None:
        """Закрыть среду и освободить ресурсы."""
        try:
            self.env.close()
            logger.info(f"Среда {self.env_id} закрыта")
        except Exception as e:
            logger.warning(f"Ошибка при закрытии среды {self.env_id}: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Получить полную информацию о среде.
        
        Returns:
            Словарь с информацией о среде
        """
        return {
            'env_id': self.env_id,
            'config': self.config,
            'render_mode': self.render_mode,
            'action_space': str(self.action_space),
            'observation_space': str(self.observation_space),
            'reward_range': self.reward_range,
            'metadata': self.metadata,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'seed': self._seed,
        }
    
    def __enter__(self) -> 'EnvironmentWrapper':
        """Контекстный менеджер для автоматического закрытия."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Закрыть среду при выходе из контекста."""
        self.close()
    
    def __str__(self) -> str:
        """Строковое представление wrapper'а."""
        return (
            f"EnvironmentWrapper(env_id='{self.env_id}', "
            f"action_space={self.action_space}, "
            f"observation_space={self.observation_space})"
        )
    
    def __repr__(self) -> str:
        """Официальное строковое представление."""
        return (
            f"EnvironmentWrapper(env_id='{self.env_id}', "
            f"config={self.config}, render_mode='{self.render_mode}')"
        )


# Алиас для обратной совместимости
GymWrapper = EnvironmentWrapper