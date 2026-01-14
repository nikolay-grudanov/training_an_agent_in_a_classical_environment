"""Специализированный handler для среды LunarLander-v3.

Этот модуль предоставляет LunarLanderEnvironment - специализированный wrapper
для среды LunarLander-v2/v3 с дополнительными функциями для RL обучения.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from src.environments.wrapper import EnvironmentWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LunarLanderEnvironment(EnvironmentWrapper):
    """Специализированный wrapper для среды LunarLander-v2/v3.
    
    Расширяет базовый EnvironmentWrapper, добавляя:
    - Валидацию специфичную для LunarLander
    - Опции формирования наград (reward shaping)
    - Критерии завершения эпизода
    - Определение успешной/неуспешной посадки
    - Статистику посадки
    
    Атрибуты:
        env: Базовая среда Gymnasium
        config: Конфигурация среды
        seed: Текущий seed
        episode_count: Счетчик эпизодов
        step_count: Счетчик шагов
        total_reward: Общая награда за эпизод
        landing_stats: Статистика посадок
        reward_shaping: Настройки формирования наград
    """
    
    # Константы для LunarLander
    SUPPORTED_VERSIONS = {'LunarLander-v2', 'LunarLander-v3'}
    CONTINUOUS_ACTION_SPACE = ('LunarLanderContinuous-v2', 'LunarLanderContinuous-v3')
    
    # Пороги для определения успешной посадки
    LANDING_SUCCESS_VELOCITY_THRESHOLD = 0.1  # м/с
    LANDING_SUCCESS_ANGLE_THRESHOLD = 0.2     # радианы
    LANDING_SUCCESS_POSITION_THRESHOLD = 0.1  # единицы
    
    # Награды по умолчанию
    DEFAULT_REWARDS = {
        'main_engine': -0.3,      # Использование основного двигателя
        'side_engine': -0.03,     # Использование бокового двигателя
        'crash': -100.0,          # Авария
        'successful_landing': 100.0,  # Успешная посадка
        'landing_leg_contact': 10.0,  # Контакт посадочной опоры
        'distance_from_target': -1.0,  # Штраф за расстояние от цели
        'velocity_penalty': -1.0,      # Штраф за скорость
        'angle_penalty': -1.0,         # Штраф за угол
    }
    
    def __init__(
        self,
        env_id: str = 'LunarLander-v3',
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        reward_shaping: Optional[Dict[str, float]] = None,
        max_episode_steps: Optional[int] = 1000,
        enable_landing_detection: bool = True,
    ) -> None:
        """Инициализировать LunarLanderEnvironment.
        
        Args:
            env_id: Идентификатор среды ('LunarLander-v2', 'LunarLander-v3',
                   'LunarLanderContinuous-v2', 'LunarLanderContinuous-v3')
            config: Конфигурация для создания среды
            render_mode: Режим рендеринга
            reward_shaping: Кастомные награды для формирования наград
            max_episode_steps: Максимальное количество шагов в эпизоде
            enable_landing_detection: Включить автоматическое определение посадки
            
        Raises:
            ValueError: Если env_id не поддерживается или конфигурация некорректна
        """
        # Валидация env_id
        self._validate_env_id(env_id)
        
        # Подготовка конфигурации
        full_config = self._prepare_config(config, max_episode_steps)
        
        # Инициализация базового wrapper'а
        super().__init__(env_id, full_config, render_mode)
        
        # Инициализация специфичных атрибутов
        self.reward_shaping = reward_shaping or self.DEFAULT_REWARDS.copy()
        self.enable_landing_detection = enable_landing_detection
        self.max_episode_steps = max_episode_steps
        
        # Статистика посадок
        self.landing_stats = {
            'successful': 0,
            'crashed': 0,
            'out_of_fuel': 0,
            'timeout': 0,
            'total': 0,
        }
        
        # Текущее состояние посадки
        self._current_landing_state = {
            'is_landing': False,
            'landing_start_step': None,
            'final_velocity': None,
            'final_angle': None,
            'final_position': None,
            'landing_outcome': None,
        }
        
        logger.info(
            f"Инициализирован LunarLanderEnvironment: {env_id}",
            env_id=env_id,
            action_space=self.action_space,
            observation_space=self.observation_space,
            max_episode_steps=max_episode_steps,
        )
    
    def _validate_env_id(self, env_id: str) -> None:
        """Проверить, что env_id поддерживается.
        
        Args:
            env_id: Идентификатор среды для проверки
            
        Raises:
            ValueError: Если env_id не поддерживается
        """
        if env_id not in self.SUPPORTED_VERSIONS and env_id not in self.CONTINUOUS_ACTION_SPACE:
            raise ValueError(
                f"Неподдерживаемый env_id: {env_id}. "
                f"Поддерживаемые: {sorted(self.SUPPORTED_VERSIONS | set(self.CONTINUOUS_ACTION_SPACE))}"
            )
    
    def _prepare_config(
        self,
        config: Optional[Dict[str, Any]],
        max_episode_steps: Optional[int],
    ) -> Dict[str, Any]:
        """Подготовить конфигурацию для создания среды.
        
        Args:
            config: Базовая конфигурация
            max_episode_steps: Максимальное количество шагов в эпизоде
            
        Returns:
            Полная конфигурация для создания среды
        """
        full_config = config.copy() if config else {}
        
        # Добавляем max_episode_steps если указано
        if max_episode_steps is not None:
            full_config['max_episode_steps'] = max_episode_steps
        
        return full_config
    
    @property
    def is_continuous(self) -> bool:
        """Проверить, использует ли среда непрерывное пространство действий.
        
        Returns:
            True если пространство действий непрерывное
        """
        return self.env_id in self.CONTINUOUS_ACTION_SPACE
    
    @property
    def observation_dimensions(self) -> Dict[str, int]:
        """Получить размерности наблюдений.
        
        Returns:
            Словарь с размерностями различных компонентов наблюдения
        """
        # LunarLander наблюдения: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        return {
            'position': 2,           # x, y
            'velocity': 2,           # vx, vy
            'angle': 1,              # angle
            'angular_velocity': 1,   # angular_velocity
            'leg_contact': 2,        # left_leg, right_leg
            'total': 8,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сбросить среду к начальному состоянию.
        
        Args:
            seed: Seed для генератора случайных чисел
            options: Дополнительные опции для сброса
            
        Returns:
            Кортеж (начальное_наблюдение, информация_о_сбросе)
        """
        observation, info = super().reset(seed, options)
        
        # Сброс состояния посадки
        self._current_landing_state = {
            'is_landing': False,
            'landing_start_step': None,
            'final_velocity': None,
            'final_angle': None,
            'final_position': None,
            'landing_outcome': None,
        }
        
        return observation, info
    
    def step(
        self,
        action: Union[int, float, np.ndarray, list],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполнить шаг в среде.
        
        Args:
            action: Действие для выполнения
            
        Returns:
            Кортеж (наблюдение, награда, завершено, усечено, информация)
        """
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Добавляем дополнительную информацию о посадке
        info = self._augment_step_info(info, observation, reward)
        
        # Обновляем состояние посадки
        if self.enable_landing_detection:
            self._update_landing_state(observation, info)
        
        # Определяем исход эпизода
        if terminated or truncated:
            self._determine_episode_outcome(info)
        
        return observation, reward, terminated, truncated, info
    
    def _augment_step_info(
        self,
        info: Dict[str, Any],
        observation: np.ndarray,
        reward: float,
    ) -> Dict[str, Any]:
        """Дополнить информацию шага специфичными для LunarLander данными.
        
        Args:
            info: Базовая информация шага
            observation: Текущее наблюдение
            reward: Полученная награда
            
        Returns:
            Дополненная информация
        """
        # Извлекаем компоненты наблюдения
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
        
        # Вычисляем дополнительные метрики
        distance_from_target = np.sqrt(x**2 + y**2)
        total_velocity = np.sqrt(vx**2 + vy**2)
        
        # Добавляем в информацию
        info.update({
            'lunar_lander': {
                'position': {'x': float(x), 'y': float(y)},
                'velocity': {'x': float(vx), 'y': float(vy), 'total': float(total_velocity)},
                'angle': float(angle),
                'angular_velocity': float(angular_velocity),
                'leg_contact': {'left': bool(left_leg), 'right': bool(right_leg)},
                'distance_from_target': float(distance_from_target),
                'is_landing': self._current_landing_state['is_landing'],
            },
            'reward_components': self._analyze_reward_components(observation, action=None),
        })
        
        return info
    
    def _update_landing_state(
        self,
        observation: np.ndarray,
        info: Dict[str, Any],
    ) -> None:
        """Обновить состояние посадки на основе текущего наблюдения.
        
        Args:
            observation: Текущее наблюдение
            info: Информация шага
        """
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
        
        # Проверяем, началась ли посадка (контакт с землей)
        is_touching_ground = y <= 0
        
        if is_touching_ground and not self._current_landing_state['is_landing']:
            # Начало посадки
            self._current_landing_state.update({
                'is_landing': True,
                'landing_start_step': self.step_count,
            })
            logger.debug(f"Начало посадки на шаге {self.step_count}")
        
        if self._current_landing_state['is_landing']:
            # Обновляем финальные значения во время посадки
            self._current_landing_state.update({
                'final_velocity': np.sqrt(vx**2 + vy**2),
                'final_angle': abs(angle),
                'final_position': np.sqrt(x**2 + y**2),
            })
    
    def _determine_episode_outcome(self, info: Dict[str, Any]) -> None:
        """Определить исход эпизода и обновить статистику.
        
        Args:
            info: Информация шага
        """
        if not self._current_landing_state['is_landing']:
            # Не было посадки (таймаут или другие причины)
            outcome = 'timeout'
            self.landing_stats['timeout'] += 1
        else:
            # Была посадка, определяем успешность
            velocity = self._current_landing_state['final_velocity']
            angle = self._current_landing_state['final_angle']
            position = self._current_landing_state['final_position']
            
            is_successful = (
                velocity is not None and velocity <= self.LANDING_SUCCESS_VELOCITY_THRESHOLD and
                angle is not None and angle <= self.LANDING_SUCCESS_ANGLE_THRESHOLD and
                position is not None and position <= self.LANDING_SUCCESS_POSITION_THRESHOLD
            )
            
            if is_successful:
                outcome = 'successful'
                self.landing_stats['successful'] += 1
            else:
                outcome = 'crashed'
                self.landing_stats['crashed'] += 1
        
        # Обновляем состояние посадки
        self._current_landing_state['landing_outcome'] = outcome
        self.landing_stats['total'] += 1
        
        # Добавляем в информацию
        info['lunar_lander']['landing_outcome'] = outcome
        info['lunar_lander']['landing_stats'] = self.landing_stats.copy()
        
        logger.info(
            f"Эпизод {self.episode_count} завершен: {outcome}",
            episode=self.episode_count,
            outcome=outcome,
            total_reward=self.total_reward,
            steps=self.step_count,
        )
    
    def _analyze_reward_components(
        self,
        observation: np.ndarray,
        action: Optional[Union[int, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Проанализировать компоненты награды.
        
        Args:
            observation: Текущее наблюдение
            action: Выполненное действие (опционально)
            
        Returns:
            Словарь с компонентами награды
        """
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
        
        components = {
            'distance_from_target': -np.sqrt(x**2 + y**2),
            'velocity_penalty': -np.sqrt(vx**2 + vy**2),
            'angle_penalty': -abs(angle),
        }
        
        # Анализ использования двигателей если действие предоставлено
        if action is not None:
            if self.is_continuous:
                # Непрерывное пространство действий: [main_engine, side_engine]
                main_engine, side_engine = action
                components['main_engine'] = -abs(main_engine)
                components['side_engine'] = -abs(side_engine)
            else:
                # Дискретное пространство действий: 0-3
                # 0: ничего, 1: левый двигатель, 2: основной двигатель, 3: правый двигатель
                if action == 2:  # Основной двигатель
                    components['main_engine'] = self.reward_shaping['main_engine']
                elif action in (1, 3):  # Боковые двигатели
                    components['side_engine'] = self.reward_shaping['side_engine']
        
        return components
    
    def is_landing_successful(self) -> Optional[bool]:
        """Проверить, была ли посадка успешной.
        
        Returns:
            True если посадка успешна, False если авария, None если посадка не завершена
        """
        outcome = self._current_landing_state['landing_outcome']
        if outcome is None:
            return None
        return outcome == 'successful'
    
    def get_landing_statistics(self) -> Dict[str, Any]:
        """Получить статистику посадок.
        
        Returns:
            Словарь со статистикой посадок
        """
        stats = self.landing_stats.copy()
        
        # Вычисляем проценты
        if stats['total'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total']
            stats['crash_rate'] = stats['crashed'] / stats['total']
            stats['timeout_rate'] = stats['timeout'] / stats['total']
        else:
            stats.update({
                'success_rate': 0.0,
                'crash_rate': 0.0,
                'timeout_rate': 0.0,
            })
        
        # Добавляем текущее состояние
        stats['current_state'] = self._current_landing_state.copy()
        
        return stats
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Получить полную информацию о среде.
        
        Returns:
            Словарь с информацией о среде
        """
        base_info = super().get_info()
        
        lunar_lander_info = {
            'is_continuous': self.is_continuous,
            'observation_dimensions': self.observation_dimensions,
            'landing_statistics': self.get_landing_statistics(),
            'reward_shaping': self.reward_shaping,
            'enable_landing_detection': self.enable_landing_detection,
            'max_episode_steps': self.max_episode_steps,
            'success_thresholds': {
                'velocity': self.LANDING_SUCCESS_VELOCITY_THRESHOLD,
                'angle': self.LANDING_SUCCESS_ANGLE_THRESHOLD,
                'position': self.LANDING_SUCCESS_POSITION_THRESHOLD,
            },
        }
        
        base_info['lunar_lander'] = lunar_lander_info
        return base_info
    
    def render(self, mode: Optional[str] = None) -> Optional[Union[np.ndarray, str]]:
        """Визуализировать текущее состояние среды.
        
        Args:
            mode: Режим рендеринга (переопределяет self.render_mode)
            
        Returns:
            Визуализация в зависимости от режима
        """
        render_mode = mode or self.render_mode
        
        if render_mode is None:
            logger.warning("Режим рендеринга не указан")
            return None
        
        try:
            # Временное переопределение режима рендеринга
            if hasattr(self.env, 'render_mode'):
                original_mode = self.env.render_mode
                self.env.render_mode = render_mode
                result = self.env.render()
                self.env.render_mode = original_mode
            else:
                result = self.env.render()
            
            return result
            
        except Exception as e:
            logger.warning(f"Ошибка при рендеринге: {e}")
            return None
    
    def close(self) -> None:
        """Закрыть среду и вывести итоговую статистику.
        """
        # Логируем итоговую статистику
        stats = self.get_landing_statistics()
        logger.info(
            "Итоговая статистика LunarLanderEnvironment",
            total_episodes=stats['total'],
            successful_landings=stats['successful'],
            crash_landings=stats['crashed'],
            timeout_episodes=stats['timeout'],
            success_rate=f"{stats.get('success_rate', 0.0):.1%}",
        )
        
        # Закрываем базовую среду
        super().close()
    
    def __str__(self) -> str:
        """Строковое представление LunarLanderEnvironment.
        """
        stats = self.get_landing_statistics()
        success_rate = stats.get('success_rate', 0.0)
        
        return (
            f"LunarLanderEnvironment(env_id='{self.env_id}', "
            f"episodes={self.episode_count}, "
            f"success_rate={success_rate:.1%}, "
            f"continuous={self.is_continuous})"
        )


# Фабричная функция для удобства

def create_lunar_lander_env(
    env_id: str = 'LunarLander-v3',
    continuous: bool = False,
    **kwargs: Any,
) -> LunarLanderEnvironment:
    """Создать LunarLanderEnvironment с предустановленными параметрами.
    
    Args:
        env_id: Базовый идентификатор среды
        continuous: Использовать непрерывное пространство действий
        **kwargs: Дополнительные аргументы для LunarLanderEnvironment
        
    Returns:
        Созданный LunarLanderEnvironment
    """
    # Выбираем правильный env_id на основе continuous
    if continuous:
        if 'v2' in env_id:
            env_id = 'LunarLanderContinuous-v2'
        else:
            env_id = 'LunarLanderContinuous-v3'
    
    return LunarLanderEnvironment(env_id, **kwargs)