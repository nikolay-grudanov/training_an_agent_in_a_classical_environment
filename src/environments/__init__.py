"""Модуль для работы с RL средами.

Предоставляет обертки и специализированные среды для обучения RL агентов.
"""

from .wrapper import EnvironmentWrapper
from .lunar_lander import LunarLanderEnvironment

__all__ = [
    "EnvironmentWrapper",
    "LunarLanderEnvironment",
]
