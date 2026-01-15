"""Модуль агентов для обучения с подкреплением.

Этот пакет содержит базовые классы и реализации RL агентов,
включая интеграцию с Stable-Baselines3 и общие интерфейсы.
"""

from .base import Agent, AgentConfig, TrainingResult
from .ppo_agent import PPOAgent, PPOConfig
from .a2c_agent import A2CAgent, A2CConfig
from .sac_agent import SACAgent, SACConfig
from .td3_agent import TD3Agent, TD3Config

__all__ = [
    # Базовые классы
    "Agent",
    "AgentConfig",
    "TrainingResult",
    # PPO агент
    "PPOAgent",
    "PPOConfig",
    # A2C агент
    "A2CAgent",
    "A2CConfig",
    # SAC агент
    "SACAgent",
    "SACConfig",
    # TD3 агент
    "TD3Agent",
    "TD3Config",
]
