"""Evaluation utilities for trained RL agents.

Provides functions for evaluating trained models and computing performance metrics.
"""

from typing import Any, Dict

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_agent(
    model_path: str,
    env_id: str = "LunarLander-v3",
    n_eval_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained agent on the specified environment.

    Args:
        model_path: Path to saved model (.zip file)
        env_id: Gymnasium environment ID
        n_eval_episodes: Number of evaluation episodes
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions

    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    if model_path.endswith(".zip"):
        if "PPO" in model_path or "ppo" in model_path.lower():
            model = PPO.load(model_path)
        elif "A2C" in model_path or "a2c" in model_path.lower():
            model = A2C.load(model_path)
        else:
            # Auto-detect based on file contents
            model = PPO.load(model_path)
    else:
        model = PPO.load(model_path)

    # Create evaluation environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        render=render,
    )

    env.close()

    # Handle both scalar and list return types
    mean_reward_val = (
        float(mean_reward[0]) if isinstance(mean_reward, list) else float(mean_reward)
    )
    std_reward_val = (
        float(std_reward[0]) if isinstance(std_reward, list) else float(std_reward)
    )

    return {
        "mean_reward": mean_reward_val,
        "std_reward": std_reward_val,
        "n_episodes": n_eval_episodes,
        "convergence_achieved": mean_reward_val >= 200.0,
    }
