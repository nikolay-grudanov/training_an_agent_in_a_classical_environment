"""Video generation functions for RL agent demonstrations."""

import argparse
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import imageio
import numpy as np
from numpy.typing import NDArray

from src.reporting.constants import (
    DEFAULT_ENV_NAME,
    DEFAULT_VIDEO_EPISODES,
    DEFAULT_VIDEO_FPS,
)
from src.reporting.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def create_env(env_name: str = DEFAULT_ENV_NAME) -> gym.Env:
    """Create Gym environment for video recording.

    Args:
        env_name: Name of the Gym environment

    Returns:
        Gym environment with rgb_array render mode
    """
    return gym.make(env_name, render_mode="rgb_array")


# ============================================================================
# T039: Load trained model
# ============================================================================


def load_trained_model(model_path: Path, env: gym.Env) -> Any:
    """Load a trained RL model from a zip file.

    Args:
        model_path: Path to the model .zip file
        env: Gym environment the model was trained on

    Returns:
        Loaded model with predict() method

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Import inside function to avoid issues if stable-baselines3 not installed
        from stable_baselines3 import PPO, A2C

        # Try to determine algorithm from filename or path
        model_path_lower = str(model_path).lower()

        if "ppo" in model_path_lower:
            model = PPO.load(str(model_path), env=env)
        elif "a2c" in model_path_lower:
            model = A2C.load(str(model_path), env=env)
        else:
            # Default to PPO
            logger.warning(f"Could not determine algorithm from path, defaulting to PPO: {model_path}")
            model = PPO.load(str(model_path), env=env)

        logger.info(f"Loaded model from {model_path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e


# ============================================================================
# T040: Record episode
# ============================================================================


def record_episode(
    env: gym.Env,
    policy: Callable[[NDArray[np.uint8]], int],
    max_steps: int | None = None,
) -> list[NDArray[np.uint8]]:
    """Record a single episode by executing the policy.

    Args:
        env: Gym environment
        policy: Function that maps observations to actions
        max_steps: Maximum number of steps to record (None for no limit)

    Returns:
        List of frame arrays (H, W, 3) from the episode
    """
    frames = []
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        # Execute action from policy
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()

        if frame is not None:
            frames.append(frame)

        step += 1
        if max_steps and step >= max_steps:
            logger.debug(f"Reached max steps ({max_steps}) for episode")
            break

    logger.info(f"Recorded {len(frames)} frames for episode")
    return frames


# ============================================================================
# T041: Generate demo video
# ============================================================================


def generate_demo_video(
    env: gym.Env,
    policy: Callable[[NDArray[np.uint8]], int],
    output_path: Path,
    num_episodes: int = DEFAULT_VIDEO_EPISODES,
    fps: int = DEFAULT_VIDEO_FPS,
    max_steps_per_episode: int | None = None,
) -> None:
    """Generate MP4 video of agent executing policy.

    Records multiple episodes and concatenates them into a single video.

    Args:
        env: Gym environment
        policy: Function that maps observations to actions
        output_path: Path to save the MP4 video
        num_episodes: Number of episodes to record
        fps: Frames per second for output video
        max_steps_per_episode: Max steps per episode (None for no limit)
    """
    logger.info(
        f"Generating demo video: {num_episodes} episodes, {fps} FPS, max steps: {max_steps_per_episode}"
    )

    all_frames = []

    # Record each episode
    for episode_idx in range(num_episodes):
        logger.info(f"Recording episode {episode_idx + 1}/{num_episodes}")
        frames = record_episode(env, policy, max_steps=max_steps_per_episode)
        all_frames.extend(frames)

        # Reset environment for next episode
        env.reset()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save video using imageio
    try:
        # Try to use ffmpeg for MP4
        try:
            # Use pillow writer which should work for basic video
            imageio.mimsave(str(output_path), all_frames, fps=fps)
        except Exception:
            # Fallback to basic save
            imageio.mimsave(str(output_path), all_frames)
        logger.info(f"Saved demo video to {output_path} ({len(all_frames)} frames)")
    except Exception as e:
        raise RuntimeError(f"Failed to save video to {output_path}: {e}") from e


# ============================================================================
# T042: Generate demo GIF
# ============================================================================


def generate_demo_gif(
    env: gym.Env,
    policy: Callable[[NDArray[np.uint8]], int],
    output_path: Path,
    num_episodes: int = 3,
    fps: int = DEFAULT_VIDEO_FPS,
    max_steps_per_episode: int | None = None,
    resize_to: tuple[int, int] = (320, 240),
) -> None:
    """Generate lightweight GIF animation of agent execution.

    Creates a smaller, lower-resolution GIF suitable for embedding in
    documentation or README files.

    Args:
        env: Gym environment
        policy: Function that maps observations to actions
        output_path: Path to save the GIF
        num_episodes: Number of episodes to record
        fps: Frames per second for output GIF
        max_steps_per_episode: Max steps per episode (None for no limit)
        resize_to: Target resolution (width, height) for GIF frames
    """
    logger.info(f"Generating demo GIF: {num_episodes} episodes, resolution: {resize_to}")

    all_frames = []

    # Record each episode
    for episode_idx in range(num_episodes):
        logger.info(f"Recording GIF episode {episode_idx + 1}/{num_episodes}")
        frames = record_episode(env, policy, max_steps=max_steps_per_episode)

        # Resize frames for smaller file size
        import cv2  # type: ignore  # cv2 is runtime dependency for GIF generation

        resized_frames = []
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Resize
            resized = cv2.resize(frame_bgr, resize_to, interpolation=cv2.INTER_AREA)
            # Convert back to RGB
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized_frames.append(resized_rgb)

        all_frames.extend(resized_frames)
        env.reset()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save GIF using imageio
    try:
        imageio.mimsave(str(output_path), all_frames, fps=fps, subrectangles=True)
        logger.info(f"Saved demo GIF to {output_path} ({len(all_frames)} frames)")
    except Exception as e:
        raise RuntimeError(f"Failed to save GIF to {output_path}: {e}") from e


# ============================================================================
# T043: Generate top-N videos
# ============================================================================


def generate_top_n_videos(
    comparison_csv: Path,
    output_dir: Path,
    top_n: int = 3,
    fps: int = DEFAULT_VIDEO_FPS,
    num_episodes: int = DEFAULT_VIDEO_EPISODES,
    max_steps_per_episode: int | None = None,
    env_name: str = DEFAULT_ENV_NAME,
) -> None:
    """Generate demo videos for top-N models from comparison CSV.

    Automatically selects the best models by evaluation reward and
    generates a video for each using the trained policy.

    Args:
        comparison_csv: Path to model comparison CSV
        output_dir: Directory to save videos
        top_n: Number of top models to generate videos for
        fps: Frames per second for output videos
        num_episodes: Number of episodes per video
        max_steps_per_episode: Max steps per episode (None for no limit)
        env_name: Name of the Gym environment
    """
    import pandas as pd

    # Load comparison data
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comparison_csv}")

    df = pd.read_csv(comparison_csv)

    # Sort by best_eval_reward and select top-N
    df_sorted = df.sort_values("best_eval_reward", ascending=False).head(top_n)

    logger.info(f"Generating videos for top {len(df_sorted)} models")

    # Create environment
    env = create_env(env_name)

    # Generate video for each top model
    for rank, (_, row) in enumerate(df_sorted.iterrows()):
        experiment_id = row["experiment_id"]
        model_path = Path(row["model_path"])
        reward = row["best_eval_reward"]

        # Generate output filename
        rank_name = {0: "best", 1: "second_best", 2: "third_best"}.get(rank, f"rank_{rank}")
        output_path = output_dir / f"demo_{rank_name}.mp4"

        logger.info(f"Generating video #{rank + 1}: {experiment_id} (reward: {reward:.2f})")

        try:
            # Load trained model
            model = load_trained_model(model_path, env=env)

            # Create policy wrapper
            def policy(obs: NDArray[np.uint8]) -> int:
                action, _ = model.predict(obs, deterministic=True)
                return int(action)

            # Generate video
            generate_demo_video(
                env=env,
                policy=policy,
                output_path=output_path,
                num_episodes=num_episodes,
                fps=fps,
                max_steps_per_episode=max_steps_per_episode,
            )

        except Exception as e:
            logger.error(f"Failed to generate video for {experiment_id}: {e}")
            continue

    # Clean up environment
    env.close()

    logger.info(f"Finished generating videos in {output_dir}")


# ============================================================================
# T044: CLI entry point
# ============================================================================


def main() -> None:
    """Main CLI entry point for video generation.

    Supports subcommands:
    - single: Generate video for a single model
    - top-n: Generate videos for top-N models from comparison
    - gif: Generate GIF animation from a model
    """
    parser = argparse.ArgumentParser(description="Generate demo videos for RL agents")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # single command
    single_parser = subparsers.add_parser("single", help="Generate video for a single model")
    single_parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to trained model (.zip)"
    )
    single_parser.add_argument(
        "--output", type=Path, required=True, help="Path to save video (.mp4)"
    )
    single_parser.add_argument(
        "--episodes", type=int, default=DEFAULT_VIDEO_EPISODES, help="Number of episodes to record"
    )
    single_parser.add_argument(
        "--fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video FPS"
    )
    single_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode",
    )
    single_parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Gym environment name",
    )

    # top-n command
    topn_parser = subparsers.add_parser("top-n", help="Generate videos for top-N models")
    topn_parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("results/reports/model_comparison.csv"),
        help="Path to comparison CSV file",
    )
    topn_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports/videos"),
        help="Directory to save videos",
    )
    topn_parser.add_argument(
        "--number",
        type=int,
        default=3,
        help="Number of top models to generate videos for",
    )
    topn_parser.add_argument(
        "--episodes", type=int, default=DEFAULT_VIDEO_EPISODES, help="Episodes per video"
    )
    topn_parser.add_argument(
        "--fps", type=int, default=DEFAULT_VIDEO_FPS, help="Video FPS"
    )
    topn_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode",
    )
    topn_parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Gym environment name",
    )

    # gif command
    gif_parser = subparsers.add_parser("gif", help="Generate GIF animation")
    gif_parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to trained model (.zip)"
    )
    gif_parser.add_argument(
        "--output", type=Path, required=True, help="Path to save GIF (.gif)"
    )
    gif_parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to record"
    )
    gif_parser.add_argument(
        "--fps", type=int, default=DEFAULT_VIDEO_FPS, help="GIF FPS"
    )
    gif_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode",
    )
    gif_parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[320, 240],
        metavar=("WIDTH", "HEIGHT"),
        help="GIF resolution (width height)",
    )
    gif_parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="Gym environment name",
    )

    args = parser.parse_args()

    # Execute command
    if args.command == "single":
        # Generate video for single model
        env = create_env(args.env)
        model = load_trained_model(args.model_path, env=env)

        def policy(obs: NDArray[np.uint8]) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        generate_demo_video(
            env=env,
            policy=policy,
            output_path=args.output,
            num_episodes=args.episodes,
            fps=args.fps,
            max_steps_per_episode=args.max_steps,
        )

        env.close()

    elif args.command == "top-n":
        # Generate videos for top-N models
        generate_top_n_videos(
            comparison_csv=args.comparison_csv,
            output_dir=args.output_dir,
            top_n=args.number,
            fps=args.fps,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            env_name=args.env,
        )

    elif args.command == "gif":
        # Generate GIF animation
        env = create_env(args.env)
        model = load_trained_model(args.model_path, env=env)

        def policy(obs: NDArray[np.uint8]) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        generate_demo_gif(
            env=env,
            policy=policy,
            output_path=args.output,
            num_episodes=args.episodes,
            fps=args.fps,
            max_steps_per_episode=args.max_steps,
            resize_to=tuple(args.resize),
        )

        env.close()

    else:
        parser.print_help()
        parser.exit(1)


if __name__ == "__main__":
    main()
