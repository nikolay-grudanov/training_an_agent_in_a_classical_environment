"""Video generation utilities for RL agent demonstration.

Provides tools for rendering trained agents to MP4 video format.
"""

import imageio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

import gymnasium as gym
from stable_baselines3 import A2C, PPO


class VideoGenerator:
    """Generator for RL agent demonstration videos.

    Attributes:
        width: Video frame width
        height: Video frame height
        fps: Frames per second
    """

    def __init__(
        self,
        width: int = 600,
        height: int = 400,
        fps: int = 30,
    ) -> None:
        """Initialize video generator.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps

    def generate_from_model(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        env_id: str = "LunarLander-v3",
        n_episodes: int = 5,
        render_mode: str = "rgb_array",
        show_scores: bool = True,
    ) -> Tuple[str, List[float]]:
        """Generate demonstration video from trained model.

        Args:
            model_path: Path to trained model (.zip file)
            output_path: Path to save MP4 video
            env_id: Gymnasium environment ID
            n_episodes: Number of episodes to record
            render_mode: Render mode ("rgb_array" or "human")
            show_scores: Overlay episode scores on frames

        Returns:
            Tuple of (output_path, list of episode scores)
        """
        # Load model
        model_path = Path(model_path)
        if "PPO" in model_path.name or "ppo" in model_path.name.lower():
            model = PPO.load(str(model_path))
        elif "A2C" in model_path.name or "a2c" in model_path.name.lower():
            model = A2C.load(str(model_path))
        else:
            # Try PPO first, fallback to A2C
            try:
                model = PPO.load(str(model_path))
            except Exception:
                model = A2C.load(str(model_path))

        # Create environment
        env = gym.make(
            env_id,
            render_mode=render_mode,
            width=self.width,
            height=self.height,
        )

        frames: List[np.ndarray] = []
        episode_scores: List[float] = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Render frame
                frame = env.render()
                if frame is None:
                    continue
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                if show_scores:
                    frame = self._overlay_score(frame, episode, episode_reward)
                frames.append(frame)

                # Take action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

                done = terminated or truncated

            episode_scores.append(episode_reward)
            if episode < n_episodes - 1:
                frames.append(
                    self._create_separator_frame(self.width, self.height, episode + 1)
                )

        env.close()

        # Save video
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            str(output_path),
            frames,  # type: ignore[arg-type]
            fps=self.fps,
            codec="libx264",
            quality=8,
        )

        return str(output_path), episode_scores

    def _overlay_score(
        self,
        frame: np.ndarray,
        episode: int,
        score: float,
    ) -> np.ndarray:
        """Overlay episode number and score on frame.

        Args:
            frame: RGB frame array
            episode: Current episode number
            score: Current episode score

        Returns:
            Frame with overlay
        """
        # Add episode info (simplified - just return original frame)
        # Full implementation would use PIL or cv2 for text overlay
        return frame

    def _create_separator_frame(
        self,
        width: int,
        height: int,
        next_episode: int,
    ) -> np.ndarray:
        """Create separator frame between episodes.

        Args:
            width: Frame width
            height: Frame height
            next_episode: Next episode number

        Returns:
            Separator frame
        """
        # Gray separator frame
        return np.full((height, width, 3), 128, dtype=np.uint8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate agent demonstration video")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output video path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes (default: 5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    args = parser.parse_args()

    generator = VideoGenerator(fps=args.fps)
    output_path, scores = generator.generate_from_model(
        model_path=args.model,
        output_path=args.output,
        n_episodes=args.episodes,
    )

    print(f"Video saved: {output_path}")
    print(f"Episode scores: {[f'{s:.1f}' for s in scores]}")
    print(f"Mean score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
