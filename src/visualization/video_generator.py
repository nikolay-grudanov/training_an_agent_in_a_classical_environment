"""Генерация видео для демонстрации RL агентов.

Этот модуль предоставляет функциональность для создания видео демонстраций
обученных RL агентов, включая запись эпизодов, создание монтажей прогресса
обучения и сравнительных видео нескольких агентов.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.agents.base import Agent
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Конфигурация для генерации видео.
    
    Содержит все параметры для настройки качества, формата и обработки видео.
    """
    
    # Основные параметры
    fps: int = 30
    format: str = "mp4"  # mp4, gif
    quality: str = "high"  # low, medium, high, presentation
    
    # Разрешение
    width: Optional[int] = None  # None = использовать оригинальное
    height: Optional[int] = None
    
    # Настройки сжатия
    compression: str = "medium"  # low, medium, high
    bitrate: Optional[str] = None  # "2M", "5M", etc.
    
    # Оверлеи и аннотации
    show_metrics: bool = True
    show_episode_info: bool = True
    show_agent_name: bool = True
    
    # Цвета для оверлеев
    text_color: Tuple[int, int, int] = (255, 255, 255)
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 128)  # RGBA
    
    # Шрифт
    font_size: int = 16
    font_path: Optional[str] = None
    
    # Дополнительные параметры
    max_episode_length: int = 1000
    memory_limit_mb: int = 1024  # Лимит памяти для буферизации кадров


class VideoGenerationError(Exception):
    """Исключение для ошибок генерации видео."""
    pass


def setup_recording_environment(
    env_name: str,
    render_mode: str = "rgb_array",
    seed: Optional[int] = None,
    **env_kwargs: Any,
) -> gym.Env:
    """Подготовка среды для записи видео.
    
    Args:
        env_name: Название среды Gymnasium
        render_mode: Режим рендеринга (rgb_array для видео)
        seed: Сид для воспроизводимости
        **env_kwargs: Дополнительные параметры среды
        
    Returns:
        Настроенная среда для записи
        
    Raises:
        VideoGenerationError: Если не удалось создать среду
    """
    try:
        env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        
        if seed is not None:
            set_seed(seed)
            env.reset(seed=seed)
            
        logger.info(f"Среда {env_name} подготовлена для записи")
        return env
        
    except Exception as e:
        logger.error(f"Ошибка создания среды {env_name}: {e}")
        raise VideoGenerationError(f"Не удалось создать среду: {e}") from e


def add_metrics_overlay(
    frame: np.ndarray,
    metrics: Dict[str, Any],
    config: VideoConfig,
) -> np.ndarray:
    """Добавление оверлея с метриками на кадр.
    
    Args:
        frame: Исходный кадр (H, W, 3)
        metrics: Словарь с метриками для отображения
        config: Конфигурация видео
        
    Returns:
        Кадр с добавленным оверлеем
    """
    if not config.show_metrics and not config.show_episode_info:
        return frame
        
    # Конвертация в PIL для работы с текстом
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    
    # Загрузка шрифта
    try:
        if config.font_path:
            font = ImageFont.truetype(config.font_path, config.font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Подготовка текста
    text_lines = []
    
    if config.show_agent_name and "agent_name" in metrics:
        text_lines.append(f"Agent: {metrics['agent_name']}")
        
    if config.show_episode_info:
        if "episode" in metrics:
            text_lines.append(f"Episode: {metrics['episode']}")
        if "step" in metrics:
            text_lines.append(f"Step: {metrics['step']}")
            
    if config.show_metrics:
        if "reward" in metrics:
            text_lines.append(f"Reward: {metrics['reward']:.2f}")
        if "total_reward" in metrics:
            text_lines.append(f"Total: {metrics['total_reward']:.2f}")
        if "action" in metrics:
            action_str = str(metrics['action'])
            if len(action_str) > 20:
                action_str = action_str[:17] + "..."
            text_lines.append(f"Action: {action_str}")
    
    # Отрисовка текста
    if text_lines:
        # Вычисление размеров текстового блока
        line_height = config.font_size + 2
        text_height = len(text_lines) * line_height + 10
        text_width = max(draw.textlength(line, font=font) for line in text_lines) + 20
        
        # Позиция (левый верхний угол)
        x, y = 10, 10
        
        # Фон для текста
        background = Image.new('RGBA', (text_width, text_height), config.background_color)
        pil_image.paste(background, (x, y), background)
        
        # Отрисовка строк
        for i, line in enumerate(text_lines):
            draw.text(
                (x + 10, y + 5 + i * line_height),
                line,
                fill=config.text_color,
                font=font
            )
    
    return np.array(pil_image)


def record_agent_episode(
    agent: Agent,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[VideoConfig] = None,
    episode_seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Запись одного эпизода агента.
    
    Args:
        agent: Обученный агент для записи
        env: Среда или название среды
        output_path: Путь для сохранения видео
        config: Конфигурация видео
        episode_seed: Сид для эпизода
        max_steps: Максимальное количество шагов
        
    Returns:
        Словарь с информацией о записанном эпизоде
        
    Raises:
        VideoGenerationError: При ошибках записи
    """
    if config is None:
        config = VideoConfig()
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Подготовка среды
    if isinstance(env, str):
        env = setup_recording_environment(env, seed=episode_seed)
    
    try:
        # Инициализация записи
        frames = []
        episode_metrics = {
            "total_reward": 0.0,
            "episode_length": 0,
            "agent_name": getattr(agent, 'name', agent.__class__.__name__),
        }
        
        # Сброс среды
        obs, info = env.reset(seed=episode_seed)
        
        # Основной цикл эпизода
        step = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            if max_steps and step >= max_steps:
                break
                
            # Получение действия от агента
            action = agent.predict(obs)
            
            # Шаг в среде
            obs, reward, done, truncated, info = env.step(action)
            
            # Захват кадра
            frame = env.render()
            if frame is None:
                logger.warning(f"Пустой кадр на шаге {step}")
                continue
                
            # Добавление метрик на кадр
            frame_metrics = {
                "step": step,
                "reward": reward,
                "total_reward": episode_metrics["total_reward"] + reward,
                "action": action,
                "agent_name": episode_metrics["agent_name"],
            }
            
            frame_with_overlay = add_metrics_overlay(frame, frame_metrics, config)
            frames.append(frame_with_overlay)
            
            # Обновление метрик
            episode_metrics["total_reward"] += reward
            episode_metrics["episode_length"] = step + 1
            
            step += 1
            
            # Проверка лимита памяти
            if len(frames) * frame.nbytes > config.memory_limit_mb * 1024 * 1024:
                logger.warning("Достигнут лимит памяти, завершение записи")
                break
        
        # Сохранение видео
        if not frames:
            raise VideoGenerationError("Не удалось захватить кадры")
            
        _save_video(frames, output_path, config)
        
        episode_metrics.update({
            "frames_recorded": len(frames),
            "output_path": str(output_path),
            "success": True,
        })
        
        logger.info(
            f"Эпизод записан: {len(frames)} кадров, "
            f"награда: {episode_metrics['total_reward']:.2f}, "
            f"длина: {episode_metrics['episode_length']}"
        )
        
        return episode_metrics
        
    except Exception as e:
        logger.error(f"Ошибка записи эпизода: {e}")
        raise VideoGenerationError(f"Не удалось записать эпизод: {e}") from e
    finally:
        if hasattr(env, 'close'):
            env.close()


def record_multiple_episodes(
    agent: Agent,
    env: Union[str, gym.Env],
    output_dir: Union[str, Path],
    num_episodes: int = 5,
    config: Optional[VideoConfig] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Запись нескольких эпизодов агента.
    
    Args:
        agent: Обученный агент
        env: Среда или название среды
        output_dir: Директория для сохранения видео
        num_episodes: Количество эпизодов для записи
        config: Конфигурация видео
        seeds: Список сидов для эпизодов
        
    Returns:
        Список с информацией о каждом записанном эпизоде
    """
    if config is None:
        config = VideoConfig()
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if seeds is None:
        seeds = list(range(42, 42 + num_episodes))
    elif len(seeds) < num_episodes:
        seeds.extend(range(max(seeds) + 1, max(seeds) + 1 + num_episodes - len(seeds)))
    
    episodes_info = []
    
    for i in range(num_episodes):
        try:
            episode_seed = seeds[i] if i < len(seeds) else None
            output_path = output_dir / f"episode_{i+1:03d}.{config.format}"
            
            logger.info(f"Запись эпизода {i+1}/{num_episodes}")
            
            episode_info = record_agent_episode(
                agent=agent,
                env=env,
                output_path=output_path,
                config=config,
                episode_seed=episode_seed,
            )
            
            episode_info["episode_number"] = i + 1
            episodes_info.append(episode_info)
            
        except Exception as e:
            logger.error(f"Ошибка записи эпизода {i+1}: {e}")
            episodes_info.append({
                "episode_number": i + 1,
                "success": False,
                "error": str(e),
            })
    
    # Сводная статистика
    successful_episodes = [ep for ep in episodes_info if ep.get("success", False)]
    
    logger.info(
        f"Записано {len(successful_episodes)}/{num_episodes} эпизодов. "
        f"Средняя награда: {np.mean([ep['total_reward'] for ep in successful_episodes]):.2f}"
    )
    
    return episodes_info


def create_training_montage(
    agent_checkpoints: List[Tuple[str, Agent]],
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[VideoConfig] = None,
    episodes_per_checkpoint: int = 1,
) -> Dict[str, Any]:
    """Создание монтажа прогресса обучения.
    
    Args:
        agent_checkpoints: Список (название_чекпоинта, агент)
        env: Среда для тестирования
        output_path: Путь для сохранения видео
        config: Конфигурация видео
        episodes_per_checkpoint: Эпизодов на чекпоинт
        
    Returns:
        Информация о созданном монтаже
    """
    if config is None:
        config = VideoConfig()
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_frames = []
    montage_info = {
        "checkpoints": [],
        "total_frames": 0,
    }
    
    try:
        for checkpoint_name, agent in agent_checkpoints:
            logger.info(f"Запись чекпоинта: {checkpoint_name}")
            
            checkpoint_frames = []
            
            for episode in range(episodes_per_checkpoint):
                # Подготовка среды
                if isinstance(env, str):
                    episode_env = setup_recording_environment(env, seed=42 + episode)
                else:
                    episode_env = env
                
                try:
                    # Запись эпизода
                    obs, _ = episode_env.reset(seed=42 + episode)
                    step = 0
                    done = False
                    truncated = False
                    total_reward = 0.0
                    
                    while not done and not truncated and step < config.max_episode_length:
                        action = agent.predict(obs)
                        obs, reward, done, truncated, _ = episode_env.step(action)
                        
                        frame = episode_env.render()
                        if frame is not None:
                            # Добавление информации о чекпоинте
                            frame_metrics = {
                                "agent_name": checkpoint_name,
                                "step": step,
                                "reward": reward,
                                "total_reward": total_reward + reward,
                                "action": action,
                            }
                            
                            frame_with_overlay = add_metrics_overlay(frame, frame_metrics, config)
                            checkpoint_frames.append(frame_with_overlay)
                        
                        total_reward += reward
                        step += 1
                    
                    logger.info(f"Чекпоинт {checkpoint_name}, эпизод {episode + 1}: "
                              f"награда {total_reward:.2f}, шагов {step}")
                    
                finally:
                    if hasattr(episode_env, 'close') and isinstance(env, str):
                        episode_env.close()
            
            # Добавление разделителя между чекпоинтами
            if checkpoint_frames and len(agent_checkpoints) > 1:
                # Добавляем несколько кадров с информацией о переходе
                transition_frame = _create_transition_frame(
                    checkpoint_frames[-1], 
                    f"Переход к: {checkpoint_name}",
                    config
                )
                checkpoint_frames.extend([transition_frame] * (config.fps // 2))  # 0.5 сек
            
            all_frames.extend(checkpoint_frames)
            
            montage_info["checkpoints"].append({
                "name": checkpoint_name,
                "frames": len(checkpoint_frames),
            })
        
        # Сохранение монтажа
        if all_frames:
            _save_video(all_frames, output_path, config)
            montage_info.update({
                "total_frames": len(all_frames),
                "output_path": str(output_path),
                "success": True,
            })
        else:
            raise VideoGenerationError("Не удалось создать кадры для монтажа")
        
        logger.info(f"Монтаж создан: {len(all_frames)} кадров, "
                   f"{len(agent_checkpoints)} чекпоинтов")
        
        return montage_info
        
    except Exception as e:
        logger.error(f"Ошибка создания монтажа: {e}")
        raise VideoGenerationError(f"Не удалось создать монтаж: {e}") from e


def generate_comparison_video(
    agents: List[Tuple[str, Agent]],
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[VideoConfig] = None,
    sync_episodes: bool = True,
) -> Dict[str, Any]:
    """Создание сравнительного видео нескольких агентов.
    
    Args:
        agents: Список (название, агент) для сравнения
        env: Среда для тестирования
        output_path: Путь для сохранения видео
        config: Конфигурация видео
        sync_episodes: Синхронизировать эпизоды по шагам
        
    Returns:
        Информация о созданном сравнительном видео
    """
    if config is None:
        config = VideoConfig()
        
    if len(agents) < 2:
        raise VideoGenerationError("Для сравнения нужно минимум 2 агента")
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Подготовка сред для каждого агента
        envs = []
        for i, (name, agent) in enumerate(agents):
            if isinstance(env, str):
                agent_env = setup_recording_environment(env, seed=42 + i)
            else:
                agent_env = env
            envs.append(agent_env)
        
        # Инициализация эпизодов
        observations = []
        done_flags = []
        truncated_flags = []
        total_rewards = []
        steps = []
        
        for agent_env in envs:
            obs, _ = agent_env.reset(seed=42)
            observations.append(obs)
            done_flags.append(False)
            truncated_flags.append(False)
            total_rewards.append(0.0)
            steps.append(0)
        
        frames = []
        step = 0
        
        # Основной цикл сравнения
        while step < config.max_episode_length:
            # Проверка завершения всех эпизодов
            if sync_episodes and all(done_flags[i] or truncated_flags[i] for i in range(len(agents))):
                break
            
            agent_frames = []
            
            # Получение кадров от каждого агента
            for i, ((name, agent), agent_env) in enumerate(zip(agents, envs)):
                if not done_flags[i] and not truncated_flags[i]:
                    # Получение действия и выполнение шага
                    action = agent.predict(observations[i])
                    obs, reward, done, truncated, _ = agent_env.step(action)
                    
                    observations[i] = obs
                    done_flags[i] = done
                    truncated_flags[i] = truncated
                    total_rewards[i] += reward
                    steps[i] += 1
                    
                    # Захват кадра
                    frame = agent_env.render()
                    if frame is not None:
                        frame_metrics = {
                            "agent_name": name,
                            "step": steps[i],
                            "reward": reward,
                            "total_reward": total_rewards[i],
                            "action": action,
                        }
                        frame_with_overlay = add_metrics_overlay(frame, frame_metrics, config)
                        agent_frames.append(frame_with_overlay)
                    else:
                        # Заполнение пустым кадром
                        if agent_frames:
                            agent_frames.append(np.zeros_like(agent_frames[0]))
                        else:
                            agent_frames.append(np.zeros((200, 200, 3), dtype=np.uint8))
                else:
                    # Агент завершил эпизод, используем последний кадр
                    if agent_frames:
                        agent_frames.append(agent_frames[-1])
                    else:
                        agent_frames.append(np.zeros((200, 200, 3), dtype=np.uint8))
            
            # Объединение кадров в один
            if agent_frames:
                combined_frame = _combine_frames_side_by_side(agent_frames, agents)
                frames.append(combined_frame)
            
            step += 1
        
        # Сохранение видео
        if frames:
            _save_video(frames, output_path, config)
        else:
            raise VideoGenerationError("Не удалось создать кадры для сравнения")
        
        # Закрытие сред
        for agent_env in envs:
            if hasattr(agent_env, 'close') and isinstance(env, str):
                agent_env.close()
        
        comparison_info = {
            "agents": [{"name": name, "total_reward": reward, "steps": step} 
                      for (name, _), reward, step in zip(agents, total_rewards, steps)],
            "total_frames": len(frames),
            "output_path": str(output_path),
            "success": True,
        }
        
        logger.info(f"Сравнительное видео создано: {len(frames)} кадров, "
                   f"{len(agents)} агентов")
        
        return comparison_info
        
    except Exception as e:
        logger.error(f"Ошибка создания сравнительного видео: {e}")
        raise VideoGenerationError(f"Не удалось создать сравнительное видео: {e}") from e


def compress_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    compression_level: str = "medium",
    target_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Сжатие видео для уменьшения размера файла.
    
    Args:
        input_path: Путь к исходному видео
        output_path: Путь для сжатого видео
        compression_level: Уровень сжатия (low, medium, high)
        target_size_mb: Целевой размер в МБ
        
    Returns:
        Информация о сжатии
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise VideoGenerationError(f"Исходный файл не найден: {input_path}")
    
    try:
        # Чтение исходного видео
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']
        
        # Настройки сжатия
        compression_settings = {
            "low": {"quality": 8, "fps_reduction": 1},
            "medium": {"quality": 6, "fps_reduction": 1},
            "high": {"quality": 4, "fps_reduction": 2},
        }
        
        settings = compression_settings.get(compression_level, compression_settings["medium"])
        new_fps = fps // settings["fps_reduction"]
        
        # Сжатие
        with imageio.get_writer(
            output_path,
            fps=new_fps,
            quality=settings["quality"],
            macro_block_size=None
        ) as writer:
            for i, frame in enumerate(reader):
                # Пропуск кадров для уменьшения FPS
                if i % settings["fps_reduction"] == 0:
                    writer.append_data(frame)
        
        reader.close()
        
        # Статистика
        original_size = input_path.stat().st_size / (1024 * 1024)  # МБ
        compressed_size = output_path.stat().st_size / (1024 * 1024)  # МБ
        compression_ratio = compressed_size / original_size
        
        compression_info = {
            "original_size_mb": original_size,
            "compressed_size_mb": compressed_size,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1 - compression_ratio) * 100,
            "original_fps": fps,
            "compressed_fps": new_fps,
            "success": True,
        }
        
        logger.info(f"Видео сжато: {original_size:.1f}МБ → {compressed_size:.1f}МБ "
                   f"({compression_info['size_reduction_percent']:.1f}% экономии)")
        
        return compression_info
        
    except Exception as e:
        logger.error(f"Ошибка сжатия видео: {e}")
        raise VideoGenerationError(f"Не удалось сжать видео: {e}") from e


def _save_video(
    frames: List[np.ndarray],
    output_path: Path,
    config: VideoConfig,
) -> None:
    """Сохранение кадров в видео файл."""
    try:
        if config.format.lower() == "gif":
            # Сохранение как GIF
            imageio.mimsave(
                output_path,
                frames,
                fps=config.fps,
                loop=0,
            )
        else:
            # Сохранение как MP4
            quality_settings = {
                "low": 4,
                "medium": 6,
                "high": 8,
                "presentation": 10,
            }
            
            quality = quality_settings.get(config.quality, 6)
            
            with imageio.get_writer(
                output_path,
                fps=config.fps,
                quality=quality,
                macro_block_size=None,
                ffmpeg_params=['-pix_fmt', 'yuv420p']  # Совместимость
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
        
        logger.info(f"Видео сохранено: {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка сохранения видео: {e}")
        raise VideoGenerationError(f"Не удалось сохранить видео: {e}") from e


def _create_transition_frame(
    base_frame: np.ndarray,
    text: str,
    config: VideoConfig,
) -> np.ndarray:
    """Создание переходного кадра с текстом."""
    # Затемнение базового кадра
    transition_frame = (base_frame * 0.3).astype(np.uint8)
    
    # Добавление текста
    pil_image = Image.fromarray(transition_frame)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype(config.font_path or "arial.ttf", config.font_size * 2)
    except Exception:
        font = ImageFont.load_default()
    
    # Центрирование текста
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (transition_frame.shape[1] - text_width) // 2
    y = (transition_frame.shape[0] - text_height) // 2
    
    # Фон для текста
    padding = 20
    background = Image.new('RGBA', 
                          (text_width + padding * 2, text_height + padding * 2), 
                          (0, 0, 0, 180))
    pil_image.paste(background, (x - padding, y - padding), background)
    
    # Текст
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return np.array(pil_image)


def _combine_frames_side_by_side(
    frames: List[np.ndarray],
    agents: List[Tuple[str, Any]],
) -> np.ndarray:
    """Объединение кадров нескольких агентов в один."""
    if not frames:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Приведение всех кадров к одному размеру
    target_height = max(frame.shape[0] for frame in frames)
    target_width = max(frame.shape[1] for frame in frames)
    
    resized_frames = []
    for frame in frames:
        if frame.shape[:2] != (target_height, target_width):
            pil_frame = Image.fromarray(frame)
            pil_frame = pil_frame.resize((target_width, target_height), Image.LANCZOS)
            resized_frames.append(np.array(pil_frame))
        else:
            resized_frames.append(frame)
    
    # Объединение по горизонтали
    if len(resized_frames) <= 2:
        # Рядом друг с другом
        combined = np.hstack(resized_frames)
    else:
        # Сетка 2x2 или больше
        rows = []
        frames_per_row = 2
        
        for i in range(0, len(resized_frames), frames_per_row):
            row_frames = resized_frames[i:i + frames_per_row]
            
            # Дополнение ряда пустыми кадрами если нужно
            while len(row_frames) < frames_per_row:
                row_frames.append(np.zeros_like(resized_frames[0]))
            
            row = np.hstack(row_frames)
            rows.append(row)
        
        combined = np.vstack(rows)
    
    return combined