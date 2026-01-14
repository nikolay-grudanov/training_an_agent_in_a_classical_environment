"""Высокоуровневые функции для демонстрации обученных RL агентов.

Этот модуль предоставляет удобные функции для создания демонстрационных видео
обученных агентов с различными режимами: одиночный агент, сравнение, прогресс
обучения. Включает автоматическую генерацию названий, пакетную обработку
и интеграцию с результатами обучения.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from src.agents.base import Agent
from src.utils.seeding import set_seed
from src.visualization.video_generator import (
    VideoConfig,
    record_agent_episode,
    record_multiple_episodes,
    create_training_montage,
    generate_comparison_video,
    compress_video,
    setup_recording_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Конфигурация для создания демонстрационных видео.
    
    Расширяет VideoConfig дополнительными параметрами для демонстраций.
    """
    
    # Базовая конфигурация видео
    video_config: VideoConfig = field(default_factory=VideoConfig)
    
    # Настройки демонстрации
    demo_type: str = "best_episode"  # best_episode, average, comparison, progress
    num_episodes: int = 5
    max_episode_length: int = 1000
    
    # Автоматическое именование
    auto_naming: bool = True
    title_prefix: str = "RL Agent Demo"
    include_timestamp: bool = True
    
    # Сжатие и оптимизация
    auto_compress: bool = True
    compression_level: str = "medium"  # low, medium, high
    target_size_mb: Optional[float] = None
    
    # Пакетная обработка
    batch_processing: bool = False
    parallel_episodes: bool = False
    
    # Интеграция с результатами обучения
    use_training_metrics: bool = True
    include_learning_curves: bool = False
    
    # Обработка ошибок
    continue_on_error: bool = True
    max_retries: int = 3
    
    # Дополнительные настройки
    seed_range: Tuple[int, int] = (42, 100)
    deterministic_policy: bool = True


class AgentDemoError(Exception):
    """Исключение для ошибок создания демонстраций агентов."""
    pass


def create_best_episode_demo(
    agent: Agent,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[DemoConfig] = None,
    num_candidates: int = 10,
) -> Dict[str, Any]:
    """Создать демо лучшего эпизода агента.
    
    Запускает несколько эпизодов и выбирает лучший по награде для записи.
    
    Args:
        agent: Обученный агент
        env: Среда или название среды
        output_path: Путь для сохранения видео
        config: Конфигурация демонстрации
        num_candidates: Количество кандидатов для выбора лучшего
        
    Returns:
        Информация о созданном демо
        
    Raises:
        AgentDemoError: При ошибке создания демо
    """
    if config is None:
        config = DemoConfig()
        
    output_path = Path(output_path)
    
    test_env = None
    try:
        logger.info(f"Поиск лучшего эпизода среди {num_candidates} кандидатов")
        
        # Подготовка среды для тестирования
        if isinstance(env, str):
            test_env = setup_recording_environment(env, render_mode="rgb_array")
        else:
            test_env = env
            
        best_reward = float('-inf')
        best_seed = None
        episode_rewards = []
        
        # Поиск лучшего эпизода
        for i in range(num_candidates):
            episode_seed = np.random.randint(*config.seed_range)
            set_seed(episode_seed)
            
            obs, _ = test_env.reset(seed=episode_seed)
            total_reward = 0.0
            step = 0
            done = False
            truncated = False
            
            while not done and not truncated and step < config.max_episode_length:
                action, _ = agent.predict(obs, deterministic=config.deterministic_policy)
                obs, reward, done, truncated, _ = test_env.step(action)
                total_reward += float(reward)
                step += 1
            
            episode_rewards.append(total_reward)
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_seed = episode_seed
                
            logger.debug(f"Кандидат {i+1}: награда {total_reward:.2f}")
        
        if best_seed is None:
            raise AgentDemoError("Не удалось найти подходящий эпизод")
            
        logger.info(f"Лучший эпизод: награда {best_reward:.2f}, сид {best_seed}")
        
        # Генерация названия
        if config.auto_naming:
            timestamp = f"_{int(time.time())}" if config.include_timestamp else ""
            agent_name = getattr(agent, 'name', agent.__class__.__name__)
            output_path = output_path.with_name(
                f"{config.title_prefix}_{agent_name}_best{timestamp}.{config.video_config.format}"
            )
        
        # Запись лучшего эпизода
        episode_info = record_agent_episode(
            agent=agent,
            env=env,
            output_path=output_path,
            config=config.video_config,
            episode_seed=best_seed,
            max_steps=config.max_episode_length,
        )
        
        # Сжатие видео если требуется
        compressed_path = None
        if config.auto_compress:
            compressed_path = output_path.with_name(
                f"{output_path.stem}_compressed{output_path.suffix}"
            )
            compression_info = compress_video(
                input_path=output_path,
                output_path=compressed_path,
                compression_level=config.compression_level,
                target_size_mb=config.target_size_mb,
            )
            logger.info(f"Видео сжато: {compression_info['size_reduction_percent']:.1f}% экономии")
        
        demo_info = {
            "demo_type": "best_episode",
            "agent_name": getattr(agent, 'name', agent.__class__.__name__),
            "best_reward": best_reward,
            "best_seed": best_seed,
            "candidates_tested": num_candidates,
            "reward_statistics": {
                "mean": float(np.mean(episode_rewards)),
                "std": float(np.std(episode_rewards)),
                "min": float(np.min(episode_rewards)),
                "max": float(np.max(episode_rewards)),
            },
            "output_path": str(output_path),
            "compressed_path": str(compressed_path) if compressed_path else None,
            "episode_info": episode_info,
            "success": True,
        }
        
        logger.info(f"Демо лучшего эпизода создано: {output_path}")
        return demo_info
        
    except Exception as e:
        logger.error(f"Ошибка создания демо лучшего эпизода: {e}")
        raise AgentDemoError(f"Не удалось создать демо: {e}") from e
    finally:
        if isinstance(env, str) and test_env is not None:
            try:
                test_env.close()
            except Exception:
                pass  # Игнорируем ошибки закрытия


def create_average_behavior_demo(
    agent: Agent,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[DemoConfig] = None,
) -> Dict[str, Any]:
    """Создать демо среднего поведения агента.
    
    Записывает несколько эпизодов и выбирает тот, который ближе всего к средней награде.
    
    Args:
        agent: Обученный агент
        env: Среда или название среды
        output_path: Путь для сохранения видео
        config: Конфигурация демонстрации
        
    Returns:
        Информация о созданном демо
    """
    if config is None:
        config = DemoConfig()
        
    output_path = Path(output_path)
    
    try:
        logger.info(f"Создание демо среднего поведения ({config.num_episodes} эпизодов)")
        
        # Запись нескольких эпизодов для анализа
        temp_dir = output_path.parent / "temp_episodes"
        episodes_info = record_multiple_episodes(
            agent=agent,
            env=env,
            output_dir=temp_dir,
            num_episodes=config.num_episodes,
            config=config.video_config,
            seeds=list(range(42, 42 + config.num_episodes)),
        )
        
        # Анализ наград
        successful_episodes = [ep for ep in episodes_info if ep.get("success", False)]
        if not successful_episodes:
            raise AgentDemoError("Не удалось записать ни одного успешного эпизода")
            
        rewards = [ep["total_reward"] for ep in successful_episodes]
        mean_reward = np.mean(rewards)
        
        # Поиск эпизода ближайшего к среднему
        closest_idx = np.argmin([abs(reward - mean_reward) for reward in rewards])
        closest_episode = successful_episodes[closest_idx]
        
        logger.info(f"Средняя награда: {mean_reward:.2f}, "
                   f"выбранный эпизод: {closest_episode['total_reward']:.2f}")
        
        # Генерация названия
        if config.auto_naming:
            timestamp = f"_{int(time.time())}" if config.include_timestamp else ""
            agent_name = getattr(agent, 'name', agent.__class__.__name__)
            output_path = output_path.with_name(
                f"{config.title_prefix}_{agent_name}_average{timestamp}.{config.video_config.format}"
            )
        
        # Копирование выбранного видео
        import shutil
        source_path = Path(closest_episode["output_path"])
        shutil.copy2(source_path, output_path)
        
        # Очистка временных файлов
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Сжатие если требуется
        compressed_path = None
        if config.auto_compress:
            compressed_path = output_path.with_name(
                f"{output_path.stem}_compressed{output_path.suffix}"
            )
            compress_video(
                input_path=output_path,
                output_path=compressed_path,
                compression_level=config.compression_level,
            )
        
        demo_info = {
            "demo_type": "average_behavior",
            "agent_name": getattr(agent, 'name', agent.__class__.__name__),
            "selected_reward": closest_episode["total_reward"],
            "mean_reward": mean_reward,
            "episodes_analyzed": len(successful_episodes),
            "reward_statistics": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
            "output_path": str(output_path),
            "compressed_path": str(compressed_path) if compressed_path else None,
            "success": True,
        }
        
        logger.info(f"Демо среднего поведения создано: {output_path}")
        return demo_info
        
    except Exception as e:
        logger.error(f"Ошибка создания демо среднего поведения: {e}")
        raise AgentDemoError(f"Не удалось создать демо: {e}") from e


def create_before_after_demo(
    untrained_agent: Agent,
    trained_agent: Agent,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[DemoConfig] = None,
) -> Dict[str, Any]:
    """Создать демо сравнения до и после обучения.
    
    Args:
        untrained_agent: Необученный агент
        trained_agent: Обученный агент
        env: Среда для тестирования
        output_path: Путь для сохранения видео
        config: Конфигурация демонстрации
        
    Returns:
        Информация о созданном сравнительном демо
    """
    if config is None:
        config = DemoConfig()
        
    output_path = Path(output_path)
    
    try:
        logger.info("Создание демо сравнения до/после обучения")
        
        # Генерация названия
        if config.auto_naming:
            timestamp = f"_{int(time.time())}" if config.include_timestamp else ""
            agent_name = getattr(trained_agent, 'name', trained_agent.__class__.__name__)
            output_path = output_path.with_name(
                f"{config.title_prefix}_{agent_name}_before_after{timestamp}.{config.video_config.format}"
            )
        
        # Создание сравнительного видео
        agents = [
            ("До обучения", untrained_agent),
            ("После обучения", trained_agent),
        ]
        
        comparison_info = generate_comparison_video(
            agents=agents,
            env=env,
            output_path=output_path,
            config=config.video_config,
            sync_episodes=True,
        )
        
        # Сжатие если требуется
        compressed_path = None
        if config.auto_compress:
            compressed_path = output_path.with_name(
                f"{output_path.stem}_compressed{output_path.suffix}"
            )
            compress_video(
                input_path=output_path,
                output_path=compressed_path,
                compression_level=config.compression_level,
            )
        
        demo_info = {
            "demo_type": "before_after",
            "agent_name": getattr(trained_agent, 'name', trained_agent.__class__.__name__),
            "comparison_info": comparison_info,
            "output_path": str(output_path),
            "compressed_path": str(compressed_path) if compressed_path else None,
            "success": True,
        }
        
        logger.info(f"Демо сравнения до/после создано: {output_path}")
        return demo_info
        
    except Exception as e:
        logger.error(f"Ошибка создания демо сравнения: {e}")
        raise AgentDemoError(f"Не удалось создать демо сравнения: {e}") from e


def create_training_progress_demo(
    checkpoint_paths: List[Union[str, Path]],
    agent_class: type,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[DemoConfig] = None,
    checkpoint_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Создать демо прогресса обучения через чекпоинты.
    
    Args:
        checkpoint_paths: Пути к чекпоинтам модели
        agent_class: Класс агента для загрузки чекпоинтов
        env: Среда для тестирования
        output_path: Путь для сохранения видео
        config: Конфигурация демонстрации
        checkpoint_names: Названия чекпоинтов (автогенерация если None)
        
    Returns:
        Информация о созданном демо прогресса
    """
    if config is None:
        config = DemoConfig()
        
    output_path = Path(output_path)
    
    try:
        logger.info(f"Создание демо прогресса обучения ({len(checkpoint_paths)} чекпоинтов)")
        
        # Загрузка чекпоинтов
        agent_checkpoints = []
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            try:
                # Загрузка агента из чекпоинта
                agent = agent_class.load(str(checkpoint_path), env=env)
                
                # Генерация названия чекпоинта
                if checkpoint_names and i < len(checkpoint_names):
                    name = checkpoint_names[i]
                else:
                    name = f"Чекпоинт {i+1}"
                    
                agent_checkpoints.append((name, agent))
                logger.debug(f"Загружен чекпоинт: {name}")
                
            except Exception as e:
                logger.warning(f"Ошибка загрузки чекпоинта {checkpoint_path}: {e}")
                if not config.continue_on_error:
                    raise
        
        if not agent_checkpoints:
            raise AgentDemoError("Не удалось загрузить ни одного чекпоинта")
        
        # Генерация названия
        if config.auto_naming:
            timestamp = f"_{int(time.time())}" if config.include_timestamp else ""
            agent_name = agent_checkpoints[0][1].__class__.__name__
            output_path = output_path.with_name(
                f"{config.title_prefix}_{agent_name}_progress{timestamp}.{config.video_config.format}"
            )
        
        # Создание монтажа прогресса
        montage_info = create_training_montage(
            agent_checkpoints=agent_checkpoints,
            env=env,
            output_path=output_path,
            config=config.video_config,
            episodes_per_checkpoint=1,
        )
        
        # Сжатие если требуется
        compressed_path = None
        if config.auto_compress:
            compressed_path = output_path.with_name(
                f"{output_path.stem}_compressed{output_path.suffix}"
            )
            compress_video(
                input_path=output_path,
                output_path=compressed_path,
                compression_level=config.compression_level,
            )
        
        demo_info = {
            "demo_type": "training_progress",
            "checkpoints_loaded": len(agent_checkpoints),
            "checkpoint_names": [name for name, _ in agent_checkpoints],
            "montage_info": montage_info,
            "output_path": str(output_path),
            "compressed_path": str(compressed_path) if compressed_path else None,
            "success": True,
        }
        
        logger.info(f"Демо прогресса обучения создано: {output_path}")
        return demo_info
        
    except Exception as e:
        logger.error(f"Ошибка создания демо прогресса: {e}")
        raise AgentDemoError(f"Не удалось создать демо прогресса: {e}") from e


def create_multi_agent_comparison(
    agents: List[Tuple[str, Agent]],
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    config: Optional[DemoConfig] = None,
) -> Dict[str, Any]:
    """Создать демо сравнения нескольких агентов.
    
    Args:
        agents: Список (название, агент) для сравнения
        env: Среда для тестирования
        output_path: Путь для сохранения видео
        config: Конфигурация демонстрации
        
    Returns:
        Информация о созданном сравнительном демо
    """
    if config is None:
        config = DemoConfig()
        
    output_path = Path(output_path)
    
    try:
        logger.info(f"Создание демо сравнения {len(agents)} агентов")
        
        # Генерация названия
        if config.auto_naming:
            timestamp = f"_{int(time.time())}" if config.include_timestamp else ""
            agent_names = "_vs_".join([name.replace(" ", "") for name, _ in agents[:3]])
            if len(agents) > 3:
                agent_names += f"_and_{len(agents)-3}_more"
            output_path = output_path.with_name(
                f"{config.title_prefix}_{agent_names}{timestamp}.{config.video_config.format}"
            )
        
        # Создание сравнительного видео
        comparison_info = generate_comparison_video(
            agents=agents,
            env=env,
            output_path=output_path,
            config=config.video_config,
            sync_episodes=True,
        )
        
        # Сжатие если требуется
        compressed_path = None
        if config.auto_compress:
            compressed_path = output_path.with_name(
                f"{output_path.stem}_compressed{output_path.suffix}"
            )
            compress_video(
                input_path=output_path,
                output_path=compressed_path,
                compression_level=config.compression_level,
            )
        
        demo_info = {
            "demo_type": "multi_agent_comparison",
            "agents_compared": len(agents),
            "agent_names": [name for name, _ in agents],
            "comparison_info": comparison_info,
            "output_path": str(output_path),
            "compressed_path": str(compressed_path) if compressed_path else None,
            "success": True,
        }
        
        logger.info(f"Демо сравнения агентов создано: {output_path}")
        return demo_info
        
    except Exception as e:
        logger.error(f"Ошибка создания демо сравнения агентов: {e}")
        raise AgentDemoError(f"Не удалось создать демо сравнения: {e}") from e


def create_batch_demos(
    agents: List[Tuple[str, Agent]],
    env: Union[str, gym.Env],
    output_dir: Union[str, Path],
    demo_types: List[str],
    config: Optional[DemoConfig] = None,
) -> Dict[str, Any]:
    """Создать пакет демонстраций для нескольких агентов.
    
    Args:
        agents: Список (название, агент) для демонстрации
        env: Среда для тестирования
        output_dir: Директория для сохранения видео
        demo_types: Типы демо для создания (best_episode, average, etc.)
        config: Конфигурация демонстрации
        
    Returns:
        Сводная информация о созданных демо
    """
    if config is None:
        config = DemoConfig()
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_info = {
        "agents_processed": 0,
        "demos_created": 0,
        "demos_failed": 0,
        "demo_results": [],
        "success": True,
    }
    
    try:
        logger.info(f"Пакетное создание демо: {len(agents)} агентов, {len(demo_types)} типов")
        
        for agent_name, agent in agents:
            agent_dir = output_dir / agent_name.replace(" ", "_")
            agent_dir.mkdir(exist_ok=True)
            
            agent_results = {
                "agent_name": agent_name,
                "demos": [],
                "success": True,
            }
            
            for demo_type in demo_types:
                try:
                    output_path = agent_dir / f"{demo_type}.{config.video_config.format}"
                    
                    if demo_type == "best_episode":
                        result = create_best_episode_demo(
                            agent=agent,
                            env=env,
                            output_path=output_path,
                            config=config,
                        )
                    elif demo_type == "average":
                        result = create_average_behavior_demo(
                            agent=agent,
                            env=env,
                            output_path=output_path,
                            config=config,
                        )
                    else:
                        logger.warning(f"Неизвестный тип демо: {demo_type}")
                        continue
                    
                    agent_results["demos"].append(result)
                    batch_info["demos_created"] += 1
                    
                    logger.info(f"Создано демо {demo_type} для {agent_name}")
                    
                except Exception as e:
                    logger.error(f"Ошибка создания демо {demo_type} для {agent_name}: {e}")
                    batch_info["demos_failed"] += 1
                    agent_results["success"] = False
                    
                    if not config.continue_on_error:
                        raise
            
            batch_info["demo_results"].append(agent_results)
            batch_info["agents_processed"] += 1
        
        logger.info(f"Пакетное создание завершено: {batch_info['demos_created']} успешно, "
                   f"{batch_info['demos_failed']} ошибок")
        
        return batch_info
        
    except Exception as e:
        logger.error(f"Ошибка пакетного создания демо: {e}")
        batch_info["success"] = False
        raise AgentDemoError(f"Не удалось создать пакет демо: {e}") from e


def generate_demo_summary(
    demo_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Создать сводный отчет о созданных демонстрациях.
    
    Args:
        demo_results: Результаты создания демо
        output_path: Путь для сохранения отчета
    """
    output_path = Path(output_path)
    
    try:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_demos": len(demo_results),
            "successful_demos": len([r for r in demo_results if r.get("success", False)]),
            "failed_demos": len([r for r in demo_results if not r.get("success", False)]),
            "demo_types": list(set(r.get("demo_type", "unknown") for r in demo_results)),
            "demos": demo_results,
        }
        
        # Сохранение в JSON
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Сводный отчет сохранен: {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка создания сводного отчета: {e}")


def auto_demo_from_training_results(
    training_results_dir: Union[str, Path],
    agent_class: type,
    env: Union[str, gym.Env],
    output_dir: Union[str, Path],
    config: Optional[DemoConfig] = None,
) -> Dict[str, Any]:
    """Автоматически создать демо на основе результатов обучения.
    
    Сканирует директорию с результатами обучения и создает демонстрации
    для найденных моделей и чекпоинтов.
    
    Args:
        training_results_dir: Директория с результатами обучения
        agent_class: Класс агента для загрузки моделей
        env: Среда для тестирования
        output_dir: Директория для сохранения демо
        config: Конфигурация демонстрации
        
    Returns:
        Информация о созданных демо
    """
    if config is None:
        config = DemoConfig()
        
    training_results_dir = Path(training_results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Автоматическое создание демо из {training_results_dir}")
        
        # Поиск моделей и чекпоинтов
        model_files = list(training_results_dir.rglob("*.zip"))  # SB3 модели
        
        if not model_files:
            logger.warning("Не найдено моделей для демонстрации")
            return {"success": False, "error": "Модели не найдены"}
        
        demo_results = []
        
        for model_path in model_files:
            try:
                # Загрузка агента
                agent = agent_class.load(str(model_path), env=env)
                agent_name = model_path.stem
                
                # Создание демо лучшего эпизода
                demo_path = output_dir / f"{agent_name}_auto_demo.{config.video_config.format}"
                
                result = create_best_episode_demo(
                    agent=agent,
                    env=env,
                    output_path=demo_path,
                    config=config,
                )
                
                result["model_path"] = str(model_path)
                demo_results.append(result)
                
                logger.info(f"Создано автоматическое демо для {agent_name}")
                
            except Exception as e:
                logger.error(f"Ошибка создания демо для {model_path}: {e}")
                if not config.continue_on_error:
                    raise
        
        # Создание сводного отчета
        summary_path = output_dir / "demo_summary.json"
        generate_demo_summary(demo_results, summary_path)
        
        auto_demo_info = {
            "models_found": len(model_files),
            "demos_created": len(demo_results),
            "demo_results": demo_results,
            "summary_path": str(summary_path),
            "success": True,
        }
        
        logger.info(f"Автоматическое создание завершено: {len(demo_results)} демо")
        return auto_demo_info
        
    except Exception as e:
        logger.error(f"Ошибка автоматического создания демо: {e}")
        raise AgentDemoError(f"Не удалось создать автоматические демо: {e}") from e


# Удобные функции-обертки для быстрого использования

def quick_demo(
    agent: Agent,
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
    demo_type: str = "best_episode",
) -> str:
    """Быстрое создание демо с настройками по умолчанию.
    
    Args:
        agent: Обученный агент
        env: Среда для тестирования
        output_path: Путь для сохранения
        demo_type: Тип демо (best_episode, average)
        
    Returns:
        Путь к созданному видео
    """
    config = DemoConfig(
        video_config=VideoConfig(fps=30, quality="high"),
        auto_compress=True,
    )
    
    if demo_type == "best_episode":
        result = create_best_episode_demo(agent, env, output_path, config)
    elif demo_type == "average":
        result = create_average_behavior_demo(agent, env, output_path, config)
    else:
        raise ValueError(f"Неподдерживаемый тип демо: {demo_type}")
    
    return result["compressed_path"] or result["output_path"]


def quick_comparison(
    agents: List[Tuple[str, Agent]],
    env: Union[str, gym.Env],
    output_path: Union[str, Path],
) -> str:
    """Быстрое создание сравнительного демо.
    
    Args:
        agents: Список (название, агент) для сравнения
        env: Среда для тестирования
        output_path: Путь для сохранения
        
    Returns:
        Путь к созданному видео
    """
    config = DemoConfig(
        video_config=VideoConfig(fps=30, quality="high"),
        auto_compress=True,
    )
    
    result = create_multi_agent_comparison(agents, env, output_path, config)
    return result["compressed_path"] or result["output_path"]