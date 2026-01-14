#!/usr/bin/env python3
"""
Пример использования модуля оценки RL агентов.

Демонстрирует основные возможности Evaluator:
- Оценка одного агента
- Сравнение нескольких агентов
- Генерация отчетов
- Экспорт в DataFrame
"""

import gymnasium as gym
import numpy as np
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.agents.base import Agent
from typing import Any, Optional, Tuple


class ProgressCallback:
    """Callback для отслеживания прогресса оценки."""
    
    def __init__(self) -> None:
        self.episodes_completed = 0
    
    def on_episode_start(self, episode: int) -> None:
        """Вызывается в начале каждого эпизода."""
        if episode % 10 == 0:
            print(f"Начало эпизода {episode}")
    
    def on_episode_end(
        self, 
        episode: int, 
        reward: float, 
        length: int, 
        success: bool
    ) -> None:
        """Вызывается в конце каждого эпизода."""
        self.episodes_completed += 1
        if episode % 10 == 0:
            print(f"Эпизод {episode}: награда={reward:.2f}, длина={length}, успех={success}")
    
    def on_evaluation_end(self, metrics) -> None:
        """Вызывается в конце оценки."""
        print(f"Оценка завершена! Обработано {self.episodes_completed} эпизодов")
        print(f"Средняя награда: {metrics.mean_reward:.3f}")


class DummyAgent(Agent):
    """Простой агент-заглушка для демонстрации."""
    
    def __init__(self, name: str, performance_level: float = 1.0) -> None:
        """
        Инициализация агента-заглушки.
        
        Args:
            name: Имя агента
            performance_level: Уровень производительности (влияет на награды)
        """
        self.name = name
        self.performance_level = performance_level
        self.step_count = 0
    
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True, 
        **kwargs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание действия."""
        self.step_count += 1
        
        # Простая стратегия: случайное действие с небольшим смещением
        if len(observation) >= 2:
            # Для CartPole: действие зависит от угла
            action = 1 if observation[2] > 0 else 0
        else:
            action = np.random.randint(0, 2)
        
        # Добавляем немного случайности в зависимости от performance_level
        if np.random.random() > self.performance_level:
            action = 1 - action  # Инвертируем действие
        
        return np.array([action]), None
    
    def _create_model(self) -> Any:
        """Заглушка для создания модели."""
        return None
    
    def train(self, *args, **kwargs) -> Any:
        """Заглушка для обучения."""
        return None
    
    def save(self, path: str) -> None:
        """Заглушка для сохранения."""
        pass
    
    @classmethod
    def load(cls, path: str, env: Optional[Any] = None, **kwargs: Any) -> "DummyAgent":
        """Заглушка для загрузки."""
        return cls("loaded_agent")


def create_dummy_agent(env: gym.Env, name: str, performance: float = 0.8) -> DummyAgent:
    """Создание агента-заглушки для демонстрации."""
    return DummyAgent(name=name, performance_level=performance)


def main() -> None:
    """Основная функция демонстрации."""
    print("🤖 Демонстрация модуля оценки RL агентов")
    print("=" * 50)
    
    # Создание среды
    env = gym.make("CartPole-v1")
    env_name = getattr(env.spec, 'id', 'CartPole-v1') if env.spec else 'CartPole-v1'
    print(f"Создана среда: {env_name}")
    
    # Создание оценщика
    evaluator = Evaluator(
        env=env,
        success_threshold=200.0,  # Для CartPole успех = награда >= 200
        confidence_level=0.95,
        random_seed=42,
    )
    print("Создан оценщик агентов")
    
    # Создание агентов для демонстрации
    print("\n📦 Создание агентов...")
    agent1 = create_dummy_agent(env, "PPO_Agent_1")
    agent2 = create_dummy_agent(env, "PPO_Agent_2")
    
    # 1. Оценка одного агента
    print("\n🔍 Оценка одного агента...")
    callback = ProgressCallback()
    
    metrics = evaluator.evaluate_agent(
        agent=agent1,
        num_episodes=50,
        agent_name="PPO_Agent_1",
        callback=callback,
    )
    
    print("\nРезультаты оценки PPO_Agent_1:")
    print(f"  Средняя награда: {metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}")
    print(f"  Средняя длина эпизода: {metrics.mean_episode_length:.1f}")
    print(f"  Доля успешных эпизодов: {metrics.success_rate:.1%}")
    print(f"  Доверительный интервал (95%): [{metrics.reward_ci_lower:.3f}, {metrics.reward_ci_upper:.3f}]")
    
    # 2. Сравнение агентов
    print("\n⚖️ Сравнение агентов...")
    comparison = evaluator.compare_agents(
        agent1=agent1,
        agent2=agent2,
        num_episodes=30,
        agent1_name="PPO_Agent_1",
        agent2_name="PPO_Agent_2",
    )
    
    print("\nРезультаты сравнения:")
    print(f"  Лучший агент: {comparison.better_agent}")
    print(f"  Статистическая значимость (награды): {comparison.reward_significant}")
    print(f"  p-value (награды): {comparison.reward_ttest_pvalue:.4f}")
    print(f"  Размер эффекта (Cohen's d): {comparison.reward_effect_size:.3f}")
    
    # 3. Оценка нескольких агентов
    print("\n📊 Оценка нескольких агентов...")
    agents = {
        "PPO_Agent_1": agent1,
        "PPO_Agent_2": agent2,
    }
    
    results = evaluator.evaluate_multiple_agents(
        agents=agents,  # type: ignore
        num_episodes=20,
    )
    
    print("\nРезультаты всех агентов:")
    for name, metrics in results.items():
        print(f"  {name}: {metrics.mean_reward:.3f} ± {metrics.std_reward:.3f}")
    
    # 4. Генерация отчета
    print("\n📄 Генерация отчета...")
    report = evaluator.generate_report(results)
    
    # Сохранение отчета
    report_path = Path("evaluation_report.txt")
    evaluator.generate_report(results, save_path=report_path)
    print(f"Отчет сохранен в: {report_path}")
    
    # Показать часть отчета
    print("\nФрагмент отчета:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # 5. Экспорт в DataFrame
    print("\n📈 Экспорт в DataFrame...")
    df = evaluator.export_to_dataframe(results)
    print("\nDataFrame с результатами:")
    print(df.to_string(index=False))
    
    # Сохранение в CSV
    csv_path = Path("evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Результаты сохранены в CSV: {csv_path}")
    
    print("\n✅ Демонстрация завершена!")
    print("Созданные файлы:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    
    # Очистка
    env.close()


if __name__ == "__main__":
    main()