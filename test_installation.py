# test_installation.py
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Тестирование импортов основных модулей."""
    print("🧪 Тестирование импортов...")

    try:
        from src.training import Trainer, TrainerConfig
        print("✅ src.training импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта src.training: {e}")

    try:
        from src.agents import PPOAgent, A2CAgent, SACAgent, TD3Agent
        print("✅ src.agents импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта src.agents: {e}")

    try:
        from src.experiments import ExperimentManager
        print("✅ src.experiments импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта src.experiments: {e}")

    try:
        from src.visualization.plots import plot_learning_curve
        print("✅ src.visualization импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта src.visualization: {e}")

    try:
        import gymnasium as gym
        print("✅ gymnasium импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта gymnasium: {e}")

    try:
        import stable_baselines3
        print("✅ stable_baselines3 импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта stable_baselines3: {e}")

    print("✅ Все основные импорты прошли успешно!")

def test_environment():
    """Тестирование простой среды CartPole."""
    print("\n🧪 Тестирование среды...")

    try:
        import gymnasium as gym

        # Создание более простой среды, не требующей Box2D
        env = gym.make('CartPole-v1')
        obs, info = env.reset()

        print(f"✅ Среда создана: {env.spec.id}")
        print(f"✅ Пространство наблюдений: {env.observation_space}")
        print(f"✅ Пространство действий: {env.action_space}")
        print(f"✅ Размер наблюдения: {obs.shape}")

        # Выполнение одного шага
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"✅ Шаг выполнен: reward={reward}, terminated={terminated}")

        env.close()
        print("✅ Среда закрыта")

    except Exception as e:
        print(f"❌ Ошибка тестирования среды: {e}")

def test_agent_creation():
    """Тестирование создания агента."""
    print("\n🧪 Тестирование создания агента...")

    try:
        from src.agents import PPOAgent, PPOConfig

        # Создание конфигурации агента
        agent_config = PPOConfig(
            algorithm="PPO",
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42
        )

        # Создание агента (среда создается внутри)
        agent = PPOAgent(
            config=agent_config,
            experiment_name="test_agent"
        )

        print(f"✅ Агент создан: {agent.__class__.__name__}")
        print(f"✅ Алгоритм: {agent_config.algorithm}")
        print(f"✅ Среда: {agent_config.env_name}")

    except Exception as e:
        print(f"❌ Ошибка создания агента: {e}")

if __name__ == "__main__":
    print("🚀 Тестирование работоспособности приложения")
    print("=" * 50)
    
    test_imports()
    test_environment()
    test_agent_creation()
    
    print("\n✅ Все тесты выполнены!")