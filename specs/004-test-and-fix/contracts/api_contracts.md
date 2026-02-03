# API Contracts: ML Project Architecture

**Feature**: 004-test-and-fix | **Date**: 2026-02-04
**Project Type**: Machine Learning (Reinforcement Learning) | **Phase**: 1 (Design & Contracts)

---

## 📋 NOTE: ML Project Architecture

**Это ML проект (Reinforcement Learning), NOT traditional web application.**

**Ключевые отличия**:
- ❌ **Нет REST API** - Модели используются напрямую через Python API (Stable-Baselines3)
- ❌ **Нет GraphQL API** - Нет запросов от клиентов
- ❌ **Нет базы данных** - Данные хранятся в файлах (CSV, JSON, ZIP)
- ✅ **Есть Python API** - Stable-Baselines3, Gymnasium, PyTorch
- ✅ **Есть CLI** - Командная строка для обучения и тестирования
- ✅ **Есть конфигурации** - YAML/JSON файлы для экспериментов

**Архитектура проекта**:
```
User (Developer/Researcher)
    │
    │ CLI (Command Line Interface)
    │ python -m src.experiments.completion.baseline_training ...
    │
    ▼
Python API
    │
    ├── Stable-Baselines3 API (PPO, A2C, TD3)
    ├── Gymnasium API (Environments: LunarLander-v3)
    └── PyTorch API (Deep Learning)
    │
    ▼
File System
    │
    ├── Models (.zip files)
    ├── Metrics (CSV files)
    ├── Configurations (JSON files)
    ├── Checkpoints (ZIP files)
    └── Visualizations (PNG, MP4 files)
```

**Полную документацию см. в папке `/docs/`**:
- [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт
- [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - Решение проблем

---

## Python API Contracts

### 1. Stable-Baselines3 API (PPO Agent)

**Библиотека**: `stable_baselines3.ppo.PPO`

**Основные методы**:

#### Конструктор
```python
def __init__(
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[str, Env, VecEnv],
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.0,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[str, th.device] = "auto",
    _init_setup_model: bool = True,
) -> None:
    """
    Proximal Policy Optimization (PPO)

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, etc.)
        env: The environment to learn from
        learning_rate: The learning rate, it can be a function
        n_steps: The number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        ent_coef: Entropy coefficient for loss calculation
        verbose: Verbosity level: 0 for no output, 1 for info messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, auto)
        _init_setup_model: Whether or not to build the network at the creation of the instance

    Returns:
        PPO agent instance
    """
```

**Использование**:
```python
from stable_baselines3 import PPO

# Создание PPO агента
model = PPO(
    policy="MlpPolicy",
    env="LunarLander-v3",
    learning_rate=3e-4,
    n_steps=1024,
    n_epochs=4,
    gamma=0.999,
    ent_coef=0.01,
    gae_lambda=0.98,
    verbose=1,
    seed=42,
    device="cpu"
)
```

---

#### Метод обучения
```python
def learn(
    self,
    total_timesteps: int,
    callback: Optional[Union[list, BaseCallback, MaybeCallback]] = None,
    log_interval: int = 100,
    tb_log_name: Optional[str] = "PPO",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
) -> "PPO":
    """
    Return a trained model.

    Args:
        total_timesteps: The total number of samples (env steps) to train on
        callback: callback(s) called at every step with state of the algorithm
        log_interval: The number of timesteps before logging
        tb_log_name: The name of the run for TensorBoard
        reset_num_timesteps: Whether or not to reset the current timestep number
        progress_bar: Display a progress bar using tqdm and rich

    Returns:
        the trained model
    """
```

**Использование**:
```python
# Обучение агента
model.learn(
    total_timesteps=500000,
    callback=[checkpoint_callback, eval_callback, metrics_callback]
)
```

---

#### Метод сохранения
```python
def save(
    self,
    path: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> None:
    """
    Save the model to the given path.

    Args:
        path: the path to save the model to
        include: name of variables to include
        exclude: name of variables to exclude
    """
```

**Использование**:
```python
# Сохранение модели
model.save("results/experiments/ppo_seed42/ppo_seed42_model.zip")
```

---

#### Метод загрузки (static)
```python
@staticmethod
def load(
    path: str,
    env: Optional[GymEnv] = None,
    device: Union[str, th.device] = "auto",
    custom_objects: Optional[Dict[str, Any]] = None,
    print_system_info: bool = False,
    force_reset: bool = True,
    **kwargs,
) -> "BaseAlgorithm":
    """
    Load the model from a zip-file.

    Args:
        path: The path to the file (or a file-like)
        env: the environment to use to evaluate the model if it was loaded with a different environment
        device: Device on which the code should run
        custom_objects: Dictionary of objects to replace upon loading
        print_system_info: Whether to print system info from the saved model
        force_reset: Force call to `reset_num_timesteps` (can be used to continue training)

    Returns:
        The loaded model
    """
```

**Использование**:
```python
# Загрузка модели
model = PPO.load("results/experiments/ppo_seed42/ppo_seed42_model.zip")
```

---

#### Метод предсказания
```python
def predict(
    self,
    observation: Union[np.ndarray, Dict[str, np.ndarray]],
    state: Optional[Tuple[np.ndarray, ...]] = None,
    episode_start: Optional[np.ndarray] = None,
    deterministic: bool = False,
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    """
    Get the model's action(s) from an observation.

    Args:
        observation: the input observation
        state: The last states (can be None, used in recurrent policies)
        episode_start: These last episode start(s) (can be None, used in recurrent policies)
        deterministic: Whether to use stochastic or deterministic actions

    Returns:
        The model's action and the next state (used in recurrent policies)
    """
```

**Использование**:
```python
# Предсказание действия
action, states = model.predict(observation, deterministic=True)
```

---

### 2. Gymnasium API (Environment)

**Библиотека**: `gymnasium`

#### Создание среды
```python
def make(
    id: str,
    max_episode_steps: Optional[int] = None,
    autoreset: Optional[bool] = None,
    disable_env_checker: Optional[bool] = None,
    **kwargs,
) -> Env:
    """
    Create an environment from an ID.

    Args:
        id: The environment ID
        max_episode_steps: The maximum number of steps that an episode lasts
        autoreset: Whether to automatically reset the environment
        disable_env_checker: Whether to disable the environment checker
        **kwargs: Additional keyword arguments

    Returns:
        An instance of the environment
    """
```

**Использование**:
```python
import gymnasium as gym

# Создание среды LunarLander-v3
env = gym.make("LunarLander-v3", render_mode="rgb_array")
```

---

#### Метод сброса
```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> Tuple[ObsType, Dict[str, Any]]:
    """
    Reset the environment to an initial state.

    Args:
        seed: The seed for the PRNG
        options: Additional info to reset the environment with

    Returns:
        The initial observation and info dictionary
    """
```

**Использование**:
```python
# Сброс среды
observation, info = env.reset(seed=42)
```

---

#### Метод шага
```python
def step(
    self,
    action: ActType,
) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
    """
    Execute one step in the environment.

    Args:
        action: The action to take

    Returns:
        observation, reward, terminated, truncated, info
    """
```

**Использование**:
```python
# Шаг среды
observation, reward, terminated, truncated, info = env.step(action)
```

---

#### Метод рендеринга
```python
def render(self) -> Optional[Union[np.ndarray, str]]:
    """
    Render the environment to the screen

    Returns:
        None or a numpy array of RGB values
    """
```

**Использование**:
```python
# Рендеринг
frame = env.render()
```

---

### 3. CLI Contract (Command Line Interface)

**Скрипт**: `src/experiments/completion/baseline_training.py`

**Основные команды**:

#### Базовое обучение (default параметры)
```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 200000 \
    --seed 42
```

**Параметры**:
- `--algo`: Алгоритм (ppo, a2c, td3)
- `--timesteps`: Количество шагов обучения (int)
- `--seed`: Random seed (int, по умолчанию 42)

---

#### Оптимизированное обучение (CPU)
```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 500000 \
    --seed 42 \
    --gamma 0.999 \
    --ent-coef 0.01 \
    --gae-lambda 0.98 \
    --n-steps 1024 \
    --n-epochs 4 \
    --device cpu
```

**Параметры**:
- `--gamma`: Discount factor (float, по умолчанию 0.99)
- `--ent-coef`: Entropy coefficient (float, по умолчанию 0.0)
- `--gae-lambda`: GAE lambda (float, по умолчанию 0.95)
- `--n-steps`: Number of steps per update (int, по умолчанию 2048)
- `--n-epochs`: Number of epochs (int, по умолчанию 10)
- `--device`: Device (auto/cpu/gpu/cuda/mps, по умолчанию auto)

---

#### Оптимизированное обучение (GPU)
```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 500000 \
    --seed 42 \
    --gamma 0.999 \
    --ent-coef 0.01 \
    --gae-lambda 0.98 \
    --n-steps 1024 \
    --n-epochs 4 \
    --device auto
```

**Переменные окружения**:
- `CUDA_VISIBLE_DEVICES`: NVIDIA GPU ID (для ROCm, установить "" для CPU)
- `HIP_VISIBLE_DEVICES`: AMD GPU ID (для ROCm, установить "" для CPU)

---

## Интеграционные потоки (Integration Flows)

### 1. Полный пайплайн обучения

```
[CLI Command]
    │
    ▼
[Initialize Environment] (gym.make)
    │
    ▼
[Initialize Agent] (PPO.__init__)
    │
    ▼
[Train Agent] (model.learn)
    │
    ├── [Checkpoint Callback] - Save every 100K steps
    ├── [Eval Callback] - Evaluate every 10K steps
    └── [Metrics Callback] - Log metrics every step
    │
    ▼
[Save Final Model] (model.save)
    │
    ▼
[Generate Plots] (matplotlib)
    │
    ▼
[Generate Video] (gymnasium录制)
```

---

### 2. Полный пайплайн инференса

```
[Load Model] (PPO.load)
    │
    ▼
[Initialize Environment] (gym.make)
    │
    ▼
[Reset Environment] (env.reset)
    │
    ▼
[Loop Episodes]
    │
    ├── [Predict Action] (model.predict)
    ├── [Step Environment] (env.step)
    └── [Check Done] (terminated or truncated)
    │
    ▼
[Calculate Statistics] (mean, std, min, max)
    │
    ▼
[Report Results]
```

---

## Error Handling

### Ошибки Stable-Baselines3

| Ошибка | Причина | Решение |
|--------|---------|----------|
| `ValueError: Unknown environment` | Невалидный ID среды | Проверить `gym.make("LunarLander-v3")` |
| `RuntimeError: CUDA out of memory` | Не хватает памяти GPU | Уменьшить `batch_size` или использовать CPU |
| `UserWarning: You are trying to run PPO on the GPU` | GPU предупреждение на CPU | Установить `CUDA_VISIBLE_DEVICES=""` и `HIP_VISIBLE_DEVICES=""` |

---

### Ошибки Gymnasium

| Ошибка | Причина | Решение |
|--------|---------|----------|
| `ImportError: No module named 'box2d'` | Box2D не установлен | `pip install gymnasium[box2d]` |
| `gymnasium.error.DependencyNotInstalled: Box2D` | Box2D не установлен | `pip install swig && pip install box2d-py` |

---

## Ссылки на документацию

**Полную документацию см. в папке `/docs/`**:

- 📄 [PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md) - Обзор проекта
- 📄 [QUICKSTART.md](../../docs/QUICKSTART.md) - Быстрый старт
- 📄 [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - Решение проблем

**Внешняя документация API**:

- 📖 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/) - Полная документация SB3
- 📖 [Gymnasium Documentation](https://gymnasium.farama.org/) - Полная документация Gymnasium
- 📖 [PyTorch Documentation](https://pytorch.org/docs/stable/) - Полная документация PyTorch

---

## Заключение

Этот документ описывает API контракты для ML проекта (Reinforcement Learning), где:
- **Primary API**: Python API (Stable-Baselines3, Gymnasium, PyTorch)
- **CLI Interface**: Командная строка для обучения
- **No REST/GraphQL API**: Нет HTTP endpoint-ов
- **No Database**: Данные в файлах

Для полной документации см. папку `/docs/`.
