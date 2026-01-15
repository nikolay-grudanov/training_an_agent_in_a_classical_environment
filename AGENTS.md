# AGENTS.md: Guidelines for Agentic Coding in RL Agent Training Repo

## 🚀 Quick Start Commands

Activate environment:
```bash
conda activate sb3-lunar-env  # From environment.yml
# or pip install -r requirements.txt
```

Install dev dependencies:
```bash
pip install -e .[dev]  # If setup.py/pyproject.toml exists
# Includes: pytest, ruff, black, mypy, types-*
```

## 🛠️ Build/Lint/Test Commands

### Linting & Formatting (Ruff - all-in-one)
```bash
# Lint (errors + style)
ruff check .

# Lint specific file/dir
ruff check src/utils/seeding.py
ruff check src/ --fix  # Auto-fix where possible

# Format code (Black-compatible)
ruff format .

# Sort imports (isort-compatible)
ruff check --select I .  # Check imports only
ruff check --select I --fix .  # Fix imports

# Stats
ruff stats .
```

### Type Checking (mypy)
```bash
mypy src/ tests/ --strict
mypy src/utils/  # Specific module
```

### Testing (pytest)
```bash
# All tests
pytest tests/ -v --cov=src/ --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Single test file
pytest tests/unit/test_seeding.py -v

# Single test function (key for agents!)
pytest tests/unit/test_seeding.py::test_set_seed_np -v

# With coverage
pytest tests/ --cov=src/utils/ --cov-report=term-missing

# Parallel tests
pytest tests/ -n auto

# Jupyter notebooks
pytest notebooks/ --nbval  # If nbval installed
```

### Pre-commit (if .pre-commit-config.yaml exists)
```bash
pre-commit install
pre-commit run --all-files
pre-commit run ruff --all-files
```

### RL-Specific Commands
```bash
# Train PPO baseline
python -m src.training.trainer --config configs/training_schema.yaml

# Run experiment
python -m src.experiments.runner --config configs/experiment_schema.yaml

# Generate plots/video
python -m src.visualization.plots.generate_all --log-dir results/logs/
```

### Clean & Rebuild
```bash
# Clean
rm -rf __pycache__/ .pytest_cache/ .ruff_cache/ htmlcov/ dist/ build/
rm -rf results/models/ results/videos/  # RL artifacts (gitignored)

# Reinstall
pip install -e . --force-reinstall
```

## 💻 Code Style Guidelines

Follow PEP 8 with Black formatting. Use Ruff for enforcement.

### Imports (isort + Ruff I001-I005)
**Absolute imports preferred:**
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

# Local
from src.utils.seeding import set_seed
from src.environments.lunar_lander import LunarLanderEnv
```

**Group order:** stdlib > third-party > local
**No wildcard imports:** `from utils import *` → Forbidden
**One import per line**

### Formatting (Black + Ruff)
- Line length: 88 (Black default)
- Black auto-formats everything
- Single quotes for strings: `'hello'` (unless escaping)
- Trailing commas always
```python
def train_agent(
    env: gym.Env,
    model: PPO,
    total_timesteps: int = 100_000,
) -> Dict[str, float]:
    ...
```

### Types (mypy strict)
- **Full type hints everywhere** (functions, classes, returns)
- Use `typing.Protocol` for callbacks
- `Any` only if unavoidable (comment why)
- `Final` for constants
```python
from typing import TypedDict

class TrainingMetrics(TypedDict):
    reward_mean: float
    episode_length: int
    loss_policy: Optional[float]

def log_metrics(metrics: TrainingMetrics) -> None: ...
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `seeding.py`, `lunar_lander.py` |
| Classes | CamelCase | `LunarLanderEnv`, `PPOAgent` |
| Methods/Functions | snake_case | `set_seed()`, `train_model()` |
| Variables/Params | snake_case | `total_timesteps`, `obs_space` |
| Constants | UPPER_SNAKE_CASE | `MAX_EPISODES = 1000` |
| Private | `_leading_underscore` | `_validate_config()` |

### Docstrings (Google Style + NumPy for Sci/ML)
```python
def set_seed(seed: int) -> None:
    \"\"\"Set global seed for reproducibility across NumPy, PyTorch, Gymnasium.

    Args:
        seed: Random seed value (0-2**32-1)

    Raises:
        ValueError: If seed out of range.
    \"\"\"
    ...
```

**ML-Specific:**
- Document shapes: `obs: np.ndarray (state_dim,)`
- Hyperparams: List ranges/defaults
- Inputs/Outputs: Exact types/shapes

### Error Handling
- **Specific exceptions:** `ValueError` for config, `RuntimeError` for env
- **Context managers:** For files, envs
- **Logging before raise:**
```python
import logging
from gymnasium import error as gym_error

logger = logging.getLogger(__name__)

try:
    env = gym.make(env_name)
except gym_error.Error as e:
    logger.error(f"Failed to create env {env_name}: {e}")
    raise ValueError(f"Invalid env: {env_name}") from e
```

**Never bare `except:`** → Use `except Exception:`
**RL-Specific:** Handle `Done`, `Truncated`, timeouts gracefully.

### Logging (structlog or logging)
- Module-level logger: `logger = logging.getLogger(__name__)`
- Levels: DEBUG (metrics), INFO (progress), WARNING (degraded), ERROR (failures)
- JSON for experiments: `structlog.get_logger().bind(experiment_id=exp_id)`
```python
logger.info("Training started", total_timesteps=1_000_000, env="LunarLander-v2")
```

### Reproducibility (Critical for RL!)
```python
def train_with_seed(seed: int = 42) -> PPO:
    set_seed(seed)  # ALWAYS first!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ...
```

### File Organization
```
src/
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── seeding.py
└── agents/
    ├── __init__.py
    └── ppo_agent.py
```
- `__init__.py` everywhere
- No circular imports
- Relative imports only in `__init__.py`

### Testing Guidelines
- **100% coverage** on utils/training
- Parametrized tests for hyperparams/seeds
```python
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_set_seed(seed: int) -> None:
    set_seed(seed)
    assert np.random.randint(0, 10) == expected
```
- Mock envs/models: `pytest-mock`, `monkeypatch`
- Fixtures for `gym.Env`, `PPO`

### ML/RL Best Practices
- **Shapes explicit:** `obs: np.ndarray = np.zeros((state_dim,))`
- **Device-agnostic:** `device = "cpu"` (no CUDA)
- **Callbacks over loops:** Use SB3 `BaseCallback`
- **Save everything:** models `.zip`, metrics JSON/CSV
- **Version pins:** In requirements.txt

### Forbidden Patterns
- `print()` → Use logger
- Global variables → Config dicts
- Magic numbers → Named constants
- `if __name__ == "__main__":` → CLI with `typer`/`click`

## 🤖 For Agents (Cursor/Copilot/You)
1. **Always run `ruff check --fix .` before commit**
2. **Test incrementally:** `pytest <file>::<func>`
3. **Read tasks.md** before editing
4. **Phase order:** Setup → Utils → Agents → Training
5. **No CUDA:** CPU-only (Linux server)
6. **Seed=42** for baselines

**Length: ~150 lines. Updated: 2026-01-14**

## Active Technologies
- Python 3.10.14 (002-project-cleanup-validation)
- Python 3.10.14 + Stable-Baselines3, Gymnasium, PyTorch, NumPy, Matplotlib (001-research-spec-update)
- Files (experiment results, trained models, videos) (001-research-spec-update)
- Python (conda environment "rocm") + Stable-Baselines3 (PPO/A2C), Gymnasium (LunarLander-v3), PyTorch, NumPy, Matplotlib (001-cleanup-ppo-a2c-experiments)
- JSON for metrics/audit reports, Pickle for trained models (001-cleanup-ppo-a2c-experiments)

## Recent Changes
- 002-project-cleanup-validation: Added Python 3.10.14
