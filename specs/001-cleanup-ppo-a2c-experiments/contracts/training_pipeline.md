# Training Pipeline Interface Contract

**Version**: 1.0.0
**Date**: 2026-01-15
**Feature**: Project Cleanup and PPO vs A2C Experiments

## Overview

This contract defines the interface for training reinforcement learning agents (PPO and A2C) on LunarLander-v3 environment with reproducible settings, checkpointing, and comprehensive metrics tracking.

## CLI Interface

### Command

```bash
python -m src.training.train --algo <algorithm> [--options]
```

### Required Options

| Option | Type | Values | Description |
|--------|------|---------|----------|-------------|
| `--algo` | string | `ppo` or `a2c` | Algorithm to train |

### Optional Options

| Option | Type | Required | Default | Description |
|--------|------|----------|----------|-------------|
| `--env` | string | No | `LunarLander-v3` | Gymnasium environment name |
| `--seed` | int | No | `42` | Random seed for reproducibility |
| `--timesteps` | int | No | `50000` | Total training timesteps |
| `--checkpoint-interval` | int | No | `1000` | Steps between checkpoints |
| `--output-dir` | string | No | `results/experiments/{algo}_seed{seed}/` | Output directory |
| `--resume-from` | string | No | `None` | Path to checkpoint for resumption |
| `--verbose` | flag | No | `False` | Enable DEBUG-level logging |

### Example Usage

```bash
# Train PPO with default settings
python -m src.training.train --algo ppo

# Train A2C with custom checkpoint interval
python -m src.training.train --algo a2c --checkpoint-interval 500

# Resume training from checkpoint
python -m src.training.train --algo ppo --resume-from results/experiments/ppo_seed42/checkpoint_10000.zip

# Custom experiment with different settings
python -m src.training.train --algo ppo --env LunarLander-v3 --seed 123 --timesteps 100000 --verbose
```

## Input Parameters

### TrainingConfig

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for RL agent training."""
    # Algorithm selection
    algo: str  # "ppo" or "a2c"

    # Environment
    env_name: str = "LunarLander-v3"

    # Reproducibility
    seed: int = 42

    # Training parameters
    total_timesteps: int = 50000
    checkpoint_interval: int = 1000

    # Output
    output_dir: Optional[Path] = None

    # Resumption
    resume_from: Optional[Path] = None

    # Logging
    verbose: bool = False

    # Conda environment (validation)
    conda_env: str = "rocm"

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path(f"results/experiments/{self.algo}_seed{self.seed}/")
```

## Training Workflow

### Phase 1: Initialization

1. Validate conda environment (must be "rocm")
2. Set random seeds (NumPy, PyTorch, Gymnasium)
3. Create output directory structure
4. Initialize environment (LunarLander-v3)
5. Load or create model

### Phase 2: Training Loop

1. For each checkpoint interval:
   - Train for `checkpoint_interval` timesteps
   - Save checkpoint to `{output_dir}/checkpoint_{step}.zip`
   - Log metrics (reward, loss, episode length)
   - Write metrics to JSON file
2. Handle interruption:
   - On KeyboardInterrupt: Save partial checkpoint, exit gracefully
   - On exception: Log error, save checkpoint if possible

### Phase 3: Finalization

1. Save final model to `{output_dir}/{algo}_seed{seed}_model.zip`
2. Generate comprehensive experiment results JSON
3. Validate training completed successfully
4. Log completion message with metrics summary

## Output Format

### Directory Structure

```text
results/experiments/{algo}_seed{seed}/
├── {algo}_seed{seed}_model.zip           # Final trained model
├── {algo}_seed{seed}_results.json        # Experiment metadata and results
├── {algo}_seed{seed}_metrics.json        # Time-series metrics
├── checkpoint_1000.zip                   # Checkpoint files
├── checkpoint_2000.zip
├── ...
├── checkpoint_50000.zip
└── logs/
    ├── training_{timestamp}.log            # Training logs (DEBUG level)
    └── metrics_{timestamp}.csv            # CSV metrics (optional)
```

### Experiment Results JSON

```json
{
  "experiment_results": {
    "metadata": {
      "experiment_id": "ppo_seed42",
      "algorithm": "PPO",
      "environment": "LunarLander-v3",
      "seed": 42,
      "start_time": "2026-01-15T10:00:00Z",
      "end_time": "2026-01-15T10:28:30Z",
      "total_timesteps": 50000,
      "conda_environment": "rocm"
    },
    "model": {
      "algorithm": "PPO",
      "policy": "MlpPolicy",
      "model_file": "ppo_seed42_model.zip",
      "model_path": "results/experiments/ppo_seed42/",
      "checkpoint_interval": 1000,
      "checkpoints": [
        "checkpoint_1000.zip",
        "checkpoint_2000.zip",
        "checkpoint_50000.zip"
      ]
    },
    "metrics": {
      "final_reward_mean": 250.5,
      "final_reward_std": 12.3,
      "episode_length_mean": 250.0,
      "total_episodes": 200,
      "training_time_seconds": 1710.0,
      "converged": true
    },
    "hyperparameters": {
      "learning_rate": 0.0003,
      "n_steps": 2048,
      "batch_size": 64,
      "n_epochs": 10,
      "gamma": 0.99,
      "gae_lambda": 0.95
    },
    "environment": {
      "name": "LunarLander-v3",
      "observation_space": "Box(8,)",
      "action_space": "Discrete(4)",
      "reward_threshold": 200.0
    }
  }
}
```

### Training Metrics JSON

```json
{
  "training_metrics": {
    "metadata": {
      "experiment_id": "ppo_seed42",
      "algorithm": "PPO",
      "environment": "LunarLander-v3",
      "seed": 42,
      "recording_interval": 100
    },
    "time_series": [
      {
        "timestep": 100,
        "episode": 5,
        "reward": -50.2,
        "episode_length": 45,
        "loss": 0.523,
        "timestamp": "2026-01-15T10:01:30Z"
      },
      {
        "timestep": 200,
        "episode": 10,
        "reward": -35.8,
        "episode_length": 52,
        "loss": 0.498,
        "timestamp": "2026-01-15T10:02:45Z"
      }
    ],
    "aggregated": {
      "reward_mean": 45.2,
      "reward_std": 15.3,
      "reward_min": -100.0,
      "reward_max": 285.0,
      "episode_length_mean": 125.0,
      "total_timesteps": 50000
    }
  }
}
```

## Checkpoint Resumption

### Resumption Logic

```python
if resume_from is not None:
    print(f"Resuming from checkpoint: {resume_from}")
    model = PPO.load(resume_from)  # or A2C.load()
    remaining_timesteps = total_timesteps - extract_step_from_filename(resume_from)
else:
    print(f"Creating new {algo.upper()} model")
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)

# Continue training (no reset of timesteps)
model.learn(total_timesteps=remaining_timesteps, reset_num_timesteps=False)
```

### Checkpoint Filename Format

- `{output_dir}/checkpoint_{timestep}.zip`
- Example: `checkpoint_1000.zip`, `checkpoint_25000.zip`
- Final checkpoint: `checkpoint_50000.zip`

### Resumption Validation

1. Checkpoint file must exist
2. Checkpoint must match algorithm (PPO.load() for PPO, A2C.load() for A2C)
3. Conda environment must match (rocm)
4. Seed must be compatible (same seed preferred)

## Algorithm Configurations

### PPO (Proximal Policy Optimization)

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=seed,
    tensorboard_log=None,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95
)
```

### A2C (Advantage Actor-Critic)

```python
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    seed=seed,
    tensorboard_log=None,
    learning_rate=0.0007,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0
)
```

## Logging Behavior

### DEBUG Level (when `--verbose`)

- Every training step logged (timestep, episode, reward)
- Checkpoint save operations logged
- Detailed model parameters logged
- Environment initialization details
- Full exception stack traces

### INFO Level (default)

- Checkpoint saves (every 1000 steps)
- Training progress updates (every 1000 steps)
- Completion message with metrics summary
- Warning messages (e.g., low reward)

### WARNING Level

- Reproducibility warnings (e.g., CUDA nondeterminism)
- Conda environment mismatch
- Checkpoint resumption warnings

### ERROR Level

- Critical failures (environment creation, model loading)
- Checkpoint save failures
- Training interruption errors

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Training completed successfully |
| 1 | Training completed with warnings |
| 2 | Training failed (critical error) |
| 3 | Invalid command-line arguments |
| 4 | Conda environment validation failed |
| 5 | Environment creation failed |
| 6 | Model load failed (resumption) |
| 7 | Checkpoint save failed (interruption) |

## Error Handling

### Expected Errors

1. **Environment Not Available**: Exit code 5, log error message
2. **Conda Environment Mismatch**: Exit code 4, require "rocm" environment
3. **Checkpoint File Not Found**: Exit code 6, verify path
4. **Algorithm Mismatch**: Exit code 6, PPO.load() on A2C checkpoint
5. **Disk Full**: Exit code 2, save partial checkpoint if possible
6. **KeyboardInterrupt**: Exit code 7, save checkpoint, exit gracefully

### Recovery Behavior

- Checkpoint saves are atomic (write to temp, then rename)
- Metrics written incrementally to prevent data loss
- Partial results saved on any interruption
- Training can be resumed from last successful checkpoint

## Performance Requirements

- Complete training within 30 minutes on CPU (SC-003)
- Checkpoint save time < 5 seconds per checkpoint
- Metrics recording overhead < 1% of training time
- Memory usage < 2GB during training

## Reproducibility Requirements

### Seeding (Constitution Compliance)

```python
import numpy as np
import torch
import random
import gymnasium as gym

def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set before environment creation
set_seed(seed)

# Create environment with seed
env = gym.make(env_name)
env.reset(seed=seed)

# Create model with seed
model = PPO("MlpPolicy", env, verbose=1, seed=seed)
```

### Validation

- Run same experiment twice with identical seed
- Compare final metrics (reward mean, std dev)
- Target: Standard deviation < 0.01 for perfect reproducibility

## Testing Requirements

### Unit Tests

```python
def test_config_validation():
    """Test training config validation."""
    config = TrainingConfig(algo="ppo", seed=42)
    assert config.algo == "ppo"
    assert config.seed == 42
    assert config.conda_env == "rocm"

def test_seeding():
    """Test seeding produces deterministic results."""
    rewards_1 = run_training(seed=42, timesteps=1000)
    rewards_2 = run_training(seed=42, timesteps=1000)
    assert np.allclose(rewards_1, rewards_2)

def test_checkpoint_save_load():
    """Test checkpoint save and load cycle."""
    config = TrainingConfig(algo="ppo", seed=42, total_timesteps=1000)
    model = train_and_save(config)
    loaded_model = PPO.load(model.checkpoint_path)
    assert loaded_model is not None
```

### Integration Tests

```python
def test_full_training_workflow():
    """Test complete training process."""
    # Run training
    run_cli(["--algo", "ppo", "--seed", "42", "--timesteps", "10000"])

    # Verify outputs
    output_dir = Path("results/experiments/ppo_seed42/")
    assert (output_dir / "ppo_seed42_model.zip").exists()
    assert (output_dir / "ppo_seed42_results.json").exists()
    assert (output_dir / "ppo_seed42_metrics.json").exists()

    # Verify metrics
    results = load_json(output_dir / "ppo_seed42_results.json")
    assert results["experiment_results"]["metrics"]["total_timesteps"] == 10000

def test_checkpoint_resumption():
    """Test training resumption from checkpoint."""
    # Initial training
    run_cli(["--algo", "ppo", "--timesteps", "5000"])

    # Resume training
    run_cli([
        "--algo", "ppo",
        "--timesteps", "10000",
        "--resume-from", "results/experiments/ppo_seed42/checkpoint_5000.zip"
    ])

    # Verify total timesteps
    results = load_json("results/experiments/ppo_seed42/ppo_seed42_results.json")
    assert results["experiment_results"]["metadata"]["total_timesteps"] == 10000
```

### Reproducibility Tests

```python
def test_training_reproducibility():
    """Test training produces identical results with same seed."""
    # Run training twice
    run_cli(["--algo", "ppo", "--seed", "42", "--timesteps", "5000"])
    results_1 = load_json("results/experiments/ppo_seed42/ppo_seed42_results.json")

    clean_output_dir("results/experiments/ppo_seed42/")

    run_cli(["--algo", "ppo", "--seed", "42", "--timesteps", "5000"])
    results_2 = load_json("results/experiments/ppo_seed42/ppo_seed42_results.json")

    # Compare metrics
    reward_1 = results_1["experiment_results"]["metrics"]["final_reward_mean"]
    reward_2 = results_2["experiment_results"]["metrics"]["final_reward_mean"]
    assert abs(reward_1 - reward_2) < 0.01  # Within tolerance
```

## Dependencies

- `stable_baselines3` (PPO, A2C implementations)
- `gymnasium` (LunarLander-v3 environment)
- `torch` (PyTorch backend)
- `numpy` (Numerical computations, seeding)
- `random` (Python random number generator)
- `logging` (Python stdlib)
- `json` (Python stdlib)
- `pathlib` (Python stdlib)
- `dataclasses` (Python stdlib)

## Performance Targets

| Metric | Target | Acceptance Criteria |
|--------|---------|-------------------|
| Training time | < 30 min | SC-003: Both PPO and A2C complete within 30 min |
| Reward threshold | > 200 | SC-006: Successful task completion in LunarLander-v3 |
| Convergence evidence | Clear progression | SC-008: Metrics show learning over 50k steps |
| Checkpoint overhead | < 5 sec | Minimal interruption of training flow |

---

**Related Contracts**:
- [Audit System](./audit_system.md)
- [Cleanup System](./cleanup_system.md)
