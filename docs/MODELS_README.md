# Trained Models - Demonstration of Experiments

This directory contains trained models from various experiments conducted during the development of the RL agent training project.

## Purpose

These models demonstrate:
1. **Hyperparameter optimization journey** - showing how we reached the best configuration
2. **Algorithm comparison** - PPO vs A2C performance on LunarLander-v3
3. **Learning rate experiments** - testing different learning rates
4. **Gamma parameter tuning** - discount factor optimization
5. **Seed variation** - reproducibility verification

## Available Models

### PPO Models (Production Algorithm)

| Model | Directory | Size | Purpose |
|-------|-----------|------|---------|
| `gamma_999_model.zip` | `results/experiments/gamma_999/` | 146 KB | **BEST MODEL** - Optimal gamma=0.999 |
| `ppo_seed123_model.zip` | `results/experiments/ppo_seed123/` | 145 KB | Reproducibility test (seed=123) |
| `ppo_seed999_model.zip` | `results/experiments/ppo_seed999/` | 146 KB | Reproducibility test (seed=999) |

### A2C Models (Legacy/Comparison)

| Model | Directory | Size | Purpose |
|-------|-----------|------|---------|
| `a2c_seed42_model.zip` | `results/experiments/a2c_seed42/` | 104 KB | Baseline A2C performance |
| `a2c_lr1e4_model.zip` | `results/experiments/a2c_lr1e4/` | 104 KB | A2C with lr=1e-4 |
| `a2c_lr3e4_model.zip` | `results/experiments/a2c_lr3e4/` | 104 KB | A2C with lr=3e-4 |

### Gamma Parameter Experiments

| Model | Gamma | Size | Performance |
|-------|-------|------|-------------|
| `gamma_900_model.zip` | 0.900 | 146 KB | Lower performance |
| `gamma_990_model.zip` | 0.990 | 146 KB | Good performance |
| `gamma_999_model.zip` | **0.999** | **146 KB** | **BEST - 216.31 ± 65.80** |

## Best Model: PPO with gamma=0.999

**Configuration:**
- Algorithm: PPO
- Environment: LunarLander-v3
- Learning Rate: 3e-4
- Batch Size: 64
- Gamma: 0.999 (KEY PARAMETER)
- Entropy Coefficient: 0.01
- Timesteps: 150,000

**Performance:**
- Mean Reward: **216.31**
- Std Dev: ± 65.80
- Target: >200 ✅ (exceeded by 8.2%)

**Location:**
```
results/experiments/gamma_999/gamma_999_model.zip
```

## Loading Models

### Load Best Model (Python)
```python
from stable_baselines3 import PPO

# Load the best model
model = PPO.load("results/experiments/gamma_999/gamma_999_model.zip")

# Use for inference
obs, _ = env.reset()
action, _states = model.predict(obs)
```

### Load Any Model
```python
from stable_baselines3 import PPO, A2C

# Load PPO model
ppo_model = PPO.load("results/experiments/ppo_seed123/ppo_seed123_model.zip")

# Load A2C model
a2c_model = A2C.load("results/experiments/a2c_lr3e4/a2c_lr3e4_model.zip")
```

## Experiment Journey

### Phase 1: Initial Testing
- Tested PPO vs A2C algorithms
- Results: PPO consistently outperformed A2C

### Phase 2: Hyperparameter Tuning
- **Learning Rate**: Tested 1e-4, 3e-4, 5e-4
- **Gamma**: Tested 0.900, 0.990, 0.999
- **Batch Size**: Tested 64, 128
- **Entropy Coefficient**: Tested 0.0, 0.01

### Phase 3: Best Configuration Found
- Best hyperparameters: gamma=0.999, lr=3e-4, batch_size=64
- Reward improvement: +181% over baseline
- Training time: ~2.5 minutes for 150K timesteps

### Phase 4: Verification
- Tested multiple seeds (42, 123, 999)
- Confirmed reproducibility and consistency
- All achieved target reward >200

## Performance Summary

| Algorithm | Best Reward | Model | Configuration |
|-----------|-------------|-------|---------------|
| **PPO** | **216.31** | `gamma_999_model.zip` | gamma=0.999, lr=3e-4 |
| PPO | ~210-215 | `ppo_seed123_model.zip` | gamma=0.999, lr=3e-4 |
| PPO | ~210-215 | `ppo_seed999_model.zip` | gamma=0.999, lr=3e-4 |
| A2C | ~150-180 | `a2c_lr3e4_model.zip` | lr=3e-4 |
| A2C | ~140-170 | `a2c_lr1e4_model.zip` | lr=1e-4 |

**Conclusion**: PPO with gamma=0.999 is the best performing configuration for LunarLander-v3.

## Training New Models

To train new models with the best configuration:

```bash
# Activate environment
conda activate rocm

# Train best model
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 150000 \
    --seed 42 \
    --gamma 0.999 \
    --ent-coef 0.01 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --device cpu
```

## Model Size

All models are ~100-146 KB each (compressed). Total: ~1.01 MB for 8 models.

## Storage in Git

These models are stored in Git for:
1. **Demonstration** - showing the progression of experiments
2. **Reproducibility** - allowing immediate loading without retraining
3. **Documentation** - providing concrete results from experiments

The models are small enough to store in Git without needing Git LFS.

## Related Documentation

- [Phase 5 Report](../specs/004-test-and-fix/phase5_report.md) - Optimized parameters
- [Phase 13 Report](../specs/004-test-and-fix/phase13_report.md) - Hyperparameter optimization
- [Grid Search Results](../GRID_SEARCH_RESULTS.md) - Detailed experiment results
- [QUICKSTART.md](./QUICKSTART.md) - Quick start guide

---
**Last Updated**: 2026-02-05
**Project**: RL Agent Training - LunarLander-v3
**Status**: ✅ All targets achieved (>200 reward)
