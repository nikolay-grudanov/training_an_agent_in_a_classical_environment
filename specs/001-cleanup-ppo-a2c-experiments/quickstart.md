# Quick Start Guide: Project Cleanup and PPO vs A2C Experiments

**Feature**: Project Cleanup and PPO vs A2C Experiments
**Branch**: `001-cleanup-ppo-a2c-experiments`
**Last Updated**: 2026-01-15

## Overview

This guide helps you quickly set up, execute, and validate the project cleanup and PPO/A2C experiments feature. The feature consists of four main components: (1) code audit, (2) project cleanup, (3) PPO training, and (4) A2C training.

## Prerequisites

### Environment Setup

```bash
# Activate the conda environment (required)
conda activate rocm

# Verify environment
python --version  # Should show Python version
conda info --envs  # Should show "rocm" as active
```

### Verify Dependencies

```bash
# Check for required packages
python -c "import stable_baselines3, gymnasium, torch, numpy, matplotlib"
echo "All dependencies installed successfully"
```

If any import fails, install missing packages:
```bash
pip install stable-baselines3 gymnasium torch numpy matplotlib
```

## Quick Execution

### 1. Run Code Audit (5 min)

Audit all modules in `src/` directory to identify working/broken code.

```bash
# Basic audit (INFO level logging)
python -m src.audit.run

# Verbose audit (DEBUG level)
python -m src.audit.run --verbose

# Audit specific directory
python -m src.audit.run --scope tests/ --output tests_audit.md

# Generate both Markdown and JSON reports
python -m src.audit.run --format both
```

**Expected Output**:
- `АУДИТ.md` - Markdown audit report in Russian
- `audit_report.json` - JSON audit report (if `--format both` or `json`)

**Verification**:
```bash
# Check report was generated
ls -lh АУДИТ.md audit_report.json

# View summary
head -20 АУДИТ.md
```

### 2. Run Project Cleanup (2 min)

Clean up root directory and organize files following ML best practices.

```bash
# Preview changes (dry run)
python -m src.cleanup.run --dry-run

# Execute cleanup
python -m src.cleanup.run

# Execute with verbose logging
python -m src.cleanup.run --verbose

# Force cleanup of protected files
python -m src.cleanup.run --force
```

**Expected Output**:
- Clean root directory (only: requirements.txt, README.md, .gitignore, src/, tests/, results/, specs/)
- `results/project_structure.json` - Project structure validation report

**Verification**:
```bash
# Check root directory
ls -la

# Should only show allowed files and directories
# No example files, demo artifacts, or temporary files

# Check structure report
cat results/project_structure.json | jq .project_structure.root_directory.validation_status
# Should output: "clean"
```

### 3. Train PPO Agent (30 min)

Train PPO agent on LunarLander-v3 with reproducible settings.

```bash
# Default training (50k steps, seed=42)
python -m src.training.train --algo ppo

# Custom checkpoint interval
python -m src.training.train --algo ppo --checkpoint-interval 500

# Resume from checkpoint
python -m src.training.train --algo ppo --resume-from results/experiments/ppo_seed42/checkpoint_10000.zip

# Verbose training
python -m src.training.train --algo ppo --verbose
```

**Expected Output**:
- `results/experiments/ppo_seed42/ppo_seed42_model.zip` - Trained model
- `results/experiments/ppo_seed42/ppo_seed42_results.json` - Experiment results
- `results/experiments/ppo_seed42/ppo_seed42_metrics.json` - Time-series metrics
- Checkpoints: `results/experiments/ppo_seed42/checkpoint_*.zip`

**Verification**:
```bash
# Check outputs exist
ls -lh results/experiments/ppo_seed42/

# View final metrics
cat results/experiments/ppo_seed42/ppo_seed42_results.json | jq .experiment_results.metrics

# Should show final_reward_mean, training_time_seconds, converged status
```

### 4. Train A2C Agent (30 min)

Train A2C agent with identical settings for comparison.

```bash
# Default training (50k steps, seed=42)
python -m src.training.train --algo a2c

# Verbose training
python -m src.training.train --algo a2c --verbose
```

**Expected Output**:
- `results/experiments/a2c_seed42/a2c_seed42_model.zip` - Trained model
- `results/experiments/a2c_seed42/a2c_seed42_results.json` - Experiment results
- `results/experiments/a2c_seed42/a2c_seed42_metrics.json` - Time-series metrics
- Checkpoints: `results/experiments/a2c_seed42/checkpoint_*.zip`

**Verification**:
```bash
# Check outputs exist
ls -lh results/experiments/a2c_seed42/

# Compare with PPO results
echo "PPO final reward:"
cat results/experiments/ppo_seed42/ppo_seed42_results.json | jq .experiment_results.metrics.final_reward_mean

echo "A2C final reward:"
cat results/experiments/a2c_seed42/a2c_seed42_results.json | jq .experiment_results.metrics.final_reward_mean
```

## Complete Workflow (All Components)

Execute all four components in sequence:

```bash
#!/bin/bash
# complete_workflow.sh - Execute all feature components

echo "=== STEP 1: Code Audit ==="
python -m src.audit.run --verbose

echo ""
echo "=== STEP 2: Project Cleanup ==="
python -m src.cleanup.run --verbose

echo ""
echo "=== STEP 3: PPO Training ==="
python -m src.training.train --algo ppo --verbose

echo ""
echo "=== STEP 4: A2C Training ==="
python -m src.training.train --algo a2c --verbose

echo ""
echo "=== COMPLETE ==="
echo "Review results in:"
echo "  - АУДИТ.md (audit report)"
echo "  - results/project_structure.json (cleanup status)"
echo "  - results/experiments/ppo_seed42/ (PPO results)"
echo "  - results/experiments/a2c_seed42/ (A2C results)"
```

Run the complete workflow:
```bash
chmod +x complete_workflow.sh
./complete_workflow.sh
```

## Validation & Testing

### Run Automated Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=src/ --cov-report=html
```

### Manual Validation Checklist

#### Audit Validation
- [ ] `АУДИТ.md` was generated
- [ ] Report contains module status table
- [ ] Summary statistics are correct
- [ ] All modules in `src/` were audited

#### Cleanup Validation
- [ ] Root directory is clean (only allowed items)
- [ ] `results/project_structure.json` exists
- [ ] No example/demo files in root
- [ ] All working code preserved

#### PPO Training Validation
- [ ] Training completed within 30 minutes
- [ ] `ppo_seed42_model.zip` saved
- [ ] `ppo_seed42_results.json` contains metrics
- [ ] `ppo_seed42_metrics.json` contains time-series data
- [ ] Checkpoints saved every 1000 steps
- [ ] Final reward > 200 (converged)

#### A2C Training Validation
- [ ] Training completed within 30 minutes
- [ ] `a2c_seed42_model.zip` saved
- [ ] `a2c_seed42_results.json` contains metrics
- [ ] `a2c_seed42_metrics.json` contains time-series data
- [ ] Checkpoints saved every 1000 steps
- [ ] Final reward > 200 (converged)

#### Reproducibility Validation
- [ ] Run PPO training twice with seed=42
- [ ] Compare final rewards (should be identical)
- [ ] Standard deviation < 0.01 between runs

## Troubleshooting

### Common Issues

#### 1. Import Errors During Audit

**Error**: `ModuleNotFoundError: No module named 'stable_baselines3'`

**Solution**:
```bash
pip install stable_baselines3
```

#### 2. Conda Environment Mismatch

**Error**: `Training failed: Conda environment must be 'rocm'`

**Solution**:
```bash
conda activate rocm
```

#### 3. Permission Denied During Cleanup

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Check file permissions
ls -la

# Use --force flag if you're sure
python -m src.cleanup.run --force
```

#### 4. Environment Not Available

**Error**: `gymnasium.error.Error: Unknown environment 'LunarLander-v3'`

**Solution**:
```bash
pip install gymnasium[box2d]
```

#### 5. Training Timeout

**Issue**: Training takes longer than 30 minutes

**Possible Causes**:
- CPU is slow/overloaded
- Checkpoint interval too small
- Environment reset is slow

**Solutions**:
```bash
# Reduce checkpoint frequency
python -m src.training.train --algo ppo --checkpoint-interval 5000

# Check CPU usage
top -u $USER
```

## Advanced Usage

### Custom Experiments

```bash
# Different environment
python -m src.training.train --algo ppo --env BipedalWalker-v3

# Custom hyperparameters (modify training script)
# Edit src/training/train.py and adjust PPO/A2C parameters

# Long training
python -m src.training.train --algo ppo --timesteps 100000
```

### Checkpoint Analysis

```python
# Load a specific checkpoint and evaluate
from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("results/experiments/ppo_seed42/checkpoint_25000.zip")
env = gym.make("LunarLander-v3")

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

### Metrics Visualization

```python
# Plot training metrics
import json
import matplotlib.pyplot as plt

with open("results/experiments/ppo_seed42/ppo_seed42_metrics.json") as f:
    data = json.load(f)

rewards = [point["reward"] for point in data["training_metrics"]["time_series"]]
plt.plot(rewards)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("PPO Training Progress")
plt.show()
```

## Performance Benchmarks

Based on SC-001 through SC-008, expected performance:

| Component | Target | Acceptance Criteria |
|-----------|---------|-------------------|
| Audit completion | < 10 min | All modules in `src/` audited |
| Cleanup completion | < 5 min | Root directory clean |
| PPO training | < 30 min | 50k steps, seed=42 |
| A2C training | < 30 min | 50k steps, seed=42 |
| Final reward (both) | > 200 | Successful task completion |
| Reproducibility | < 0.01 std | Identical seed runs |

## Next Steps

After successful execution:

1. **Review Audit Report**: Check `АУДИТ.md` for broken modules and fix if needed
2. **Compare Algorithms**: Compare PPO vs A2C final rewards and learning curves
3. **Validate Reproducibility**: Run training twice to verify deterministic behavior
4. **Document Findings**: Summarize performance comparison in experiment report

## Support

- **Feature Spec**: See `specs/001-cleanup-ppo-a2c-experiments/spec.md`
- **Data Model**: See `specs/001-cleanup-ppo-a2c-experiments/data-model.md`
- **Contracts**: See `specs/001-cleanup-ppo-a2c-experiments/contracts/`
- **Constitution**: See `.specify/memory/constitution.md`

---

**Ready to start?** Begin with Step 1 (Code Audit) and proceed sequentially through the workflow.
