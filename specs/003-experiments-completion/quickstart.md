# Quickstart: RL Experiments Completion & Convergence

**Branch**: `003-experiments-completion`  
**Date**: 15 января 2026

This guide provides commands for training RL agents to convergence, generating visualizations, running controlled experiments, and compiling reports on LunarLander-v3.

---

## 1. Prerequisites

```bash
# Activate conda environment
conda activate rocm

# Verify installation
python -c "import stable_baselines3; print(f'SB3 version: {stable_baselines3.__version__}')"
python -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
```

---

## 2. Training Baseline Models (User Story 1)

### 2.1 Train A2C Agent (200K timesteps)

```bash
python -m src.experiments.completion.baseline_training \
    --algo a2c \
    --timesteps 200000 \
    --seed 42 \
    --gamma 0.99
```

### 2.2 Train PPO Agent (200K timesteps)

```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 200000 \
    --seed 42 \
    --gamma 0.99
```

### 2.3 Train with Custom Parameters

```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 200000 \
    --seed 123 \
    --gamma 0.99 \
    --checkpoint-freq 50000
```

### 2.4 Resume Training from Checkpoint

```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo \
    --timesteps 200000 \
    --seed 42 \
    --resume-from results/experiments/ppo_seed42/checkpoints/checkpoint_100000
```

### 2.5 Expected Output

```
Experiment: ppo_seed42
Model: results/experiments/ppo_seed42/ppo_seed42_model.zip
Mean reward: 245.32 ± 35.67
Convergence: YES
Duration: 1423.5s
```

---

## 3. Evaluating Trained Models

### 3.1 Evaluate Single Model

```bash
python -c "
from src.training.evaluation import evaluate_agent

result = evaluate_agent(
    model_path='results/experiments/ppo_seed42/ppo_seed42_model.zip',
    env_id='LunarLander-v3',
    n_eval_episodes=10,
)
print(f'Mean reward: {result[\"mean_reward\"]:.2f} ± {result[\"std_reward\"]:.2f}')
print(f'Convergence: {\"YES\" if result[\"convergence_achieved\"] else \"NO\"}')
"
```

### 3.2 Evaluate All Baselines

```bash
python -c "
from src.training.evaluation import evaluate_agent

for exp_id in ['a2c_seed42', 'ppo_seed42']:
    result = evaluate_agent(f'results/experiments/{exp_id}/{exp_id}_model.zip')
    print(f'{exp_id}: {result[\"mean_reward\"]:.2f} ± {result[\"std_reward\"]:.2f}')
"
```

---

## 4. Generating Performance Graphs (User Story 2)

### 4.1 Generate Learning Curve for Single Experiment

```bash
python -m src.visualization.graphs \
    --experiment a2c_seed42 \
    --type learning_curve \
    --output results/experiments/a2c_seed42/reward_curve.png \
    --title "A2C Learning Curve (Seed=42)"
```

### 4.2 Generate Learning Curve for PPO

```bash
python -m src.visualization.graphs \
    --experiment ppo_seed42 \
    --type learning_curve \
    --output results/experiments/ppo_seed42/reward_curve.png \
    --title "PPO Learning Curve (Seed=42)"
```

### 4.3 Generate Comparison Graph (A2C vs PPO)

```bash
python -m src.visualization.graphs \
    --experiment a2c_seed42,ppo_seed42 \
    --type comparison \
    --output results/comparison/a2c_vs_ppo.png \
    --title "Algorithm Comparison: A2C vs PPO"
```

### 4.4 Generate Gamma Comparison Graph

```bash
python -m src.visualization.graphs \
    --experiment gamma_090,gamma_099,gamma_0999 \
    --type gamma_comparison \
    --output results/comparison/gamma_comparison.png \
    --title "Gamma Hyperparameter Comparison"
```

### 4.5 Graph Types Reference

| Type | Description | Output |
|------|-------------|--------|
| `learning_curve` | Single experiment reward progression | `*_seed*/reward_curve.png` |
| `comparison` | Compare 2+ experiments | `comparison/*.png` |
| `gamma_comparison` | Compare gamma variations | `comparison/gamma_*.png` |

---

## 5. Generating Demonstration Videos (User Story 3)

### 5.1 Generate Video for A2C Model

```bash
python -m src.visualization.video \
    --model results/experiments/a2c_seed42/a2c_seed42_model.zip \
    --output results/experiments/a2c_seed42/video.mp4 \
    --episodes 5 \
    --fps 30
```

### 5.2 Generate Video for PPO Model

```bash
python -m src.visualization.video \
    --model results/experiments/ppo_seed42/ppo_seed42_model.zip \
    --output results/experiments/ppo_seed42/video.mp4 \
    --episodes 5 \
    --fps 30
```

### 5.3 Generate Video with Custom Settings

```bash
python -m src.visualization.video \
    --model results/experiments/ppo_seed42/ppo_seed42_model.zip \
    --output results/videos/ppo_demo.mp4 \
    --episodes 10 \
    --fps 30 \
    --show-scores
```

### 5.4 Video Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Path to trained model (.zip) |
| `--output` | Required | Output video path (.mp4) |
| `--episodes` | 5 | Number of episodes to record |
| `--fps` | 30 | Frames per second |
| `--show-scores` | False | Overlay episode scores |

---

## 6. Running Gamma Hyperparameter Experiment (User Story 4)

### 6.1 Run Default Gamma Experiment (0.90, 0.99, 0.999)

```bash
python -m src.experiments.completion.gamma_experiment \
    --gamma 0.90 0.99 0.999 \
    --timesteps 100000 \
    --seed 42
```

### 6.2 Run Custom Gamma Values

```bash
python -m src.experiments.completion.gamma_experiment \
    --gamma 0.95 0.99 0.995 \
    --timesteps 100000 \
    --seed 42
```

### 6.3 Run with Extended Training

```bash
python -m src.experiments.completion.gamma_experiment \
    --gamma 0.90 0.99 0.999 \
    --timesteps 200000 \
    --seed 42
```

### 6.4 Expected Output

```
============================================================
GAMMA EXPERIMENT RESULTS
============================================================

Best gamma: 0.99 (reward: 252.34)

Hypothesis: gamma=0.99 provides best balance between immediate and long-term rewards
Result: SUPPORTED
Evidence: gamma=0.99 achieved highest mean reward (252.34)

ANOVA: F=4.237, p=0.0213
Significant: Yes

| Gamma | Mean Reward | Std Dev | Convergence | Duration (s) |
|-------|-------------|---------|-------------|--------------|
| 0.900 | 198.45      | 42.31   | ✗           | 812.3        |
| 0.990 | 252.34      | 28.67   | ✓           | 834.1        |
| 0.999 | 241.89      | 31.24   | ✓           | 856.7        |
```

---

## 7. Generating Experiment Reports (User Story 5)

### 7.1 Generate Report for Baseline Experiments

```bash
python -m src.reporting.report_generator \
    --output results/reports/experiment_report.md \
    --experiments a2c_seed42 ppo_seed42 \
    --include-graphs \
    --include-videos
```

### 7.2 Generate Report with Gamma Experiment

```bash
python -m src.reporting.report_generator \
    --output results/reports/full_report.md \
    --experiments a2c_seed42 ppo_seed42 gamma_090 gamma_099 gamma_0999 \
    --include-graphs \
    --include-videos
```

### 7.3 Generate Report without Media

```bash
python -m src.reporting.report_generator \
    --output results/reports/quick_report.md \
    --experiments a2c_seed42 ppo_seed42 \
    --no-include-graphs \
    --no-include-videos
```

### 7.4 Report Parameters

| Parameter | Description |
|-----------|-------------|
| `--output` | Path to output markdown file |
| `--experiments` | Space-separated list of experiment IDs |
| `--include-graphs` | Include graph references (default: True) |
| `--include-videos` | Include video references (default: True) |

---

## 8. Running Tests

### 8.1 Run All Unit Tests

```bash
pytest tests/unit/ -v --cov=src/
```

### 8.2 Run Specific Test Files

```bash
# Callback tests
pytest tests/unit/test_callbacks.py -v

# Evaluation tests
pytest tests/unit/test_evaluation.py -v

# Graph tests
pytest tests/unit/test_graphs.py -v

# Statistics tests
pytest tests/unit/test_statistics.py -v
```

### 8.3 Run Integration Tests

```bash
# Baseline training integration
pytest tests/integration/test_baseline_training.py -v

# Video generation integration
pytest tests/integration/test_video.py -v

# Full pipeline test
pytest tests/integration/test_full_pipeline.py -v
```

### 8.4 Run All Tests with Coverage

```bash
pytest tests/ -v --cov=src/ --cov-report=html
```

---

## 9. Linting and Code Quality

### 9.1 Check Code Style

```bash
# Check for errors
ruff check .

# Auto-fix errors
ruff check . --fix

# Format code
ruff format .
```

### 9.2 Type Checking

```bash
mypy src/ --strict
```

### 9.3 Full Quality Check

```bash
ruff check . --fix && ruff format . && mypy src/ --strict
```

---

## 10. Command Reference Summary

### Training Commands

| Command | Purpose |
|---------|---------|
| `python -m src.experiments.completion.baseline_training --algo a2c --timesteps 200000` | Train A2C baseline |
| `python -m src.experiments.completion.baseline_training --algo ppo --timesteps 200000` | Train PPO baseline |

### Visualization Commands

| Command | Purpose |
|---------|---------|
| `python -m src.visualization.graphs --experiment a2c_seed42 --type learning_curve` | Generate learning curve |
| `python -m src.visualization.graphs --experiment a2c_seed42,ppo_seed42 --type comparison` | Compare algorithms |
| `python -m src.visualization.video --model ... --output video.mp4` | Generate demo video |

### Experiment Commands

| Command | Purpose |
|---------|---------|
| `python -m src.experiments.completion.gamma_experiment --gamma 0.90 0.99 0.999` | Run gamma experiment |
| `python -m src.reporting.report_generator --output report.md --experiments ...` | Generate report |

### Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/ -v` | Run unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --cov=src/` | Run all tests with coverage |

---

## 11. Expected Results

### Convergence Targets

| Algorithm | Target Reward | Expected Time (CPU) |
|-----------|---------------|---------------------|
| A2C | ≥200 | ~15-20 minutes |
| PPO | ≥200 | ~25-30 minutes |

### Gamma Experiment Hypothesis

**Hypothesis**: gamma=0.99 provides best balance between immediate and long-term rewards

| Gamma | Behavior | Expected Reward |
|-------|----------|-----------------|
| 0.90 | Faster initial learning, lower final | 180-200 |
| 0.99 | Balanced, good final performance | 220-260 |
| 0.999 | Slower initial, stable landing | 200-240 |

---

## 12. Directory Structure After Training

```
results/
├── experiments/
│   ├── a2c_seed42/
│   │   ├── a2c_seed42_model.zip
│   │   ├── config.json
│   │   ├── metrics.csv
│   │   ├── eval_log.csv
│   │   ├── reward_curve.png
│   │   ├── video.mp4
│   │   └── checkpoints/
│   │       ├── checkpoint_50000.zip
│   │       ├── checkpoint_100000.zip
│   │       └── checkpoint_150000.zip
│   ├── ppo_seed42/
│   │   └── ...
│   └── gamma_experiments/
│       ├── gamma_090/
│       ├── gamma_099/
│       └── gamma_0999/
├── comparison/
│   ├── a2c_vs_ppo.png
│   └── gamma_comparison.png
└── reports/
    └── experiment_report.md
```

---

## 13. Troubleshooting

### Training Doesn't Converge

```bash
# Check seed is set correctly
python -c "from src.utils.seeding import set_seed; set_seed(42)"

# Try longer training
python -m src.experiments.completion.baseline_training --algo ppo --timesteps 300000
```

### Video Generation Fails

```bash
# Check memory
free -h

# Verify model exists
ls -la results/experiments/*/*_model.zip

# Use lower resolution
python -m src.visualization.video --model ... --width 400 --height 300
```

### Evaluation Returns NaN

```bash
# Reload model with environment
python -c "
from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load('results/experiments/ppo_seed42/ppo_seed42_model.zip')
env = gym.make('LunarLander-v3')
"
```

---

## 14. References

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Project Constitution](../.specify/memory/constitution.md)
