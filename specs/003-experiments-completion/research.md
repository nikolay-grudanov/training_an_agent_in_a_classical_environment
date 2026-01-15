# Research Report: RL Experiments Completion & Convergence

**Branch**: `003-experiments-completion`  
**Date**: 15 января 2026  
**Research Focus**: Technical decisions for 200K timestep training, video rendering, and hyperparameter experiments

---

## 1. Hyperparameter Configuration for LunarLander-v3

### Decision: Default hyperparameters with seed=42 for reproducibility

**Rationale**:
- Stable-Baselines3 provides well-tuned default hyperparameters for both PPO and A2C
- For LunarLander-v3, 200K timesteps is a standard convergence target (as shown in SB3 documentation example)
- Using defaults ensures reproducibility across runs and comparison with published results

**PPO Default Hyperparameters** (from Stable-Baselines3):
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `n_epochs`: 10
- `gamma`: 0.99 (default, matches experiment requirement)
- `gae_lambda`: 0.95
- `clip_range`: 0.2
- `ent_coef`: 0.0

**A2C Default Hyperparameters**:
- `learning_rate`: 7e-4
- `n_steps`: 5
- `gamma`: 0.99
- `rms_prop_eps`: 1e-5
- `normalize_advantage`: True

**Alternatives Considered**:
- Custom tuned hyperparameters: Rejected - defaults are well-tuned for LunarLander-v3
- Optuna optimization: Could improve results but not required for baseline experiments

---

## 2. Checkpoint and Resume Functionality

### Decision: Use Stable-Baselines3 built-in `save()` and `load()` methods with frequent checkpoints

**Rationale**:
- SB3 provides native model serialization via `model.save(path)` and `model.load(path)`
- Checkpoints can be saved at regular intervals (every 50K timesteps) using callback
- Resume works seamlessly - just load the checkpoint before calling `learn()`

**Implementation Pattern**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_path}/checkpoint_{self.num_timesteps}"
            self.model.save(path)
        return True
```

**Checkpoint Frequency Decision**: Every 50K timesteps
- Balances storage overhead (minimal for zipped models) with progress protection
- 200K training = 4 checkpoints maximum

**Alternatives Considered**:
- Custom serialization: Rejected - SB3 format is standardized and widely supported
- Cloud storage for checkpoints: Out of scope - local storage sufficient for single-server setup

---

## 3. Video Rendering with imageio

### Decision: Use `gymnasium[mujoco,image]` extras and imageio for video generation

**Rationale**:
- `gymnasium` provides `record_video` wrapper for automatic video capture
- `imageio` enables saving frames to MP4 format without external codecs
- Both are standard dependencies in the RL ecosystem

**Implementation Pattern**:
```python
import gymnasium as gym
from stable_baselines3 import PPO
import imageio

def generate_video(model_path, env_id, video_path, n_episodes=5):
    model = PPO.load(model_path)
    env = gym.make(env_id, render_mode="rgb_array")

    frames = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    imageio.mimsave(video_path, frames, fps=30)
    env.close()
```

**Video Settings**:
- Format: MP4 (H.264 codec, widely compatible)
- FPS: 30 (smooth playback, standard)
- Resolution: 600x400 (per requirements)
- Episodes: 5 (per requirements)

**Memory Considerations**:
- Each frame ~600x400x3 bytes = ~720KB
- 5 episodes × ~500 steps = 2500 frames max
- Total memory: ~1.8GB peak - acceptable for standard Linux server

**Alternatives Considered**:
- FFmpeg directly: More complex, requires external installation
- OpenAI Baselines video wrapper: Outdated, Gymnasium version preferred

---

## 4. Metrics Collection and Visualization

### Decision: Use `stable_baselines3.common.monitor.Monitor` wrapper and Matplotlib

**Rationale**:
- `Monitor` wrapper automatically logs episode rewards, lengths, and timestamps
- CSV output format is human-readable and easy to parse
- Matplotlib provides publication-quality graphs with minimal code

**Metrics to Collect**:
- Timesteps (x-axis)
- Mean reward (rolling window of 100 episodes)
- Standard deviation
- Episode length (for convergence analysis)

**Graph Types Required**:
1. **Learning Curve**: Mean reward vs. timesteps (single experiment)
2. **Comparison Plot**: A2C vs. PPO on same graph (baseline experiments)
3. **Hyperparameter Study**: Gamma variations comparison

**Implementation Pattern**:
```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(csv_path, output_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['timesteps'], df['mean_reward'], label='Mean Reward')
    plt.fill_between(df['timesteps'],
                     df['mean_reward'] - df['std_reward'],
                     df['mean_reward'] + df['std_reward'],
                     alpha=0.2)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

**Alternatives Considered**:
- TensorBoard: More complex, requires additional setup
- Weights & Biases: External service, not in scope
- Seaborn: Aesthetic but adds dependency; Matplotlib sufficient

---

## 5. Statistical Analysis for Hyperparameter Experiments

### Decision: Use scipy.stats for t-test and effect size calculation

**Rationale**:
- scipy provides well-validated statistical tests
- t-test for comparing means between gamma configurations
- Effect size (Cohen's d) for practical significance

**Statistical Tests**:
1. **Independent t-test**: Compare final rewards between gamma values
2. **One-way ANOVA**: Compare all three gamma configurations simultaneously
3. **Effect size**: Cohen's d to quantify practical difference

**Implementation Pattern**:
```python
from scipy import stats
import numpy as np

def analyze_experiment_results(rewards_a, rewards_b):
    """Compare two experimental conditions."""
    t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)
    cohens_d = (np.mean(rewards_a) - np.mean(rewards_b)) / np.sqrt(
        (np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2
    )
    return {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d}
```

**Significance Threshold**: p < 0.05 (per requirements)

---

## 6. LunarLander-v3 Environment Details

### Decision: Standard configuration with render_mode="rgb_array" for training, "human" for videos

**Environment Parameters**:
- **Observation Space**: Box(8,) - [x, y, velocity_x, velocity_y, angle, angular_velocity, left_leg_contact, right_leg_contact]
- **Action Space**: Discrete(4) - [main_engine, left_engine, right_engine, no_op]
- **Reward Structure**: Shaped rewards for fuel efficiency, landing precision, and leg contact
- **Success Criterion**: Average reward ≥ 200 points indicates reliable landing capability

**Termination Conditions**:
- Lander goes out of bounds
- Lander crashes (non-soft landing)
- Episode timeout (1000 steps)

**Convergence Indicators**:
- Smooth learning curve (reward increases steadily)
- Low variance between episodes (std < 50)
- Plateau at ≥200 average reward

---

## 7. Experiment Design Summary

### Baseline Experiments

| Experiment | Algorithm | Seed | Timesteps | Hypothesis |
|------------|-----------|------|-----------|------------|
| A2C Baseline | A2C | 42 | 200K | Default hyperparameters achieve convergence |
| PPO Baseline | PPO | 42 | 200K | PPO achieves better final performance than A2C |

### Controlled Hyperparameter Experiment

| Configuration | Gamma | Seed | Timesteps | Hypothesis |
|---------------|-------|------|-----------|------------|
| Gamma Low | 0.90 | 42 | 100K* | Lower gamma prioritizes immediate rewards, faster initial learning but potentially suboptimal final performance |
| Gamma Medium | 0.99 | 42 | 100K* | Default gamma provides balance between immediate and long-term rewards |
| Gamma High | 0.999 | 42 | 100K* | Higher gamma emphasizes long-term planning, slower initial learning but potentially more stable landing |

*100K timesteps for gamma experiment sufficient to observe convergence patterns while keeping total experiment time reasonable.

---

## 8. Implementation Recommendations

### Code Structure

```
src/experiments/completion/
├── __init__.py
├── baseline_training.py      # A2C and PPO baseline training
├── gamma_experiment.py       # Controlled gamma variations
├── metrics_collector.py      # Metrics logging and parsing
├── visualization.py          # Graph generation
├── video_generator.py        # MP4 video rendering
└── report_generator.py       # Markdown report creation
```

### Key Files to Create/Modify

1. **New**: `src/experiments/completion/baseline_training.py` - Main training script
2. **New**: `src/experiments/completion/gamma_experiment.py` - Hyperparameter study
3. **New**: `src/experiments/completion/metrics_collector.py` - Metrics handling
4. **New**: `src/experiments/completion/visualization.py` - Matplotlib graphs
5. **New**: `src/experiments/completion/video_generator.py` - Video rendering
6. **New**: `src/experiments/completion/report_generator.py` - Report creation
7. **Existing**: `src/training/trainer.py` - May need modifications for 200K support

---

## 9. Resolved Clarifications

| Question | Answer |
|----------|--------|
| Optimal learning rate for A2C/PPO? | Use SB3 defaults: 7e-4 (A2C), 3e-4 (PPO) |
| Checkpoint frequency? | Every 50K timesteps balances safety and storage |
| Video rendering memory? | ~1.8GB peak for 5 episodes, acceptable on Linux server |
| Gamma default? | 0.99 (standard in SB3, matches experiment requirements) |
| Convergence threshold? | 200 points average reward indicates reliable landing |

---

## 10. References

- Stable-Baselines3 Documentation: https://github.com/dlr-rm/stable-baselines3
- Gymnasium LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- RL Baselines3 Zoo (hyperparameter reference): https://github.com/dlr-rm/rl-baselines3-zoo
