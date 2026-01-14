# Experiment Configuration Guide

This guide explains how to create and use experiment configuration files for controlled RL experiments.

## Overview

The experiment configuration system uses YAML files to define:
- Experiment metadata and execution settings
- Baseline and variant algorithm configurations
- Evaluation and comparison parameters
- Output and visualization settings

## Configuration Structure

### Main Schema File

`configs/experiment_schema.yaml` - Contains the complete schema with all available options and defaults.

### Experiment-Specific Files

- `configs/ppo_vs_a2c_experiment.yaml` - PPO vs A2C comparison
- `configs/hyperparameter_tuning_experiment.yaml` - Hyperparameter sensitivity analysis

## Configuration Sections

### 1. Experiment Metadata

```yaml
experiment:
  name: "my_experiment"
  description: "Description of what this experiment tests"
  hypothesis: "Expected outcome and reasoning"
  
  execution:
    mode: "sequential"  # sequential, parallel, validation
    timeout_minutes: 60
    max_retries: 3
    enable_monitoring: true
    save_checkpoints: true
    
  output:
    base_directory: "results/experiments"
    save_models: true
    save_videos: true
    save_plots: true
    export_format: "json"  # json, csv, both
```

**Execution Modes:**
- `sequential`: Run baseline first, then variant (safer, slower)
- `parallel`: Run both configurations simultaneously (faster, more resource intensive)
- `validation`: Dry-run mode for testing configurations

### 2. Algorithm Configurations

#### Baseline Configuration
```yaml
baseline:
  name: "baseline_ppo"
  algorithm: "PPO"  # PPO, A2C, SAC, TD3
  environment: "LunarLander-v2"
  seed: 42
  training_steps: 100000
  evaluation_frequency: 10000
  
  hyperparameters:
    learning_rate: 0.0003
    # ... algorithm-specific parameters
```

#### Variant Configuration
```yaml
variant:
  name: "variant_a2c"
  algorithm: "A2C"
  environment: "LunarLander-v2"  # Must match baseline
  seed: 42  # Must match baseline for fair comparison
  training_steps: 100000  # Should match baseline
  evaluation_frequency: 10000
  
  hyperparameters:
    learning_rate: 0.0007
    # ... algorithm-specific parameters
```

**Important:** For controlled experiments:
- `environment` and `seed` should be identical between baseline and variant
- `training_steps` should typically be the same
- Only change the specific parameter(s) you want to test

### 3. Evaluation Settings

```yaml
evaluation:
  num_episodes: 20  # Episodes for final evaluation
  
  metrics:
    - "mean_reward"
    - "std_reward"
    - "episode_length"
    - "convergence_steps"
    - "final_performance"
    - "peak_performance"
    - "stability_score"
  
  statistical_tests:
    significance_level: 0.05
    use_bonferroni_correction: true
    bootstrap_samples: 1000
    confidence_interval: 0.95
```

### 4. Comparison Settings

```yaml
comparison:
  thresholds:
    convergence_reward: 200.0  # Environment-specific success threshold
    minimum_improvement: 0.05  # 5% minimum improvement
    stability_threshold: 0.1   # Maximum acceptable std deviation
  
  plots:
    learning_curves: true
    reward_distributions: true
    statistical_comparison: true
    convergence_analysis: true
    save_format: "png"  # png, svg, pdf
    dpi: 300
    
  report:
    generate_html: true
    generate_markdown: true
    include_raw_data: false
    include_statistical_details: true
```

## Supported Algorithms

### PPO (Proximal Policy Optimization)
```yaml
hyperparameters:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
```

### A2C (Advantage Actor-Critic)
```yaml
hyperparameters:
  learning_rate: 0.0007
  n_steps: 5
  gamma: 0.99
  gae_lambda: 1.0
  ent_coef: 0.01
  vf_coef: 0.25
  max_grad_norm: 0.5
```

### SAC (Soft Actor-Critic)
```yaml
hyperparameters:
  learning_rate: 0.0003
  buffer_size: 1000000
  learning_starts: 100
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: "auto"
  target_update_interval: 1
```

### TD3 (Twin Delayed DDPG)
```yaml
hyperparameters:
  learning_rate: 0.001
  buffer_size: 1000000
  learning_starts: 100
  batch_size: 100
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  policy_delay: 2
  target_policy_noise: 0.2
  target_noise_clip: 0.5
```

## Supported Environments

### LunarLander-v2
- **Type**: Continuous control
- **Convergence threshold**: 200.0
- **Max episode steps**: 1000
- **Description**: Land a spacecraft safely

### Pendulum-v1
- **Type**: Continuous control
- **Convergence threshold**: -200.0
- **Max episode steps**: 200
- **Description**: Swing up and balance an inverted pendulum

### MountainCarContinuous-v0
- **Type**: Continuous control
- **Convergence threshold**: 90.0
- **Max episode steps**: 999
- **Description**: Drive a car up a steep hill

### Acrobot-v1
- **Type**: Discrete control
- **Convergence threshold**: -100.0
- **Max episode steps**: 500
- **Description**: Swing up an underactuated pendulum

## Usage Examples

### 1. Algorithm Comparison

Create `my_algorithm_comparison.yaml`:
```yaml
experiment:
  name: "ppo_vs_sac_comparison"
  hypothesis: "SAC will outperform PPO on continuous control tasks"

baseline:
  algorithm: "PPO"
  environment: "Pendulum-v1"
  
variant:
  algorithm: "SAC"
  environment: "Pendulum-v1"
```

### 2. Hyperparameter Tuning

Create `learning_rate_study.yaml`:
```yaml
experiment:
  name: "learning_rate_sensitivity"
  hypothesis: "Higher learning rate will converge faster but be less stable"

baseline:
  algorithm: "PPO"
  hyperparameters:
    learning_rate: 0.0003
    
variant:
  algorithm: "PPO"
  hyperparameters:
    learning_rate: 0.001
```

### 3. Environment Generalization

Create `environment_generalization.yaml`:
```yaml
experiment:
  name: "ppo_environment_generalization"
  hypothesis: "PPO will perform consistently across different environments"

baseline:
  algorithm: "PPO"
  environment: "LunarLander-v2"
  
variant:
  algorithm: "PPO"
  environment: "Pendulum-v1"
```

## Running Experiments

### Command Line
```bash
# Run with specific config
python -m src.experiments.runner --config configs/my_experiment.yaml

# Run with different execution modes
python -m src.experiments.runner --config configs/my_experiment.yaml --mode parallel
python -m src.experiments.runner --config configs/my_experiment.yaml --mode validation

# Override specific parameters
python -m src.experiments.runner \
  --config configs/my_experiment.yaml \
  --training-steps 50000 \
  --output-dir results/quick_test
```

### Python Code
```python
from src.experiments.runner import ExperimentRunner
from src.experiments.config import Configuration
from src.experiments.experiment import Experiment

# Load configuration
baseline_config = Configuration.load("configs/baseline.yaml")
variant_config = Configuration.load("configs/variant.yaml")

# Create experiment
experiment = Experiment(
    baseline_config=baseline_config,
    variant_config=variant_config,
    hypothesis="My hypothesis"
)

# Run experiment
runner = ExperimentRunner(experiment)
success = runner.run()

if success:
    print(f"Results: {experiment.compare_results()}")
```

## Best Practices

### 1. Controlled Experiments
- Change only one variable at a time
- Use identical seeds for reproducibility
- Match training steps and evaluation frequency
- Use appropriate sample sizes (num_episodes ≥ 20)

### 2. Statistical Rigor
- Set appropriate significance levels (α = 0.05)
- Use Bonferroni correction for multiple comparisons
- Include confidence intervals
- Report effect sizes, not just p-values

### 3. Resource Management
- Use sequential mode for limited resources
- Set appropriate timeouts
- Monitor memory usage
- Clean up intermediate files

### 4. Documentation
- Write clear hypotheses
- Document parameter choices
- Include experimental rationale
- Save configuration with results

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   - Check algorithm and environment compatibility
   - Verify hyperparameter names and types
   - Ensure required fields are present

2. **Resource Constraints**
   - Reduce training steps for testing
   - Use sequential execution mode
   - Increase timeout values
   - Monitor system resources

3. **Statistical Issues**
   - Increase number of evaluation episodes
   - Check for non-normal distributions
   - Use appropriate statistical tests
   - Consider effect size vs significance

4. **Reproducibility Problems**
   - Verify seed consistency
   - Check for non-deterministic operations
   - Document environment versions
   - Save complete configuration

## Configuration Validation

The system automatically validates:
- Algorithm and environment compatibility
- Required parameter presence
- Parameter type checking
- Value range validation
- Configuration consistency between baseline and variant

Invalid configurations will be rejected with clear error messages.