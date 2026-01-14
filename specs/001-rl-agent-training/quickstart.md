# Quickstart Guide: RL Agent Training System

## Prerequisites

- Python 3.10.14
- Conda package manager
- Access to the project repository

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd training_an_agent_in_a_classical_environment
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate rocm
```

3. Install additional dependencies via pip if needed:
```bash
pip install <additional-packages>
```

## Running Your First Training Session

1. Navigate to the project directory:
```bash
cd /path/to/project
```

2. Activate the environment:
```bash
conda activate rocm
```

3. Run the basic training script:
```bash
python -m src.training.train_ppo_lunarlander --env LunarLander-v3 --algo PPO --seed 42
```

## Running Experiments

1. To run a controlled experiment comparing two configurations:
```bash
python -m src.experiments.run_experiment --baseline-config configs/baseline_ppo.yaml --variant-config configs/variant_ppo.yaml
```

2. To run the default experiment (PPO vs A2C on LunarLander-v3):
```bash
python -m src.experiments.default_comparison
```

## Generating Visualizations

1. To generate performance graphs from training logs:
```bash
python -m src.visualization.plot_training_curves --log-dir results/logs --output-dir results/plots
```

2. To create a video of the trained agent:
```bash
python -m src.visualization.generate_video --model-path results/models/final_model.zip --env LunarLander-v3 --output-path results/videos/agent_demo.mp4
```

## Reproducibility

1. To ensure reproducible results, always use fixed seeds:
```python
import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
```

2. To generate pip freeze output for dependency tracking:
```bash
pip freeze > requirements_snapshot.txt
```

## Jupyter Notebook Development

1. Start Jupyter server:
```bash
jupyter lab
```

2. Navigate to the notebooks directory and open `exploration.ipynb` to begin experimenting with different algorithms and hyperparameters.

## Running Tests

1. To run all unit tests:
```bash
pytest tests/unit/
```

2. To run integration tests:
```bash
pytest tests/integration/
```

## Key Files and Directories

- `src/agents/`: Contains agent implementations
- `src/environments/`: Environment wrappers and configurations
- `src/training/`: Training pipelines and utilities
- `src/experiments/`: Experiment configurations and runners
- `src/utils/`: Helper functions for reproducibility and logging
- `src/visualization/`: Plotting and video generation utilities
- `notebooks/`: Jupyter notebooks for exploration
- `configs/`: Configuration files for different experiments
- `results/`: Output directory for models, logs, and visualizations