# QWEN.md - Training Agent in Classical Environment

## Project Overview

This is a reinforcement learning (RL) project focused on training agents in classical environments using established RL algorithms. The project is part of an educational curriculum at MEPHI (Moscow Engineering Physics Institute) and aims to demonstrate the application of ready-made RL algorithms to solve standard tasks.

The project involves:
- Training agents using algorithms like PPO, A2C, SAC, TD3, etc.
- Working with classical RL environments such as LunarLander-v2, MountainCarContinuous-v0, Acrobot-v1, or Pendulum-v1
- Conducting controlled experiments to compare algorithms, hyperparameters, or architectures
- Creating reproducible results with proper documentation and evaluation

## Technologies and Dependencies

The project uses Python 3.10.14 with dependencies managed through conda. Key packages include:
- `stable-baselines3` - Main RL algorithm library
- `gymnasium` - RL environment library
- `torch` - PyTorch for neural networks
- Various visualization and experiment tracking libraries

The environment is configured via `environment.yml` and requires activation with `conda activate rocm`.

## Project Structure

```
training_an_agent_in_a_classical_environment/
├── .specify/                 # Specification and planning tools
│   ├── memory/               # Memory for planning
│   ├── scripts/              # Scripts for automation
│   │   └── bash/
│   └── templates/            # Templates for specs and plans
│       ├── agent-file-template.md
│       ├── checklist-template.md
│       ├── plan-template.md
│       ├── spec-template.md
│       └── tasks-template.md
├── environment.yml           # Conda environment configuration
├── info_project.md           # Project description and requirements
├── .gitignore
├── .llm-context/             # LLM context configuration
├── .opencode/                # OpenCode configuration
└── .qwen/                    # Qwen configuration
```

## Building and Running

### Environment Setup
1. Install conda/miniconda if not already installed
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate rocm
   ```
3. Install any additional dependencies via pip as needed (avoid using conda install)

### Project Execution
The project involves implementing RL solutions with the following key components:
1. Select an environment from the approved list (LunarLander-v2, MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1)
2. Choose an RL algorithm (PPO, A2C, SAC, TD3, etc.)
3. Conduct two controlled experiments with clear hypotheses
4. Generate required outputs: graphs, animations/videos, and quantitative evaluations

## Development Conventions

### Mandatory Requirements
1. **Graph**: Average reward vs timestep/episode plot
2. **Animation/Video**: Demonstration of the final trained agent
3. **Quantitative Evaluation**: Average reward across 10-20 episodes
4. **Reproducibility**: Fixed seed, pip freeze output, complete training code

### Experiment Design
- Conduct two controlled experiments comparing:
  - Different algorithms on the same environment
  - Hyperparameter effects (e.g., target network update frequency)
  - Architecture changes
  - Reward shaping techniques
  - Exploration strategies

Each experiment must have a clear hypothesis (e.g., "I expect that increasing gamma will lead to longer planning horizons").

### Code Quality
- Use test-first approach with unit tests for all RL components
- Follow reproducible experiment practices with fixed seeds
- Include proper documentation and visualization of results
- Track and visualize training metrics appropriately

## Project Goals

The primary goal is to demonstrate proficiency in applying existing RL algorithms to solve standard tasks, conduct controlled experiments, and analyze their impact on training performance. Students should be able to train an agent to convergence within 30 minutes on CPU and produce comprehensive analysis of their results.

## Key Entities

- **Environment**: RL environment with observation/action spaces, reward function, and episode termination conditions
  - Properties: name, observation_space, action_space, max_episode_steps
- **Agent**: RL agent with policy network, learning algorithm, and action selection mechanism
  - Properties: algorithm_type, policy_network_architecture, hyperparameters, training_state
- **Experiment**: Controlled comparison between two or more algorithm/hyperparameter configurations
  - Properties: experiment_id, baseline_config, variant_config, hypothesis, results
- **Metrics**: Training metrics including average reward, episode length, convergence indicators
  - Properties: timestep, episode_number, reward, loss_values, episode_length, evaluation_score
- **TrainedModel**: Saved agent parameters and configuration for reproducible inference
  - Properties: model_weights, algorithm_config, training_environment, seed_value, creation_timestamp
- **Configuration**: Parameters defining the training setup including algorithm, hyperparameters, and environment
  - Properties: algorithm, environment, hyperparameters, seed, training_steps, evaluation_frequency
- **Results**: Collected data from training sessions including performance metrics and visualizations
  - Properties: experiment_id, metrics_log, performance_graph, evaluation_metrics, video_demonstration

## Functional Requirements

- **FR-001**: System MUST support training with established RL algorithms (PPO, A2C, SAC, TD3) in classical environments (LunarLander-v2, MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1)
- **FR-002**: System MUST enable conducting at least two controlled experiments with clearly formulated hypotheses
- **FR-003**: System MUST generate performance graphs showing average reward vs timesteps/episodes
- **FR-004**: System MUST produce animations/videos demonstrating the trained agent's behavior
- **FR-005**: System MUST calculate quantitative performance metrics across 10-20 episodes
- **FR-006**: System MUST ensure reproducibility through fixed seeds and dependency documentation
- **FR-007**: System MUST complete training within 30 minutes on CPU for standard environments
- **FR-008**: System MUST allow comparison of different algorithm configurations (hyperparameters, architectures)
- **FR-009**: System MUST track and store training metrics throughout the learning process
- **FR-010**: System MUST provide clear visualization of experimental results for analysis
- **FR-011**: System MUST implement comprehensive logging for debugging and monitoring purposes
- **FR-012**: System MUST provide real-time metrics dashboard for training progress monitoring
- **FR-013**: System MUST ensure experiments are comparable with standardized conditions and baselines
- **FR-014**: System MUST require each experiment to have a clear, testable hypothesis with predicted outcomes
- **FR-015**: System MUST implement comprehensive error handling for all edge cases
- **FR-016**: System MUST handle training timeouts gracefully with appropriate notifications
- **FR-017**: System MUST manage computational resource limitations during training
- **FR-018**: System MUST provide interactive visualizations for enhanced analysis
- **FR-019**: System MUST allow users to customize visualization parameters and views

## Edge Cases

- What happens when an environment takes longer than 30 minutes to converge?
- How does the system handle insufficient computational resources during training?
- What occurs when an agent fails to learn or performs worse over time?
- How does the system handle environments with sparse rewards or long horizons?
- How does the system handle training process interruption or failure?
- What happens when there are insufficient system resources (CPU, memory)?
- How does the system handle corrupted model checkpoints?
- What occurs when external dependencies are unavailable?

## Active Technologies
- Python 3.10.14 + Stable-Baselines3 (PPO as primary algorithm), Gymnasium (LunarLander-v2 environment), PyTorch, NumPy, Matplotlib, Plotly (for interactive visualizations), imageio (for video generation) (001-rl-agent-training)
- Files (experiment results, trained models, videos, configuration files) (001-rl-agent-training)

## Recent Changes
- 001-rl-agent-training: Added Python 3.10.14 + Stable-Baselines3 (PPO as primary algorithm), Gymnasium (LunarLander-v2 environment), PyTorch, NumPy, Matplotlib, Plotly (for interactive visualizations), imageio (for video generation)
