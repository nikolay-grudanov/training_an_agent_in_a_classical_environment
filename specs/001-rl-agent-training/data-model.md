# Data Model: RL Agent Training System

## Entity: Environment
**Description**: RL environment with observation/action spaces, reward function, and episode termination conditions
**Fields**:
- name: str (e.g., "LunarLander-v2")
- observation_space: dict (shape, type, bounds of observations)
- action_space: dict (shape, type, bounds of actions)
- max_episode_steps: int (maximum steps per episode)
- reward_range: tuple (min and max possible rewards)

## Entity: Agent
**Description**: RL agent with policy network, learning algorithm, and action selection mechanism
**Fields**:
- algorithm_type: str (e.g., "PPO", "A2C", "SAC", "TD3")
- policy_network_architecture: dict (layers, activations, parameters)
- hyperparameters: dict (learning_rate, gamma, epsilon, etc.)
- training_state: dict (current weights, optimizer state, episode count)

## Entity: Configuration
**Description**: Parameters defining the training setup including algorithm, hyperparameters, and environment
**Fields**:
- algorithm: str (selected RL algorithm)
- environment: str (selected environment name)
- hyperparameters: dict (algorithm-specific parameters)
- seed: int (random seed for reproducibility)
- training_steps: int (total number of training steps)
- evaluation_frequency: int (evaluate every N steps)

## Entity: Experiment
**Description**: Controlled comparison between two or more algorithm/hyperparameter configurations
**Fields**:
- experiment_id: str (unique identifier)
- baseline_config: Configuration (reference configuration)
- variant_config: Configuration (modified configuration for comparison)
- hypothesis: str (predicted outcome of the experiment)
- results: dict (collected metrics and outcomes)

## Entity: Metrics
**Description**: Training metrics including average reward, episode length, convergence indicators
**Fields**:
- timestep: int (current training timestep)
- episode_number: int (current episode number)
- reward: float (current reward value)
- loss_values: dict (various loss values during training)
- episode_length: int (length of current episode)
- evaluation_score: float (score from evaluation run)

## Entity: TrainedModel
**Description**: Saved agent parameters and configuration for reproducible inference
**Fields**:
- model_weights: bytes (serialized model weights)
- algorithm_config: dict (configuration used for training)
- training_environment: str (environment used for training)
- seed_value: int (seed used for training)
- creation_timestamp: datetime (when the model was saved)

## Entity: Results
**Description**: Collected data from training sessions including performance metrics and visualizations
**Fields**:
- experiment_id: str (reference to the experiment)
- metrics_log: list[Metric] (complete training metrics history)
- performance_graph: str (path to performance graph file)
- evaluation_metrics: dict (final evaluation results)
- video_demonstration: str (path to video file showing trained agent)
- pip_freeze_output: str (dependencies snapshot for reproducibility)

## Relationships
- Configuration 1 -> * Experiment (one configuration can be used in multiple experiments as baseline or variant)
- Experiment 1 -> * Metrics (one experiment generates many metrics records)
- Experiment 1 -> 1 TrainedModel (one experiment produces one final model)
- Experiment 1 -> 1 Results (one experiment produces one results set)
- Environment 1 -> * Configuration (one environment can be used in multiple configurations)

## Validation Rules
- Environment.name must be one of the approved environments (LunarLander-v2, MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1)
- Agent.algorithm_type must be one of the supported algorithms (PPO, A2C, SAC, TD3)
- Configuration.seed must be a positive integer
- Metrics.timestep must be non-negative
- Experiment.hypothesis must be a non-empty string
- TrainedModel.creation_timestamp must be in ISO format