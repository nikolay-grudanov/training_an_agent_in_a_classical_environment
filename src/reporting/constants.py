"""Constants for reporting module."""

# Fixed seed for all experiments (for reproducibility)
FIXED_SEED = 42

# Evaluation parameters
MIN_EVAL_EPISODES = 10
MAX_EVAL_EPISODES = 20

# Reward threshold for convergence
REWARD_THRESHOLD = 200.0

# Plot settings
DEFAULT_FIGSIZE_LEARNING_CURVE = (12, 6)
DEFAULT_FIGSIZE_COMPARISON = (14, 6)
DEFAULT_DPI = 300

# Video settings
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_EPISODES = 5
DEFAULT_VIDEO_CODEC = "libx264"

# CSV column names (from existing experiments)
COL_TIMESTEPS = "timesteps"
COL_MEAN_REWARD = "reward_mean"
COL_STD_REWARD = "reward_std"
COL_EPISODE = "episode_count"
COL_WALLTIME = "walltime"

# CSV column names for eval_log.csv (evaluation data)
EVAL_COL_TIMESTEPS = "timesteps"
EVAL_COL_MEAN_REWARD = "mean_reward"
EVAL_COL_STD_REWARD = "std_reward"

# Convergence statuses
STATUS_CONVERGED = "CONVERGED"
STATUS_NOT_CONVERGED = "NOT_CONVERGED"
STATUS_UNKNOWN = "UNKNOWN"

# Environment
DEFAULT_ENV_NAME = "LunarLander-v3"

# Algorithm names
ALGO_PPO = "PPO"
ALGO_A2C = "A2C"
