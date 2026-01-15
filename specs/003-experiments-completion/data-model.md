# Data Model: RL Experiments Completion & Convergence

**Branch**: `003-experiments-completion`  
**Date**: 15 января 2026  
**Related**: [spec.md](./spec.md), [research.md](./research.md)

---

## Entity Relationship Overview

```
┌─────────────────────┐       ┌─────────────────────┐
│ ExperimentConfig    │──────▶│ TrainedAgent        │
│ (training setup)    │       │ (model with weights)│
└─────────────────────┘       └─────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
┌─────────────────────┐       ┌─────────────────────┐
│ ExperimentResults   │◀──────│ PerformanceGraph    │
│ (complete outcomes) │       │ (PNG visualization) │
└─────────────────────┘       └─────────────────────┘
         │
         │         ┌─────────────────────┐
         └────────▶│ AgentVideo          │
                   │ (MP4 demonstration) │
                   └─────────────────────┘
```

---

## Core Entities

### 1. TrainedAgent

RL agent model with weights, hyperparameters, training history, and final performance metrics.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `agent_id` | string | Yes | UUID format | Unique identifier |
| `algorithm` | enum | Yes | "PPO" \| "A2C" | Learning algorithm |
| `version` | string | Yes | Semantic versioning | SB3 version used |
| `timesteps_trained` | integer | Yes | > 0 | Total environment steps |
| `final_reward_mean` | float | Yes | -∞ to +∞ | Average reward over eval episodes |
| `final_reward_std` | float | Yes | ≥ 0 | Standard deviation |
| `seed` | integer | Yes | 0 to 2^32-1 | Random seed for reproducibility |
| `convergence_achieved` | boolean | Yes | - | Whether ≥200 threshold met |
| `training_duration_seconds` | integer | Yes | > 0 | Wall-clock training time |
| `created_at` | datetime | Yes | ISO 8601 | Timestamp of creation |
| `hyperparameters` | object | Yes | Valid dict | Algorithm-specific params |

**Hyperparameters Schema** (PPO example):

```json
{
  "type": "object",
  "properties": {
    "learning_rate": {"type": "number", "default": 0.0003},
    "n_steps": {"type": "integer", "default": 2048},
    "batch_size": {"type": "integer", "default": 64},
    "n_epochs": {"type": "integer", "default": 10},
    "gamma": {"type": "number", "default": 0.99},
    "gae_lambda": {"type": "number", "default": 0.95},
    "clip_range": {"type": "number", "default": 0.2},
    "ent_coef": {"type": "number", "default": 0.0}
  },
  "required": ["gamma"]
}
```

**Hyperparameters Schema** (A2C example):

```json
{
  "type": "object",
  "properties": {
    "learning_rate": {"type": "number", "default": 0.0007},
    "n_steps": {"type": "integer", "default": 5},
    "gamma": {"type": "number", "default": 0.99},
    "rms_prop_eps": {"type": "number", "default": 1e-5},
    "normalize_advantage": {"type": "boolean", "default": true}
  },
  "required": ["gamma"]
}
```

**State Machine**:

```
Created → Training → Evaluating → Completed
                ↓           ↓
            Failed     Failed
```

---

### 2. ExperimentConfiguration

Training setup defining algorithm, environment, hyperparameters, and seeds.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `config_id` | string | Yes | UUID format | Unique identifier |
| `experiment_type` | enum | Yes | "baseline" \| "hyperparameter" | Type of experiment |
| `algorithm` | enum | Yes | "PPO" \| "A2C" | Learning algorithm |
| `environment` | string | Yes | Environment ID | "LunarLander-v3" |
| `timesteps` | integer | Yes | > 0 | Training duration |
| `seed` | integer | Yes | 0 to 2^32-1 | Random seed |
| `gamma` | number | Yes | 0.0 to 1.0 | Discount factor |
| `learning_rate` | number | No | > 0 | Override default if specified |
| `checkpoint_freq` | integer | No | > 0 | Default: 50000 |
| `eval_freq` | integer | No | > 0 | Evaluation interval |
| `n_eval_episodes` | integer | No | > 0 | Default: 10 |
| `hypothesis` | string | No | Max 500 chars | Expected outcome |
| `notes` | string | No | - | Additional notes |

**Example**:

```json
{
  "config_id": "exp-gamma-099-001",
  "experiment_type": "hyperparameter",
  "algorithm": "PPO",
  "environment": "LunarLander-v3",
  "timesteps": 100000,
  "seed": 42,
  "gamma": 0.99,
  "learning_rate": 0.0003,
  "checkpoint_freq": 25000,
  "eval_freq": 5000,
  "n_eval_episodes": 10,
  "hypothesis": "gamma=0.99 provides best balance between immediate and long-term rewards"
}
```

---

### 3. ExperimentResults

Complete outcomes from single experiment including metrics, models, and analysis.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `experiment_id` | string | Yes | UUID format | Unique identifier |
| `config_used` | object | Yes | Valid ExperimentConfiguration | Configuration applied |
| `model_path` | string | Yes | Valid path | Path to saved model `.zip` |
| `metrics_history` | array | Yes | Array of MetricPoint | Time series data |
| `final_evaluation` | object | Yes | EvaluationResult | Final performance |
| `training_logs` | string | Yes | Valid path | Path to training log file |
| `graph_paths` | object | Yes | Valid paths | Paths to generated graphs |
| `video_path` | string | No | Valid path | Path to demonstration video |
| `statistical_summary` | object | No | Valid StatsSummary | Analysis results |
| `status` | enum | Yes | "completed" \| "failed" \| "interrupted" | Final state |
| `completed_at` | datetime | Yes | ISO 8601 | End timestamp |

**MetricPoint Schema**:

```json
{
  "type": "object",
  "properties": {
    "timesteps": {"type": "integer", "description": "Cumulative timesteps"},
    "walltime": {"type": "number", "description": "Wall-clock time in seconds"},
    "reward_mean": {"type": "number", "description": "Mean reward"},
    "reward_std": {"type": "number", "description": "Standard deviation"},
    "episode_count": {"type": "integer", "description": "Episodes in window"},
    "fps": {"type": "number", "description": "Training speed"}
  },
  "required": ["timesteps", "reward_mean"]
}
```

**GraphPaths Schema**:

```json
{
  "type": "object",
  "properties": {
    "learning_curve": {"type": "string", "format": "uri"},
    "comparison": {"type": "string", "format": "uri"},
    "aggregate": {"type": "string", "format": "uri"}
  },
  "required": ["learning_curve"]
}
```

**StatsSummary Schema**:

```json
{
  "type": "object",
  "properties": {
    "convergence_rate": {"type": "number", "description": "Episodes to convergence"},
    "final_vs_baseline": {"type": "number", "description": "Percent improvement"},
    "significance_test": {
      "type": "object",
      "properties": {
        "test_name": {"type": "string"},
        "t_statistic": {"type": "number"},
        "p_value": {"type": "number"},
        "effect_size": {"type": "number"}
      }
    }
  }
}
```

---

### 4. PerformanceGraph

Visualization artifact showing training progress.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `graph_id` | string | Yes | UUID format | Unique identifier |
| `graph_type` | enum | Yes | "learning_curves" \| "comparison" \| "aggregate" | Type of visualization |
| `x_axis` | enum | Yes | "timesteps" \| "episodes" \| "time" | Independent variable |
| `y_axis` | enum | Yes | "reward" \| "length" \| "value_loss" | Dependent variable |
| `experiments_included` | array | Yes | Array of experiment IDs | Data sources |
| `file_path` | string | Yes | Valid path | PNG file location |
| `width` | integer | Yes | ≥ 100 | Pixels |
| `height` | integer | Yes | ≥ 100 | Pixels |
| `dpi` | integer | Yes | ≥ 72 | Resolution |
| "format" | enum | Yes | "PNG" \| "SVG" \| "PDF" | File format |
| "annotations" | array | No | Array of Annotation | Statistical overlays |

**Annotation Schema**:

```json
{
  "type": "object",
  "properties": {
    "type": {"type": "string", "enum": ["line", "text", "region"]},
    "x": {"type": "number"},
    "y": {"type": "number"},
    "text": {"type": "string"},
    "style": {"type": "string"}
  }
}
```

---

### 5. AgentDemonstrationVideo

Video artifact showing agent behavior in environment.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `video_id` | string | Yes | UUID format | Unique identifier |
| `model_used` | string | Yes | Valid TrainedAgent ID | Agent being demonstrated |
| `model_path` | string | Yes | Valid path | Path to model file |
| `episodes_rendered` | integer | Yes | ≥ 1 | Number of episodes |
| `video_path` | string | Yes | Valid path | MP4 file location |
| `format` | enum | Yes | "MP4" \| "GIF" \| "WEBM" | Container format |
| "codec" | string | Yes | - | "H.264" |
| `width` | integer | Yes | ≥ 100 | Pixels |
| `height` | integer | Yes | ≥ 100 | Pixels |
| `fps` | integer | Yes | 1 to 60 | Frames per second |
| `duration_seconds` | number | Yes | > 0 | Total duration |
| `score_mean` | float | Yes | -∞ to +∞ | Average episode score |
| `score_std` | float | Yes | ≥ 0 | Score variation |
| `generated_at` | datetime | Yes | ISO 8601 | Creation timestamp |

---

### 6. ExperimentalReport

Documentation consolidating all experiments and findings.

**Fields**:

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `report_id` | string | Yes | UUID format | Unique identifier |
| `report_date` | datetime | Yes | ISO 8601 | Generation date |
| `experiments_included` | array | Yes | Array of experiment IDs | Coverage |
| `hypotheses` | array | Yes | Array of Hypothesis | Tested hypotheses |
| `methodology` | string | Yes | - | Experimental design description |
| `statistical_summary` | object | Yes | Valid StatsSummary | Cross-experiment analysis |
| `conclusions` | array | Yes | Array of string | Key findings |
| `recommendations` | array | No | Array of string | Future work suggestions |
| `embedded_artifacts` | object | Yes | Valid ArtifactsRef | References to graphs/videos |
| `file_path` | string | Yes | Valid path | Markdown file location |

**Hypothesis Schema**:

```json
{
  "type": "object",
  "properties": {
    "id": {"type": "string"},
    "statement": {"type": "string"},
    "experiment_id": {"type": "string"},
    "result": {"type": "string", "enum": ["supported", "refuted", "inconclusive"]},
    "evidence": {"type": "string"}
  },
  "required": ["statement", "result"]
}
```

**ArtifactsRef Schema**:

```json
{
  "type": "object",
  "properties": {
    "graphs": {"type": "array", "items": {"type": "string"}},
    "videos": {"type": "array", "items": {"type": "string"}},
    "models": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["graphs"]
}
```

---

## Validation Rules

### Cross-Field Validation

1. **Agent-Configuration Consistency**: `TrainedAgent.algorithm` must match `ExperimentConfiguration.algorithm`
2. **Metrics Completeness**: `ExperimentResults.metrics_history` must have at least 10 data points for valid analysis
3. **Video-Agent Link**: `AgentDemonstrationVideo.model_used` must reference existing `TrainedAgent.agent_id`
4. **Experiment Coverage**: `ExperimentalReport.experiments_included` must reference at least one completed experiment

### State Transition Rules

| From State | To State | Condition |
|------------|----------|-----------|
| Created | Training | `start()` called |
| Training | Evaluating | `timesteps` reached |
| Evaluating | Completed | Evaluation finished |
| Training | Failed | Exception raised |
| Evaluating | Failed | Evaluation error |
| Training | Interrupted | User stop requested |

---

## File Storage Schema

```
results/
└── experiments/
    ├── {experiment_id}/
    │   ├── config.json           # ExperimentConfiguration
    │   ├── model.zip             # Saved SB3 model (TrainedAgent)
    │   ├── metrics.csv           # Metrics history
    │   ├── training.log          # Training logs
    │   ├── reward_curve.png      # PerformanceGraph
    │   ├── video.mp4             # AgentDemonstrationVideo
    │   └── results.json          # ExperimentResults
    └── reports/
        └── experiment_report.md  # ExperimentalReport
```

---

## API Data Contracts

### Training Request

```json
{
  "type": "object",
  "properties": {
    "config": {"$ref": "#/ExperimentConfiguration"},
    "resume_from": {"type": "string", "format": "uri"},
    " callbacks": {"type": "array"}
  },
  "required": ["config"]
}
```

### Training Response

```json
{
  "type": "object",
  "properties": {
    "experiment_id": {"type": "string"},
    "status": {"type": "string", "enum": ["started", "resumed"]},
    "estimated_duration_seconds": {"type": "integer"},
    "checkpoints_will_be_saved": {"type": "boolean"}
  },
  "required": ["experiment_id", "status"]
}
```

### Evaluation Response

```json
{
  "type": "object",
  "properties": {
    "agent_id": {"type": "string"},
    "episodes_evaluated": {"type": "integer"},
    "reward_mean": {"type": "number"},
    "reward_std": {"type": "number"},
    "success_rate": {"type": "number"},
    "convergence_achieved": {"type": "boolean"}
  },
  "required": ["agent_id", "reward_mean"]
}
```
