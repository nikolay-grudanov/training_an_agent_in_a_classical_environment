# Data Model: Project Cleanup and PPO vs A2C Experiments

**Feature**: Project Cleanup and PPO vs A2C Experiments
**Date**: 2026-01-15
**Phase**: Phase 1 - Design & Contracts

## Overview

This document defines the data entities, their relationships, and validation rules for the project cleanup and PPO vs A2C experiments feature. The data model covers audit reports, experiment results, project structure, trained agents, and training metrics.

## Core Entities

### 1. AuditReport

**Description**: Document containing status assessment of all project components with categorized listings.

**File Format**: JSON

**Structure**:
```json
{
  "audit_report": {
    "metadata": {
      "date": "2026-01-15T10:30:00Z",
      "auditor": "Automated Audit System",
      "scope": "src/",
      "version": "1.0.0"
    },
    "summary": {
      "total_modules": 16,
      "working": 12,
      "broken": 2,
      "needs_fixing": 2
    },
    "modules": [
      {
        "module_name": "base_agent",
        "file_path": "src/agents/base.py",
        "import_status": "success",
        "functionality_test": "pass",
        "status": "working",
        "status_icon": "✅",
        "notes": "No issues detected"
      },
      {
        "module_name": "ppo_agent",
        "file_path": "src/agents/ppo_agent.py",
        "import_status": "error",
        "error_message": "ModuleNotFoundError: No module named 'stable_baselines3'",
        "functionality_test": "skip",
        "status": "broken",
        "status_icon": "❌",
        "notes": "Missing dependency: stable_baselines3"
      }
    ]
  }
}
```

**Fields**:
- `metadata`: Audit execution information
  - `date` (datetime, required): When audit was run
  - `auditor` (string, required): System or person running audit
  - `scope` (string, required): Directory path audited
  - `version` (string, required): Audit tool version
- `summary` (object, required): High-level statistics
  - `total_modules` (int, required): Total modules audited
  - `working` (int, required): Count of working modules
  - `broken` (int, required): Count of broken modules
  - `needs_fixing` (int, required): Count of modules needing fixes
- `modules` (array, required): Individual module assessments
  - `module_name` (string, required): Name of the module
  - `file_path` (string, required): Relative path from project root
  - `import_status` (enum: success/error/warning, required): Import test result
  - `error_message` (string, optional): Error details if import failed
  - `functionality_test` (enum: pass/fail/skip, required): Smoke test result
  - `status` (enum: working/broken/needs_fixing, required): Final status
  - `status_icon` (enum: ✅/❌/⚠️, required): Visual indicator
  - `notes` (string, optional): Additional information

**Validation Rules**:
- All `status` counts must sum to `total_modules`
- If `import_status` is "error", `error_message` must be provided
- If `functionality_test` is "skip", `import_status` must be "error"

---

### 2. ExperimentResults

**Description**: Structured data containing training metrics, configuration parameters, and references to saved models.

**File Format**: JSON

**Structure**:
```json
{
  "experiment_results": {
    "metadata": {
      "experiment_id": "ppo_seed42",
      "algorithm": "PPO",
      "environment": "LunarLander-v3",
      "seed": 42,
      "start_time": "2026-01-15T10:00:00Z",
      "end_time": "2026-01-15T10:28:30Z",
      "total_timesteps": 50000,
      "conda_environment": "rocm"
    },
    "model": {
      "algorithm": "PPO",
      "policy": "MlpPolicy",
      "model_file": "ppo_seed42_model.zip",
      "model_path": "results/experiments/ppo_seed42/",
      "checkpoint_interval": 1000,
      "checkpoints": [
        "checkpoint_1000.zip",
        "checkpoint_2000.zip",
        "...",
        "checkpoint_50000.zip"
      ]
    },
    "metrics": {
      "final_reward_mean": 250.5,
      "final_reward_std": 12.3,
      "episode_length_mean": 250.0,
      "total_episodes": 200,
      "training_time_seconds": 1710.0,
      "converged": true
    },
    "hyperparameters": {
      "learning_rate": 0.0003,
      "n_steps": 2048,
      "batch_size": 64,
      "n_epochs": 10,
      "gamma": 0.99,
      "gae_lambda": 0.95
    },
    "environment": {
      "name": "LunarLander-v3",
      "observation_space": "Box(8,)",
      "action_space": "Discrete(4)",
      "reward_threshold": 200.0
    }
  }
}
```

**Fields**:
- `metadata` (object, required): Experiment identification
  - `experiment_id` (string, required): Unique identifier (e.g., "ppo_seed42")
  - `algorithm` (enum: PPO/A2C, required): Algorithm used
  - `environment` (string, required): Gymnasium environment name
  - `seed` (int, required): Random seed for reproducibility
  - `start_time` (datetime, required): Training start timestamp
  - `end_time` (datetime, required): Training end timestamp
  - `total_timesteps` (int, required): Total training timesteps
  - `conda_environment` (string, required): Conda environment name
- `model` (object, required): Model information
  - `algorithm` (string, required): Algorithm class
  - `policy` (string, required): Policy architecture
  - `model_file` (string, required): Final model filename
  - `model_path` (string, required): Directory containing model
  - `checkpoint_interval` (int, required): Steps between checkpoints
  - `checkpoints` (array of strings, required): Checkpoint filenames
- `metrics` (object, required): Training performance metrics
  - `final_reward_mean` (float, required): Mean reward over evaluation episodes
  - `final_reward_std` (float, required): Standard deviation of rewards
  - `episode_length_mean` (float, required): Mean episode length
  - `total_episodes` (int, required): Total episodes completed
  - `training_time_seconds` (float, required): Training duration
  - `converged` (boolean, required): Whether agent converged
- `hyperparameters` (object, optional): Algorithm hyperparameters
  - Various algorithm-specific parameters
- `environment` (object, required): Environment details
  - `name` (string, required): Environment name
  - `observation_space` (string, required): Observation space spec
  - `action_space` (string, required): Action space spec
  - `reward_threshold` (float, optional): Success threshold

**Validation Rules**:
- `total_timesteps` must be 50000 for PPO and A2C experiments
- `seed` must be 42 for reproducibility
- `conda_environment` must be "rocm"
- `end_time` must be after `start_time`
- `final_reward_std` must be non-negative

---

### 3. ProjectStructure

**Description**: Organized directory layout with clear separation of source code, tests, results, and specifications.

**File Format**: JSON

**Structure**:
```json
{
  "project_structure": {
    "metadata": {
      "timestamp": "2026-01-15T10:00:00Z",
      "root_path": "/home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment",
      "cleanup_status": "completed"
    },
    "root_directory": {
      "allowed_files": [
        "requirements.txt",
        "README.md",
        ".gitignore"
      ],
      "allowed_directories": [
        "src/",
        "tests/",
        "results/",
        "specs/"
      ],
      "actual_files": [
        "requirements.txt",
        "README.md",
        ".gitignore"
      ],
      "actual_directories": [
        "src/",
        "tests/",
        "results/",
        "specs/"
      ],
      "validation_status": "clean"
    },
    "cleanup_actions": [
      {
        "action": "moved",
        "source": "example_training.py",
        "destination": "src/examples/example_training.py",
        "status": "completed"
      },
      {
        "action": "removed",
        "source": "verify_setup.py",
        "status": "completed"
      },
      {
        "action": "moved",
        "source": "demo_checkpoints/",
        "destination": "results/demo_checkpoints/",
        "status": "completed"
      }
    ]
  }
}
```

**Fields**:
- `metadata` (object, required): Cleanup execution info
  - `timestamp` (datetime, required): When cleanup was run
  - `root_path` (string, required): Project root directory
  - `cleanup_status` (enum: pending/in_progress/completed/failed, required)
- `root_directory` (object, required): Root directory state
  - `allowed_files` (array of strings, required): Permitted file list
  - `allowed_directories` (array of strings, required): Permitted directory list
  - `actual_files` (array of strings, required): Files present after cleanup
  - `actual_directories` (array of strings, required): Directories present after cleanup
  - `validation_status` (enum: clean/dirty, required): Root directory state
- `cleanup_actions` (array of objects, optional): Actions taken
  - `action` (enum: moved/removed/skipped, required): Type of action
  - `source` (string, required): Source path
  - `destination` (string, optional): Destination path (for "moved")
  - `status` (enum: pending/in_progress/completed/failed, required): Action status

**Validation Rules**:
- All `actual_files` must be in `allowed_files`
- All `actual_directories` must be in `allowed_directories`
- `validation_status` is "clean" only if no violations exist

---

### 4. TrainedAgent

**Description**: Pickle-serialized model parameters and configuration for both PPO and A2C algorithms.

**File Format**: Pickle (.zip, SB3 internal format)

**Structure**: Binary format (SB3 proprietary), not human-readable.

**Metadata File** (JSON):
```json
{
  "agent_metadata": {
    "model_file": "ppo_seed42_model.zip",
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "environment": "LunarLander-v3",
    "seed": 42,
    "training_timesteps": 50000,
    "training_date": "2026-01-15T10:28:30Z",
    "conda_environment": "rocm",
    "file_size_bytes": 1048576,
    "checksum": "sha256:abc123..."
  }
}
```

**Validation Rules**:
- Model file must be loadable with `PPO.load()` or `A2C.load()`
- File must exist in `results/experiments/{algo}_seed42/` directory
- Metadata must be consistent with loaded model properties

---

### 5. TrainingMetrics

**Description**: JSON time-series data of rewards, episode lengths, and other performance indicators during training.

**File Format**: JSON

**Structure**:
```json
{
  "training_metrics": {
    "metadata": {
      "experiment_id": "ppo_seed42",
      "algorithm": "PPO",
      "environment": "LunarLander-v3",
      "seed": 42,
      "recording_interval": 100
    },
    "time_series": [
      {
        "timestep": 100,
        "episode": 5,
        "reward": -50.2,
        "episode_length": 45,
        "loss": 0.523,
        "timestamp": "2026-01-15T10:01:30Z"
      },
      {
        "timestep": 200,
        "episode": 10,
        "reward": -35.8,
        "episode_length": 52,
        "loss": 0.498,
        "timestamp": "2026-01-15T10:02:45Z"
      }
    ],
    "aggregated": {
      "reward_mean": 45.2,
      "reward_std": 15.3,
      "reward_min": -100.0,
      "reward_max": 285.0,
      "episode_length_mean": 125.0,
      "total_timesteps": 50000
    }
  }
}
```

**Fields**:
- `metadata` (object, required): Metrics collection info
  - `experiment_id` (string, required): Experiment identifier
  - `algorithm` (string, required): Algorithm used
  - `environment` (string, required): Environment name
  - `seed` (int, required): Random seed
  - `recording_interval` (int, required): Timesteps between recordings
- `time_series` (array of objects, required): Individual data points
  - `timestep` (int, required): Current timestep
  - `episode` (int, required): Current episode number
  - `reward` (float, required): Reward at this point
  - `episode_length` (int, required): Length of current episode
  - `loss` (float, optional): Training loss
  - `timestamp` (datetime, required): When recorded
- `aggregated` (object, required): Summary statistics
  - `reward_mean` (float, required): Mean reward
  - `reward_std` (float, required): Standard deviation
  - `reward_min` (float, required): Minimum reward
  - `reward_max` (float, required): Maximum reward
  - `episode_length_mean` (float, required): Mean episode length
  - `total_timesteps` (int, required): Total timesteps

**Validation Rules**:
- `time_series` must be sorted by `timestep` ascending
- All `timestep` values must be multiples of `recording_interval`
- `reward_std` must be non-negative
- Aggregated stats must match time-series data

---

## Entity Relationships

```
AuditReport (1)
    └── references (0..*) ProjectStructure files

ProjectStructure (1)
    ├── contains (1..*) source modules
    └── organized into (1) ExperimentResults

ExperimentResults (1)
    ├── contains (1) TrainedAgent
    ├── includes (1) TrainingMetrics
    └── references (1) ProjectStructure
```

## Data Flow

1. **Audit Phase**: Generate `AuditReport` → Validate project structure → Output `ProjectStructure`
2. **Cleanup Phase**: Transform `ProjectStructure` → Reorganize files → Update `ProjectStructure` state
3. **Training Phase**: Execute training with seed=42 → Generate `TrainingMetrics` → Save `TrainedAgent` → Aggregate `ExperimentResults`
4. **Validation Phase**: Compare `ExperimentResults` across runs → Validate reproducibility

## File Locations

| Entity | Directory | Filename Pattern |
|---------|-----------|------------------|
| AuditReport | project root | `АУДИТ.md` (Markdown), `audit_report.json` (JSON) |
| ProjectStructure | results/ | `project_structure.json` |
| ExperimentResults | results/experiments/{algo}_seed42/ | `{experiment_id}_results.json` |
| TrainedAgent | results/experiments/{algo}_seed42/ | `{experiment_id}_model.zip` |
| TrainingMetrics | results/experiments/{algo}_seed42/logs/ | `{experiment_id}_metrics.json` |

---

## Glossary

- **Audit**: Systematic examination of code to assess functionality and identify issues
- **Checkpoint**: Saved model state at specific timestep during training
- **Converged**: Agent has achieved stable performance meeting problem requirements
- **Reproducibility**: Ability to obtain identical results when repeating experiments with same seed
- **Timestep**: Single step of interaction between agent and environment

---

**Next**: See `/contracts/` for interface definitions and `quickstart.md` for usage instructions.
