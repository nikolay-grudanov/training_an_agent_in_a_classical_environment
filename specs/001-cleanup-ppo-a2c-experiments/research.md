# Research Report: Project Cleanup and PPO vs A2C Experiments

**Feature**: Project Cleanup and PPO vs A2C Experiments
**Date**: 2026-01-15
**Phase**: Phase 0 - Outline & Research

## Executive Summary

This research document resolves all technical unknowns identified in the implementation plan for the feature "Project Cleanup and PPO vs A2C Experiments". Research covered audit methodology, project organization patterns, SB3 checkpoint handling, audit report formats, and reproducibility validation. All findings support the existing conda environment "rocm" constraint and align with the project's constitution principles for reproducible research.

## Research Tasks Completed

### 1. Audit Methodology Research

**Objective**: Determine best practices for Python module import testing and functionality verification in RL codebases.

**Decision**: Use `importlib.util.spec_from_file_location` for import testing, execute basic smoke tests for main functions.

**Rationale**:
- Python's `importlib` provides robust module loading without side effects
- Smoke tests (basic instantiation, method calls) catch critical failures quickly
- Allows granular control over what gets tested vs. imported
- Works well for partially broken codebases

**Implementation Approach**:
```python
import importlib.util
import sys
from pathlib import Path

def test_module_import(module_path: Path) -> tuple[bool, str]:
    """Test if module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            return False, "No spec/loader found"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = module
        spec.loader.exec_module(module)
        return True, "Import successful"
    except Exception as e:
        return False, str(e)
```

**Alternatives Considered**:
1. **subprocess execution**: Too heavyweight, creates separate Python processes, hard to capture detailed errors
2. **Static analysis tools (pylint, mypy)**: Miss runtime issues, requires complete type hints
3. **Direct `import` statements**: Side effects from executed code at import time, hard to isolate failures

**Recommendation**: Proceed with importlib + smoke tests approach.

---

### 2. Project Cleanup Patterns Research

**Objective**: Research standard ML/RL project organization best practices and file categorization.

**Decision**: Follow PyTorch project guidelines: src/ for all code, tests/ at root, results/ for outputs.

**Rationale**:
- Well-established pattern in ML community (PyTorch, HuggingFace, etc.)
- Matches existing directory structure (minimal refactoring needed)
- Clear separation between source, tests, and results
- Easy to package for distribution if needed

**Project Structure Best Practices**:
```text
project_root/
├── src/              # All source code
│   ├── agents/       # Agent implementations
│   ├── training/     # Training pipelines
│   ├── environments/ # Environment configs
│   └── utils/        # Utilities
├── tests/            # All tests (no src/tests/)
├── results/          # Experiment outputs
├── docs/             # Documentation
├── notebooks/        # Jupyter notebooks
├── requirements.txt   # Dependencies
└── README.md         # Project overview
```

**Cleanup Rules** (from research):
1. **Keep in root**: requirements.txt, README.md, .gitignore, src/, tests/, results/, specs/
2. **Move to src/**: All Python scripts, modules, working code
3. **Remove**: Example files, validation scripts, temporary experiment files
4. **Archive**: Old demo checkpoints (move to results/demo_checkpoints/ or delete)

**Alternatives Considered**:
1. **Monolithic package structure** (setup.py with everything in package/): Overkill for research project, adds complexity
2. **Flat structure** (all .py files in root): Poor organization, hard to navigate, not scalable
3. **Docker-centric structure** (dockerfile, docker-compose): Unnecessary for this feature, violates "rocm environment" constraint

**Recommendation**: Preserve existing src/ structure, clean root directory, maintain results/ separation.

---

### 3. SB3 Checkpoint Resumption Research

**Objective**: Verify Stable-Baselines3 checkpoint/resume functionality for training interruption handling.

**Decision**: Use SB3's built-in `.save()` and `.load()` methods with `load_path` parameter.

**Rationale**:
- Native SB3 functionality handles model, optimizer, and replay buffer state correctly
- Well-documented and battle-tested in RL community
- Supports both full saves and periodic checkpoints
- Compatible with both PPO and A2C algorithms

**Implementation Approach**:
```python
from stable_baselines3 import PPO
import time

# Training with checkpointing
model = PPO("MlpPolicy", env, verbose=1)
checkpoint_dir = "results/experiments/ppo_seed42/"
checkpoint_interval = 1000

for i in range(0, 50000, checkpoint_interval):
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{i}.zip"
    model.save(checkpoint_path)
    print(f"Saved checkpoint at {i} steps")

# Resuming from checkpoint
model = PPO.load("results/experiments/ppo_seed42/checkpoint_10000.zip")
model.learn(total_timesteps=40000, reset_num_timesteps=False)  # Continue to 50k
```

**Key Considerations**:
- Use `reset_num_timesteps=False` when resuming to preserve learning progress
- Checkpoint format is `.zip` (SB3 internal format, can be loaded with `.load()`)
- Models are algorithm-agnostic for loading (e.g., PPO.load() on PPO checkpoint)

**Alternatives Considered**:
1. **Manual serialization** (pickle model state directly): Error-prone, misses optimizer state, no SB3 guarantees
2. **External checkpoint managers** (MLflow, Weights & Biases): Adds dependency complexity, not needed for this feature
3. **No checkpointing** (restart on failure): Violates edge case handling requirements, wastes compute time

**Recommendation**: Use native SB3 checkpointing with 1000-step intervals.

---

### 4. Audit Report Format Research

**Objective**: Research industry-standard formats for code audit reports.

**Decision**: Markdown with tabular format for module status (✅/❌/⚠️ icons, notes column).

**Rationale**:
- Human-readable and easy to review manually
- Git-friendly (diff-able, merge-able)
- Supports both manual review and simple automated parsing
- No additional dependencies required

**Report Format Template**:
```markdown
# Code Audit Report

**Date**: 2026-01-15
**Auditor**: Automated Audit System
**Scope**: All modules in src/

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Working ✅ | 12 | 75% |
| Broken ❌ | 2 | 12.5% |
| Needs Fixing ⚠️ | 2 | 12.5% |

## Module Details

| Module | Path | Import Status | Functionality Test | Status | Notes |
|--------|------|---------------|-------------------|--------|-------|
| base_agent | src/agents/base.py | ✅ Success | ✅ Pass | ✅ Working | - |
| ppo_agent | src/agents/ppo_agent.py | ✅ Success | ❌ Fail | ❌ Broken | Missing import in line 45 |
| a2c_agent | src/agents/a2c_agent.py | ✅ Success | ✅ Pass | ✅ Working | - |
| trainer | src/training/trainer.py | ⚠️ Warning | ❌ Fail | ⚠️ Needs Fixing | Deprecation warning in line 78 |

## Recommendations

1. Fix broken imports in ppo_agent.py
2. Update trainer.py to resolve deprecation warnings
3. Add unit tests for all working modules
```

**Alternatives Considered**:
1. **JSON only**: Less human-readable, harder to review manually
2. **HTML with styling**: Adds complexity, requires additional tooling
3. **Plain text**: No structure, hard to parse programmatically
4. **Excel/CSV**: Good for data, poor for narrative and mixed content

**Recommendation**: Use Markdown format with table structure as shown above.

---

### 5. Reproducibility Validation Research

**Objective**: Methods to validate training reproducibility across multiple runs with same seed.

**Decision**: Compare final metrics (mean reward, std dev) across 2-3 runs with identical seeds.

**Rationale**:
- Statistical comparison catches subtle non-determinism
- Practical for CPU training (30min per run is acceptable)
- Validates both model behavior and training stability
- Aligns with "Experiment-Driven Development" principle

**Validation Approach**:
```python
import numpy as np
from pathlib import Path
import json

def validate_reproducibility(results_dirs: list[Path]) -> dict:
    """Compare metrics across multiple runs."""
    all_rewards = []
    all_times = []

    for results_dir in results_dirs:
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)
        all_rewards.append(metrics["final_reward"])
        all_times.append(metrics["training_time"])

    rewards = np.array(all_rewards)
    reward_std = np.std(rewards)
    reward_mean = np.mean(rewards)

    # Check if all runs produce identical results (deterministic)
    if reward_std < 1e-10:
        status = "PERFECT REPRODUCIBILITY"
    elif reward_std < 0.01:
        status = "GOOD REPRODUCIBILITY"
    else:
        status = "POOR REPRODUCIBILITY"

    return {
        "status": status,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "time_std": np.std(all_times),
        "runs": len(results_dirs)
    }
```

**Reproducibility Requirements** (from Constitution):
1. Fixed random seed (42) for all experiments
2. Document all dependencies (pip freeze)
3. Deterministic environment (LunarLander-v3)
4. No GPU nondeterminism (CPU training)

**Alternatives Considered**:
1. **Bit-level output comparison** (exact file hashes): Too strict, fails on minor timing differences
2. **Single run only**: Insufficient validation, misses stochastic behavior
3. **Visual comparison only** (plot overlays): Subjective, not quantitative
4. **Full statistical analysis** (t-tests, etc.): Overkill for baseline experiments

**Recommendation**: Compare mean reward ± standard deviation across 2-3 runs; target standard deviation < 0.01 for perfect reproducibility.

---

## Conclusions

All research tasks completed successfully. Key findings:

1. **Audit**: Use importlib + smoke tests for robust module verification
2. **Cleanup**: Preserve existing src/ structure, clean root directory
3. **Checkpoints**: Native SB3 checkpointing with 1000-step intervals
4. **Reporting**: Markdown with tables for audit reports
5. **Reproducibility**: Compare metrics across 2-3 identical seed runs

**No NEEDS CLARIFICATION items remain**. Ready to proceed to Phase 1 (Design & Contracts).

---

**Next Steps**:
- Generate data-model.md based on entity definitions
- Create contract definitions for audit, cleanup, and training interfaces
- Write quickstart.md with setup and execution instructions
- Update agent context with new technology findings
