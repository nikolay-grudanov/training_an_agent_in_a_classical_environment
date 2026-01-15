<!-- SYNC IMPACT REPORT:
Version change: N/A (initial version) → 1.0.0
Modified principles: N/A
Added sections: All sections (initial implementation)
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
Follow-up TODOs: None
-->

# Training Agent in Classical Environment Constitution

## Core Principles

### I. Reproducible Research (NON-NEGOTIABLE)
All experiments must be fully reproducible with fixed seeds, documented dependencies (`pip freeze`), and complete training code. Every experiment requires quantitative evaluation across 10-20 episodes with clear metrics.

### II. Experiment-Driven Development
Every algorithm or hyperparameter change must be validated through controlled experiments with clear hypotheses (e.g., "I expect increasing gamma will lead to longer planning horizons"). Experiments must compare at least two conditions.

### III. Test-First for RL Components (NON-NEGOTIABLE)
RL environments, agents, and training pipelines must have unit tests covering core functionality. Tests written before implementation, following red-green-refactor cycle to ensure correctness of reward functions, action spaces, and training loops.

### IV. Performance Monitoring
Continuous tracking of training metrics (average reward vs timesteps/episodes) with visualization capabilities. Clear stopping criteria and convergence detection mechanisms must be implemented.

### V. Scientific Documentation
All experiments require hypothesis statements, methodology descriptions, quantitative results, and reproducible code samples. Documentation must include video/animations of trained agents and quantitative performance assessments.

## Technology Stack Requirements

Python 3.10.14 ecosystem with conda environment management. Dependencies managed via environment.yml with pip for additional packages. Jupyter notebooks for exploratory analysis and experiment prototyping. Stable-Baselines3 as primary RL framework.

## Development Workflow

Use conda environment activation (`conda activate rocm`) for all development. Follow Git branching strategy with feature branches for experiments. Code reviews must verify reproducibility requirements and experimental design validity. All notebooks must be converted to scripts for production deployment.

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, and migration plan. All PRs/reviews must verify compliance with reproducibility standards. Complexity must be justified with clear performance or research benefits.

**Version**: 1.0.0 | **Ratified**: 2026-01-14 | **Last Amended**: 2026-01-14