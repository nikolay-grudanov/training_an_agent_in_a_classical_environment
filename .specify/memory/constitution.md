<!-- SYNC IMPACT REPORT:
Version change: 1.0.0 → 1.1.0
Modified principles: N/A
Added sections:
  - VI. Continuous Task Completion (новый обязательный принцип)
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

### VI. Continuous Task Completion (MANDATORY) ⚠️

**Principle**: The AI agent MUST NOT stop until ALL assigned tasks are completed at the highest possible quality level. This is a non-negotiable requirement.

**Success Criteria**:
- All tests pass at 100%
- All models demonstrate acceptable performance metrics
- The complete training pipeline works flawlessly
- All documentation is up-to-date and accurate

**Agent Behavior**:
1. After each task completion, the agent MUST check tasks.md for remaining tasks
2. After each context optimization, the agent MUST review all task files (tasks.md, spec.md, etc.)
3. The agent CANNOT declare work "complete" unless:
   - All items in tasks.md are checked off [x]
   - All acceptance criteria from spec.md are met
   - All tests pass (100% pass rate required)
   - All model performance metrics meet or exceed targets

**Mandatory Tool Usage**:
1. **todo tool**: MUST be used to update task status after EVERY task completion
   - Use `todowrite` to mark tasks as "completed"
   - Use `todoread` before starting new work to check current status
2. **think tool**: MUST be used for EVERY action (planning, analysis, criticism)
   - Use `think-mcp_think` for planning and analysis
   - Use `think-mcp_criticize` for self-criticism after major actions
   - This ensures context-aware decision making

**Task File Verification**:
- After ANY task completion, context optimization, or code generation:
  1. Read the relevant tasks.md file
  2. Review all remaining unchecked tasks
  3. Verify acceptance criteria for completed user stories
  4. Only proceed if no critical blockers exist

**Blocking Conditions** (Work CANNOT stop until these are met):
- [ ] All unit tests pass (100% pass rate)
- [ ] All integration tests pass
- [ ] Model achieves target reward (>200 for LunarLander-v3)
- [ ] Training pipeline completes without errors
- [ ] All code quality checks pass (ruff, mypy)
- [ ] Documentation is complete and accurate

## Technology Stack Requirements

Python 3.10.14 ecosystem with conda environment management. Dependencies managed via environment.yml with pip for additional packages. Jupyter notebooks for exploratory analysis and experiment prototyping. Stable-Baselines3 as primary RL framework.

## Development Workflow

Use conda environment activation (`conda activate rocm`) for all development. Follow Git branching strategy with feature branches for experiments. Code reviews must verify reproducibility requirements and experimental design validity. All notebooks must be converted to scripts for production deployment.

**Mandatory Tools**:
- **todo**: Task tracking - MUST use after every task completion
- **think**: Context-aware decision making - MUST use for every action

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, and migration plan. All PRs/reviews must verify compliance with reproducibility standards. Complexity must be justified with clear performance or research benefits.

**Version**: 1.1.0 | **Ratified**: 2026-02-04 | **Last Amended**: 2026-02-04

**Mandatory Amendments**:
- Added Principle VI: Continuous Task Completion (blocking condition for agent)
- Added mandatory tool usage requirements (todo, think)
- Added task file verification after each action
