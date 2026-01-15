# Implementation Plan: Project Cleanup and PPO vs A2C Experiments

**Branch**: `001-cleanup-ppo-a2c-experiments` | **Date**: 2026-01-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-cleanup-ppo-a2c-experiments/spec.md`

**Note**: This template is filled in by `/speckit.plan` command. See `.specify/templates/commands/plan.md` for execution workflow.

## Summary

Clean up the existing RL agent training codebase and establish baseline performance metrics for PPO and A2C algorithms on LunarLander-v3 environment. The feature consists of four main components: (1) comprehensive code audit to identify working/broken modules, (2) project structure cleanup following best practices, (3) PPO agent training with reproducible settings, and (4) A2C agent training for direct performance comparison. All experiments use seed=42 for 50,000 timesteps with JSON metrics and Pickle model storage. Technical approach prioritizes conda environment "rocm" compatibility, verbose DEBUG logging, and checkpoint-based resumption capabilities.

## Technical Context

**Language/Version**: Python (conda environment "rocm")
**Primary Dependencies**: Stable-Baselines3 (PPO/A2C), Gymnasium (LunarLander-v3), PyTorch, NumPy, Matplotlib
**Storage**: JSON for metrics/audit reports, Pickle for trained models
**Testing**: pytest for unit/integration tests
**Target Platform**: Linux server with ROCm support (conda environment "rocm")
**Project Type**: Research/Experimentation - classical control environment RL
**Performance Goals**: Audit completes within 10 minutes, training completes within 30 minutes per algorithm
**Constraints**: Must use existing conda environment "rocm", no new environments, CPU training with ROCm backend availability
**Scale/Scope**: Single-agent reinforcement learning (PPO vs A2C comparison) in LunarLander-v3 environment

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification
- [x] **Reproducible Research**: Fixed seeds (42), documented dependencies (pip freeze), complete training code planned
- [x] **Experiment-Driven**: Clear comparison hypothesis (PPO vs A2C), controlled experiments with identical settings
- [x] **Test-First**: Unit tests required for audit process, training pipelines, and agent functionality
- [x] **Performance Monitoring**: JSON metrics tracking (reward, training_time, episodes), checkpoint resumption planned
- [x] **Scientific Documentation**: Audit report format, experiment results structure, and reproducibility requirements defined

**Gate Status**: ✅ PASSED - All constitution principles addressed

## Project Structure

### Documentation (this feature)

```text
specs/001-cleanup-ppo-a2c-experiments/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
├── checklists/
│   └── requirements.md # Feature acceptance criteria
└── spec.md             # Clarified feature specification
```

### Source Code (repository root)

```text
# RL Agent Training Project Structure (existing, to be cleaned up)
src/
├── agents/               # Agent implementations (PPO, A2C, TD3, SAC)
│   ├── base.py          # Base agent class
│   ├── ppo_agent.py     # PPO implementation
│   ├── a2c_agent.py     # A2C implementation
│   └── td3_agent.py     # TD3 implementation
├── environments/          # Environment configurations and wrappers
├── training/             # Training pipelines
│   └── trainer.py       # Main training orchestration
├── utils/                # Helper functions
│   ├── seeding.py       # Reproducibility utilities
│   └── result_exporter.py # Results export utilities
├── api/                  # REST API (out of scope for this feature)
│   └── app.py
└── visualization/        # Plotting and video generation (if exists)

tests/
├── unit/                 # Unit tests for components
│   ├── test_ppo_agent.py
│   ├── test_a2c_agent.py
│   └── test_td3_agent.py
└── integration/          # Integration tests (if exists)

results/
├── experiments/          # Experiment outputs
│   ├── ppo_seed42/     # PPO experiment results
│   └── a2c_seed42/     # A2C experiment results
├── metrics/              # Aggregated metrics
└── logs/                # Training logs

demo_checkpoints/        # Demo experiment checkpoints (to be cleaned)
```

**Structure Decision**: Existing RL training project structure will be preserved and cleaned. Key components (src/, tests/, results/) will remain. Root directory will be cleaned to only contain: requirements.txt, README.md, .gitignore, src/, tests/, results/, specs/. Demo checkpoints and example files will be moved or removed.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. Constitution compliance met.

## Phase 0: Outline & Research

### Research Tasks

Based on Technical Context analysis, the following research tasks identified:

1. **Audit Methodology Research**: Determine best practices for Python module import testing and functionality verification in RL codebases
2. **Project Cleanup Patterns**: Research standard ML/RL project organization best practices and file categorization
3. **SB3 Checkpoint Resumption**: Verify Stable-Baselines3 checkpoint/resume functionality for training interruption handling
4. **Audit Report Format**: Research industry-standard formats for code audit reports (Markdown structure, categorization schemes)
5. **Reproducibility Validation**: Methods to validate training reproducibility across multiple runs with same seed

### Research Findings

**Task 1: Audit Methodology**
- **Decision**: Use `importlib.util.spec_from_file_location` for import testing, execute basic smoke tests for main functions
- **Rationale**: Python's importlib provides robust module loading without side effects; smoke tests catch critical failures
- **Alternatives Considered**: `subprocess` execution (too heavyweight), static analysis tools (misses runtime issues)

**Task 2: Project Cleanup**
- **Decision**: Follow PyTorch project guidelines: src/ for all code, tests/ at root, results/ for outputs
- **Rationale**: Well-established pattern in ML community; matches existing directory structure
- **Alternatives Considered**: Monolithic package structure (overkill for research), flat structure (poor organization)

**Task 3: SB3 Checkpoint Resumption**
- **Decision**: Use SB3's built-in `.save()` and `.load()` methods with `load_path` parameter
- **Rationale**: Native SB3 functionality handles model, optimizer, and replay buffer state correctly
- **Alternatives Considered**: Manual serialization (error-prone), external checkpoint managers (unnecessary complexity)

**Task 4: Audit Report Format**
- **Decision**: Markdown with tabular format for module status (✅/❌/⚠️ icons, notes column)
- **Rationale**: Human-readable, Git-friendly, supports both manual review and automated parsing
- **Alternatives Considered**: JSON only (less readable), HTML (adds complexity), plain text (no structure)

**Task 5: Reproducibility Validation**
- **Decision**: Compare final metrics (mean reward, std dev) across 2-3 runs with identical seeds
- **Rationale**: Statistical comparison catches subtle non-determinism while being practical
- **Alternatives Considered**: Bit-level output comparison (too strict), single run only (insufficient validation)

## Phase 1: Design & Contracts

### Data Model

See `data-model.md` for detailed entity definitions, relationships, and validation rules.

### API Contracts

This feature primarily uses CLI/script interfaces rather than REST APIs. Contract definitions in `/contracts/` specify:

- **Audit Script Interface**: `python -m src.audit.run` - outputs `АУДИТ.md`
- **Cleanup Script Interface**: `python -m src.cleanup.run` - modifies project structure
- **Training Script Interface**: `python -m src.training.train --algo ppo --seed 42 --steps 50000`
- **Result Export Interface**: `python -m src.utils.export_results` - outputs JSON metrics

### Quick Start Guide

See `quickstart.md` for setup, execution, and validation instructions.

## Execution Phases

### Phase 0: Research ✅ COMPLETED
- ✅ Generated research findings resolving all technical unknowns
- ✅ No NEEDS CLARIFICATION items remain
- ✅ Output: `research.md`

### Phase 1: Design & Contracts ✅ COMPLETED
- ✅ Generated `data-model.md` (5 entities defined: AuditReport, ExperimentResults, ProjectStructure, TrainedAgent, TrainingMetrics)
- ✅ Created `/contracts/` directory with 3 interface contracts:
  - ✅ `contracts/audit_system.md` (audit CLI, output formats, validation rules)
  - ✅ `contracts/cleanup_system.md` (cleanup CLI, dry-run mode, safety features)
  - ✅ `contracts/training_pipeline.md` (training CLI, checkpointing, reproducibility)
- ✅ Generated `quickstart.md` (setup, execution, validation, troubleshooting)
- ✅ Updated agent context (AGENTS.md) with Python/rocm, SB3, Gymnasium technologies

### Phase 2: Task Breakdown (NEXT - requires `/speckit.tasks` command)
- This phase will be handled by `/speckit.tasks` command
- Will generate `tasks.md` with executable implementation tasks

## Constitution Check (Post-Design Verification)

*GATE: Re-checked after Phase 1 design completion.*

### Compliance Verification
- [x] **Reproducible Research**: Fixed seeds (42), documented dependencies (pip freeze), complete training code, checkpoint resumption designed
- [x] **Experiment-Driven**: Clear comparison hypothesis (PPO vs A2C), controlled experiments with identical settings, metrics comparison defined
- [x] **Test-First**: Unit tests planned for audit process, training pipelines, and agent functionality (see contracts for test requirements)
- [x] **Performance Monitoring**: JSON metrics tracking (reward, training_time, episodes), checkpoint resumption, time-series data collection designed
- [x] **Scientific Documentation**: Audit report format, experiment results structure, quickstart guide, and reproducibility validation defined

**Gate Status**: ✅ PASSED - All constitution principles addressed in design phase

---

**Plan Status**: Phase 1 completed successfully, ready for `/speckit.tasks` command to generate implementation tasks

## Artifacts Generated

| Artifact | Location | Description |
|----------|----------|-------------|
| plan.md | `specs/001-cleanup-ppo-a2c-experiments/plan.md` | This implementation plan |
| research.md | `specs/001-cleanup-ppo-a2c-experiments/research.md` | Research findings (Phase 0) |
| data-model.md | `specs/001-cleanup-ppo-a2c-experiments/data-model.md` | Entity definitions (Phase 1) |
| contracts/ | `specs/001-cleanup-ppo-a2c-experiments/contracts/` | Interface contracts (Phase 1) |
| quickstart.md | `specs/001-cleanup-ppo-a2c-experiments/quickstart.md` | User guide (Phase 1) |
