# Task Breakdown: Project Cleanup and PPO vs A2C Experiments

**Feature**: Project Cleanup and PPO vs A2C Experiments
**Branch**: `001-cleanup-ppo-a2c-experiments`
**Generated**: 2026-01-15
**Source Spec**: [spec.md](./spec.md) | [plan.md](./plan.md) | [data-model.md](./data-model.md) | [contracts/](./contracts/)

## Overview

This document provides executable implementation tasks for the feature "Project Cleanup and PPO vs A2C Experiments". Tasks are organized by phase and user story to enable independent implementation and testing.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Tasks | 18 |
| Parallelizable Tasks | 8 |
| User Stories | 4 (P1 → P2 → P3 → P4) |

## Dependency Graph

```
Phase 1: Setup
    │
    ▼
Phase 2: Foundational
    │
    ▼
Phase 3: US1 Code Audit ──────────────► Independent Test: Audit report generated
    │                                      (АУДИТ.md + audit_report.json)
    ▼
Phase 4: US2 Project Cleanup ─────────► Independent Test: Root directory clean
    │                                      (results/project_structure.json)
    ▼
Phase 5: US3 PPO Training ────────────► Independent Test: Model saved
    │                                      (results/experiments/ppo_seed42/)
    ▼
Phase 6: US4 A2C Training ────────────► Independent Test: Model saved
                                       (results/experiments/a2c_seed42/)
    │
    ▼
Phase 7: Polish & Cross-Cutting ──────► Final validation
```

**Story Dependencies**: US1 → US2 → US3 → US4 (sequential, each builds on previous)

## Implementation Strategy

**MVP Scope**: Phase 3 (User Story 1 - Code Audit) delivers immediate value by identifying project health

**Incremental Delivery**:
1. MVP: Code audit system (US1) - 5 tasks
2. Phase 2: Utility foundations (seeding, logging) - 2 tasks  
3. Phase 3: Cleanup system (US2) - 4 tasks
4. Phase 4: Training utilities (metrics, checkpoints) - 3 tasks
5. Phase 5: PPO training (US3) - 2 tasks
6. Phase 6: A2C training (US4) - 2 tasks
7. Final: Polish, documentation, reproducibility validation

---

## Phase 1: Setup

**Goal**: Project initialization and directory structure preparation

- [X] T001 Create directory structure per implementation plan:
  - Create `src/audit/` directory for audit module
  - Create `src/cleanup/` directory for cleanup module
  - Create `src/training/` directory for training modules
  - Create `results/logs/` directory for DEBUG-level logging
  - Create `results/experiments/ppo_seed42/` directory for PPO results
  - Create `results/experiments/a2c_seed42/` directory for A2C results
  - Verify `tests/unit/` and `tests/integration/` directories exist

- [X] T002 Initialize __init__.py files for new Python packages:
  - Create `src/audit/__init__.py`
  - Create `src/cleanup/__init__.py`
  - Create `src/training/__init__.py` (already exists)
  - Verify `src/agents/__init__.py` and `src/utils/__init__.py` exist

- [X] T003 Configure logging per NFR-001 and NFR-002 requirements:
  - Create `src/utils/logging_config.py` with DEBUG-level logging
  - Configure console (stdout) and file output to `results/logs/`
  - Add timestamp formatting to all log messages
  - Implement log rotation if logs exceed 10MB

- [X] T004 Verify conda environment "rocm" compatibility:
  - Confirm stable-baselines3 imports successfully (v2.7.1)
  - Confirm gymnasium and LunarLander-v3 available
  - Confirm PyTorch imports with ROCm backend detection (2.5.1+rocm6.2)
  - Document any environment-specific adjustments needed

- [X] T005 [P] Create requirements.txt verification script:
  - Script to capture `pip freeze` output
  - Save to `results/dependencies/snapshot_initial.txt`
  - Verify all contract dependencies listed

---

## Phase 2: Foundational

**Goal**: Blocking prerequisites for all user stories (reproducibility utilities)

- [X] T006 Implement seeding utilities for reproducibility per FR-009:
  - Create `src/utils/seeding.py` with `set_seed(seed: int)` function
  - Set Python random, NumPy, PyTorch, Gymnasium seeds
  - Configure PyTorch for deterministic operations
  - Add CUDA/ROCm backend handling for reproducibility
  - Include unit tests for seeding consistency

- [X] T007 Implement metrics exporter per FR-008 and data-model.md:
  - Create `src/utils/metrics_exporter.py`
  - Export training metrics to JSON format
  - Implement `TrainingMetrics` entity serialization
  - Implement `ExperimentResults` entity serialization
  - Add timestamp and metadata fields per data-model.md

---

## Phase 3: User Story 1 - Code Audit and Project Health Assessment

**Priority**: P1  
**Goal**: Conduct comprehensive audit of all modules in `src/` directory  
**Independent Test Criteria**: Run `python -m src.audit.run` and verify `АУДИТ.md` and `audit_report.json` generated with module status table

**Parallel Execution Example**: T008 and T009 can run in parallel (different components)

- [ ] T008 [P] [US1] Create audit module in src/audit/:
  - Create `src/audit/core.py` with `AuditConfig` dataclass
  - Implement `test_module_import(module_path: Path)` using importlib.util per research.md
  - Implement smoke test execution for basic functionality verification
  - Handle import errors and capture detailed error messages

- [ ] T009 [P] [US1] Implement module assessment logic:
  - Create `src/audit/assessor.py` with `ModuleAssessment` dataclass
  - Implement status determination (working/broken/needs_fixing) per FR-012
  - Implement status icon assignment (✅/❌/⚠️)
  - Add notes generation for issues and fixes needed

- [ ] T010 [P] [US1] Generate audit report per FR-002 and contract:
  - Create `src/audit/report_generator.py`
  - Generate Markdown report (`АУДИТ.md`) with table format per research.md
  - Generate JSON report (`audit_report.json`) per data-model.md
  - Include summary statistics (total, working, broken, needs_fixing)
  - Add metadata (date, auditor, scope, version)

- [ ] T011 [US1] Create audit CLI entry point:
  - Create `src/audit/run.py` with CLI interface per contracts/audit_system.md
  - Implement `--scope` option (default: src/)
  - Implement `--output` option (default: АУДИТ.md)
  - Implement `--format` option (markdown/json/both)
  - Implement `--verbose` flag for DEBUG-level logging
  - Implement `--skip-smoke-tests` flag

- [ ] T012 [US1] Create unit tests for audit module:
  - Create `tests/unit/test_audit_core.py` for import testing logic
  - Create `tests/unit/test_audit_assessor.py` for status determination
  - Create `tests/unit/test_audit_report_generator.py` for report generation
  - Create `tests/unit/test_audit_run.py` for CLI interface
  - Test on existing modules in src/ to verify audit accuracy

---

## Phase 4: User Story 2 - Project Structure Cleanup

**Priority**: P2  
**Goal**: Clean root directory following best practices, reorganize files  
**Independent Test Criteria**: Run `python -m src.cleanup.run` and verify `results/project_structure.json` shows validation_status: "clean" and root contains only 7 approved items

**Parallel Execution Example**: T013 and T014 can run in parallel (different components)

- [ ] T013 [P] [US2] Create cleanup module in src/cleanup/:
  - Create `src/cleanup/core.py` with `CleanupConfig` dataclass
  - Define allowed files list: requirements.txt, README.md, .gitignore
  - Define allowed directories list: src/, tests/, results/, specs/
  - Implement root directory scanner to identify unexpected items

- [ ] T014 [P] [US2] Implement file categorization logic:
  - Create `src/cleanup/categorizer.py`
  - Categorize files as: move to src/, move to tests/, remove, archive
  - Apply rules: *.py scripts → src/, test_*.py → tests/, demo_*.py → remove
  - Handle edge cases: files in use, permission denied, locked files

- [ ] T015 [US2] Implement cleanup execution:
  - Create `src/cleanup/executor.py`
  - Execute file moves, removals, and archives per categorization
  - Generate `ProjectStructure` entity JSON per data-model.md
  - Log warnings for skipped/protected files
  - Handle KeyboardInterrupt gracefully (partial state saved)

- [ ] T016 [US2] Create cleanup CLI entry point:
  - Create `src/cleanup/run.py` with CLI interface per contracts/cleanup_system.md
  - Implement `--dry-run` flag for preview mode
  - Implement `--force` flag for protected file removal
  - Implement `--output` option (default: results/project_structure.json)
  - Implement `--verbose` flag for DEBUG-level logging

- [ ] T017 [US2] Create unit tests for cleanup module:
  - Create `tests/unit/test_cleanup_core.py` for config validation
  - Create `tests/unit/test_cleanup_categorizer.py` for file categorization
  - Create `tests/unit/test_cleanup_executor.py` for execution logic
  - Create `tests/unit/test_cleanup_run.py` for CLI interface
  - Test with temporary directory simulating root cleanup scenario

---

## Phase 5: User Story 3 - PPO Agent Training Experiment

**Priority**: P3  
**Goal**: Train PPO agent on LunarLander-v3 with seed=42 for 50,000 steps  
**Independent Test Criteria**: Run `python -m src.training.train --algo ppo` and verify `results/experiments/ppo_seed42/ppo_seed42_model.zip` exists with final_reward_mean > 200

**Parallel Execution Example**: T018 and T019 can run in parallel (different components, but both needed before T020)

- [X] T018 [P] [US3] Implement checkpointing system per research.md:
  - Create `src/training/checkpoint.py`
  - Implement `CheckpointManager` class with save/load methods
  - Use SB3 native `.save()` and `.load()` for model serialization
  - Save checkpoints every 1,000 steps to `results/experiments/ppo_seed42/checkpoint_*.zip`
  - Implement resume capability from last checkpoint

- [X] T019 [P] [US3] Implement metrics collector:
  - Create `src/training/metrics_collector.py`
  - Collect time-series data: timestep, episode, reward, episode_length, loss
  - Implement recording at configurable interval (default: 100 steps)
  - Generate `TrainingMetrics` entity JSON per data-model.md
  - Calculate aggregated statistics (mean, std, min, max)

- [X] T020 [US3] Implement PPO training script:
  - Create `src/training/trainer.py` with PPO-specific training logic
  - Implement `--algo` option (ppo/a2c) per contracts/training_pipeline.md
  - Set seed=42 using `src/utils/seeding.py`
  - Configure PPO with default hyperparameters (lr=0.0003, n_steps=2048, gamma=0.99)
  - Train for exactly 50,000 timesteps
  - Save final model to `{algo}_seed{seed}_model.zip`
  - Generate `ExperimentResults` entity JSON per data-model.md
  - Implement verbose DEBUG-level logging per NFR-001

- [X] T021 [US3] Create unit tests for training module:
  - Create `tests/unit/test_checkpoint.py` for checkpoint save/load
  - Create `tests/unit/test_metrics_collector.py` for metrics collection
  - Create `tests/unit/test_trainer_ppo.py` for PPO training logic
  - Mock Gymnasium environment for fast unit tests
  - Test reproducibility with identical seed

---

## Phase 6: User Story 4 - A2C Agent Training Experiment

**Priority**: P4  
**Goal**: Train A2C agent on LunarLander-v3 with identical settings to PPO  
**Independent Test Criteria**: Run `python -m src.training.train --algo a2c` and verify `results/experiments/a2c_seed42/a2c_seed42_model.zip` exists with same structure as PPO results

- [X] T022 [US4] Add A2C training support to trainer.py:
  - Extend `src/training/trainer.py` to support A2C algorithm
  - Configure A2C with default hyperparameters (lr=0.0007, n_steps=5, gamma=0.99)
  - Use identical seed=42, timesteps=50,000, checkpoint_interval=1000
  - Save to `results/experiments/a2c_seed42/` directory structure
  - Ensure output format matches PPO for comparison

- [X] T023 [US4] Create unit tests for A2C training:
  - Create `tests/unit/test_trainer_a2c.py` for A2C training logic
  - Mock Gymnasium environment for fast unit tests
  - Verify output structure matches PPO format
  - Test reproducibility with identical seed

---

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Final validation, documentation, and reproducibility verification

- [ ] T024 Create integration test suite:
  - Create `tests/integration/test_full_workflow.py`
  - Test complete workflow: audit → cleanup → PPO → A2C
  - Verify all output files generated correctly
  - Test on clean repository state

- [ ] T025 Validate reproducibility per SC-004:
  - Run PPO training twice with seed=42
  - Compare final_reward_mean and reward_std across runs
  - Verify std deviation < 0.01 (PERFECT REPRODUCIBILITY)
  - Document reproducibility validation results

- [ ] T026 Update quickstart.md with execution verification:
  - Add validation checklist for each phase
  - Add troubleshooting section for common errors
  - Add expected output examples for each command

- [ ] T027 Final cleanup and documentation:
  - Verify all paths in contracts match implemented paths
  - Update any outdated paths in documentation
  - Ensure AGENTS.md reflects all new technologies
  - Run linter (ruff) on all new code

- [ ] T028 [P] Performance validation per success criteria:
  - Verify SC-001: Audit completes within 10 minutes
  - Verify SC-003: PPO and A2C training complete within 30 minutes each
  - Verify SC-006: Final reward > 200 for both algorithms
  - Document performance benchmarks

---

## Parallel Execution Examples

### Within User Story 1 (Code Audit)
```bash
# T008 and T009 can be implemented in parallel
# T010 and T011 can be implemented in parallel (both use output from T008-T009)
# T012 must run after T011
```

### Within User Story 2 (Project Cleanup)
```bash
# T013 and T014 can be implemented in parallel
# T015 must run after T014
# T016 must run after T015
# T017 must run after T016
```

### Within User Story 3 (PPO Training)
```bash
# T018 and T019 can be implemented in parallel
# T020 must run after both T018 and T019
# T021 must run after T020
```

### Within Phase 7 (Polish)
```bash
# T024, T025, T026 can run in parallel
# T027 and T028 depend on T024-T026 completion
```

---

## File Paths Reference

| Task | File Path |
|------|-----------|
| T001 | `src/audit/`, `src/cleanup/`, `results/logs/`, `results/experiments/ppo_seed42/`, `results/experiments/a2c_seed42/` |
| T002 | `src/audit/__init__.py`, `src/cleanup/__init__.py` |
| T003 | `src/utils/logging_config.py` |
| T004 | `results/dependencies/snapshot_initial.txt` |
| T005 | `requirements.txt` (verification) |
| T006 | `src/utils/seeding.py` |
| T007 | `src/utils/metrics_exporter.py` |
| T008 | `src/audit/core.py` |
| T009 | `src/audit/assessor.py` |
| T010 | `src/audit/report_generator.py` |
| T011 | `src/audit/run.py` |
| T012 | `tests/unit/test_audit_*.py` |
| T013 | `src/cleanup/core.py` |
| T014 | `src/cleanup/categorizer.py` |
| T015 | `src/cleanup/executor.py` |
| T016 | `src/cleanup/run.py` |
| T017 | `tests/unit/test_cleanup_*.py` |
| T018 | `src/training/checkpoint.py` |
| T019 | `src/training/metrics_collector.py` |
| T020 | `src/training/trainer.py` |
| T021 | `tests/unit/test_trainer_ppo.py` |
| T022 | `src/training/trainer.py` (A2C extension) |
| T023 | `tests/unit/test_trainer_a2c.py` |
| T024 | `tests/integration/test_full_workflow.py` |
| T025 | `results/reproducibility_validation.json` |
| T026 | `quickstart.md` (update) |
| T027 | Various documentation files |
| T028 | `results/performance_benchmarks.json` |

---

## Success Criteria Verification Checklist

| ID | Criteria | Verification Command |
|----|----------|---------------------|
| SC-001 | Audit completes < 10 min | `time python -m src.audit.run` |
| SC-002 | Root clean (7 items) | Check `results/project_structure.json` |
| SC-003 | Training < 30 min each | `time python -m src.training.train --algo ppo` |
| SC-004 | Reproducible (std < 0.01) | Run T025 validation |
| SC-005 | All artifacts present | Check results/experiments/{algo}_seed42/ |
| SC-006 | Reward > 200 | Check *_results.json metrics |
| SC-007 | Code preserved after cleanup | Compare pre/post cleanup |
| SC-008 | Learning progression | Check *_metrics.json time_series |

---

**Next Steps**: Execute tasks in dependency order. Start with Phase 1 (Setup), then proceed sequentially through phases. Use parallel execution where tasks are marked [P].
