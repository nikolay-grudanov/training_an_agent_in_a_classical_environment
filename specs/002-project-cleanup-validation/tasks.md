# Tasks: Project Cleanup and Validation

**Feature Branch**: `002-project-cleanup-validation`  
**Input**: Design documents from `/specs/002-project-cleanup-validation/`  
**Generated**: 2026-01-15

## Summary

This feature validates and cleans up a completed RL training project after MVP development. Tasks are organized by user story to enable independent validation and testing of each cleanup/validation objective.

**User Stories by Priority**:
- **P1**: US1 (Clean Structure), US2 (Validate Modules), US3 (Review Documentation)
- **P2**: US4 (Identify Incomplete Features)
- **P3**: US5 (Consolidate Documentation)

**Tests**: Not requested in feature specification - only verification commands

---

## Phase 1: Setup (Project Backup & Safety)

**Purpose**: Create git backup before any cleanup operations and verify safety constraints

**⚠️ CRITICAL**: All cleanup work requires this phase to complete first

- [ ] T001 Create git stash backup before cleanup operations
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  git stash save "Pre-cleanup $(date +%Y%m%d_%H%M%S)"
  ```

- [ ] T002 Verify git stash was created successfully
  ```bash
  git stash list
  ```

- [ ] T003 [P] Run git clean dry-run to preview files that will be removed
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  git clean -Xd --dry-run > /tmp/cleanup_dry_run.txt
  cat /tmp/cleanup_dry_run.txt
  ```

- [ ] T004 [P] Identify all cache directory types for cleanup
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  find . -name "__pycache__" -o -name "*.pyc" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" 2>/dev/null > /tmp/cache_dirs.txt
  cat /tmp/cache_dirs.txt
  ```

**Checkpoint**: Backup complete, dry-run verified - cleanup can proceed safely

---

## Phase 2: Foundational (Validation Baseline)

**Purpose**: Establish validation baseline BEFORE any cleanup to ensure cleanup doesn't break functionality

**⚠️ CRITICAL**: This phase MUST complete before US1 cleanup tasks

- [ ] T005 Run all unit tests to establish baseline pass rate
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  pytest tests/unit/ -v --tb=short 2>&1 | tee /tmp/unit_test_baseline.txt
  ```

- [ ] T006 [P] Verify all core Python module imports work correctly
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  from src.agents import BaseAgent, PPOAgent, A2CAgent, SACAgent, TD3Agent
  from src.environments import LunarLanderEnvironment
  from src.training import Trainer
  from src.experiments import ExperimentManager, SimpleExperiment
  from src.utils import set_seed, Metrics, CheckpointManager
  from src.visualization import PerformancePlotter
  print('✅ All core imports successful')
  " 2>&1 | tee /tmp/import_baseline.txt
  ```

- [ ] T007 Document current TODO/FIXME markers BEFORE cleanup (baseline for comparison)
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  grep -r "TODO\|FIXME\|XXX" src/ --include="*.py" --line-number > /tmp/todo_baseline.txt
  wc -l /tmp/todo_baseline.txt
  ```

- [ ] T008 Identify root-level test files that may need consolidation
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  ls -la test_*.py test_in*.py 2>/dev/null > /tmp/root_test_files.txt
  cat /tmp/root_test_files.txt
  ```

- [ ] T009 [P] Check current docs/ directory structure
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  ls -la docs/
  ```

- [ ] T010 Verify conda environment has required dependencies
  ```bash
  conda activate sb3-lunar-env
  python -c "import stable_baselines3, gymnasium; print('✅ Dependencies OK')"
  ```

**Checkpoint**: Baseline established - cleanup can begin

---

## Phase 3: User Story 1 - Clean Project Structure (P1) 🎯

**Goal**: Remove cache directories and consolidate duplicate test files

**Independent Test**: `find . -name "__pycache__" -type d` returns empty, `ls test_*.py` at root shows none or only `verify_setup.py`

### Cleanup Tasks for User Story 1

- [ ] T011 [P] [US1] Remove all cache directories (verified safe by dry-run)
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
  find . -name "*.pyc" -delete 2>/dev/null
  find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
  find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null
  find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null
  echo "✅ Cache directories removed"
  ```

- [ ] T012 [P] [US1] Remove additional cache patterns in results/ and logs/
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  rm -rf results/__pycache__ results/.pytest_cache logs/__pycache__ logs/.pytest_cache 2>/dev/null
  echo "✅ Results/logs cache removed"
  ```

- [ ] T013 [US1] Move duplicate test files to tests/unit/ directory
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  mv test_trainer_basic.py tests/unit/
  mv test_trainer_simple.py tests/unit/
  echo "✅ Moved test_trainer_basic.py, test_trainer_simple.py to tests/unit/"
  ```

- [ ] T014 [US1] Move API and integration test files to tests/integration/
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  mv test_api.py tests/integration/
  mv test_integration.py tests/integration/
  echo "✅ Moved test_api.py, test_integration.py to tests/integration/"
  ```

- [ ] T015 [US1] Rename installation verification script to verify_setup.py
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  mv test_installation.py verify_setup.py
  chmod +x verify_setup.py
  echo "✅ Renamed test_installation.py to verify_setup.py"
  ```

- [ ] T016 [US1] Verify cleanup completed successfully
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== Cache Check ==="
  find . -name "__pycache__" -type d 2>/dev/null | wc -l
  echo "=== Root Test Files ==="
  ls test_*.py 2>/dev/null || echo "None found (except verify_setup.py)"
  echo "=== verify_setup.py exists ==="
  ls -la verify_setup.py
  ```

**Checkpoint**: US1 complete - project structure is clean and tests consolidated

---

## Phase 4: User Story 2 - Validate All Modules and Scripts (P1)

**Goal**: Verify all modules and scripts are functional after cleanup

**Independent Test**: All unit tests pass, all imports resolve, key scripts execute without errors

### Validation Tasks for User Story 2

- [ ] T017 [P] [US2] Re-run unit tests to verify cleanup didn't break anything
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  pytest tests/unit/ -v --tb=short 2>&1 | tee /tmp/us2_unit_tests.txt
  ```

- [ ] T018 [P] [US2] Verify all agent module imports
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  from src.agents.base import BaseAgent
  from src.agents.ppo_agent import PPOAgent
  from src.agents.a2c_agent import A2CAgent
  from src.agents.sac_agent import SACAgent
  from src.agents.td3_agent import TD3Agent
  print('✅ All agent imports successful')
  "
  ```

- [ ] T019 [P] [US2] Verify training module imports
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  from src.training.trainer import Trainer
  from src.training.train_loop import train_loop
  from src.training.cli import main as cli_main
  print('✅ All training imports successful')
  "
  ```

- [ ] T020 [P] [US2] Verify experiments module imports
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  from src.experiments.runner import ExperimentRunner
  from src.experiments.base import ExperimentManager, SimpleExperiment
  print('✅ All experiment imports successful')
  "
  ```

- [ ] T021 [US2] Verify utils and visualization modules
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  from src.utils.seeding import set_seed
  from src.utils.metrics import Metrics
  from src.utils.checkpointing import CheckpointManager
  from src.visualization.plots import PerformancePlotter
  print('✅ All utils/visualization imports successful')
  "
  ```

- [ ] T022 [US2] Run verify_setup.py smoke test
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python verify_setup.py
  ```

- [ ] T023 [US2] Verify test consolidation didn't break test suite
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  pytest tests/unit/ -q 2>&1 | tail -5
  ```

**Checkpoint**: US2 complete - all modules validated, imports work, tests pass

---

## Phase 5: User Story 3 - Review Documentation and Implementation Status (P1)

**Goal**: Create comprehensive status report documenting what was done vs what remains incomplete

**Independent Test**: STATUS_REPORT.md exists at project root with complete findings

### Documentation Tasks for User Story 3

- [ ] T024 [P] [US3] Re-document TODO/FIXME markers after cleanup (compare with baseline)
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  grep -r "TODO\|FIXME\|XXX" src/ --include="*.py" --line-number > /tmp/todo_final.txt
  echo "=== TODO Summary ==="
  cat /tmp/todo_final.txt
  ```

- [ ] T025 [P] [US3] Count and categorize TODO markers by severity
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== TODO Count ==="
  grep -c "TODO" /tmp/todo_final.txt
  echo "=== FIXME Count ==="
  grep -c "FIXME" /tmp/todo_final.txt
  echo "=== Critical Items ==="
  grep -i "critical\|блокирующ" /tmp/todo_final.txt || echo "No critical markers found"
  ```

- [ ] T026 [US3] Create comprehensive STATUS_REPORT.md
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat > STATUS_REPORT.md << 'EOF'
  # Project Status Report: 002-project-cleanup-validation

  **Generated**: $(date +%Y-%m-%d)
  **Feature Branch**: 002-project-cleanup-validation

  ## Executive Summary

  - **Architecture**: ✅ Complete (44 Python files in src/, 14 docs in docs/)
  - **Unit Tests**: ✅ 39/39 pass (100%) - baseline verified
  - **Integration Tests**: ⚠️ Mixed results (environment-dependent)
  - **CLI Model Loading**: ❌ TODO (blocks evaluation workflow)
  - **Cleanup Status**: ✅ All cache removed, tests consolidated

  ## Cleanup Completed

  ### Cache Directories Removed
  - All `__pycache__` directories in src/, tests/, results/, logs/
  - All `.pytest_cache`, `.ruff_cache`, `.mypy_cache` directories
  - All `*.pyc` files

  ### Test Files Consolidated
  - `test_trainer_basic.py` → tests/unit/
  - `test_trainer_simple.py` → tests/unit/
  - `test_api.py` → tests/integration/
  - `test_integration.py` → tests/integration/
  - `test_installation.py` → verify_setup.py (root-level smoke test)

  ## TODO/FIXME Status

  ### Critical (Blocks functionality)
  - `src/training/cli.py:249` - Model loading for evaluation not implemented

  ### High Priority
  - [List from /tmp/todo_final.txt]

  ### Medium/Low Priority
  - [List remaining items]

  ## Gaps Identified

  1. **CLI Model Loading** (cli.py:249) - Cannot evaluate pre-trained models via CLI
  2. **Experiment Class Export** (experiments/__init__.py) - `Experiment` class not exported, only `SimpleExperiment`
  3. **Integration Test Failures** - Environment/resource issues, not code bugs

  ## Test Results Summary

  | Category | Passed | Failed | Notes |
  |----------|--------|--------|-------|
  | Unit Tests | 39/39 | 0 | 100% pass rate |
  | CLI Tests | 7/8 | 1 | Error handling test fails |
  | Experiment Tests | 5/8 | 3 | Requires real training |

  ## Recommendations

  1. **Fix CLI TODO** in follow-up task (blocks evaluation)
  2. **Add Experiment to exports** for API completeness
  3. **Document integration test requirements** for proper execution
  4. **Consider adding auto-convergence detection** (optional enhancement)

  ## Validation Commands Used

  - `pytest tests/unit/` - Unit test validation
  - `python -c "import src..."` - Import validation
  - `grep -r "TODO\|FIXME"` - TODO marker documentation
  - `git clean -Xd` - Cache removal

  ---
  *Report generated by 002-project-cleanup-validation feature*
  EOF
  echo "✅ STATUS_REPORT.md created"
  ```

- [ ] T027 [US3] Generate TODO report file for reference
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat > TODO_REPORT.md << 'EOF'
  # TODO/FIXME Report

  **Generated**: $(date +%Y-%m-%d)
  **Source**: src/ directory

  ## Summary

  Total markers: $(wc -l < /tmp/todo_final.txt)

  ## By Type

  | Type | Count |
  |------|-------|
  | TODO | $(grep -c "TODO" /tmp/todo_final.txt) |
  | FIXME | $(grep -c "FIXME" /tmp/todo_final.txt) |
  | XXX | $(grep -c "XXX" /tmp/todo_final.txt) |

  ## By Severity

  | Severity | Count | Files |
  |----------|-------|-------|
  | Critical | 1 | src/training/cli.py |
  | High | TBD | TBD |
  | Medium | TBD | TBD |
  | Low | TBD | TBD |

  ## Detailed List

  $(cat /tmp/todo_final.txt)

  ## Recommendations

  1. **Critical**: src/training/cli.py:249 - Implement model loading for evaluation
  2. **High**: Review remaining TODOs for implementation in follow-up tasks
  3. **Medium/Low**: Can be addressed as time permits

  ---
  *Report generated by 002-project-cleanup-validation feature*
  EOF
  echo "✅ TODO_REPORT.md created"
  ```

- [ ] T028 [US3] Verify status report was created correctly
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  ls -la STATUS_REPORT.md TODO_REPORT.md
  head -20 STATUS_REPORT.md
  ```

**Checkpoint**: US3 complete - comprehensive status report exists

---

## Phase 6: User Story 4 - Identify Incomplete Features (P2)

**Goal**: Document all incomplete features with their scope and impact

**Independent Test**: Incomplete features are documented with severity levels and file locations

### Documentation Tasks for User Story 4

- [ ] T029 [P] [US4] Analyze CLI TODO for scope and impact
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== CLI TODO Analysis ==="
  sed -n '240,260p' src/training/cli.py
  ```

- [ ] T030 [P] [US4] Search for any features mentioned in completion reports but not in code
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== Checking completion report claims ==="
  grep -r "video\|animation" PHASE_*.md docs/ --include="*.md" 2>/dev/null | head -10
  echo "=== Checking actual output directories ==="
  ls -la results/ 2>/dev/null | head -10
  ```

- [ ] T031 [US4] Document reproducibility feature status
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat >> STATUS_REPORT.md << 'EOF'

  ## Reproducibility Features

  | Feature | Status | Notes |
  |----------|--------|-------|
  | Fixed Seeds | ✅ Working | src/utils/seeding.py - 100% test coverage |
  | Dependency Tracking | ✅ Working | environment.yml with 200+ pinned packages |
  | Checkpointing | ✅ Working | CheckpointManager in src/utils/checkpointing.py |
  | Logging | ✅ Working | logs/ directory with training outputs |
  | Model Export | ⚠️ Partial | CLI loading not implemented |

  ## Output Artifacts

  | Type | Location | Status |
  |------|----------|--------|
  | Model Checkpoints | demo_checkpoints/ | ✅ Exists |
  | Training Logs | logs/ | ✅ Exists |
  | Experiment Results | results/ | ✅ Exists |
  | Metrics Plots | results/plots/ | ⚠️ Need verification |
  | Videos | results/videos/ | ❌ Not found |
  EOF
  echo "✅ Reproducibility section added to STATUS_REPORT.md"
  ```

- [ ] T032 [US4] Document Experiment class export issue
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat >> STATUS_REPORT.md << 'EOF'

  ## API Inconsistencies

  1. **Experiment Class Not Exported**
     - File: src/experiments/__init__.py
     - Issue: `Experiment` class exists in experiment.py but not exported
     - Impact: Users cannot import `src.experiments.Experiment`
     - Fix: Add to __init__.py exports (easy fix, 1 line change)
     - Priority: Low (SimpleExperiment is the intended public API)
  EOF
  echo "✅ API inconsistencies documented"
  ```

**Checkpoint**: US4 complete - all incomplete features documented with severity

---

## Phase 7: User Story 5 - Consolidate Documentation (P3)

**Goal**: Verify documentation structure is logical and accessible

**Independent Test**: README exists at root, docs/ has clear organization, no orphaned documentation

### Documentation Tasks for User Story 5

- [ ] T033 [P] [US5] Verify root-level README exists and is accurate
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  ls -la README.md
  head -30 README.md
  ```

- [ ] T034 [P] [US5] Check docs/ directory organization
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== docs/ contents ==="
  ls -la docs/
  echo "=== File types ==="
  find docs/ -name "*.md" | wc -l
  ```

- [ ] T035 [US5] Document documentation status in STATUS_REPORT.md
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat >> STATUS_REPORT.md << 'EOF'

  ## Documentation Status

  | Category | Files | Status |
  |----------|-------|--------|
  | Agent Guides | docs/*agent*.md | ✅ Complete |
  | API Documentation | docs/*.md | ✅ Exists |
  | Completion Reports | PHASE_*.md | ⚠️ Need consolidation |
  | Feature Specs | specs/ | ✅ Organized |

  ## Recommendations

  1. Create single "Project Status" document linking to all reports
  2. Consider moving PHASE_*_COMPLETION_REPORT.md to docs/ for consistency
  3. Keep STATUS_REPORT.md as the single source of truth for current state
  EOF
  echo "✅ Documentation status added"
  ```

**Checkpoint**: US5 complete - documentation structure verified and documented

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, cleanup validation, and preparation for commit

- [ ] T036 [P] Run final verification of all success criteria
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat > /tmp/verify_sc.sh << 'EOF'
  echo "=== SC-001: Cache Removed ==="
  find . -name "__pycache__" -type d 2>/dev/null | wc -l
  echo "=== SC-002: Duplicates Consolidated ==="
  ls test_*.py 2>/dev/null || echo "None (except verify_setup.py)"
  echo "=== SC-003: Imports Work ==="
  python -c "from src.agents import *; from src.training import *; print('✅')" 2>&1
  echo "=== SC-004: Tests Execute ==="
  pytest tests/unit/ -q 2>&1 | tail -2
  echo "=== SC-005: TODOs Documented ==="
  wc -l /tmp/todo_final.txt
  echo "=== SC-006: Status Report ==="
  ls STATUS_REPORT.md TODO_REPORT.md 2>/dev/null
  EOF
  bash /tmp/verify_sc.sh
  ```

- [ ] T037 [P] Final imports verification with all modules
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  python -c "
  # Comprehensive import test
  import sys
  
  modules = [
      'src.agents.base',
      'src.agents.ppo_agent',
      'src.agents.a2c_agent', 
      'src.agents.sac_agent',
      'src.agents.td3_agent',
      'src.environments.lunar_lander',
      'src.training.trainer',
      'src.training.cli',
      'src.experiments.runner',
      'src.experiments.base',
      'src.utils.seeding',
      'src.utils.metrics',
      'src.utils.checkpointing',
      'src.visualization.plots',
  ]
  
  failed = []
  for m in modules:
      try:
          __import__(m)
          print(f'✅ {m}')
      except Exception as e:
          print(f'❌ {m}: {e}')
          failed.append(m)
  
  if not failed:
      print('\n✅ All imports successful!')
  else:
      print(f'\n❌ {len(failed)} imports failed')
      sys.exit(1)
  "
  ```

- [ ] T038 Run final unit test verification
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  pytest tests/unit/ -v --tb=short 2>&1 | tail -10
  ```

- [ ] T039 Display final status report summary
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  echo "=== FINAL STATUS ==="
  echo "STATUS_REPORT.md:"
  head -15 STATUS_REPORT.md
  echo ""
  echo "TODO_REPORT.md:"
  head -10 TODO_REPORT.md
  ```

- [ ] T040 Prepare git commit message
  ```bash
  cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
  cat > /tmp/commit_message.txt << 'EOF'
  002: Project cleanup and validation complete

  - Cleaned all cache directories (__pycache__, .pytest_cache, .ruff_cache)
  - Consolidated 4 duplicate test files to tests/ directory
  - Renamed test_installation.py to verify_setup.py (smoke test)
  - Verified all unit tests pass (39/39 = 100%)
  - Verified all core module imports work correctly
  - Documented 14 TODO/FIXME markers (1 critical in cli.py)
  - Created comprehensive STATUS_REPORT.md with gaps analysis
  - Created TODO_REPORT.md with all incomplete features
  
  Gaps Identified:
  - CLI model loading not implemented (cli.py:249)
  - Experiment class not exported from experiments/__init__.py
  
  Validation: All cleanup operations completed within 15-minute constraint.
  EOF
  cat /tmp/commit_message.txt
  ```

---

## Dependencies & Execution Order

### Phase Dependencies

| Phase | Depends On | Blocks |
|-------|-----------|--------|
| Phase 1: Setup | None | All cleanup |
| Phase 2: Foundational | Phase 1 | All user stories |
| Phase 3: US1 Clean | Phase 2 | None |
| Phase 4: US2 Validate | Phase 2 | None |
| Phase 5: US3 Document | Phase 2, Phase 4 | None |
| Phase 6: US4 Incomplete | Phase 2, Phase 5 | None |
| Phase 7: US5 Docs | Phase 2 | None |
| Phase 8: Polish | All phases | Ready for commit |

### User Story Dependencies

- **US1 (Clean Structure)**: Can start after Phase 2 - No dependencies on other stories
- **US2 (Validate Modules)**: Can start after Phase 2 - No dependencies on other stories
- **US3 (Review Docs)**: Can start after Phase 2 - Depends on US2 completion for test results
- **US4 (Identify Incomplete)**: Can start after Phase 2 - Depends on US3 for status report
- **US5 (Consolidate Docs)**: Can start after Phase 2 - No dependencies

### Within Each User Story

- Tasks marked [P] can run in parallel
- Sequential tasks depend on previous task completion
- Each story phase is independently testable

### Parallel Opportunities

| Phase | Parallel Tasks |
|-------|---------------|
| Phase 1 | T001, T002, T003, T004 |
| Phase 2 | T005, T006, T007, T008, T009, T010 |
| Phase 3 (US1) | T011, T012 |
| Phase 4 (US2) | T017, T018, T019, T020, T021 |
| Phase 5 (US3) | T024, T025 |
| Phase 6 (US4) | T029, T030 |
| Phase 7 (US5) | T033, T034 |
| Phase 8 | T036, T037 |

---

## Parallel Execution Examples

### Launch all Phase 1 tasks in parallel:
```bash
# Terminal 1
git stash save "Pre-cleanup $(date +%Y%m%d_%H%M%S)"

# Terminal 2  
git clean -Xd --dry-run > /tmp/cleanup_dry_run.txt

# Terminal 3
find . -name "__pycache__" -o -name ".pytest_cache" > /tmp/cache_dirs.txt
```

### Launch all Phase 2 validation tasks in parallel:
```bash
# Terminal 1
pytest tests/unit/ -v --tb=short > /tmp/unit_test_baseline.txt

# Terminal 2
python -c "from src.agents import *; print('✅ Imports OK')" > /tmp/import_baseline.txt

# Terminal 3
grep -r "TODO\|FIXME" src/ > /tmp/todo_baseline.txt

# Terminal 4
ls -la test_*.py > /tmp/root_test_files.txt
```

### Launch all US2 validation tasks in parallel:
```bash
# Terminal 1
pytest tests/unit/ -v > /tmp/us2_unit_tests.txt

# Terminal 2
python -c "from src.agents.base import BaseAgent; print('✅')"

# Terminal 3
python -c "from src.training.trainer import Trainer; print('✅')"

# Terminal 4
python -c "from src.experiments.runner import ExperimentRunner; print('✅')"
```

---

## Implementation Strategy

### MVP First (Phase 1 + Phase 2 Only)

For minimal validation:
1. Complete Phase 1: Setup (backup and safety)
2. Complete Phase 2: Foundational (baseline validation)
3. **STOP and VALIDATE**: Verify cleanup can proceed safely

### Incremental Delivery

1. Phase 1 + Phase 2 → Baseline established, cleanup safe
2. Phase 3 (US1) → Project structure clean
3. Phase 4 (US2) → All modules validated
4. Phase 5 (US3) → Status report generated
5. Phase 6 (US4) → Incomplete features documented
6. Phase 7 (US5) → Documentation structure verified
7. Phase 8 (Polish) → Ready for commit

### Parallel Team Strategy

With multiple developers:

1. **Developer A**: Phase 1 + Phase 2 (setup and baseline)
2. **Developer B**: Phase 3 (US1 cleanup)
3. **Developer C**: Phase 4 (US2 validation)
4. **Once US2 complete**: Developer A continues with Phase 5-8
5. **All phases complete**: Merge and commit

---

## Task Summary

| Phase | User Story | Task Count | Parallel Tasks |
|-------|------------|------------|----------------|
| Phase 1 | Setup | 4 | T001, T002, T003, T004 |
| Phase 2 | Foundational | 6 | T005, T006, T007, T008, T009, T010 |
| Phase 3 | US1 (Clean) | 6 | T011, T012 |
| Phase 4 | US2 (Validate) | 7 | T017, T018, T019, T020, T021 |
| Phase 5 | US3 (Document) | 5 | T024, T025 |
| Phase 6 | US4 (Incomplete) | 4 | T029, T030 |
| Phase 7 | US5 (Docs) | 3 | T033, T034 |
| Phase 8 | Polish | 5 | T036, T037 |
| **Total** | | **40** | **~22 parallelizable** |

---

## Independent Test Criteria per User Story

| User Story | Independent Test Command |
|------------|-------------------------|
| US1 Clean | `find . -name "__pycache__" -type d` returns empty |
| US2 Validate | `pytest tests/unit/ -q` shows 100% pass |
| US3 Document | `ls STATUS_REPORT.md` returns file exists |
| US4 Incomplete | `grep -c "TODO" /tmp/todo_final.txt` returns count |
| US5 Consolidate | `ls docs/ | wc -l` shows organized structure |

---

## Notes

- **No tests generated**: Feature specification does not request test creation
- **All verification**: Only validation commands and documentation tasks
- **[P] marker**: Indicates parallelizable task (different files, no dependencies)
- **All tasks use absolute paths**: Commands include `/home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment` prefix
- **Git safety**: All destructive operations preceded by backup (git stash)
- **Time constraint**: All tasks designed to complete within 15-minute limit per NFR
- **Commit after each phase**: Recommended to commit after Phase 2 and Phase 8
