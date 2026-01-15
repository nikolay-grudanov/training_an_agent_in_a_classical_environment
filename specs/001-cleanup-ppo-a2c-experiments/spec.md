# Feature Specification: Project Cleanup and PPO vs A2C Experiments

**Feature Branch**: `001-cleanup-ppo-a2c-experiments`  
**Created**: 2026-01-15  
**Status**: Draft  
**Input**: User description: "Очистка проекта и эксперименты PPO vs A2C"

## Clarifications

### Session 2026-01-15

- Q: What data format should be used for saving training metrics and audit reports? → A: JSON for metrics and audit reports, Pickle for trained agent models
- Q: How should the system handle training interruption (e.g., user stops training, system crash, timeout)? → A: Save partial results and checkpoint - Enable resume capability from interruption point
- Q: What level of logging should the system provide for training and audit processes? → A: Verbose logging - DEBUG level with detailed step-by-step progress
- Q: What are the technical constraints for Python version and execution environment? → A: Existing conda environment named "rocm"
- Q: What specific information should the audit report (АУДИТ.md) contain for each module? → A: Module name, file path, import status, functionality test result, status icon (✅/❌/⚠️), and brief notes on issues or fixes needed

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Code Audit and Project Health Assessment (Priority: P1)

A research engineer needs to understand the current state of the codebase before conducting experiments. They want to know which components are functional, which are broken, and what needs to be fixed to ensure reliable experimentation.

**Why this priority**: Without a clear understanding of the codebase health, any experiments conducted may produce unreliable results or fail entirely. This is foundational to all subsequent work.

**Independent Test**: Can be fully tested by running the audit process and generating a comprehensive report that categorizes all code components by functionality status, delivering immediate value for project planning.

**Acceptance Scenarios**:

1. **Given** a project with mixed working/broken code, **When** the audit process is executed, **Then** a detailed report is generated listing all modules in `src/` with their status (working ✅, broken ❌, needs fixing ⚠️)
2. **Given** the audit is complete, **When** reviewing the report, **Then** basic import tests have been executed and documented for all core modules
3. **Given** root directory scripts exist, **When** audit runs, **Then** all scripts are tested and their functionality status is documented

---

### User Story 2 - Project Structure Cleanup (Priority: P2)

A research engineer wants a clean, organized project structure that follows best practices, removing clutter from the root directory and organizing files into appropriate directories.

**Why this priority**: Clean project structure improves maintainability and reduces confusion when conducting experiments. It's essential before running experiments but can be done after understanding what exists.

**Independent Test**: Can be fully tested by verifying the final directory structure matches the specified layout and all unwanted files are removed from root, delivering immediate organizational value.

**Acceptance Scenarios**:

1. **Given** a cluttered root directory, **When** cleanup is performed, **Then** only approved files remain in root (requirements.txt, README.md, .gitignore, src/, tests/, results/, specs/)
2. **Given** scattered test files, **When** reorganization occurs, **Then** all test files are moved to appropriate locations in tests/ directory
3. **Given** various script files in root, **When** cleanup runs, **Then** working code is moved to src/, example files are removed, and validation scripts are removed

---

### User Story 3 - PPO Agent Training Experiment (Priority: P3)

A research engineer wants to train a PPO agent on LunarLander-v3 environment with reproducible settings to establish a baseline performance metric.

**Why this priority**: PPO is often considered a reliable baseline algorithm. Having this experiment provides a reference point for comparison, but it depends on having a clean, working codebase.

**Independent Test**: Can be fully tested by executing the training process and verifying that a trained model is saved with documented performance metrics, delivering a functional PPO agent.

**Acceptance Scenarios**:

1. **Given** a clean project setup, **When** PPO training is initiated with seed=42, **Then** training runs for exactly 50,000 steps and saves results to `results/experiments/ppo_seed42/`
2. **Given** training is complete, **When** checking the results directory, **Then** metrics (reward, training_time, episodes) are saved in a structured format
3. **Given** the experiment finishes, **When** reviewing outputs, **Then** training is fully reproducible with the same seed producing identical results

---

### User Story 4 - A2C Agent Training Experiment (Priority: P4)

A research engineer wants to train an A2C agent on the same environment with identical settings to enable direct comparison with PPO performance.

**Why this priority**: A2C provides an important comparison point to PPO, but it's secondary to establishing the PPO baseline and requires the same clean infrastructure.

**Independent Test**: Can be fully tested by executing A2C training and verifying comparable output structure to PPO experiment, delivering a second trained agent for comparison.

**Acceptance Scenarios**:

1. **Given** PPO experiment is complete, **When** A2C training is initiated with seed=42, **Then** training runs for exactly 50,000 steps and saves results to `results/experiments/a2c_seed42/`
2. **Given** A2C training finishes, **When** comparing output structure, **Then** metrics are saved in the same format as PPO for easy comparison
3. **Given** both experiments are complete, **When** reviewing results, **Then** both agents can be loaded and evaluated independently

---

### Edge Cases

- **Training interruption**: System saves partial results and checkpoint to `results/experiments/{algo}_seed42/checkpoint_{step}.zip` and enables resume from interruption point
- **Corrupted or missing dependencies during audit**: System identifies missing dependencies in audit report with broken ❌ status
- **LunarLander-v3 environment unavailable or incompatible**: System logs error and skips training experiments, noting environment issue in audit report
- **Cleanup handling of files in use or locked**: System logs warning for locked files and skips removal of files currently in use
- **Insufficient disk space for saving results**: System logs error and terminates training before attempting to save, preserving partial results in memory

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST conduct a comprehensive audit of all modules in `src/` directory, testing basic imports and functionality
- **FR-002**: System MUST generate an audit report ("АУДИТ.md") containing for each module: module name, file path, import status, functionality test result, status icon (✅/❌/⚠️), and brief notes on issues or fixes needed
- **FR-003**: System MUST remove specified files from root directory while preserving approved project structure
- **FR-004**: System MUST reorganize project files into appropriate directories (src/, tests/, results/, specs/)
- **FR-005**: System MUST train PPO agent on LunarLander-v3 for exactly 50,000 steps with seed=42
- **FR-006**: System MUST train A2C agent on LunarLander-v3 for exactly 50,000 steps with seed=42
- **FR-007**: System MUST save experiment results in structured directories (`results/experiments/ppo_seed42/`, `results/experiments/a2c_seed42/`)
- **FR-008**: System MUST track and save metrics (reward, training_time, episodes) for both experiments
- **FR-009**: System MUST ensure reproducibility by using fixed seeds and documenting all dependencies
- **FR-010**: System MUST preserve working code during cleanup and reorganization
- **FR-011**: System MUST provide verbose DEBUG-level logging with detailed step-by-step progress for all training and audit processes
- **FR-012**: System MUST check for and report missing dependencies during audit, flagging affected modules as broken ❌

### Key Entities

- **AuditReport**: JSON document containing status assessment of all project components with categorized listings (working ✅, broken ❌, needs fixes ⚠️)
- **ExperimentResults**: JSON structured data containing training metrics, configuration parameters, and references to saved models
- **ProjectStructure**: Organized directory layout with clear separation of source code, tests, results, and specifications
- **TrainedAgent**: Pickle-serialized model parameters and configuration for both PPO and A2C algorithms
- **TrainingMetrics**: JSON time-series data of rewards, episode lengths, and other performance indicators during training

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Complete audit report is generated within 10 minutes, covering 100% of modules in src/ directory
- **SC-002**: Project root directory contains only 7 approved items (requirements.txt, README.md, .gitignore, src/, tests/, results/, specs/)
- **SC-003**: Both PPO and A2C agents complete training in under 30 minutes each on standard hardware
- **SC-004**: Training results are reproducible with identical performance metrics when using the same seed
- **SC-005**: Experiment directories contain all required artifacts (saved models, metrics files, configuration)
- **SC-006**: Both trained agents demonstrate successful task completion in LunarLander-v3 environment
- **SC-007**: All working code identified in audit remains functional after cleanup and reorganization
- **SC-008**: Training metrics show clear learning progression over the 50,000 step training period

## Assumptions *(optional)*

- LunarLander-v3 environment is available and compatible with the current Gymnasium installation
- Sufficient disk space exists for saving experiment results and model checkpoints
- Existing conda environment "rocm" is available and properly configured with required dependencies
- The current codebase contains some functional components that can be preserved
- 50,000 training steps provide sufficient time for meaningful learning in LunarLander-v3

## Non-Functional Requirements *(optional)*

- **NFR-001**: System MUST log all operations at DEBUG level with timestamps to enable step-by-step progress tracking
- **NFR-002**: System MUST write logs to both console (stdout) and file at `results/logs/` for debugging and audit trails

## Dependencies *(optional)*

- Gymnasium environment with LunarLander-v3
- Stable-Baselines3 or equivalent RL library for PPO and A2C implementations
- NumPy for numerical computations and random seeding
- Matplotlib for potential visualization of training metrics
- Python environment with all required ML/RL dependencies installed

## Constraints *(optional)*

- **CNT-001**: All training and audit operations MUST use existing conda environment named "rocm"
- **CNT-002**: No new virtual environments should be created; must work within "rocm" constraints

## Out of Scope *(optional)*

- API server development
- Web interface creation
- Training algorithms other than PPO and A2C
- Automated performance comparison tools (manual comparison expected)
- Hyperparameter optimization beyond default settings
- Multi-environment testing (only LunarLander-v3)
- Real-time training monitoring dashboards