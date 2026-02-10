# Implementation Plan: Финальное тестирование, отладка и оптимизация RL проекта

**Branch**: `004-test-and-fix` | **Date**: 2026-02-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-test-and-fix/spec.md`

**Note**: This template is filled in by `/speckit.plan` command. See `.specify/templates/commands/plan.md` for execution workflow.

## Summary

Финальный этап тестирования, отладки и оптимизации RL проекта для PPO агента на LunarLander-v3. Основная цель: обеспечить надежную работу всех компонентов, достичь 100% прохождения тестов, подтвердить воспроизводимость результатов и обновить документацию. Проект уже достиг целевой награды >200 (229.15 GPU / 203.15 CPU), но требует финальной верификации и исправления 33 failed тестов.

## Technical Context

**Language/Version**: Python 3.10.14
**Primary Dependencies**: Stable-Baselines3 2.7.1, Gymnasium 1.2.3 (with Box2D), PyTorch 2.5.1+rocm6.2, NumPy 1.26.4, Matplotlib 3.9.4, pytest, ruff, mypy
**Storage**: Files (experiment results, trained models .zip, metrics CSV, checkpoints, plots PNG, videos MP4)
**Testing**: pytest for unit/integration tests, coverage measurement
**Target Platform**: Linux server with AMD GPU (RX 7800 XT) via ROCm 6.2, CPU training primary
**Project Type**: RL Research/Experimentation (single-agent, classical control environment)
**Performance Goals**: 500K timesteps training <15 min on CPU, <2GB memory, reward >200 stable
**Constraints**: Reproducibility (fixed seeds), ROCm GPU support, 33 failed unit tests need analysis
**Scale/Scope**: Single PPO agent on LunarLander-v3, 13 testing phases, full pipeline verification

**Known Issues**:
- 33 failed unit tests (outdated A2C/TD3 agent tests - non-critical)
- GPU warnings on CPU (FIXED: set CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES)
- Integration tests not yet verified

**Key Technologies**:
- **RL Framework**: Stable-Baselines3 2.7.1 (PPO algorithm)
- **Environment**: Gymnasium LunarLander-v3 (Box2D physics)
- **Deep Learning**: PyTorch 2.5.1+rocm6.2 (AMD GPU support)
- **Code Quality**: Ruff (linting, formatting), mypy (type checking)
- **Testing**: pytest (unit/integration), coverage reporting
- **Environment Manager**: conda (environment "rocm")

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification
- [x] **Reproducible Research**: Fixed seeds (42, 123, 999), documented dependencies (requirements.txt), complete training code (baseline_training.py)
  - Status: ✅ COMPLIANT - seed utils exist, experiments documented
- [x] **Experiment-Driven**: Clear hypotheses ("optimal params achieve >200 reward"), controlled comparisons (CPU vs GPU, default vs optimized)
  - Status: ✅ COMPLIANT - 8 experiments conducted with documented results
- [x] **Test-First**: Unit tests exist for RL components (603/637 passed, 94.7%)
  - Status: ✅ COMPLIANT - comprehensive test suite exists
- [x] **Performance Monitoring**: Metrics tracking in CSV (metrics.csv, eval_log.csv), visualization (plots PNG), convergence detection
  - Status: ✅ COMPLIANT - MetricsLoggingCallback implemented
- [x] **Scientific Documentation**: Hypothesis statements, methodology, quantitative results, reproducible code samples, video demonstrations
  - Status: ✅ COMPLIANT - ~4,500 lines of documentation created

## Project Structure

### Documentation (this feature)

```text
specs/004-test-and-fix/
 ├── spec.md              # Feature specification (created by @documentation-writer)
 ├── plan.md              # This file (implementation plan)
 ├── research.md          # Phase 0 output (research findings)
 ├── data-model.md        # Phase 1 output (data entities)
 ├── quickstart.md        # Phase 1 output (quickstart guide)
 ├── contracts/           # Phase 1 output (API contracts)
 └── tasks.md             # Phase 2 output (task breakdown - NOT created by plan command)
```

### Source Code (repository root - existing structure, no changes)

```text
# RL Agent Training Project Structure (existing)
src/
├── experiments/
│   └── completion/
│       ├── baseline_training.py      # Main training script (modified with device params)
│       └── config.py               # Configuration management
├── training/
│   ├── callbacks.py               # MetricsLoggingCallback, CheckpointCallback
│   ├── evaluation.py              # Evaluation utilities
│   └── results.py                # Results parsing
├── utils/
│   └── seeding.py                # set_seed() for reproducibility
├── agents/
│   ├── ppo_agent.py              # PPO agent wrapper
│   ├── a2c_agent.py              # A2C agent (legacy)
│   └── td3_agent.py              # TD3 agent (legacy)
└── visualization/
    └── plots/
        └── generate_all.py        # Plot generation script

tests/
├── unit/
│   ├── test_seeding.py           # Seed reproducibility tests
│   ├── test_ppo_agent.py         # PPO agent tests
│   ├── test_a2c_agent.py         # A2C agent tests (33 failed here)
│   └── test_td3_agent.py         # TD3 agent tests (failed)
├── integration/
│   └── test_full_pipeline.py     # End-to-end pipeline tests
└── experiment/                   # Experiment-specific tests

docs/
├── QUICKSTART.md                 # Quick start guide
├── TROUBLESHOOTING.md           # Troubleshooting documentation
├── PROJECT_CONTEXT.md            # Project context and status
├── CPU_vs_GPU_Comparison.md      # CPU vs GPU comparison report
└── PROJECT_COMPLETION_REPORT.md  # Final project report

results/
├── experiments/
│   └── ppo_seed42/             # Experiment artifacts
│       ├── ppo_seed42_model.zip   # Trained model
│       ├── metrics.csv           # Training metrics
│       ├── eval_log.csv          # Evaluation metrics
│       ├── config.json          # Configuration
│       ├── checkpoints/         # Model checkpoints
│       ├── reward_curve.png      # Reward visualization
│       └── video.mp4           # Episode demonstration
├── comparison/                  # Comparison results
├── reports/                    # Generated reports
└── videos/                     # Video demonstrations
```

**Structure Decision**: No structural changes required. This is a testing/debugging phase, not a feature development phase. All 13 testing phases will operate within the existing structure:
- Training scripts in `src/experiments/completion/`
- Tests in `tests/unit/` and `tests/integration/`
- Artifacts in `results/experiments/`
- Documentation updates in root `docs/`

The existing structure is already optimized for RL research workflows and complies with the project's constitution (reproducibility, experiment-driven development, test-first approach).
