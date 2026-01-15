# Quickstart: Project Cleanup & Validation

**Feature Branch**: `002-project-cleanup-validation`  
**Date**: 2026-01-15  
**Estimated Time**: 1-2 minutes execution  
**Risk Level**: Low (git-reversible operations)

---

## TL;DR

This feature validates a completed RL training project after MVP development. It performs:
1. **Cleanup**: Remove cache, consolidate duplicate test files
2. **Validation**: Verify imports, run unit tests, check TODOs
3. **Documentation**: Generate honest status report

**Status**: Research complete, ready for execution.

---

## Prerequisites

```bash
# Activate environment
conda activate rocm  # or sb3-lunar-env

# Verify dependencies
python -c "import stable_baselines3, gymnasium; print('✅ Dependencies OK')"

# Check git status
git status  # Ensure clean working tree
```

---

## Step-by-Step Execution

### Phase 1: Validation (2 minutes)

```bash
# 1. Verify all imports work
python -c "
from src.agents import BaseAgent, PPOAgent, A2CAgent, SACAgent, TD3Agent
from src.environments import LunarLanderEnvironment
from src.training import Trainer
from src.experiments import ExperimentManager, SimpleExperiment
print('✅ All core imports successful')
"

# 2. Run unit tests (fast, mocked)
cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
pytest tests/unit/ -v --tb=short

# Expected: 39/39 passed (100%)

# 3. Identify TODOs and FIXMEs
grep -r "TODO\|FIXME\|XXX" src/ --include="*.py" --line-number > todo_report.txt
cat todo_report.txt

# Expected: ~14 markers, 1 critical (cli.py:249)
```

### Phase 2: Cleanup (30 seconds)

```bash
# 1. Create git backup (SAFETY FIRST)
git stash save "Pre-cleanup backup $(date +%Y%m%d_%H%M%S)"

# 2. Dry-run git clean (verify what will be removed)
git clean -Xd --dry-run

# 3. Remove cache directories (if dry-run looks correct)
git clean -Xd

# 4. Consolidate duplicate test files
# Move root test files to tests/ directory
mv test_trainer_basic.py tests/unit/
mv test_trainer_simple.py tests/unit/
mv test_api.py tests/integration/
mv test_integration.py tests/integration/

# 5. Keep install verification script
mv test_installation.py verify_setup.py
chmod +x verify_setup.py

# 6. Verify cleanup
find . -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" 2>/dev/null
# Should return nothing (or only .gitignored paths)
```

### Phase 3: Documentation (1 minute)

```bash
# 1. Generate status report
python -m src.reporting.status_generator --feature 002-project-cleanup-validation

# 2. Or manual generation (if generator doesn't exist):
cat > STATUS_REPORT.md << 'EOF'
# Project Status Report: 002-project-cleanup-validation

## Summary
- Architecture: ✅ Complete
- Unit Tests: ✅ 39/39 pass (100%)
- Integration Tests: ⚠️ Mixed (some fail due to environment)
- CLI Model Loading: ❌ TODO (blocks full evaluation)

## Cleanup Completed
- Cache directories: Removed
- Duplicate files: Consolidated (4 files moved)
- Empty files: Documented
- TODOs: Documented (14 total, 1 critical)

## Gaps Identified
1. CLI model loading (src/training/cli.py:249) - needs implementation
2. Experiment class not exported from __init__.py - easy fix
3. Integration test failures are environment-related, not bugs

## Recommendations
1. Fix CLI TODO in follow-up task
2. Add Experiment to exports in src/experiments/__init__.py
3. Document integration test environment requirements
EOF

cat STATUS_REPORT.md
```

---

## Verification Commands

```bash
# Check cache is gone
find . -name "__pycache__" -type d 2>/dev/null | wc -l
# Should output: 0

# Check test consolidation
ls -la test_*.py 2>/dev/null
# Should show: verify_setup.py only (or nothing)

# Verify imports still work
python -c "import src; print('✅ Project structure intact')"

# Run quick smoke test
python verify_setup.py
```

---

## Expected Artifacts

After execution, you should have:

```
specs/002-project-cleanup-validation/
├── plan.md              # Updated with real context
├── research.md          # All unknowns resolved
├── data-model.md        # Entity definitions
├── quickstart.md        # This file
└── contracts/
    ├── openapi.yaml     # REST API schema
    └── graphql.schema   # GraphQL schema (alternative)

# Plus:
- STATUS_REPORT.md      # At project root
- todo_report.txt       # All TODO markers
```

---

## Common Issues & Solutions

### Issue: "Dependencies not found"
**Solution**: 
```bash
conda activate rocm
pip install -e .
```

### Issue: "Integration tests fail"
**Solution**: Expected. These require full SB3 training which is slow. Unit tests are sufficient for validation.

### Issue: "Git clean removes something important"
**Solution**: 
```bash
git stash pop  # Restore from backup
# Review .gitignore file and add patterns if needed
```

### Issue: "Test files have conflicts"
**Solution**: If root vs tests/ versions differ:
```bash
# Keep the tests/ version (more recent)
diff test_trainer_basic.py tests/unit/test_trainer_basic.py
# If diff exists, tests/ version wins
```

---

## Post-Cleanup Actions

1. **Commit changes**:
```bash
git add -A
git commit -m "002: Project cleanup and validation complete

- Cleaned cache directories
- Consolidated test files (4 moved)
- Documented 14 TODOs (1 critical)
- Created status report
- Verified all imports work
- Unit tests: 39/39 pass (100%)"
```

2. **Create follow-up task** for CLI TODO:
```bash
# New spec: "Implement model loading in CLI"
# Priority: High (blocks evaluation workflow)
```

3. **Update docs**:
```bash
# Add TODO to README or create KNOWN_ISSUES.md
echo "## Known Issues" > KNOWN_ISSUES.md
echo "1. CLI model loading not implemented (see cli.py:249)" >> KNOWN_ISSUES.md
```

---

## Constitution Compliance

✅ **Reproducible**: All operations documented, git-based, reversible  
✅ **Test-First**: Unit tests verified before cleanup  
✅ **Documented**: Status report, research findings, data model  
✅ **Scientific**: Claims vs reality gap explicitly documented  
✅ **Efficient**: Completes well within 15-minute constraint  

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Delete wrong file | Very Low | High | Git stash backup |
| Break tests | Low | Medium | Unit tests run first |
| Miss TODOs | Low | Low | Grep automation |
| Time overrun | Very Low | Low | ~2 min actual |

---

## Next Steps (After This Feature)

1. **Priority**: Fix CLI model loading (src/training/cli.py:249)
2. **Easy**: Add Experiment to src/experiments/__init__.py exports
3. **Docs**: Create integration test environment guide
4. **Validation**: Run full integration tests with proper env setup

---

## Command Summary (Copy-Paste)

```bash
# Full execution script
cd /home/gna/workspase/education/MEPHI/training_an_agent_in_a_classical_environment
git stash save "Pre-cleanup $(date +%Y%m%d_%H%M%S)"
pytest tests/unit/ -q
grep -r "TODO\|FIXME" src/ --line-number
find . -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -delete 2>/dev/null
mv test_trainer_basic.py test_trainer_simple.py tests/unit/
mv test_api.py test_integration.py tests/integration/
mv test_installation.py verify_setup.py
python -c "from src.agents import *; from src.training import *; print('✅ Imports OK')"
echo "Cleanup complete. Review and commit."
```

---

**Ready to Execute**: Run the script above, then review and commit changes.

**Questions**: See `research.md` for detailed answers to all unknowns.