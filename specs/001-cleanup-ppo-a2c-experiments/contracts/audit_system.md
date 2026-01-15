# Audit System Interface Contract

**Version**: 1.0.0
**Date**: 2026-01-15
**Feature**: Project Cleanup and PPO vs A2C Experiments

## Overview

This contract defines the interface for the automated code audit system that scans project modules, tests imports, and generates comprehensive audit reports.

## CLI Interface

### Command

```bash
python -m src.audit.run [--options]
```

### Options

| Option | Type | Required | Default | Description |
|--------|------|----------|----------|-------------|
| `--scope` | string | No | `src/` | Directory to audit (relative to project root) |
| `--output` | string | No | `АУДИТ.md` | Output file path for audit report |
| `--format` | enum | No | `markdown` | Output format: `markdown`, `json`, or `both` |
| `--verbose` | flag | No | `False` | Enable DEBUG-level logging |
| `--skip-smoke-tests` | flag | No | `False` | Skip functionality smoke tests |

### Example Usage

```bash
# Default audit of src/ directory
python -m src.audit.run

# Audit tests/ directory with JSON output
python -m src.audit.run --scope tests/ --output audit_tests.json --format json

# Verbose audit with both formats
python -m src.audit.run --verbose --format both
```

## Input Parameters

### AuditConfig

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AuditConfig:
    """Configuration for code audit process."""
    scope: Path = Path("src/")
    output_file: Path = Path("АУДИТ.md")
    format: str = "markdown"  # "markdown", "json", "both"
    verbose: bool = False
    skip_smoke_tests: bool = False
```

## Output Format

### Markdown Format (`АУДИТ.md`)

```markdown
# Code Audit Report

**Date**: 2026-01-15T10:30:00Z
**Auditor**: Automated Audit System
**Scope**: src/

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
| ppo_agent | src/agents/ppo_agent.py | ❌ Error | - | ❌ Broken | ModuleNotFoundError: No module named 'stable_baselines3' |

## Recommendations

1. Install missing dependency: stable_baselines3
2. Fix import errors in ppo_agent.py
```

### JSON Format (`audit_report.json`)

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
      }
    ]
  }
}
```

## Module Assessment Criteria

### Import Status Categories

| Category | Description | Exit Code |
|----------|-------------|------------|
| `success` | Module imported without errors | 0 |
| `warning` | Imported with deprecation warnings | 1 |
| `error` | Import failed with exception | 2 |

### Functionality Test Categories

| Category | Description | Trigger |
|----------|-------------|----------|
| `pass` | Smoke test completed successfully | Main functions callable |
| `fail` | Smoke test failed with exception | Runtime error |
| `skip` | Test skipped (import failed) | Import status is `error` |

### Final Status Determination

| Import | Test | Final Status |
|--------|-------|-------------|
| success | pass | Working ✅ |
| success | fail | Needs Fixing ⚠️ |
| success | skip | Needs Fixing ⚠️ |
| warning | pass | Needs Fixing ⚠️ |
| warning | fail | Needs Fixing ⚠️ |
| error | skip | Broken ❌ |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Audit completed successfully |
| 1 | Audit completed with warnings |
| 2 | Audit failed (critical error) |
| 3 | Invalid command-line arguments |

## Logging Behavior

### DEBUG Level (when `--verbose`)

- Every module import attempt logged
- Full exception stack traces for errors
- Detailed smoke test execution logs

### INFO Level (default)

- Module assessment results
- Summary statistics
- Completion message

### ERROR Level

- Critical failures during audit
- File system errors
- Invalid configuration

## Error Handling

### Expected Errors

1. **Module Import Errors**: Logged as "broken" status, execution continues
2. **File Access Errors**: Logged with full path, audit continues with remaining files
3. **Missing Directory**: Exit code 2 with error message

### Recovery Behavior

- Audit continues after individual module failures
- Only critical system errors halt execution
- Partial reports saved even if audit incomplete

## Testing Requirements

### Unit Tests

```python
def test_audit_success():
    """Test audit on valid module."""
    config = AuditConfig(scope=Path("tests/fixtures/working_module.py"))
    result = run_audit(config)
    assert result.summary.working == 1

def test_audit_failure():
    """Test audit on broken module."""
    config = AuditConfig(scope=Path("tests/fixtures/broken_module.py"))
    result = run_audit(config)
    assert result.summary.broken == 1

def test_import_test():
    """Test import testing logic."""
    success, msg = test_module_import(Path("src/agents/base.py"))
    assert success is True
```

### Integration Tests

```python
def test_full_audit_workflow():
    """Test complete audit process."""
    # Run audit
    run_cli(["--scope", "tests/fixtures/"])

    # Verify output files
    assert Path("АУДИТ.md").exists()
    assert Path("audit_report.json").exists()

    # Verify content
    report = load_json("audit_report.json")
    assert report["audit_report"]["summary"]["total_modules"] > 0
```

## Dependencies

- `importlib.util` (Python stdlib)
- `logging` (Python stdlib)
- `pathlib` (Python stdlib)
- `json` (Python stdlib)
- `dataclasses` (Python stdlib)

## Performance Requirements

- Complete audit of 20 modules within 10 minutes (SC-001)
- Support 100+ modules without degradation
- Memory usage < 500MB

---

**Related Contracts**:
- [Cleanup System](./cleanup_system.md)
- [Training Pipeline](./training_pipeline.md)
