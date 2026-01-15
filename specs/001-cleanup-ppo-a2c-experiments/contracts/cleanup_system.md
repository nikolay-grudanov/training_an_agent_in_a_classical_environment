# Cleanup System Interface Contract

**Version**: 1.0.0
**Date**: 2026-01-15
**Feature**: Project Cleanup and PPO vs A2C Experiments

## Overview

This contract defines the interface for project cleanup system that reorganizes files, removes unnecessary items from root directory, and establishes proper project structure following ML best practices.

## CLI Interface

### Command

```bash
python -m src.cleanup.run [--options]
```

### Options

| Option | Type | Required | Default | Description |
|--------|------|----------|----------|-------------|
| `--dry-run` | flag | No | `False` | Show actions without executing |
| `--force` | flag | No | `False` | Force removal of protected files |
| `--output` | string | No | `results/project_structure.json` | Output file for structure report |
| `--verbose` | flag | No | `False` | Enable DEBUG-level logging |

### Example Usage

```bash
# Dry run to preview changes
python -m src.cleanup.run --dry-run

# Execute cleanup with verbose logging
python -m src.cleanup.run --verbose

# Force cleanup of protected files
python -m src.cleanup.run --force
```

## Input Parameters

### CleanupConfig

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class CleanupConfig:
    """Configuration for project cleanup process."""
    dry_run: bool = False
    force: bool = False
    output_file: Path = Path("results/project_structure.json")
    verbose: bool = False

    # Root directory structure rules
    allowed_files: List[str] = None
    allowed_directories: List[str] = None

    def __post_init__(self):
        if self.allowed_files is None:
            self.allowed_files = [
                "requirements.txt",
                "README.md",
                ".gitignore"
            ]
        if self.allowed_directories is None:
            self.allowed_directories = [
                "src/",
                "tests/",
                "results/",
                "specs/"
            ]
```

## Cleanup Rules

### Root Directory Validation

**Allowed Files**:
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

**Allowed Directories**:
- `src/` - Source code
- `tests/` - Test files
- `results/` - Experiment results
- `specs/` - Feature specifications

**Prohibited Items**:
- Example scripts (e.g., `example_training.py`, `demo_*.py`)
- Validation scripts (e.g., `verify_setup.py`, `test_installation.py`)
- Temporary files (e.g., `*.tmp`, `*.bak`)
- Demo artifacts (e.g., `demo_checkpoints/`, `demo_experiment/`)
- Documentation in root (should be in `docs/`)
- Notebooks in root (should be in `notebooks/`)

### File Categorization

| Action | Pattern | Destination |
|--------|----------|-------------|
| Move to src/ | `*.py` (working scripts) | `src/` |
| Move to tests/ | `test_*.py` | `tests/` |
| Move to results/ | `*_checkpoints/`, `*_experiment/` | `results/` |
| Remove | `*.tmp`, `*.bak`, `demo_*.py` | Delete |
| Archive | `info_project.md`, `*_SUMMARY.md` | `docs/archives/` |

## Output Format

### Project Structure JSON

```json
{
  "project_structure": {
    "metadata": {
      "timestamp": "2026-01-15T10:00:00Z",
      "root_path": "/home/gna/.../project_root",
      "cleanup_status": "completed"
    },
    "root_directory": {
      "allowed_files": [
        "requirements.txt",
        "README.md",
        ".gitignore"
      ],
      "allowed_directories": [
        "src/",
        "tests/",
        "results/",
        "specs/"
      ],
      "actual_files": [
        "requirements.txt",
        "README.md",
        ".gitignore"
      ],
      "actual_directories": [
        "src/",
        "tests/",
        "results/",
        "specs/"
      ],
      "validation_status": "clean"
    },
    "cleanup_actions": [
      {
        "action": "moved",
        "source": "example_training.py",
        "destination": "src/examples/example_training.py",
        "status": "completed"
      },
      {
        "action": "removed",
        "source": "verify_setup.py",
        "status": "completed"
      },
      {
        "action": "moved",
        "source": "demo_checkpoints/",
        "destination": "results/demo_checkpoints/",
        "status": "completed"
      }
    ]
  }
}
```

### Dry Run Output

```text
=== PROJECT CLEANUP - DRY RUN ===

Root Directory Analysis:
  Allowed files: 3/3 (OK)
  Allowed directories: 4/4 (OK)
  Unexpected files: 5
  Unexpected directories: 2

Planned Actions:
  [MOVE] example_training.py → src/examples/example_training.py
  [MOVE] test_installation.py → tests/test_installation.py
  [MOVE] demo_checkpoints/ → results/demo_checkpoints/
  [REMOVE] verify_setup.py
  [REMOVE] .jupyter_ystore.db
  [ARCHIVE] info_project.md → docs/archives/

Total: 3 moves, 2 removals, 1 archive

Run without --dry-run to execute changes.
```

## Cleanup Actions

### Action Types

| Type | Description | Risk Level |
|------|-------------|------------|
| `moved` | File/directory moved to new location | Low |
| `removed` | File/directory deleted | High |
| `skipped` | Item ignored (protected or excluded) | None |

### Action Status

| Status | Description |
|--------|-------------|
| `pending` | Action planned but not yet executed |
| `in_progress` | Action currently executing |
| `completed` | Action finished successfully |
| `failed` | Action failed (error logged) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Cleanup completed successfully |
| 1 | Cleanup completed with warnings |
| 2 | Cleanup failed (critical error) |
| 3 | Invalid command-line arguments |
| 4 | Root directory validation failed |
| 5 | Dry run mode (no changes made) |

## Logging Behavior

### DEBUG Level (when `--verbose`)

- Every file/directory scanned logged
- Detailed action execution logs
- File move/copy progress
- Permission/access errors

### INFO Level (default)

- Cleanup summary statistics
- Action categories (moved, removed, archived)
- Final validation status
- Output file location

### WARNING Level

- Protected files skipped (use `--force` to override)
- Unexpected files found
- Archive operations

### ERROR Level

- Critical failures (disk full, permission denied)
- Root directory violations
- Action execution failures

## Error Handling

### Expected Errors

1. **Permission Denied**: Log error, skip file, continue cleanup
2. **File in Use**: Log warning, skip file, continue cleanup
3. **Disk Full**: Stop cleanup, exit code 2, report error
4. **Path Too Long**: Truncate to 255 chars, log warning
5. **Root Directory Violation**: Exit code 4, do not continue

### Recovery Behavior

- Individual file failures don't halt cleanup
- Protected files require `--force` flag
- Critical system errors halt execution
- Partial cleanup results saved to output file

## Dry Run Mode

### Behavior

- Scans project root directory
- Categorizes all items (allowed, unexpected)
- Generates action plan without executing
- Returns exit code 5 (dry run indicator)

### Output Format

- Table format for planned actions
- Summary statistics
- Explicit message requiring `--dry-run` removal to execute

## Testing Requirements

### Unit Tests

```python
def test_allowed_files_validation():
    """Test root directory file validation."""
    config = CleanupConfig()
    config.allowed_files = ["requirements.txt", "README.md"]
    is_valid = validate_root_files(["requirements.txt", "README.md"], config)
    assert is_valid is True

def test_action_categorization():
    """Test file action categorization."""
    action = categorize_file("example_training.py", config)
    assert action.type == "move"
    assert action.destination == Path("src/examples/example_training.py")

def test_dry_run_mode():
    """Test dry run doesn't modify files."""
    config = CleanupConfig(dry_run=True)
    result = run_cleanup(config)
    assert result.cleanup_status == "dry_run"
    assert len(result.cleanup_actions) == 0
```

### Integration Tests

```python
def test_full_cleanup_workflow():
    """Test complete cleanup process."""
    # Create test structure
    create_test_project_with_clutter()

    # Run cleanup
    run_cli([])

    # Verify root directory
    root_files = list(Path(".").iterdir())
    assert all(f.name in ["requirements.txt", "README.md", ".gitignore", "src/", "tests/", "results/", "specs/"] for f in root_files)

    # Verify output report
    report = load_json("results/project_structure.json")
    assert report["project_structure"]["metadata"]["cleanup_status"] == "completed"
```

### Safety Tests

```python
def test_protected_files_require_force():
    """Test protected files need --force flag."""
    config = CleanupConfig(force=False)
    with open("protected_file.txt", "w") as f:
        f.write("protected content")

    result = run_cleanup(config)
    assert any(action.status == "skipped" for action in result.cleanup_actions)

def test_no_force_flag_errors():
    """Test removal fails without --force."""
    config = CleanupConfig(force=False)
    result = run_cleanup(config)
    assert result.cleanup_status == "failed"
```

## Dependencies

- `shutil` (Python stdlib)
- `pathlib` (Python stdlib)
- `logging` (Python stdlib)
- `json` (Python stdlib)
- `dataclasses` (Python stdlib)
- `typing` (Python stdlib)

## Performance Requirements

- Complete cleanup of 100 files within 5 minutes
- Support 1000+ files without degradation
- Memory usage < 200MB

## Safety Features

### Protection Mechanisms

1. **Dry Run Mode**: Preview changes before execution
2. **Force Flag**: Required for removing protected files
3. **Validation Check**: Verify root directory structure before cleanup
4. **Partial Results**: Save cleanup status even if process fails
5. **Backup Warnings**: Warn before removing large directories

### Validation Rules

1. **Root Directory**: Must contain at least requirements.txt or README.md
2. **Source Directory**: Must exist (src/)
3. **Results Directory**: Must exist or be creatable (results/)
4. **Git Repository**: Must not delete .git directory (implicit protection)

---

**Related Contracts**:
- [Audit System](./audit_system.md)
- [Training Pipeline](./training_pipeline.md)
