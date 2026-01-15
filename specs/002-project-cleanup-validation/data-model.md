# Data Model: Project Cleanup and Validation

**Feature**: 002-project-cleanup-validation  
**Date**: 2026-01-15  
**Based on**: Feature Specification and Research Findings

---

## Entity Definitions

### 1. CacheDirectory

**Description**: Temporary files and directories that accumulate during development and should be cleaned

**Fields**:
- `path`: string - Full filesystem path
- `type`: enum[pycache, pytest_cache, ruff_cache, other] - Cache type
- `size_kb`: number - Size in kilobytes
- `file_count`: number - Number of files contained
- `git_status`: enum[ignored, tracked] - Whether git tracks these files

**Validation Rules**:
- Must exist in project root or subdirectories
- Type must match recognized patterns (`__pycache__`, `.pytest_cache`, etc.)
- Size must be >= 0
- Git status determines cleanup method (ignored = safe to delete, tracked = backup required)

**State Transitions**:
```
[Discovered] → [BackedUp] → [Deleted] → [Verified]
       ↓
[Ignored] (if git-ignored, skip backup)
```

**Example from Project**:
```json
{
  "path": "/home/.../src/__pycache__",
  "type": "pycache",
  "size_kb": 245,
  "file_count": 12,
  "git_status": "ignored"
}
```

---

### 2. DuplicateFile

**Description**: Test files that exist at root level and should be consolidated into `tests/` directory

**Fields**:
- `root_path`: string - Path at project root
- `tests_path`: string - Corresponding path in tests/ directory
- `category`: enum[unit, integration, api, installation] - Test category
- `status`: enum[duplicate, unique, conflict] - Duplication status
- `root_hash`: string - Content hash of root file
- `tests_hash`: string - Content hash of tests file (if exists)

**Validation Rules**:
- Root files must be in `test_*.py` pattern
- Category determined by file content and location
- Status requires content comparison (hash)
- Conflict if hashes differ but names match

**State Transitions**:
```
[Discovered] → [Compared] → [Action]
    ↓
[Duplicate] → [Consolidated] → [Verified]
[Unique]    → [Kept or Moved]
[Conflict]  → [Manual Review Required]
```

**Example from Project**:
```json
{
  "root_path": "./test_trainer_basic.py",
  "tests_path": "tests/unit/test_trainer_basic.py",
  "category": "unit",
  "status": "duplicate",
  "root_hash": "abc123...",
  "tests_hash": "abc123..."
}
```

---

### 3. IncompleteFeature

**Description**: Features with TODO markers, incomplete implementation, or partial functionality

**Fields**:
- `file_path`: string - Source file location
- `line_number`: number - Line number of marker
- `type`: enum[todo, fixme, xxx, incomplete] - Marker type
- `severity`: enum[critical, high, medium, low] - Impact on functionality
- `description`: string - What needs to be done
- `scope`: enum[cli, training, evaluation, validation] - Affected area
- `status`: enum[unresolved, documented, deferred, resolved] - Current state

**Validation Rules**:
- Must be found via grep/search for markers
- Severity based on impact on core workflows
- Scope must match project structure
- Status updated based on resolution

**State Transitions**:
```
[Discovered] → [Assessed] → [Action]
    ↓
[Unresolved] → [Documented] → [Deferred/Fixed]
```

**Example from Project**:
```json
{
  "file_path": "src/training/cli.py",
  "line_number": 249,
  "type": "todo",
  "severity": "high",
  "description": "Реализовать загрузку модели в тренере",
  "scope": "cli",
  "status": "documented"
}
```

---

### 4. DocumentationFile

**Description**: Markdown documentation files that need organization

**Fields**:
- `path`: string - File path
- `category`: enum[api, agents, environments, training, experiments, cleanup] - Topic
- `status`: enum[consolidated, scattered, orphaned] - Organization state
- `size_kb`: number - File size
- `last_modified`: date - Last update date

**Validation Rules**:
- Must be .md file
- Category inferred from content and location
- Status based on location relative to docs/ directory

**Example from Project**:
```json
{
  "path": "docs/ppo_agent_guide.md",
  "category": "agents",
  "status": "consolidated",
  "size_kb": 12,
  "last_modified": "2026-01-14"
}
```

---

### 5. TestResult

**Description**: Results from running pytest with pass/fail status

**Fields**:
- `test_file`: string - Test file path
- `test_name`: string - Test function name
- `category`: enum[unit, integration, experiment] - Test type
- `status`: enum[passed, failed, skipped, error] - Execution result
- `duration_ms`: number - Execution time
- `is_mocked`: boolean - Whether test uses mocking
- `failure_reason`: string|null - Error message if failed

**Validation Rules**:
- Must come from pytest output
- Duration >= 0
- Failure reason required if status is failed/error

**State Transitions**:
```
[Executed] → [Recorded] → [Analyzed]
    ↓
[Passed] → [Verified]
[Failed] → [Investigated] → [Bug/Env]
```

**Example from Project**:
```json
{
  "test_file": "tests/unit/test_seeding.py",
  "test_name": "test_set_seed_np",
  "category": "unit",
  "status": "passed",
  "duration_ms": 15,
  "is_mocked": true,
  "failure_reason": null
}
```

---

### 6. ModuleImport

**Description**: Status of Python module imports

**Fields**:
- `module_path`: string - Import path (e.g., "src.agents.base")
- `file_path`: string - Actual file location
- `status`: enum[success, failed, missing_dependency] - Import result
- `error_type`: string|null - Exception type if failed
- `error_message`: string|null - Exception message
- `dependencies`: string[] - Required modules for import

**Validation Rules**:
- Must attempt actual import
- Error details required for failures
- Dependencies list all required imports

**Example from Project**:
```json
{
  "module_path": "src.agents.ppo_agent",
  "file_path": "src/agents/ppo_agent.py",
  "status": "success",
  "error_type": null,
  "error_message": null,
  "dependencies": ["stable_baselines3", "src.agents.base"]
}
```

---

### 7. ReproducibilityFeature

**Description**: Features related to seeding, dependency tracking, and result reproducibility

**Fields**:
- `feature_name`: string - Name of reproducibility feature
- `category`: enum[seeding, dependencies, checkpointing, logging] - Type
- `status`: enum[working, broken, incomplete, not_tested] - Verification state
- `test_coverage`: number - Percentage of test coverage
- `verification_method`: string - How it was verified

**Validation Rules**:
- Must have corresponding test file
- Test coverage calculated from pytest coverage
- Verification method documented

**Example from Project**:
```json
{
  "feature_name": "Global Seed Setting",
  "category": "seeding",
  "status": "working",
  "test_coverage": 100,
  "verification_method": "Unit tests in tests/unit/test_seeding.py"
}
```

---

### 8. OutputArtifact

**Description**: Generated files from training, evaluation, or cleanup processes

**Fields**:
- `path`: string - File path
- `type`: enum[checkpoint, video, plot, metrics, report, log] - Artifact type
- `size_kb`: number - File size
- `generation_date`: date - When it was created
- `associated_experiment`: string|null - Experiment ID if applicable

**Validation Rules**:
- Must exist on filesystem
- Type must match file extension/pattern
- Size must be >= 0

**Example from Project**:
```json
{
  "path": "results/ppo_lunarlander/model.zip",
  "type": "checkpoint",
  "size_kb": 1245,
  "generation_date": "2026-01-14",
  "associated_experiment": "ppo_lunarlander_v1"
}
```

---

## Relationships

### Primary Entity Relationships

```
CacheDirectory (1) → (N) FileOperations
DuplicateFile (1) → (N) FileOperations
IncompleteFeature (1) → (0..1) Resolution
DocumentationFile (1) → (N) ConsolidationOperations
TestResult (N) → (1) TestExecution
ModuleImport (N) → (1) ImportVerification
ReproducibilityFeature (1) → (N) VerificationTest
OutputArtifact (N) → (1) GenerationProcess
```

### Cleanup Workflow Entity Flow

```
[CacheDirectory] → [Discovery] → [Validation] → [Backup] → [Deletion] → [Verification]
[DuplicateFile] → [Discovery] → [Comparison] → [Consolidation] → [UpdateRefs] → [Verification]
[IncompleteFeature] → [Search] → [Assessment] → [Documentation] → [Reporting]
[ModuleImport] → [Attempt] → [Success/Failure] → [Record] → [Reporting]
[TestResult] → [Execute] → [Collect] → [Analyze] → [Reporting]
```

---

## Validation Rules (Cross-Entity)

### Cleanup Phase Rules

1. **Cache Removal**: All CacheDirectory entities with `git_status: ignored` must be deleted
2. **Duplicate Consolidation**: All DuplicateFile entities with `status: duplicate` must be moved
3. **Safety Check**: No file deletion without git stash backup (NFR-005)

### Validation Phase Rules

1. **Import Success**: All ModuleImport entities must have `status: success`
2. **Test Pass Rate**: Unit tests must have ≥95% pass rate
3. **Critical TODOs**: All IncompleteFeature entities with `severity: critical` must be flagged

### Documentation Phase Rules

1. **Status Report**: Must include all entities with state changes
2. **JSON Export**: All entities must be serializable to JSON
3. **Gap Documentation**: Claims vs reality must be explicitly documented

---

## State Machines

### CacheDirectory State Machine

```
        [create]
            ↓
      [DISCOVERED] → [IGNORED] (if git-ignored)
            ↓
        [SCANNED]
            ↓
      [BACKED_UP] ← [SKIPPED]
            ↓
       [DELETED]
            ↓
      [VERIFIED]
```

### IncompleteFeature State Machine

```
      [found]
         ↓
    [UNRESOLVED] → [ASSESSING]
         ↓
    [DOCUMENTED] → [DEFERRED]
         ↓
    [RESOLVED] (by code change)
```

---

## Data Constraints

### Numerical Constraints
- `size_kb`: Must be ≥ 0, integer
- `file_count`: Must be ≥ 0, integer
- `line_number`: Must be ≥ 1, integer
- `test_coverage`: Must be 0-100, integer
- `duration_ms`: Must be ≥ 0, integer

### String Constraints
- `path`: Must be valid filesystem path, absolute or relative
- `description`: Max 500 characters
- `error_message`: Max 1000 characters

### Enum Constraints
- All enum fields must match predefined values exactly
- Case-sensitive where specified

---

## JSON Schema for Status Report

```json
{
  "status_report": {
    "generated_at": "datetime",
    "feature_branch": "string",
    "summary": {
      "cache_cleaned": "number",
      "duplicates_consolidated": "number",
      "incomplete_features": "number",
      "tests_passed": "number",
      "tests_failed": "number",
      "imports_verified": "number"
    },
    "entities": {
      "cache_directories": "CacheDirectory[]",
      "duplicate_files": "DuplicateFile[]",
      "incomplete_features": "IncompleteFeature[]",
      "documentation_files": "DocumentationFile[]",
      "test_results": "TestResult[]",
      "module_imports": "ModuleImport[]",
      "reproducibility_features": "ReproducibilityFeature[]",
      "output_artifacts": "OutputArtifact[]"
    },
    "gaps": {
      "documentation_claims": "string[]",
      "actual_status": "string[]",
      "discrepancies": "string[]"
    }
  }
}
```

---

## Usage in Cleanup Workflow

### Phase 1: Discovery
1. Scan for CacheDirectory → Create entities
2. Find DuplicateFile → Create entities  
3. Search IncompleteFeature → Create entities
4. List DocumentationFile → Create entities

### Phase 2: Validation
1. Execute all ModuleImport → Update status
2. Run all TestResult → Update status
3. Verify ReproducibilityFeature → Update status

### Phase 3: Action
1. For CacheDirectory: backup → delete
2. For DuplicateFile: consolidate → update refs
3. For IncompleteFeature: document → report

### Phase 4: Reporting
1. Generate JSON from all entities
2. Create human-readable report
3. Export status summary

---

## Notes

### Design Decisions
- **Why entities?**: Allows structured tracking of all cleanup/validation artifacts
- **Why state machines?**: Ensures safe operations (backup before delete)
- **Why enums?**: Enforces valid states and transitions

### Trade-offs
- **Simplicity vs Detail**: Entities capture needed detail without over-engineering
- **Flexibility vs Safety**: State machines enforce safety at cost of flexibility
- **Single source of truth**: All validation flows through entity states

### Future Extensions
- Add `severity_score` to IncompleteFeature for prioritization
- Add `dependencies` to ModuleImport for dependency graph
- Add `impact` to TestResult for risk assessment

---

**End of Data Model**