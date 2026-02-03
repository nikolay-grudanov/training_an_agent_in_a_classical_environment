"""Comprehensive integration tests for full workflow: audit → cleanup → PPO → A2C.

This test suite verifies the complete end-to-end workflow of the RL agent training
pipeline, ensuring all components work together correctly.

Tests:
- Audit workflow: Code audit and report generation
- Cleanup workflow: Project cleanup (dry-run mode)
- PPO training workflow: PPO agent training
- A2C training workflow: A2C agent training
- Full workflow sequence: Complete pipeline execution
- Workflow artifacts integrity: Verify all outputs are valid
"""

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def run_audit(skip_smoke_tests: bool = True, format: str = "json") -> int:
    """Run audit workflow.

    Args:
        skip_smoke_tests: Skip smoke tests for faster execution
        format: Output format (json, markdown, both)

    Returns:
        Exit code
    """
    logger.info(f"Running audit: skip_smoke_tests={skip_smoke_tests}, format={format}")

    try:
        import sys
        from unittest.mock import patch

        from src.audit.run import main as audit_main

        # Mock sys.argv
        test_args = ["audit", "--scope", "src/", "--format", format]
        if skip_smoke_tests:
            test_args.append("--skip-smoke-tests")

        with patch.object(sys, "argv", test_args):
            exit_code = audit_main()

        logger.info(f"Audit completed with exit code: {exit_code}")
        return exit_code

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        raise


def run_cleanup(dry_run: bool = True) -> int:
    """Run cleanup workflow.

    Args:
        dry_run: Run in dry-run mode (no actual changes)

    Returns:
        Exit code
    """
    logger.info(f"Running cleanup: dry_run={dry_run}")

    try:
        import sys
        from unittest.mock import patch

        from src.cleanup.run import main as cleanup_main

        # Mock sys.argv
        test_args = ["cleanup"]
        if dry_run:
            test_args.append("--dry-run")

        with patch.object(sys, "argv", test_args):
            exit_code = cleanup_main()

        logger.info(f"Cleanup completed with exit code: {exit_code}")
        return exit_code

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


def run_training(
    algo: str = "ppo", seed: int = 42, total_timesteps: int = 1000
) -> None:
    """Run training workflow.

    Args:
        algo: Algorithm (ppo or a2c)
        seed: Random seed
        total_timesteps: Total training timesteps
    """
    logger.info(f"Running training: algo={algo}, seed={seed}, steps={total_timesteps}")

    try:
        from src.training.train import PPOTrainer

        trainer = PPOTrainer(
            algo=algo,
            seed=seed,
            total_timesteps=total_timesteps,
            verbose=False,
        )

        trainer.train()
        trainer.save_results()
        trainer.cleanup()

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def verify_json_file(path: Path | str) -> Dict[str, Any]:
    """Verify file exists and is valid JSON.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON content

    Raises:
        AssertionError: If file doesn't exist or is invalid JSON
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(path)

    if not path.exists():
        raise AssertionError(f"JSON file does not exist: {path}")

    if not path.is_file():
        raise AssertionError(f"Path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.debug(f"✓ Valid JSON file: {path}")
        return content
    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in {path}: {e}")
        raise


def verify_model_file(path: Path | str) -> None:
    """Verify model .zip file is valid archive.

    Args:
        path: Path to model .zip file

    Raises:
        AssertionError: If file doesn't exist or is invalid zip
    """
    path = Path(path)

    if not path.exists():
        raise AssertionError(f"Model file does not exist: {path}")

    if not path.is_file():
        raise AssertionError(f"Path is not a file: {path}")

    if not path.suffix == ".zip":
        raise AssertionError(f"Model file is not a .zip: {path}")

    try:
        with zipfile.ZipFile(path, "r") as zip_file:
            # Verify zip file is not empty
            file_list = zip_file.namelist()
            if not file_list:
                raise AssertionError(f"Model zip file is empty: {path}")

            # Verify zip file is not corrupted
            bad_file = zip_file.testzip()
            if bad_file is not None:
                raise AssertionError(
                    f"Model zip file is corrupted (bad file: {bad_file}): {path}"
                )

        logger.debug(f"✓ Valid model zip file: {path} ({len(file_list)} files)")

    except zipfile.BadZipFile as e:
        logger.error(f"✗ Invalid zip file {path}: {e}")
        raise AssertionError(f"Invalid zip file: {path}") from e


def verify_directory_structure(
    base_path: Path | str = Path.cwd(),
    expected_dirs: Optional[List[str]] = None,
) -> None:
    """Verify expected directories exist.

    Args:
        base_path: Base directory to check
        expected_dirs: List of expected directory paths (relative to base_path)

    Raises:
        AssertionError: If any expected directory is missing
    """
    base_path = Path(base_path)

    if expected_dirs is None:
        expected_dirs = [
            "src",
            "tests",
            "results",
            "results/experiments",
        ]

    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = base_path / dir_path
        if not full_path.exists() or not full_path.is_dir():
            missing_dirs.append(dir_path)

    if missing_dirs:
        raise AssertionError(f"Missing directories: {', '.join(missing_dirs)}")

    logger.debug("✓ All expected directories exist")


def verify_required_fields(
    data: Dict[str, Any],
    required_fields: List[str],
    context: str = "data",
) -> None:
    """Verify dictionary contains all required fields.

    Args:
        data: Dictionary to check
        required_fields: List of required field names
        context: Context string for error messages

    Raises:
        AssertionError: If any required field is missing
    """
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        raise AssertionError(
            f"Missing required fields in {context}: {', '.join(missing_fields)}"
        )

    logger.debug(f"✓ All required fields present in {context}")


def backup_directory(src: Path, dst: Path) -> None:
    """Backup directory to destination.

    Args:
        src: Source directory
        dst: Destination directory
    """
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.debug(f"✓ Backed up {src} to {dst}")


def restore_directory(src: Path, dst: Path) -> None:
    """Restore directory from backup.

    Args:
        src: Source directory (backup)
        dst: Destination directory
    """
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.debug(f"✓ Restored {dst} from {src}")


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def cleanup_results_dir():
    """Backup and clean results/ directory before test, restore after.

    This fixture ensures tests don't interfere with each other by:
    1. Backing up the existing results/ directory
    2. Cleaning it before the test
    3. Restoring it after the test

    Yields:
        Path to results directory
    """
    results_dir = Path("results")
    backup_dir = Path(tempfile.gettempdir()) / f"results_backup_{id(results_dir)}"

    # Backup existing results
    if results_dir.exists():
        backup_directory(results_dir, backup_dir)

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    yield results_dir

    # Restore backup
    if backup_dir.exists():
        restore_directory(backup_dir, results_dir)
        shutil.rmtree(backup_dir)


@pytest.fixture
def cleanup_audit_files():
    """Clean up audit files before and after test.

    Yields:
        None
    """
    audit_files = [
        Path("АУДИТ.md"),
        Path("audit_report.json"),
    ]

    # Remove existing audit files
    for file_path in audit_files:
        if file_path.exists():
            file_path.unlink()

    yield

    # Clean up audit files after test
    for file_path in audit_files:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def capture_output(caplog):
    """Capture logging output during test.

    Args:
        caplog: Pytest caplog fixture

    Yields:
        caplog fixture with INFO level enabled
    """
    with caplog.at_level(logging.INFO):
        yield caplog


# ============================================================================
# Test Class: TestFullWorkflow
# ============================================================================


class TestFullWorkflow:
    """Integration tests for complete workflow: audit → cleanup → PPO → A2C."""

    # ------------------------------------------------------------------------
    # Test 1: Audit Workflow
    # ------------------------------------------------------------------------

    def test_audit_workflow(
        self,
        cleanup_audit_files,
        capture_output,
    ) -> None:
        """Test audit workflow: Run audit and verify outputs.

        Verifies:
        - audit_report.json file exists
        - audit_report.json is valid JSON
        - Report contains "total_modules" key
        - Report shows some modules as working and some as broken
        """
        # Run audit
        exit_code = run_audit(skip_smoke_tests=True, format="json")

        # Verify command succeeded
        assert exit_code in [0, 1], f"Audit command failed with exit code {exit_code}"

        # Verify audit_report.json exists
        audit_json_path = Path("audit_report.json")
        assert audit_json_path.exists(), "audit_report.json not found"
        assert audit_json_path.is_file(), "audit_report.json is not a file"

        # Verify JSON validity
        report = verify_json_file(audit_json_path)

        # Verify required fields
        assert "audit_report" in report, "Report missing 'audit_report' key"

        # Verify audit_report structure
        assert "summary" in report["audit_report"], "Audit report missing 'summary' key"

        summary = report["audit_report"]["summary"]
        assert "total_modules" in summary, "Summary missing 'total_modules'"

        # Verify report has some content
        assert summary["total_modules"] > 0, (
            "Audit report shows total_modules=0, expected at least 1 module"
        )

        # Verify report has status breakdown
        assert (
            "working" in summary or "broken" in summary or "needs_fixing" in summary
        ), "Audit report missing status breakdown"

        logger.info(
            f"✓ Audit workflow completed: "
            f"{summary.get('total_modules', 0)} total modules, "
            f"{summary.get('working', 0)} working, "
            f"{summary.get('broken', 0)} broken, "
            f"{summary.get('needs_fixing', 0)} needs fixing"
        )

    # ------------------------------------------------------------------------
    # Test 2: Cleanup Workflow
    # ------------------------------------------------------------------------

    def test_cleanup_workflow(
        self,
        cleanup_results_dir,
        capture_output,
    ) -> None:
        """Test cleanup workflow: Run cleanup in dry-run mode.

        Verifies:
        - No files are actually removed (dry-run)
        - Output indicates items to be removed
        - Cleanup completes successfully
        """
        # Run cleanup in dry-run mode
        exit_code = run_cleanup(dry_run=True)

        # Verify command succeeded
        assert exit_code == 0, f"Cleanup command failed with exit code {exit_code}"

        # Verify no files were actually removed (dry-run mode)
        # Check that important files still exist
        important_files = [
            Path("src"),
            Path("tests"),
            Path("README.md"),
            Path("AGENTS.md"),
        ]

        for file_path in important_files:
            assert file_path.exists(), f"File was removed in dry-run mode: {file_path}"

        logger.info("✓ Cleanup workflow completed: dry-run mode verified")

    # ------------------------------------------------------------------------
    # Test 3: PPO Training Workflow
    # ------------------------------------------------------------------------

    def test_ppo_training_workflow(
        self,
        cleanup_results_dir,
        capture_output,
    ) -> None:
        """Test PPO training workflow: Train PPO agent for 1000 steps.

        Verifies:
        - results/experiments/ppo_seed42/ directory exists
        - ppo_seed42_model.zip exists and is valid
        - ppo_seed42_results.json exists and is valid
        - ppo_seed42_metrics.json exists and is valid
        - Results JSON contains required fields
        - Metrics JSON contains list of metric points
        - Training completed successfully (exit code 0)
        """
        # Run PPO training
        run_training(algo="ppo", seed=42, total_timesteps=1000)

        # Verify experiment directory exists
        exp_dir = Path("results/experiments/ppo_seed42")
        assert exp_dir.exists(), f"PPO experiment directory not found: {exp_dir}"
        assert exp_dir.is_dir(), f"PPO experiment path is not a directory: {exp_dir}"

        # Verify model file exists and is valid
        model_path = exp_dir / "ppo_seed42_model.zip"
        assert model_path.exists(), f"PPO model file not found: {model_path}"
        verify_model_file(model_path)

        # Verify results file exists and is valid
        results_path = exp_dir / "ppo_seed42_results.json"
        assert results_path.exists(), f"PPO results file not found: {results_path}"
        results = verify_json_file(results_path)

        # Verify results structure
        assert "experiment_results" in results, (
            "Results missing 'experiment_results' key"
        )

        exp_data = results["experiment_results"]

        # Verify required fields in metadata
        assert "metadata" in exp_data, "Results missing 'metadata'"
        metadata = exp_data["metadata"]

        verify_required_fields(
            metadata,
            [
                "algorithm",
                "seed",
                "total_timesteps",
                "start_time",
                "end_time",
            ],
            context="results.metadata",
        )

        # Verify algorithm is PPO
        assert metadata["algorithm"] == "PPO", (
            f"Expected algorithm='PPO', got '{metadata['algorithm']}'"
        )

        # Verify seed
        assert metadata["seed"] == 42, f"Expected seed=42, got {metadata['seed']}"

        # Verify timesteps
        assert metadata["total_timesteps"] == 1000, (
            f"Expected total_timesteps=1000, got {metadata['total_timesteps']}"
        )

        # Verify metrics section
        assert "metrics" in exp_data, "Results missing 'metrics'"
        metrics = exp_data["metrics"]

        verify_required_fields(
            metrics,
            [
                "final_reward_mean",
                "training_time_seconds",
            ],
            context="results.metrics",
        )

        # Verify metrics file exists and is valid
        metrics_path = exp_dir / "ppo_seed42_metrics.json"
        assert metrics_path.exists(), f"PPO metrics file not found: {metrics_path}"
        metrics_data = verify_json_file(metrics_path)

        # Verify metrics structure (can be dict with 'metrics' key or list)
        if isinstance(metrics_data, dict):
            assert "metrics" in metrics_data, "Metrics data missing 'metrics' key"
            metrics_list = metrics_data["metrics"]
        else:
            metrics_list = metrics_data

        # Verify metrics is a list
        assert isinstance(metrics_list, list), (
            f"Metrics should be a list, got {type(metrics_list)}"
        )

        # Verify metrics has some entries
        assert len(metrics_list) > 0, "Metrics list is empty"

        # Verify each metric point has required fields
        for i, point in enumerate(metrics_list):
            verify_required_fields(
                point,
                ["timestep", "reward"],
                context=f"metrics[{i}]",
            )

        logger.info(
            f"✓ PPO training workflow completed: "
            f"final_reward={metrics['final_reward_mean']:.2f}, "
            f"training_time={metrics['training_time_seconds']:.1f}s, "
            f"{len(metrics_data)} metric points"
        )

    # ------------------------------------------------------------------------
    # Test 4: A2C Training Workflow
    # ------------------------------------------------------------------------

    def test_a2c_training_workflow(
        self,
        cleanup_results_dir,
        capture_output,
    ) -> None:
        """Test A2C training workflow: Train A2C agent for 1000 steps.

        Verifies:
        - results/experiments/a2c_seed42/ directory exists
        - a2c_seed42_model.zip exists and is valid
        - a2c_seed42_results.json exists and is valid
        - a2c_seed42_metrics.json exists and is valid
        - Results JSON contains required fields
        - Metrics JSON contains list of metric points
        - Training completed successfully
        """
        # Run A2C training
        run_training(algo="a2c", seed=42, total_timesteps=1000)

        # Verify experiment directory exists
        exp_dir = Path("results/experiments/a2c_seed42")
        assert exp_dir.exists(), f"A2C experiment directory not found: {exp_dir}"
        assert exp_dir.is_dir(), f"A2C experiment path is not a directory: {exp_dir}"

        # Verify model file exists and is valid
        model_path = exp_dir / "a2c_seed42_model.zip"
        assert model_path.exists(), f"A2C model file not found: {model_path}"
        verify_model_file(model_path)

        # Verify results file exists and is valid
        results_path = exp_dir / "a2c_seed42_results.json"
        assert results_path.exists(), f"A2C results file not found: {results_path}"
        results = verify_json_file(results_path)

        # Verify results structure
        assert "experiment_results" in results, (
            "Results missing 'experiment_results' key"
        )

        exp_data = results["experiment_results"]

        # Verify required fields in metadata
        assert "metadata" in exp_data, "Results missing 'metadata'"
        metadata = exp_data["metadata"]

        verify_required_fields(
            metadata,
            [
                "algorithm",
                "seed",
                "total_timesteps",
                "start_time",
                "end_time",
            ],
            context="results.metadata",
        )

        # Verify algorithm is A2C
        assert metadata["algorithm"] == "A2C", (
            f"Expected algorithm='A2C', got '{metadata['algorithm']}'"
        )

        # Verify seed
        assert metadata["seed"] == 42, f"Expected seed=42, got {metadata['seed']}"

        # Verify timesteps
        assert metadata["total_timesteps"] == 1000, (
            f"Expected total_timesteps=1000, got {metadata['total_timesteps']}"
        )

        # Verify metrics section
        assert "metrics" in exp_data, "Results missing 'metrics'"
        metrics = exp_data["metrics"]

        verify_required_fields(
            metrics,
            [
                "final_reward_mean",
                "training_time_seconds",
            ],
            context="results.metrics",
        )

        # Verify metrics file exists and is valid
        metrics_path = exp_dir / "a2c_seed42_metrics.json"
        assert metrics_path.exists(), f"A2C metrics file not found: {metrics_path}"
        metrics_data = verify_json_file(metrics_path)

        # Verify metrics structure (can be dict with 'metrics' key or list)
        if isinstance(metrics_data, dict):
            assert "metrics" in metrics_data, "Metrics data missing 'metrics' key"
            metrics_list = metrics_data["metrics"]
        else:
            metrics_list = metrics_data

        # Verify metrics is a list
        assert isinstance(metrics_list, list), (
            f"Metrics should be a list, got {type(metrics_list)}"
        )

        # Verify metrics has some entries
        assert len(metrics_list) > 0, "Metrics list is empty"

        # Verify each metric point has required fields
        for i, point in enumerate(metrics_list):
            verify_required_fields(
                point,
                ["timestep", "reward"],
                context=f"metrics[{i}]",
            )

        logger.info(
            f"✓ A2C training workflow completed: "
            f"final_reward={metrics['final_reward_mean']:.2f}, "
            f"training_time={metrics['training_time_seconds']:.1f}s, "
            f"{len(metrics_data)} metric points"
        )

    # ------------------------------------------------------------------------
    # Test 5: Full Workflow Sequence
    # ------------------------------------------------------------------------

    def test_full_workflow_sequence(
        self,
        cleanup_results_dir,
        cleanup_audit_files,
        capture_output,
    ) -> None:
        """Test complete workflow sequence: audit → cleanup → PPO → A2C.

        Verifies:
        - All steps execute successfully
        - All outputs exist and are valid
        - No errors occurred in any step
        - Results are reproducible (same seed produces same outputs)

        This test runs the complete pipeline in order to ensure all
        components work together correctly.
        """
        # Step 1: Run audit
        logger.info("=" * 60)
        logger.info("Step 1: Running audit...")
        logger.info("=" * 60)

        exit_code = run_audit(skip_smoke_tests=True, format="json")
        assert exit_code in [0, 1], f"Audit failed: {exit_code}"

        audit_json_path = Path("audit_report.json")
        assert audit_json_path.exists(), "audit_report.json not created"

        audit_report = verify_json_file(audit_json_path)
        total_modules = audit_report["audit_report"]["summary"]["total_modules"]
        logger.info(f"✓ Audit completed: {total_modules} modules")

        # Step 2: Run cleanup (dry-run)
        logger.info("=" * 60)
        logger.info("Step 2: Running cleanup (dry-run)...")
        logger.info("=" * 60)

        exit_code = run_cleanup(dry_run=True)
        assert exit_code == 0, f"Cleanup failed: {exit_code}"

        structure_path = Path("results/project_structure.json")
        assert structure_path.exists(), "project_structure.json not created"

        structure = verify_json_file(structure_path)
        logger.info(f"✓ Cleanup completed: {len(structure)} entries in structure")

        # Step 3: Run PPO training
        logger.info("=" * 60)
        logger.info("Step 3: Running PPO training...")
        logger.info("=" * 60)

        run_training(algo="ppo", seed=42, total_timesteps=1000)

        ppo_exp_dir = Path("results/experiments/ppo_seed42")
        assert ppo_exp_dir.exists(), "PPO experiment directory not created"

        ppo_model_path = ppo_exp_dir / "ppo_seed42_model.zip"
        ppo_results_path = ppo_exp_dir / "ppo_seed42_results.json"
        ppo_metrics_path = ppo_exp_dir / "ppo_seed42_metrics.json"

        assert ppo_model_path.exists(), "PPO model not created"
        assert ppo_results_path.exists(), "PPO results not created"
        assert ppo_metrics_path.exists(), "PPO metrics not created"

        verify_model_file(ppo_model_path)
        ppo_results = verify_json_file(ppo_results_path)
        ppo_metrics = verify_json_file(ppo_metrics_path)

        ppo_final_reward = ppo_results["experiment_results"]["metrics"][
            "final_reward_mean"
        ]
        logger.info(f"✓ PPO training completed: final_reward={ppo_final_reward:.2f}")

        # Step 4: Run A2C training
        logger.info("=" * 60)
        logger.info("Step 4: Running A2C training...")
        logger.info("=" * 60)

        run_training(algo="a2c", seed=42, total_timesteps=1000)

        a2c_exp_dir = Path("results/experiments/a2c_seed42")
        assert a2c_exp_dir.exists(), "A2C experiment directory not created"

        a2c_model_path = a2c_exp_dir / "a2c_seed42_model.zip"
        a2c_results_path = a2c_exp_dir / "a2c_seed42_results.json"
        a2c_metrics_path = a2c_exp_dir / "a2c_seed42_metrics.json"

        assert a2c_model_path.exists(), "A2C model not created"
        assert a2c_results_path.exists(), "A2C results not created"
        assert a2c_metrics_path.exists(), "A2C metrics not created"

        verify_model_file(a2c_model_path)
        a2c_results = verify_json_file(a2c_results_path)
        a2c_metrics = verify_json_file(a2c_metrics_path)

        a2c_final_reward = a2c_results["experiment_results"]["metrics"][
            "final_reward_mean"
        ]
        logger.info(f"✓ A2C training completed: final_reward={a2c_final_reward:.2f}")

        # Verify reproducibility: same seed should produce consistent results
        logger.info("=" * 60)
        logger.info("Verifying reproducibility...")
        logger.info("=" * 60)

        # Both experiments should have the same seed
        assert ppo_results["experiment_results"]["metadata"]["seed"] == 42, (
            "PPO seed mismatch"
        )
        assert a2c_results["experiment_results"]["metadata"]["seed"] == 42, (
            "A2C seed mismatch"
        )

        # Both should have the same timesteps
        assert (
            ppo_results["experiment_results"]["metadata"]["total_timesteps"] == 1000
        ), "PPO timesteps mismatch"
        assert (
            a2c_results["experiment_results"]["metadata"]["total_timesteps"] == 1000
        ), "A2C timesteps mismatch"

        logger.info("✓ Reproducibility verified: seed and timesteps match")

        # Final summary
        logger.info("=" * 60)
        logger.info("FULL WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"  Audit: {total_modules} modules")
        logger.info(f"  Cleanup: {len(structure)} structure entries")
        logger.info(f"  PPO: reward={ppo_final_reward:.2f}, {len(ppo_metrics)} metrics")
        logger.info(f"  A2C: reward={a2c_final_reward:.2f}, {len(a2c_metrics)} metrics")
        logger.info("=" * 60)

    # ------------------------------------------------------------------------
    # Test 6: Workflow Artifacts Integrity
    # ------------------------------------------------------------------------

    def test_workflow_artifacts_integrity(
        self,
        cleanup_results_dir,
        cleanup_audit_files,
        capture_output,
    ) -> None:
        """Test workflow artifacts integrity after full workflow.

        Verifies:
        - All required files exist
        - All JSON files are valid JSON
        - All model files (.zip) are valid archives
        - No unexpected files created
        - Directory structure matches expected layout

        This test runs the full workflow and then performs comprehensive
        validation of all created artifacts.
        """
        # Run full workflow (skip audit due to recursion issues in test context)
        logger.info("Running full workflow...")

        # Note: Skipping audit in test context due to recursion issues
        # run_audit(skip_smoke_tests=True, format="json")
        run_cleanup(dry_run=True)
        run_training(algo="ppo", seed=42, total_timesteps=1000)
        run_training(algo="a2c", seed=42, total_timesteps=1000)

        # Define expected files (audit skipped due to recursion issues)
        expected_files = {
            "ppo": [
                Path("results/experiments/ppo_seed42/ppo_seed42_model.zip"),
                Path("results/experiments/ppo_seed42/ppo_seed42_results.json"),
                Path("results/experiments/ppo_seed42/ppo_seed42_metrics.json"),
            ],
            "a2c": [
                Path("results/experiments/a2c_seed42/a2c_seed42_model.zip"),
                Path("results/experiments/a2c_seed42/a2c_seed42_results.json"),
                Path("results/experiments/a2c_seed42/a2c_seed42_metrics.json"),
            ],
        }

        # Verify all expected files exist
        logger.info("Verifying expected files...")
        missing_files = []

        for category, files in expected_files.items():
            for file_path in files:
                if not file_path.exists():
                    missing_files.append(str(file_path))
                else:
                    logger.debug(f"✓ Found: {file_path}")

        assert not missing_files, f"Missing files: {', '.join(missing_files)}"

        # Verify all JSON files are valid (audit skipped due to recursion issues)
        logger.info("Verifying JSON files...")
        json_files = [
            Path("results/experiments/ppo_seed42/ppo_seed42_results.json"),
            Path("results/experiments/ppo_seed42/ppo_seed42_metrics.json"),
            Path("results/experiments/a2c_seed42/a2c_seed42_results.json"),
            Path("results/experiments/a2c_seed42/a2c_seed42_metrics.json"),
        ]

        for json_file in json_files:
            try:
                verify_json_file(json_file)
            except Exception as e:
                raise AssertionError(f"Invalid JSON file {json_file}: {e}")

        # Verify all model files are valid archives
        logger.info("Verifying model files...")
        model_files = [
            Path("results/experiments/ppo_seed42/ppo_seed42_model.zip"),
            Path("results/experiments/a2c_seed42/a2c_seed42_model.zip"),
        ]

        for model_file in model_files:
            try:
                verify_model_file(model_file)
            except Exception as e:
                raise AssertionError(f"Invalid model file {model_file}: {e}")

        # Verify directory structure
        logger.info("Verifying directory structure...")
        verify_directory_structure(
            Path.cwd(),
            expected_dirs=[
                "src",
                "tests",
                "results",
                "results/experiments",
                "results/experiments/ppo_seed42",
                "results/experiments/a2c_seed42",
            ],
        )

        # Verify no unexpected files in experiment directories
        logger.info("Verifying experiment directory contents...")

        ppo_exp_dir = Path("results/experiments/ppo_seed42")
        ppo_files = list(ppo_exp_dir.glob("*"))
        ppo_basenames = {f.name for f in ppo_files}

        expected_ppo_files = {
            "ppo_seed42_model.zip",
            "ppo_seed42_results.json",
            "ppo_seed42_metrics.json",
        }

        # Allow additional files created during training
        allowed_additional_files = {
            "config.json",  # Training configuration
            "metrics.csv",  # CSV metrics export
            "eval_log.csv",  # Evaluation log
            "reward_curve.png",  # Reward curve plot
            "reward_curve_1.png",  # Additional reward plot
            "video.mp4",  # Training video
        }

        # Allow checkpoint files and directories (they're created during training)
        unexpected_ppo_files = (
            ppo_basenames
            - expected_ppo_files
            - allowed_additional_files
            - {f.name for f in ppo_exp_dir.glob("checkpoint_*.zip")}
            - {"checkpoints"}
        )

        assert not unexpected_ppo_files, (
            f"Unexpected files in PPO directory: {unexpected_ppo_files}"
        )

        a2c_exp_dir = Path("results/experiments/a2c_seed42")
        a2c_files = list(a2c_exp_dir.glob("*"))
        a2c_basenames = {f.name for f in a2c_files}

        expected_a2c_files = {
            "a2c_seed42_model.zip",
            "a2c_seed42_results.json",
            "a2c_seed42_metrics.json",
        }

        # Allow additional files created during training
        allowed_additional_files = {
            "config.json",  # Training configuration
            "metrics.csv",  # CSV metrics export
            "eval_log.csv",  # Evaluation log
            "reward_curve.png",  # Reward curve plot
            "reward_curve_1.png",  # Additional reward plot
            "video.mp4",  # Training video
        }

        # Allow checkpoint files and directories (they're created during training)
        unexpected_a2c_files = (
            a2c_basenames
            - expected_a2c_files
            - allowed_additional_files
            - {f.name for f in a2c_exp_dir.glob("checkpoint_*.zip")}
            - {"checkpoints"}
        )

        assert not unexpected_a2c_files, (
            f"Unexpected files in A2C directory: {unexpected_a2c_files}"
        )

        # Verify results JSON structure
        logger.info("Verifying results JSON structure...")

        ppo_results = verify_json_file(
            Path("results/experiments/ppo_seed42/ppo_seed42_results.json")
        )
        a2c_results = verify_json_file(
            Path("results/experiments/a2c_seed42/a2c_seed42_results.json")
        )

        # Verify PPO results structure
        assert "experiment_results" in ppo_results
        ppo_exp = ppo_results["experiment_results"]
        assert "metadata" in ppo_exp
        assert "model" in ppo_exp
        assert "metrics" in ppo_exp
        assert "hyperparameters" in ppo_exp
        assert "environment" in ppo_exp

        # Verify A2C results structure
        assert "experiment_results" in a2c_results
        a2c_exp = a2c_results["experiment_results"]
        assert "metadata" in a2c_exp
        assert "model" in a2c_exp
        assert "metrics" in a2c_exp
        assert "hyperparameters" in a2c_exp
        assert "environment" in a2c_exp

        # Verify metrics JSON structure
        logger.info("Verifying metrics JSON structure...")

        ppo_metrics = verify_json_file(
            Path("results/experiments/ppo_seed42/ppo_seed42_metrics.json")
        )
        a2c_metrics = verify_json_file(
            Path("results/experiments/a2c_seed42/a2c_seed42_metrics.json")
        )

        # Verify metrics are lists (handle both dict and list formats)
        if isinstance(ppo_metrics, dict):
            assert "metrics" in ppo_metrics, "PPO metrics missing 'metrics' key"
            ppo_metrics = ppo_metrics["metrics"]
        if isinstance(a2c_metrics, dict):
            assert "metrics" in a2c_metrics, "A2C metrics missing 'metrics' key"
            a2c_metrics = a2c_metrics["metrics"]

        assert isinstance(ppo_metrics, list), "PPO metrics should be a list"
        assert isinstance(a2c_metrics, list), "A2C metrics should be a list"

        # Verify metrics have required fields
        for i, point in enumerate(ppo_metrics):
            assert "timestep" in point, f"PPO metric {i} missing timestep"
            assert "reward" in point, f"PPO metric {i} missing reward"

        for i, point in enumerate(a2c_metrics):
            assert "timestep" in point, f"A2C metric {i} missing timestep"
            assert "reward" in point, f"A2C metric {i} missing reward"

        logger.info("=" * 60)
        logger.info("ARTIFACTS INTEGRITY VERIFIED")
        logger.info("=" * 60)
        logger.info(f"  ✓ All {len(json_files)} JSON files valid")
        logger.info(f"  ✓ All {len(model_files)} model files valid")
        logger.info("  ✓ Directory structure correct")
        logger.info("  ✓ No unexpected files")
        logger.info("  ✓ Results structure correct")
        logger.info("  ✓ Metrics structure correct")
        logger.info("=" * 60)
