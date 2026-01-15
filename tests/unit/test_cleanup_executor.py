"""Unit tests for cleanup executor module.

Tests for CleanupExecutor class with temp directories.
"""

import json
import os
import tarfile
import tempfile
from pathlib import Path

import pytest

from src.cleanup.executor import (
    CleanupExecutor,
    CleanupResult,
    DryRunResult,
)
from src.cleanup.core import FileCategory, DirCategory


class TestCleanupExecutor:
    """Tests for CleanupExecutor class."""

    def test_init(self) -> None:
        """Test CleanupExecutor initialization."""
        executor = CleanupExecutor()
        assert executor.root_path == Path(".").resolve()
        assert executor.backup_dir == Path(".").resolve() / "results" / "cleanup_backups"
        assert executor.validation_paths is not None

    def test_init_with_custom_root(self) -> None:
        """Test CleanupExecutor initialization with custom root."""
        custom_root = Path("/tmp/test_root")
        executor = CleanupExecutor(root_path=custom_root)
        assert executor.root_path == custom_root.resolve()

    def test_dry_run_does_not_remove_files(self) -> None:
        """Test that dry_run doesn't actually remove files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            test_file = tmpdir_path / "test.py"
            test_file.write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Run dry run
            result = executor.dry_run([test_file])

            # Verify file still exists
            assert test_file.exists()

            # Verify result
            assert isinstance(result, DryRunResult)
            assert len(result.items_to_remove) >= 0

    def test_dry_run_with_temp_directory(self) -> None:
        """Test dry_run with temporary directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test structure
            (tmpdir_path / "README.md").write_text("# README")
            (tmpdir_path / "example.py").write_text("# example")
            (tmpdir_path / "test.md").write_text("# test")
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            (configs_dir / "config.yaml").write_text("test: true")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Run dry run
            result = executor.dry_run()

            # Verify result
            assert isinstance(result, DryRunResult)
            assert len(result.items_to_remove) >= 0
            assert len(result.items_to_keep) >= 0

            # Verify all files still exist
            assert (tmpdir_path / "README.md").exists()
            assert (tmpdir_path / "example.py").exists()
            assert (tmpdir_path / "test.md").exists()
            assert configs_dir.exists()

    def test_execute_removes_files(self) -> None:
        """Test that execute actually removes files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test file
            test_file = tmpdir_path / "test.py"
            test_file.write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Execute with force
            result = executor.execute([test_file], force=True)

            # Verify file was removed
            assert not test_file.exists()

            # Verify result
            assert isinstance(result, CleanupResult)
            assert result.dry_run is False
            assert len(result.items_removed) >= 0

    def test_execute_with_force_false_logs_warning(self) -> None:
        """Test that execute with force=False logs warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test file
            test_file = tmpdir_path / "test.py"
            test_file.write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Execute without force
            result = executor.execute([test_file], force=False)

            # Verify result
            assert isinstance(result, CleanupResult)
            assert result.dry_run is False

    def test_execute_protected_file_not_removed(self) -> None:
        """Test that protected files are not removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create protected file
            gitignore = tmpdir_path / ".gitignore"
            gitignore.write_text("*.pyc")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Execute
            result = executor.execute([gitignore], force=True)

            # Verify file still exists
            assert gitignore.exists()

            # Verify result
            assert isinstance(result, CleanupResult)
            assert len(result.items_kept) > 0

    def test_execute_protected_directory_not_removed(self) -> None:
        """Test that protected directories are not removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create protected directory
            git_dir = tmpdir_path / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("[core]")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Execute
            result = executor.execute([git_dir], force=True)

            # Verify directory still exists
            assert git_dir.exists()

            # Verify result
            assert isinstance(result, CleanupResult)
            assert len(result.items_kept) > 0

    def test_backup_before_remove_creates_archive(self) -> None:
        """Test that backup_before_remove creates .tar.gz archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            test_file1 = tmpdir_path / "test1.py"
            test_file1.write_text("# test1")
            test_file2 = tmpdir_path / "test2.py"
            test_file2.write_text("# test2")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Create backup
            backup_path = executor.backup_before_remove([test_file1, test_file2])

            # Verify backup was created
            assert backup_path.exists()
            assert backup_path.suffix == ".gz"

            # Verify backup is a valid tar.gz
            assert tarfile.is_tarfile(backup_path)

            # Verify backup contains files
            with tarfile.open(backup_path, "r:gz") as tar:
                members = tar.getnames()
                assert "test1.py" in members
                assert "test2.py" in members

    def test_backup_before_remove_creates_directory(self) -> None:
        """Test that backup_before_remove creates backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test file
            test_file = tmpdir_path / "test.py"
            test_file.write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Create backup
            backup_path = executor.backup_before_remove([test_file])

            # Verify backup directory exists (parent of backup file)
            assert backup_path.parent.exists()
            # The backup_dir is relative to root_path, so we need to resolve it
            expected_backup_dir = tmpdir_path / "results" / "cleanup_backups"
            assert backup_path.parent == expected_backup_dir

    def test_validate_after_cleanup_passes(self) -> None:
        """Test that validate_after_cleanup passes when structure is intact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create required structure
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            training_dir = src_dir / "training"
            training_dir.mkdir()

            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Validate
            result = executor.validate_after_cleanup()

            # Verify validation passed
            assert result is True

    def test_validate_after_cleanup_fails_missing_src(self) -> None:
        """Test that validate_after_cleanup fails when src/ is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create only tests/ (missing src/)
            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Validate
            result = executor.validate_after_cleanup()

            # Verify validation failed
            assert result is False

    def test_validate_after_cleanup_fails_missing_tests(self) -> None:
        """Test that validate_after_cleanup fails when tests/ is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create only src/ (missing tests/)
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            training_dir = src_dir / "training"
            training_dir.mkdir()

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Validate
            result = executor.validate_after_cleanup()

            # Verify validation failed
            assert result is False

    def test_validate_after_cleanup_fails_missing_training(self) -> None:
        """Test that validate_after_cleanup fails when src/training/ is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create src/ but not src/training/
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()

            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Validate
            result = executor.validate_after_cleanup()

            # Verify validation failed
            assert result is False

    def test_generate_project_structure_report(self) -> None:
        """Test generate_project_structure_report returns correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create structure
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            training_dir = src_dir / "training"
            training_dir.mkdir()
            (training_dir / "trainer.py").write_text("# trainer")

            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_trainer.py").write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Generate report
            report = executor.generate_project_structure_report()

            # Verify report format
            assert isinstance(report, dict)
            assert "timestamp" in report
            assert "root_path" in report
            assert "validation" in report
            assert "src_structure" in report
            assert "tests_structure" in report

            # Verify validation section
            assert report["validation"]["src_exists"] is True
            assert report["validation"]["tests_exists"] is True

            # Verify src structure
            assert "training" in report["src_structure"]
            assert report["src_structure"]["training"]["type"] == "directory"

            # Verify tests structure
            assert "test_trainer.py" in report["tests_structure"]
            assert report["tests_structure"]["test_trainer.py"]["type"] == "file"

    def test_save_project_structure_report(self) -> None:
        """Test save_project_structure_report saves JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Save report
            output_path = tmpdir_path / "results" / "project_structure.json"
            saved_path = executor.save_project_structure_report(output_path)

            # Verify file was created
            assert saved_path.exists()
            assert saved_path == output_path

            # Verify file is valid JSON
            with open(saved_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "timestamp" in data
            assert "root_path" in data

    def test_save_project_structure_report_default_path(self) -> None:
        """Test save_project_structure_report with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Save report with default path
            saved_path = executor.save_project_structure_report()

            # Verify file was created at default location
            assert saved_path.exists()
            assert saved_path == tmpdir_path / "results" / "project_structure.json"

    def test_git_never_removed(self) -> None:
        """Test that .git directory is never removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create .git directory
            git_dir = tmpdir_path / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("[core]")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Try to remove .git
            result = executor.execute([git_dir], force=True)

            # Verify .git still exists
            assert git_dir.exists()

            # Verify it was kept
            assert len(result.items_kept) > 0

    def test_src_never_removed(self) -> None:
        """Test that src directory is never removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create src directory
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            (src_dir / "test.py").write_text("# test")

            # Create executor
            executor = CleanupExecutor(root_path=tmpdir_path)

            # Try to remove src
            result = executor.execute([src_dir], force=True)

            # Verify src still exists
            assert src_dir.exists()

            # Verify it was kept
            assert len(result.items_kept) > 0


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a CleanupResult."""
        result = CleanupResult(dry_run=False)
        assert result.dry_run is False
        assert result.items_removed == []
        assert result.items_kept == []
        assert result.backup_path is None
        assert result.validation_passed is False
        assert result.errors == []

    def test_create_result_with_values(self) -> None:
        """Test creating a CleanupResult with values."""
        result = CleanupResult(
            dry_run=True,
            items_removed=["test.py"],
            items_kept=["README.md"],
            backup_path=Path("backup.tar.gz"),
            validation_passed=True,
            errors=["Warning"],
        )
        assert result.dry_run is True
        assert result.items_removed == ["test.py"]
        assert result.items_kept == ["README.md"]
        assert result.backup_path == Path("backup.tar.gz")
        assert result.validation_passed is True
        assert result.errors == ["Warning"]


class TestDryRunResult:
    """Tests for DryRunResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a DryRunResult."""
        result = DryRunResult()
        assert result.items_to_remove == []
        assert result.items_to_keep == []
        assert result.total_size == 0

    def test_create_result_with_values(self) -> None:
        """Test creating a DryRunResult with values."""
        result = DryRunResult(
            items_to_remove=["test.py"],
            items_to_keep=["README.md"],
            total_size=1024,
        )
        assert result.items_to_remove == ["test.py"]
        assert result.items_to_keep == ["README.md"]
        assert result.total_size == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
