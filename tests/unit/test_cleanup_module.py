"""Unit tests for cleanup module components."""

import pytest
from pathlib import Path
import tempfile
import os
import json

from src.cleanup.core import (
    CleanupConfig,
    CleanupAction,
    CleanupStatus,
    RootDirectoryAnalyzer,
    FileItem,
    CleanupActionItem,
    analyze_root_directory,
)
from src.cleanup.categorizer import (
    FileCategory,
    FileCategorizer,
    categorize_file,
)
from src.cleanup.executor import (
    CleanupExecutor,
    ProjectStructureReport,
    execute_cleanup,
)


class TestCleanupConfig:
    """Tests for CleanupConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CleanupConfig()
        assert "requirements.txt" in config.allowed_files
        assert "src/" in config.allowed_directories
        assert config.dry_run is False
        assert config.force is False
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CleanupConfig(
            allowed_files=["custom.txt"],
            allowed_directories=["custom/"],
            dry_run=True,
            force=True,
        )
        assert config.allowed_files == ["custom.txt"]
        assert config.allowed_directories == ["custom/"]
        assert config.dry_run is True
        assert config.force is True


class TestFileCategory:
    """Tests for FileCategory enum."""

    def test_all_categories(self):
        """Test all file categories exist."""
        categories = list(FileCategory)
        assert FileCategory.KEEP in categories
        assert FileCategory.MOVE_TO_SRC in categories
        assert FileCategory.MOVE_TO_TESTS in categories
        assert FileCategory.REMOVE in categories
        assert FileCategory.ARCHIVE in categories
        assert FileCategory.SKIP in categories


class TestFileCategorizer:
    """Tests for FileCategorizer class."""

    def test_categorize_allowed_file(self):
        """Test categorizing allowed files."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("requirements.txt"))
        assert category == FileCategory.KEEP

    def test_categorize_allowed_directory(self):
        """Test categorizing allowed directories."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("src/"))
        assert category == FileCategory.KEEP

    def test_categorize_python_file(self):
        """Test categorizing Python files to src."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("example_training.py"))
        assert category == FileCategory.MOVE_TO_SRC

    def test_categorize_test_file(self):
        """Test categorizing test files to tests."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("test_ppo_agent.py"))
        assert category == FileCategory.MOVE_TO_TESTS

    def test_categorize_remove_pattern(self):
        """Test categorizing files for removal."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("verify_setup.py"))
        assert category == FileCategory.REMOVE

    def test_categorize_archive_pattern(self):
        """Test categorizing files for archiving."""
        categorizer = FileCategorizer()

        category = categorizer.categorize(Path("info_project.md"))
        assert category == FileCategory.ARCHIVE

    def test_get_destination(self):
        """Test getting destination paths."""
        categorizer = FileCategorizer()

        # Move to src
        dest = categorizer.get_destination(
            Path("example.py"), FileCategory.MOVE_TO_SRC
        )
        assert dest == Path("src/example.py")

        # Move to tests
        dest = categorizer.get_destination(
            Path("test_example.py"), FileCategory.MOVE_TO_TESTS
        )
        assert dest == Path("tests/test_example.py")

        # Archive
        dest = categorizer.get_destination(
            Path("info.md"), FileCategory.ARCHIVE
        )
        assert dest == Path("docs/archives/info.md")

        # No destination for keep
        dest = categorizer.get_destination(
            Path("requirements.txt"), FileCategory.KEEP
        )
        assert dest is None


class TestCategorizeFile:
    """Tests for categorize_file function."""

    def test_categorize_existing_file(self):
        """Test categorizing an existing file."""
        if Path("requirements.txt").exists():
            category, dest = categorize_file(Path("requirements.txt"))
            assert category == FileCategory.KEEP
            assert dest is None

    def test_categorize_nonexistent_file(self):
        """Test categorizing a non-existent file."""
        category, dest = categorize_file(Path("/tmp/nonexistent_12345.py"))
        assert category == FileCategory.SKIP
        assert dest is None


class TestRootDirectoryAnalyzer:
    """Tests for RootDirectoryAnalyzer class."""

    def test_scan_root_with_allowed_items(self):
        """Test scanning root with allowed items."""
        config = CleanupConfig()
        analyzer = RootDirectoryAnalyzer(config)

        allowed, unexpected = analyzer.scan_root()

        # Check that allowed items are found
        allowed_names = [item.path.name for item in allowed]
        for name in config.allowed_files:
            if Path(name).exists():
                assert name in allowed_names

    def test_categorize_items(self):
        """Test categorizing items."""
        config = CleanupConfig()
        analyzer = RootDirectoryAnalyzer(config)

        # Create a temporary file to test
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="test_", delete=False
        ) as f:
            f.write("# test file")
            temp_path = Path(f.name)

        try:
            # Move to project root temporarily
            original_cwd = Path.cwd()
            import shutil

            # Create a test directory
            test_dir = Path("/tmp/cleanup_test")
            test_dir.mkdir(exist_ok=True)

            # Copy file to test directory
            shutil.copy(temp_path, test_dir / "test_example.py")

            # Change to test directory
            original_cwd = Path.cwd().resolve()
            os.chdir(test_dir)

            try:
                analyzer = RootDirectoryAnalyzer(config)
                allowed, unexpected = analyzer.scan_root()

                # Find test_example.py in unexpected
                test_items = [i for i in unexpected if "test_example" in i.path.name]
                assert len(test_items) > 0

                # Categorize
                actions = analyzer.categorize_items(test_items)
                assert len(actions) > 0
                assert actions[0].action == CleanupAction.MOVE
            finally:
                os.chdir(original_cwd)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestCleanupActionItem:
    """Tests for CleanupActionItem dataclass."""

    def test_create_move_action(self):
        """Test creating a move action."""
        action = CleanupActionItem(
            action=CleanupAction.MOVE,
            source=Path("example.py"),
            destination=Path("src/example.py"),
            reason="Matches move pattern",
        )
        assert action.action == CleanupAction.MOVE
        assert action.source == Path("example.py")
        assert action.destination == Path("src/example.py")
        assert action.reason == "Matches move pattern"
        assert action.status == CleanupStatus.PENDING

    def test_create_remove_action(self):
        """Test creating a remove action."""
        action = CleanupActionItem(
            action=CleanupAction.REMOVE,
            source=Path("temp.tmp"),
            reason="Temporary file",
        )
        assert action.action == CleanupAction.REMOVE
        assert action.destination is None


class TestProjectStructureReport:
    """Tests for ProjectStructureReport class."""

    def test_default_report(self):
        """Test default report values."""
        report = ProjectStructureReport()

        assert "timestamp" in report.metadata
        assert report.metadata["cleanup_status"] == "completed"
        assert len(report.cleanup_actions) == 0

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = ProjectStructureReport()
        report.root_directory = {
            "allowed_files": ["requirements.txt"],
            "allowed_directories": ["src/"],
            "actual_files": ["requirements.txt"],
            "actual_directories": ["src/"],
            "validation_status": "clean",
        }
        report.cleanup_actions = [
            {"action": "moved", "source": "test.py", "destination": "src/test.py"}
        ]

        data = report.to_dict()

        assert "project_structure" in data
        assert data["project_structure"]["root_directory"]["validation_status"] == "clean"
        assert len(data["project_structure"]["cleanup_actions"]) == 1

    def test_save_report(self):
        """Test saving report to file."""
        report = ProjectStructureReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.json"
            report.save(filepath)

            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)

            assert "project_structure" in data


class TestCleanupExecutor:
    """Tests for CleanupExecutor class."""

    def test_dry_run_mode(self):
        """Test executor in dry run mode."""
        config = CleanupConfig()
        executor = CleanupExecutor(config, dry_run=True)

        assert executor.dry_run is True
        assert executor.report.metadata["cleanup_status"] == "dry_run"

    def test_execute_dry_run(self):
        """Test execute in dry run mode."""
        config = CleanupConfig()
        config.allowed_files = ["requirements.txt"]
        config.allowed_directories = ["src/"]

        executor = CleanupExecutor(config, dry_run=True)

        # Create temp file to test
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="test_", delete=False
        ) as f:
            f.write("# test")
            temp_file = Path(f.name)

        try:
            original_cwd = Path.cwd().resolve()
            os.chdir(temp_file.parent)

            try:
                report = executor.execute()

                # In dry run, no actual cleanup should happen
                assert report.metadata["cleanup_status"] == "dry_run"
            finally:
                os.chdir(original_cwd)
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestExecuteCleanup:
    """Tests for execute_cleanup function."""

    def test_execute_cleanup_dry_run(self):
        """Test execute_cleanup in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CleanupConfig()
            config.allowed_files = ["requirements.txt"]
            config.allowed_directories = ["src/"]

            # Create temp requirements.txt
            (Path(tmpdir) / "requirements.txt").write_text("pytest")

            report = execute_cleanup(
                config=config,
                dry_run=True,
                output_path=Path(tmpdir) / "test_report.json",
            )

            assert report.metadata["cleanup_status"] == "dry_run"
            assert (Path(tmpdir) / "test_report.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
