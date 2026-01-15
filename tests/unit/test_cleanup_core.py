"""Unit tests for cleanup core module.

Tests for FILE_CATEGORIES enum and constants.
"""

import pytest
from pathlib import Path

from src.cleanup.core import (
    FileCategory,
    DirCategory,
    KEEP_ROOT_ITEMS,
    REMOVE_ROOT_DIRS,
    REMOVE_ROOT_FILES,
    KEEP_ROOT_FILES_EXCEPTIONS,
    REMOVE_SRC_DIRS,
    KEEP_SRC_DIRS,
    REMOVE_SRC_FILES,
    PROTECTED_DIRECTORIES,
    PROTECTED_FILES,
    BACKUP_DIR,
    VALIDATION_PATHS,
    get_keep_root_items,
    get_remove_root_dirs,
    get_remove_root_files,
    get_keep_root_files_exceptions,
    get_remove_src_dirs,
    get_keep_src_dirs,
    get_remove_src_files,
    get_protected_directories,
    get_protected_files,
    get_backup_dir,
    get_validation_paths,
    is_protected,
    is_keep_root_item,
    is_remove_root_dir,
    is_remove_root_file,
    is_remove_src_dir,
    is_keep_src_dir,
    is_remove_src_file,
)


class TestFileCategory:
    """Tests for FileCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test that all expected file categories exist."""
        categories = list(FileCategory)
        assert FileCategory.KEEP in categories
        assert FileCategory.REMOVE in categories
        assert FileCategory.ARCHIVE in categories

    def test_category_values(self) -> None:
        """Test that category values are correct."""
        assert FileCategory.KEEP.value == "keep"
        assert FileCategory.REMOVE.value == "remove"
        assert FileCategory.ARCHIVE.value == "archive"


class TestDirCategory:
    """Tests for DirCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test that all expected directory categories exist."""
        categories = list(DirCategory)
        assert DirCategory.KEEP in categories
        assert DirCategory.REMOVE in categories
        assert DirCategory.ARCHIVE in categories
        assert DirCategory.PROTECTED in categories

    def test_category_values(self) -> None:
        """Test that category values are correct."""
        assert DirCategory.KEEP.value == "keep"
        assert DirCategory.REMOVE.value == "remove"
        assert DirCategory.ARCHIVE.value == "archive"
        assert DirCategory.PROTECTED.value == "protected"


class TestKeepRootItems:
    """Tests for KEEP_ROOT_ITEMS constant."""

    def test_keep_root_items_non_empty(self) -> None:
        """Test that KEEP_ROOT_ITEMS is not empty."""
        assert len(KEEP_ROOT_ITEMS) > 0

    def test_keep_root_items_contains_expected(self) -> None:
        """Test that KEEP_ROOT_ITEMS contains expected items."""
        assert "README.md" in KEEP_ROOT_ITEMS
        assert "requirements.txt" in KEEP_ROOT_ITEMS
        assert ".gitignore" in KEEP_ROOT_ITEMS
        assert "pytest.ini" in KEEP_ROOT_ITEMS
        assert "src/" in KEEP_ROOT_ITEMS
        assert "tests/" in KEEP_ROOT_ITEMS
        assert "results/" in KEEP_ROOT_ITEMS
        assert "specs/" in KEEP_ROOT_ITEMS

    def test_get_keep_root_items(self) -> None:
        """Test get_keep_root_items function."""
        items = get_keep_root_items()
        assert isinstance(items, list)
        assert len(items) > 0
        assert "README.md" in items


class TestRemoveRootDirs:
    """Tests for REMOVE_ROOT_DIRS constant."""

    def test_remove_root_dirs_non_empty(self) -> None:
        """Test that REMOVE_ROOT_DIRS is not empty."""
        assert len(REMOVE_ROOT_DIRS) > 0

    def test_remove_root_dirs_contains_expected(self) -> None:
        """Test that REMOVE_ROOT_DIRS contains expected directories."""
        assert "configs/" in REMOVE_ROOT_DIRS
        assert "data/" in REMOVE_ROOT_DIRS
        assert "demo_checkpoints/" in REMOVE_ROOT_DIRS
        assert "docs/" in REMOVE_ROOT_DIRS
        assert "examples/" in REMOVE_ROOT_DIRS
        assert "logs/" in REMOVE_ROOT_DIRS
        assert "notebooks/" in REMOVE_ROOT_DIRS
        assert "scripts/" in REMOVE_ROOT_DIRS

    def test_get_remove_root_dirs(self) -> None:
        """Test get_remove_root_dirs function."""
        dirs = get_remove_root_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) > 0
        assert "configs/" in dirs


class TestRemoveRootFiles:
    """Tests for REMOVE_ROOT_FILES constant."""

    def test_remove_root_files_non_empty(self) -> None:
        """Test that REMOVE_ROOT_FILES is not empty."""
        assert len(REMOVE_ROOT_FILES) > 0

    def test_remove_root_files_contains_expected(self) -> None:
        """Test that REMOVE_ROOT_FILES contains expected patterns."""
        assert "*.md" in REMOVE_ROOT_FILES
        assert "*.py" in REMOVE_ROOT_FILES
        assert "environment.yml" in REMOVE_ROOT_FILES
        assert ".env.example" in REMOVE_ROOT_FILES
        assert "install.sh" in REMOVE_ROOT_FILES
        assert "Makefile" in REMOVE_ROOT_FILES
        assert ".jupyter_ystore.db" in REMOVE_ROOT_FILES

    def test_get_remove_root_files(self) -> None:
        """Test get_remove_root_files function."""
        files = get_remove_root_files()
        assert isinstance(files, list)
        assert len(files) > 0
        assert "*.md" in files


class TestKeepRootFilesExceptions:
    """Tests for KEEP_ROOT_FILES_EXCEPTIONS constant."""

    def test_keep_root_files_exceptions_non_empty(self) -> None:
        """Test that KEEP_ROOT_FILES_EXCEPTIONS is not empty."""
        assert len(KEEP_ROOT_FILES_EXCEPTIONS) > 0

    def test_keep_root_files_exceptions_contains_expected(self) -> None:
        """Test that KEEP_ROOT_FILES_EXCEPTIONS contains expected files."""
        assert "README.md" in KEEP_ROOT_FILES_EXCEPTIONS
        assert "AGENTS.md" in KEEP_ROOT_FILES_EXCEPTIONS

    def test_get_keep_root_files_exceptions(self) -> None:
        """Test get_keep_root_files_exceptions function."""
        exceptions = get_keep_root_files_exceptions()
        assert isinstance(exceptions, list)
        assert len(exceptions) > 0
        assert "README.md" in exceptions


class TestRemoveSrcDirs:
    """Tests for REMOVE_SRC_DIRS constant."""

    def test_remove_src_dirs_non_empty(self) -> None:
        """Test that REMOVE_SRC_DIRS is not empty."""
        assert len(REMOVE_SRC_DIRS) > 0

    def test_remove_src_dirs_contains_expected(self) -> None:
        """Test that REMOVE_SRC_DIRS contains expected directories."""
        assert "src/api/" in REMOVE_SRC_DIRS
        assert "src/visualization/" in REMOVE_SRC_DIRS

    def test_get_remove_src_dirs(self) -> None:
        """Test get_remove_src_dirs function."""
        dirs = get_remove_src_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) > 0
        assert "src/api/" in dirs


class TestKeepSrcDirs:
    """Tests for KEEP_SRC_DIRS constant."""

    def test_keep_src_dirs_non_empty(self) -> None:
        """Test that KEEP_SRC_DIRS is not empty."""
        assert len(KEEP_SRC_DIRS) > 0

    def test_keep_src_dirs_contains_expected(self) -> None:
        """Test that KEEP_SRC_DIRS contains expected directories."""
        assert "src/training/" in KEEP_SRC_DIRS
        assert "src/agents/" in KEEP_SRC_DIRS
        assert "src/utils/" in KEEP_SRC_DIRS
        assert "src/environments/" in KEEP_SRC_DIRS
        assert "src/audit/" in KEEP_SRC_DIRS
        assert "src/cleanup/" in KEEP_SRC_DIRS

    def test_get_keep_src_dirs(self) -> None:
        """Test get_keep_src_dirs function."""
        dirs = get_keep_src_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) > 0
        assert "src/training/" in dirs


class TestRemoveSrcFiles:
    """Tests for REMOVE_SRC_FILES constant."""

    def test_remove_src_files_non_empty(self) -> None:
        """Test that REMOVE_SRC_FILES is not empty."""
        assert len(REMOVE_SRC_FILES) > 0

    def test_remove_src_files_contains_expected(self) -> None:
        """Test that REMOVE_SRC_FILES contains expected files."""
        assert "src/utils/rl_logging.py" in REMOVE_SRC_FILES
        assert "src/utils/logging_setup.py" in REMOVE_SRC_FILES

    def test_get_remove_src_files(self) -> None:
        """Test get_remove_src_files function."""
        files = get_remove_src_files()
        assert isinstance(files, list)
        assert len(files) > 0
        assert "src/utils/rl_logging.py" in files


class TestProtectedDirectories:
    """Tests for PROTECTED_DIRECTORIES constant."""

    def test_protected_directories_non_empty(self) -> None:
        """Test that PROTECTED_DIRECTORIES is not empty."""
        assert len(PROTECTED_DIRECTORIES) > 0

    def test_protected_directories_contains_expected(self) -> None:
        """Test that PROTECTED_DIRECTORIES contains expected directories."""
        assert ".git/" in PROTECTED_DIRECTORIES
        assert "src/" in PROTECTED_DIRECTORIES

    def test_get_protected_directories(self) -> None:
        """Test get_protected_directories function."""
        dirs = get_protected_directories()
        assert isinstance(dirs, list)
        assert len(dirs) > 0
        assert ".git/" in dirs


class TestProtectedFiles:
    """Tests for PROTECTED_FILES constant."""

    def test_protected_files_non_empty(self) -> None:
        """Test that PROTECTED_FILES is not empty."""
        assert len(PROTECTED_FILES) > 0

    def test_protected_files_contains_expected(self) -> None:
        """Test that PROTECTED_FILES contains expected files."""
        assert ".gitignore" in PROTECTED_FILES
        assert "pytest.ini" in PROTECTED_FILES

    def test_get_protected_files(self) -> None:
        """Test get_protected_files function."""
        files = get_protected_files()
        assert isinstance(files, list)
        assert len(files) > 0
        assert ".gitignore" in files


class TestBackupDir:
    """Tests for BACKUP_DIR constant."""

    def test_backup_dir_is_path(self) -> None:
        """Test that BACKUP_DIR is a Path object."""
        assert isinstance(BACKUP_DIR, Path)

    def test_backup_dir_value(self) -> None:
        """Test that BACKUP_DIR has correct value."""
        assert BACKUP_DIR == Path("results/cleanup_backups")

    def test_get_backup_dir(self) -> None:
        """Test get_backup_dir function."""
        backup_dir = get_backup_dir()
        assert isinstance(backup_dir, Path)
        assert backup_dir == Path("results/cleanup_backups")


class TestValidationPaths:
    """Tests for VALIDATION_PATHS constant."""

    def test_validation_paths_non_empty(self) -> None:
        """Test that VALIDATION_PATHS is not empty."""
        assert len(VALIDATION_PATHS) > 0

    def test_validation_paths_contains_expected(self) -> None:
        """Test that VALIDATION_PATHS contains expected paths."""
        assert Path("src/training/") in VALIDATION_PATHS
        assert Path("tests/") in VALIDATION_PATHS

    def test_get_validation_paths(self) -> None:
        """Test get_validation_paths function."""
        paths = get_validation_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert Path("src/training/") in paths


class TestIsProtected:
    """Tests for is_protected function."""

    def test_protected_git_directory(self) -> None:
        """Test that .git directory is protected."""
        assert is_protected(Path(".git"))

    def test_protected_src_directory(self) -> None:
        """Test that src directory is protected."""
        assert is_protected(Path("src"))

    def test_protected_gitignore_file(self) -> None:
        """Test that .gitignore file is protected."""
        assert is_protected(Path(".gitignore"))

    def test_protected_pytest_ini_file(self) -> None:
        """Test that pytest.ini file is protected."""
        assert is_protected(Path("pytest.ini"))

    def test_non_protected_file(self) -> None:
        """Test that non-protected file is not protected."""
        assert not is_protected(Path("example.py"))

    def test_non_protected_directory(self) -> None:
        """Test that non-protected directory is not protected."""
        assert not is_protected(Path("configs"))


class TestIsKeepRootItem:
    """Tests for is_keep_root_item function."""

    def test_keep_readme(self) -> None:
        """Test that README.md is a keep root item."""
        assert is_keep_root_item(Path("README.md"))

    def test_keep_requirements(self) -> None:
        """Test that requirements.txt is a keep root item."""
        assert is_keep_root_item(Path("requirements.txt"))

    def test_keep_gitignore(self) -> None:
        """Test that .gitignore is a keep root item."""
        assert is_keep_root_item(Path(".gitignore"))

    def test_keep_pytest_ini(self) -> None:
        """Test that pytest.ini is a keep root item."""
        assert is_keep_root_item(Path("pytest.ini"))

    def test_keep_src_directory(self) -> None:
        """Test that src/ is a keep root item."""
        assert is_keep_root_item(Path("src"))

    def test_keep_tests_directory(self) -> None:
        """Test that tests/ is a keep root item."""
        assert is_keep_root_item(Path("tests"))

    def test_not_keep_example_py(self) -> None:
        """Test that example.py is not a keep root item."""
        assert not is_keep_root_item(Path("example.py"))


class TestIsRemoveRootDir:
    """Tests for is_remove_root_dir function."""

    def test_remove_configs_directory(self) -> None:
        """Test that configs/ is a remove root directory."""
        assert is_remove_root_dir(Path("configs"))

    def test_remove_data_directory(self) -> None:
        """Test that data/ is a remove root directory."""
        assert is_remove_root_dir(Path("data"))

    def test_remove_demo_checkpoints_directory(self) -> None:
        """Test that demo_checkpoints/ is a remove root directory."""
        assert is_remove_root_dir(Path("demo_checkpoints"))

    def test_remove_docs_directory(self) -> None:
        """Test that docs/ is a remove root directory."""
        assert is_remove_root_dir(Path("docs"))

    def test_remove_examples_directory(self) -> None:
        """Test that examples/ is a remove root directory."""
        assert is_remove_root_dir(Path("examples"))

    def test_not_remove_src_directory(self) -> None:
        """Test that src/ is not a remove root directory."""
        assert not is_remove_root_dir(Path("src"))

    def test_not_remove_file(self) -> None:
        """Test that a file is not a remove root directory."""
        assert not is_remove_root_dir(Path("README.md"))


class TestIsRemoveRootFile:
    """Tests for is_remove_root_file function."""

    def test_remove_md_files(self) -> None:
        """Test that .md files (except exceptions) are remove root files."""
        assert is_remove_root_file(Path("test.md"))
        assert is_remove_root_file(Path("SUMMARY.md"))

    def test_remove_py_files(self) -> None:
        """Test that .py files are remove root files."""
        assert is_remove_root_file(Path("example.py"))
        assert is_remove_root_file(Path("test.py"))

    def test_remove_environment_yml(self) -> None:
        """Test that environment.yml is a remove root file."""
        assert is_remove_root_file(Path("environment.yml"))

    def test_remove_env_example(self) -> None:
        """Test that .env.example is a remove root file."""
        assert is_remove_root_file(Path(".env.example"))

    def test_remove_install_sh(self) -> None:
        """Test that install.sh is a remove root file."""
        assert is_remove_root_file(Path("install.sh"))

    def test_remove_makefile(self) -> None:
        """Test that Makefile is a remove root file."""
        assert is_remove_root_file(Path("Makefile"))

    def test_not_remove_readme(self) -> None:
        """Test that README.md is not a remove root file (exception)."""
        assert not is_remove_root_file(Path("README.md"))

    def test_not_remove_agents_md(self) -> None:
        """Test that AGENTS.md is not a remove root file (exception)."""
        assert not is_remove_root_file(Path("AGENTS.md"))

    def test_not_remove_directory(self) -> None:
        """Test that a directory is not a remove root file."""
        assert not is_remove_root_file(Path("src"))


class TestIsRemoveSrcDir:
    """Tests for is_remove_src_dir function."""

    def test_remove_api_directory(self) -> None:
        """Test that src/api/ is a remove source directory."""
        assert is_remove_src_dir(Path("src/api"))

    def test_remove_visualization_directory(self) -> None:
        """Test that src/visualization/ is a remove source directory."""
        assert is_remove_src_dir(Path("src/visualization"))

    def test_not_remove_training_directory(self) -> None:
        """Test that src/training/ is not a remove source directory."""
        assert not is_remove_src_dir(Path("src/training"))

    def test_not_remove_agents_directory(self) -> None:
        """Test that src/agents/ is not a remove source directory."""
        assert not is_remove_src_dir(Path("src/agents"))

    def test_not_remove_utils_directory(self) -> None:
        """Test that src/utils/ is not a remove source directory."""
        assert not is_remove_src_dir(Path("src/utils"))

    def test_not_remove_non_src_directory(self) -> None:
        """Test that a non-src directory is not a remove source directory."""
        assert not is_remove_src_dir(Path("tests"))


class TestIsKeepSrcDir:
    """Tests for is_keep_src_dir function."""

    def test_keep_training_directory(self) -> None:
        """Test that src/training/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/training"))

    def test_keep_agents_directory(self) -> None:
        """Test that src/agents/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/agents"))

    def test_keep_utils_directory(self) -> None:
        """Test that src/utils/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/utils"))

    def test_keep_environments_directory(self) -> None:
        """Test that src/environments/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/environments"))

    def test_keep_audit_directory(self) -> None:
        """Test that src/audit/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/audit"))

    def test_keep_cleanup_directory(self) -> None:
        """Test that src/cleanup/ is a keep source directory."""
        assert is_keep_src_dir(Path("src/cleanup"))

    def test_not_keep_api_directory(self) -> None:
        """Test that src/api/ is not a keep source directory."""
        assert not is_keep_src_dir(Path("src/api"))

    def test_not_keep_visualization_directory(self) -> None:
        """Test that src/visualization/ is not a keep source directory."""
        assert not is_keep_src_dir(Path("src/visualization"))

    def test_not_keep_non_src_directory(self) -> None:
        """Test that a non-src directory is not a keep source directory."""
        assert not is_keep_src_dir(Path("tests"))


class TestIsRemoveSrcFile:
    """Tests for is_remove_src_file function."""

    def test_remove_logging_py(self) -> None:
        """Test that src/utils/rl_logging.py is a remove source file."""
        assert is_remove_src_file(Path("src/utils/rl_logging.py"))

    def test_remove_logging_config_py(self) -> None:
        """Test that src/utils/logging_setup.py is a remove source file."""
        assert is_remove_src_file(Path("src/utils/logging_setup.py"))

    def test_not_remove_seeding_py(self) -> None:
        """Test that src/utils/seeding.py is not a remove source file."""
        assert not is_remove_src_file(Path("src/utils/seeding.py"))

    def test_not_remove_non_src_file(self) -> None:
        """Test that a non-src file is not a remove source file."""
        assert not is_remove_src_file(Path("example.py"))

    def test_not_remove_directory(self) -> None:
        """Test that a directory is not a remove source file."""
        assert not is_remove_src_file(Path("src/utils"))


class TestNoOverlapBetweenKeepAndRemove:
    """Tests for no overlap between KEEP and REMOVE lists."""

    def test_no_overlap_root_items_and_dirs(self) -> None:
        """Test that there's no overlap between KEEP_ROOT_ITEMS and REMOVE_ROOT_DIRS."""
        keep_dirs = [item.rstrip("/") for item in KEEP_ROOT_ITEMS if item.endswith("/")]
        remove_dirs = [item.rstrip("/") for item in REMOVE_ROOT_DIRS]
        overlap = set(keep_dirs) & set(remove_dirs)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_no_overlap_src_dirs(self) -> None:
        """Test that there's no overlap between KEEP_SRC_DIRS and REMOVE_SRC_DIRS."""
        keep_dirs = [item.rstrip("/") for item in KEEP_SRC_DIRS]
        remove_dirs = [item.rstrip("/") for item in REMOVE_SRC_DIRS]
        overlap = set(keep_dirs) & set(remove_dirs)
        assert len(overlap) == 0, f"Overlap found: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
