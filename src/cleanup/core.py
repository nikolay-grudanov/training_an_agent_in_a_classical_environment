"""Cleanup module core - Configuration constants and rules.

This module defines the cleanup rules for the RL project, including:
- Items to keep in root directory
- Directories and files to remove
- File categorization rules
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """Categories for file categorization during cleanup.

    Attributes:
        KEEP: File should be kept in place
        REMOVE: File should be removed
        ARCHIVE: File should be archived
    """

    KEEP = "keep"
    REMOVE = "remove"
    ARCHIVE = "archive"


class DirCategory(Enum):
    """Categories for directory categorization during cleanup.

    Attributes:
        KEEP: Directory should be kept
        REMOVE: Directory should be removed
        ARCHIVE: Directory should be archived
        PROTECTED: Directory is protected and cannot be removed
    """

    KEEP = "keep"
    REMOVE = "remove"
    ARCHIVE = "archive"
    PROTECTED = "protected"


# Root directory items to keep
KEEP_ROOT_ITEMS: Final = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "pytest.ini",
    "AGENTS.md",
    "src/",
    "tests/",
    "results/",
    "specs/",
]

# Root directories to remove
REMOVE_ROOT_DIRS: Final = [
    "configs/",
    "data/",
    "demo_checkpoints/",
    "docs/",
    "examples/",
    "logs/",
    "notebooks/",
    "scripts/",
]

# Root files to remove (all .md except README.md and AGENTS.md)
REMOVE_ROOT_FILES: Final = [
    "*.md",  # Will be filtered to exclude README.md and AGENTS.md
    "*.py",  # All Python files in root
    "*.png",  # Test/demo images
    "*.json",  # Audit reports and test files
    "environment.yml",
    ".env.example",
    "install.sh",
    "Makefile",
    ".jupyter_ystore.db",
]

# Root files to explicitly keep (exceptions to patterns above)
KEEP_ROOT_FILES_EXCEPTIONS: Final = [
    "README.md",
    "AGENTS.md",
]

# Source directories to remove
REMOVE_SRC_DIRS: Final = [
    "src/api/",
    "src/visualization/",
]

# Source directories to keep
KEEP_SRC_DIRS: Final = [
    "src/training/",
    "src/agents/",
    "src/utils/",
    "src/environments/",
    "src/audit/",
    "src/cleanup/",
]

# Source files to remove
REMOVE_SRC_FILES: Final = [
    "src/utils/rl_logging.py",
    "src/utils/logging_setup.py",
]

# Protected directories that should never be removed
PROTECTED_DIRECTORIES: Final = [
    ".git/",
    "src/",
]

# Protected files that should never be removed
PROTECTED_FILES: Final = [
    ".gitignore",
    "pytest.ini",
]

# Backup directory for cleanup operations
BACKUP_DIR: Final = Path("results/cleanup_backups")

# Validation paths that must exist after cleanup
VALIDATION_PATHS: Final = [
    Path("src/training/"),
    Path("tests/"),
]


def get_keep_root_items() -> list[str]:
    """Get list of root items to keep.

    Returns:
        List of root items that should be preserved.
    """
    return KEEP_ROOT_ITEMS.copy()


def get_remove_root_dirs() -> list[str]:
    """Get list of root directories to remove.

    Returns:
        List of root directories that should be removed.
    """
    return REMOVE_ROOT_DIRS.copy()


def get_remove_root_files() -> list[str]:
    """Get list of root files to remove.

    Returns:
        List of root file patterns that should be removed.
    """
    return REMOVE_ROOT_FILES.copy()


def get_keep_root_files_exceptions() -> list[str]:
    """Get list of root files to keep despite matching remove patterns.

    Returns:
        List of root files that are exceptions to remove patterns.
    """
    return KEEP_ROOT_FILES_EXCEPTIONS.copy()


def get_remove_src_dirs() -> list[str]:
    """Get list of source directories to remove.

    Returns:
        List of source directories that should be removed.
    """
    return REMOVE_SRC_DIRS.copy()


def get_keep_src_dirs() -> list[str]:
    """Get list of source directories to keep.

    Returns:
        List of source directories that should be preserved.
    """
    return KEEP_SRC_DIRS.copy()


def get_remove_src_files() -> list[str]:
    """Get list of source files to remove.

    Returns:
        List of source files that should be removed.
    """
    return REMOVE_SRC_FILES.copy()


def get_protected_directories() -> list[str]:
    """Get list of protected directories.

    Returns:
        List of directories that should never be removed.
    """
    return PROTECTED_DIRECTORIES.copy()


def get_protected_files() -> list[str]:
    """Get list of protected files.

    Returns:
        List of files that should never be removed.
    """
    return PROTECTED_FILES.copy()


def get_backup_dir() -> Path:
    """Get backup directory path.

    Returns:
        Path to backup directory.
    """
    return BACKUP_DIR


def get_validation_paths() -> list[Path]:
    """Get list of paths that must exist after cleanup.

    Returns:
        List of paths that must be validated after cleanup.
    """
    return VALIDATION_PATHS.copy()


def is_protected(path: Path) -> bool:
    """Check if a path is protected.

    Args:
        path: Path to check.

    Returns:
        True if path is protected, False otherwise.
    """
    # Check if it's a protected directory (by name)
    for protected_dir in PROTECTED_DIRECTORIES:
        if path.name == protected_dir.rstrip("/"):
            return True

    # Check if it's a protected file (by name)
    for protected_file in PROTECTED_FILES:
        if path.name == protected_file:
            return True

    return False


def is_keep_root_item(path: Path) -> bool:
    """Check if a path is a keep root item.

    Args:
        path: Path to check.

    Returns:
        True if path is a keep root item, False otherwise.
    """
    name = path.name

    # Check exact match
    if name in KEEP_ROOT_ITEMS:
        return True

    # Check directory with trailing slash
    if path.is_dir():
        if f"{name}/" in KEEP_ROOT_ITEMS:
            return True

    return False


def is_remove_root_dir(path: Path) -> bool:
    """Check if a path is a remove root directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a remove root directory, False otherwise.
    """
    name = path.name

    # Check exact match
    if name in [d.rstrip("/") for d in REMOVE_ROOT_DIRS]:
        return True

    # Check with trailing slash
    if f"{name}/" in REMOVE_ROOT_DIRS:
        return True

    return False


def is_remove_root_file(path: Path) -> bool:
    """Check if a path is a remove root file.

    Args:
        path: Path to check.

    Returns:
        True if path is a remove root file, False otherwise.
    """
    if path.is_dir():
        return False

    name = path.name

    # Check exceptions first
    if name in KEEP_ROOT_FILES_EXCEPTIONS:
        return False

    # Check patterns
    import fnmatch

    for pattern in REMOVE_ROOT_FILES:
        if fnmatch.fnmatch(name, pattern):
            return True

    return False


def is_remove_src_dir(path: Path) -> bool:
    """Check if a path is a remove source directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a remove source directory, False otherwise.
    """
    # Check if path is under src/
    try:
        relative = path.relative_to(Path("src/"))
    except ValueError:
        return False

    # Check if it matches any remove pattern
    for remove_dir in REMOVE_SRC_DIRS:
        remove_path = Path(remove_dir.rstrip("/"))
        try:
            if relative == remove_path.relative_to(Path("src/")):
                return True
        except ValueError:
            continue

    return False


def is_keep_src_dir(path: Path) -> bool:
    """Check if a path is a keep source directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a keep source directory, False otherwise.
    """
    if not path.is_dir():
        return False

    # Check if path is under src/
    try:
        relative = path.relative_to(Path("src/"))
    except ValueError:
        return False

    # Check if it matches any keep pattern
    for keep_dir in KEEP_SRC_DIRS:
        keep_path = Path(keep_dir.rstrip("/"))
        try:
            if relative == keep_path.relative_to(Path("src/")):
                return True
        except ValueError:
            continue

    return False


def is_remove_src_file(path: Path) -> bool:
    """Check if a path is a remove source file.

    Args:
        path: Path to check.

    Returns:
        True if path is a remove source file, False otherwise.
    """
    if path.is_dir():
        return False

    # Check if path is under src/
    try:
        path.relative_to(Path("src/"))
    except ValueError:
        return False

    # Check exact match
    for remove_file in REMOVE_SRC_FILES:
        if path == Path(remove_file):
            return True

    return False


if __name__ == "__main__":
    print("Cleanup Core Module")
    print(f"Keep root items: {len(KEEP_ROOT_ITEMS)}")
    print(f"Remove root dirs: {len(REMOVE_ROOT_DIRS)}")
    print(f"Remove root files: {len(REMOVE_ROOT_FILES)}")
    print(f"Remove src dirs: {len(REMOVE_SRC_DIRS)}")
    print(f"Keep src dirs: {len(KEEP_SRC_DIRS)}")
    print(f"Remove src files: {len(REMOVE_SRC_FILES)}")
    print(f"Protected directories: {len(PROTECTED_DIRECTORIES)}")
    print(f"Protected files: {len(PROTECTED_FILES)}")
    print(f"Validation paths: {len(VALIDATION_PATHS)}")
