"""Cleanup module - File categorization utilities.

Per cleanup_system.md contract:
- Categorize files for cleanup actions
- Handle edge cases (files in use, permissions)
"""

import fnmatch
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """Categories for file categorization."""

    KEEP = "keep"  # Keep in place (allowed files)
    MOVE_TO_SRC = "move_to_src"
    MOVE_TO_TESTS = "move_to_tests"
    MOVE_TO_RESULTS = "move_to_results"
    MOVE_TO_DOCS = "move_to_docs"
    REMOVE = "remove"
    ARCHIVE = "archive"
    SKIP = "skip"  # Skip (unknown/protected)


class FileCategorizer:
    """Categorizes files based on cleanup rules."""

    def __init__(self):
        # Allowed items in root (from spec)
        self.allowed_files = {
            "requirements.txt",
            "README.md",
            ".gitignore",
        }
        self.allowed_directories = {
            "src/",
            "tests/",
            "results/",
            "specs/",
        }

        # Removal patterns
        self.remove_patterns = {
            "verify_setup.py",
            "test_installation.py",
            "*.tmp",
            "*.bak",
            ".jupyter_ystore.db",
        }

        # Archive patterns
        self.archive_patterns = {
            "info_project.md",
            "*_SUMMARY.md",
            "*_COMPLETION_REPORT.md",
            "QWEN.md",
            "README_UTILS.md",
            "TRAIN_LOOP_README.md",
        }

        # Move to src patterns
        self.move_to_src_patterns = {
            "*.py",
            "example_*.py",
        }

        # Move to tests patterns
        self.move_to_tests_patterns = {
            "test_*.py",
        }

        # Move to results patterns
        self.move_to_results_patterns = {
            "demo_checkpoints/",
            "demo_experiment/",
        }

    def categorize(self, path: Path) -> FileCategory:
        """Determine category for a file or directory.

        Args:
            path: Path to categorize

        Returns:
            FileCategory for the path
        """
        name = path.name

        # Check if it's an allowed file
        if not path.is_dir():
            if name in self.allowed_files:
                return FileCategory.KEEP

        # Check if it's an allowed directory
        if path.is_dir():
            if name in self.allowed_directories:
                return FileCategory.KEEP
            # Check with trailing slash
            if f"{name}/" in self.allowed_directories:
                return FileCategory.KEEP

        # Check remove patterns (high priority)
        for pattern in self.remove_patterns:
            if self._matches(name, pattern):
                return FileCategory.REMOVE

        # Check move to tests patterns BEFORE *.py (more specific)
        for pattern in self.move_to_tests_patterns:
            if self._matches(name, pattern):
                return FileCategory.MOVE_TO_TESTS

        # Check move to src patterns (general *.py)
        for pattern in self.move_to_src_patterns:
            if self._matches(name, pattern):
                return FileCategory.MOVE_TO_SRC

        # Check archive patterns
        for pattern in self.archive_patterns:
            if self._matches(name, pattern):
                return FileCategory.ARCHIVE

        # Check move to results patterns
        for pattern in self.move_to_results_patterns:
            if self._matches(name, pattern):
                return FileCategory.MOVE_TO_RESULTS

        # Unknown item - skip with warning
        if not path.is_dir() or name not in [".git", ".vscode", ".idea"]:
            logger.warning(f"Unknown item, skipping: {path}")
        return FileCategory.SKIP

    def get_destination(self, path: Path, category: FileCategory) -> Optional[Path]:
        """Get destination path for a categorized item.

        Args:
            path: Original path
            category: File category

        Returns:
            Destination path or None
        """
        name = path.name

        if category == FileCategory.MOVE_TO_SRC:
            return Path("src/") / name
        elif category == FileCategory.MOVE_TO_TESTS:
            return Path("tests/") / name
        elif category == FileCategory.MOVE_TO_RESULTS:
            return Path("results/") / name
        elif category == FileCategory.ARCHIVE:
            return Path("docs/archives/") / name
        else:
            return None

    def _matches(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern (supports wildcards).

        Args:
            name: File/directory name
            pattern: Pattern with potential wildcards

        Returns:
            True if pattern matches name
        """
        return fnmatch.fnmatch(name, pattern)


def categorize_file(path: Path) -> tuple[FileCategory, Optional[Path]]:
    """Categorize a single file or directory.

    Args:
        path: Path to categorize

    Returns:
        Tuple of (FileCategory, destination_path)
    """
    categorizer = FileCategorizer()
    category = categorizer.categorize(path)
    destination = categorizer.get_destination(path, category)
    return category, destination


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()

    print("Testing file categorization...")

    categorizer = FileCategorizer()
    test_paths = [
        Path("requirements.txt"),
        Path("README.md"),
        Path("src/"),
        Path("example_training.py"),
        Path("test_ppo_agent.py"),
        Path("verify_setup.py"),
        Path("info_project.md"),
        Path("demo_checkpoints/"),
        Path("QWEN.md"),
        Path("unknown_file.txt"),
    ]

    for path in test_paths:
        if path.exists():
            category, dest = categorize_file(path)
            arrow = "→" if dest else ""
            print(f"   {path.name:30} → {category.value:15} {arrow} {dest or ''}")

    print("\n✅ Categorization complete!")
