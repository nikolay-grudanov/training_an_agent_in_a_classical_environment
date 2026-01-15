"""Cleanup module - File and directory categorization.

This module provides the CleanupCategorizer class that determines
which files and directories should be kept, removed, or archived.
"""

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .core import (
    FileCategory,
    DirCategory,
    get_keep_root_items,
    get_remove_root_dirs,
    get_remove_root_files,
    get_keep_root_files_exceptions,
    get_remove_src_dirs,
    get_keep_src_dirs,
    get_remove_src_files,
    get_protected_directories,
    get_protected_files,
    is_protected,
    is_keep_root_item,
    is_remove_root_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class CategorizationResult:
    """Result of file/directory categorization.

    Attributes:
        path: The path being categorized.
        category: The category (KEEP, REMOVE, ARCHIVE, PROTECTED).
        reason: The reason for this categorization.
    """

    path: Path
    category: FileCategory | DirCategory
    reason: str


class CleanupCategorizer:
    """Categorizes files and directories for cleanup operations.

    This class determines which files and directories should be kept,
    removed, or archived based on the cleanup rules defined in core.py.
    """

    def __init__(self, root_path: Path = Path(".")) -> None:
        """Initialize the categorizer.

        Args:
            root_path: Root path of the project (default: current directory).
        """
        self.root_path: Final = root_path.resolve()
        self._keep_root_items: Final = get_keep_root_items()
        self._remove_root_dirs: Final = get_remove_root_dirs()
        self._remove_root_files: Final = get_remove_root_files()
        self._keep_root_files_exceptions: Final = get_keep_root_files_exceptions()
        self._remove_src_dirs: Final = get_remove_src_dirs()
        self._keep_src_dirs: Final = get_keep_src_dirs()
        self._remove_src_files: Final = get_remove_src_files()
        self._protected_directories: Final = get_protected_directories()
        self._protected_files: Final = get_protected_files()

    def categorize_file(self, path: Path) -> CategorizationResult:
        """Determine if a file should be kept, removed, or archived.

        Args:
            path: Path to the file.

        Returns:
            CategorizationResult with the category and reason.

        Raises:
            ValueError: If path is not a file.
        """
        if path.is_dir():
            raise ValueError(f"Path {path} is a directory, not a file")

        # Check if protected
        if is_protected(path):
            return CategorizationResult(
                path=path,
                category=FileCategory.KEEP,
                reason="Protected file",
            )

        # Resolve path to absolute
        abs_path = path.resolve()

        # Check if it's a root file
        try:
            relative = abs_path.relative_to(self.root_path)
        except ValueError:
            # Path is not under root, keep it
            return CategorizationResult(
                path=path,
                category=FileCategory.KEEP,
                reason="Not under project root",
            )

        # Check if it's a source file
        try:
            relative.relative_to(Path("src/"))
        except ValueError:
            # Not under src/, check root file rules
            return self._categorize_root_file(path, relative)
        else:
            # Under src/, check source file rules
            return self._categorize_src_file(path, relative)

    def categorize_directory(self, path: Path) -> CategorizationResult:
        """Determine if a directory should be kept, removed, or archived.

        Args:
            path: Path to the directory.

        Returns:
            CategorizationResult with the category and reason.

        Raises:
            ValueError: If path is not a directory.
        """
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        # Check if protected
        if is_protected(path):
            return CategorizationResult(
                path=path,
                category=DirCategory.PROTECTED,
                reason="Protected directory",
            )

        # Check if it's a root directory (relative path or under root)
        try:
            relative = path.relative_to(self.root_path)
        except ValueError:
            # Path is not under root, categorize based on name only
            return self._categorize_root_directory(path, path)

        # Check if it's a source directory
        try:
            relative.relative_to(Path("src/"))
        except ValueError:
            # Not under src/, check root directory rules
            return self._categorize_root_directory(path, relative)
        else:
            # Under src/, check source directory rules
            return self._categorize_src_directory(path, relative)

    def get_all_items_to_remove(self) -> list[Path]:
        """Get all items marked for removal.

        Returns:
            List of paths that should be removed.
        """
        items_to_remove: list[Path] = []

        # Scan root directory
        for item in self.root_path.iterdir():
            if item.is_file():
                result = self.categorize_file(item)
                if result.category == FileCategory.REMOVE:
                    items_to_remove.append(item)
            elif item.is_dir():
                result = self.categorize_directory(item)
                if result.category == DirCategory.REMOVE:
                    items_to_remove.append(item)

        # Scan src/ subdirectories
        src_path = self.root_path / "src"
        if src_path.exists() and src_path.is_dir():
            for item in src_path.rglob("*"):
                if item.is_file():
                    result = self.categorize_file(item)
                    if result.category == FileCategory.REMOVE:
                        items_to_remove.append(item)
                elif item.is_dir():
                    result = self.categorize_directory(item)
                    if result.category == DirCategory.REMOVE:
                        items_to_remove.append(item)

        return items_to_remove

    def get_removal_summary(self) -> dict[str, int | list[str]]:
        """Get a summary of items to remove by category.

        Returns:
            Dictionary with counts and lists of items by category.
        """
        items_to_remove = self.get_all_items_to_remove()

        files: list[str] = []
        directories: list[str] = []
        root_items: list[str] = []
        src_items: list[str] = []

        for item in items_to_remove:
            if item.is_file():
                files.append(str(item))
            else:
                directories.append(str(item))

            # Check if root or src item
            try:
                item.relative_to(self.root_path / "src")
                src_items.append(str(item))
            except ValueError:
                try:
                    item.relative_to(self.root_path)
                    root_items.append(str(item))
                except ValueError:
                    pass

        summary: dict[str, int | list[str]] = {
            "total_count": len(items_to_remove),
            "files": files,
            "directories": directories,
            "root_items": root_items,
            "src_items": src_items,
        }

        return summary

    def _categorize_root_file(self, path: Path, relative: Path) -> CategorizationResult:
        """Categorize a file in the root directory.

        Args:
            path: Full path to the file.
            relative: Path relative to root.

        Returns:
            CategorizationResult with the category and reason.
        """
        name = path.name

        # Check exceptions first
        if name in self._keep_root_files_exceptions:
            return CategorizationResult(
                path=path,
                category=FileCategory.KEEP,
                reason="Exception to remove patterns",
            )

        # Check remove patterns
        for pattern in self._remove_root_files:
            if fnmatch.fnmatch(name, pattern):
                return CategorizationResult(
                    path=path,
                    category=FileCategory.REMOVE,
                    reason=f"Matches remove pattern: {pattern}",
                )

        # Default to keep
        return CategorizationResult(
            path=path,
            category=FileCategory.KEEP,
            reason="Not matching any remove pattern",
        )

    def _categorize_src_file(self, path: Path, relative: Path) -> CategorizationResult:
        """Categorize a file in the src/ directory.

        Args:
            path: Full path to the file.
            relative: Path relative to root.

        Returns:
            CategorizationResult with the category and reason.
        """
        # Check if it's a remove source file by relative path
        for remove_file_pattern in self._remove_src_files:
            # Convert pattern to relative path under root
            remove_path = Path(remove_file_pattern)
            # Compare full relative paths
            if relative == remove_path:
                return CategorizationResult(
                    path=path,
                    category=FileCategory.REMOVE,
                    reason="In remove source files list",
                )

        # Default to keep
        return CategorizationResult(
            path=path,
            category=FileCategory.KEEP,
            reason="Not in remove source files list",
        )

    def _categorize_root_directory(
        self, path: Path, relative: Path
    ) -> CategorizationResult:
        """Categorize a directory in the root directory.

        Args:
            path: Full path to the directory.
            relative: Path relative to root.

        Returns:
            CategorizationResult with the category and reason.
        """
        # Check if it's a remove root directory
        if is_remove_root_dir(path):
            return CategorizationResult(
                path=path,
                category=DirCategory.REMOVE,
                reason="In remove root directories list",
            )

        # Check if it's a keep root item
        if is_keep_root_item(path):
            return CategorizationResult(
                path=path,
                category=DirCategory.KEEP,
                reason="In keep root items list",
            )

        # Default to keep
        return CategorizationResult(
            path=path,
            category=DirCategory.KEEP,
            reason="Not in remove root directories list",
        )

    def _categorize_src_directory(
        self, path: Path, relative: Path
    ) -> CategorizationResult:
        """Categorize a directory in the src/ directory.

        Args:
            path: Full path to the directory.
            relative: Path relative to root.

        Returns:
            CategorizationResult with the category and reason.
        """
        # Check if it's a remove source directory by relative path
        for remove_dir_pattern in self._remove_src_dirs:
            # Convert pattern to relative path under root
            remove_path = Path(remove_dir_pattern.rstrip("/"))
            # Compare full relative paths
            if relative == remove_path:
                return CategorizationResult(
                    path=path,
                    category=DirCategory.REMOVE,
                    reason="In remove source directories list",
                )

        # Check if it's a keep source directory
        for keep_dir_pattern in self._keep_src_dirs:
            # Convert pattern to relative path under root
            keep_path = Path(keep_dir_pattern.rstrip("/"))
            # Compare full relative paths
            if relative == keep_path:
                return CategorizationResult(
                    path=path,
                    category=DirCategory.KEEP,
                    reason="In keep source directories list",
                )

        # Default to keep
        return CategorizationResult(
            path=path,
            category=DirCategory.KEEP,
            reason="Not in remove source directories list",
        )


def categorize_file(path: Path) -> CategorizationResult:
    """Categorize a single file.

    Args:
        path: Path to the file.

    Returns:
        CategorizationResult with the category and reason.
    """
    categorizer = CleanupCategorizer()
    return categorizer.categorize_file(path)


def categorize_directory(path: Path) -> CategorizationResult:
    """Categorize a single directory.

    Args:
        path: Path to the directory.

    Returns:
        CategorizationResult with the category and reason.
    """
    categorizer = CleanupCategorizer()
    return categorizer.categorize_directory(path)


if __name__ == "__main__":
    from src.utils.logging_setup import setup_logging

    setup_logging()

    print("Testing CleanupCategorizer...")

    categorizer = CleanupCategorizer()

    # Test categorization
    test_paths = [
        Path("README.md"),
        Path("requirements.txt"),
        Path("example_training.py"),
        Path("configs/"),
        Path("src/api/"),
        Path("src/training/"),
        Path("src/utils/logging.py"),
    ]

    for path in test_paths:
        if path.exists():
            if path.is_file():
                result = categorizer.categorize_file(path)
            else:
                result = categorizer.categorize_directory(path)

            print(f"   {path}: {result.category.value} - {result.reason}")

    # Get removal summary
    summary = categorizer.get_removal_summary()
    print("\nRemoval Summary:")
    print(f"   Total items to remove: {summary['total_count']}")
    print(
        f"   Files: {len(summary['files']) if isinstance(summary['files'], list) else 0}"
    )
    print(
        f"   Directories: {len(summary['directories']) if isinstance(summary['directories'], list) else 0}"
    )
    print(
        f"   Root items: {len(summary['root_items']) if isinstance(summary['root_items'], list) else 0}"
    )
    print(
        f"   Source items: {len(summary['src_items']) if isinstance(summary['src_items'], list) else 0}"
    )

    print("\n✅ Categorization test complete!")
