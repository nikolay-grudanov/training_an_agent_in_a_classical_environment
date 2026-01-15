"""Cleanup module core - Configuration and root directory analysis.

Per contracts/cleanup_system.md:
- Define allowed files and directories
- Implement root directory scanner
- Create CleanupConfig dataclass
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CleanupAction(Enum):
    """Types of cleanup actions."""

    MOVE = "moved"
    REMOVE = "removed"
    ARCHIVE = "archived"
    SKIP = "skipped"


class CleanupStatus(Enum):
    """Status of cleanup operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass
class CleanupConfig:
    """Configuration for project cleanup process."""

    # Allowed items in root directory
    allowed_files: list[str] = field(
        default_factory=lambda: [
            "requirements.txt",
            "README.md",
            ".gitignore",
        ]
    )
    allowed_directories: list[str] = field(
        default_factory=lambda: [
            "src/",
            "tests/",
            "results/",
            "specs/",
        ]
    )
    dry_run: bool = False
    force: bool = False
    verbose: bool = False

    # File categorization rules
    move_to_src: list[str] = field(
        default_factory=lambda: [
            "*.py",
            "example_*.py",
        ]
    )
    move_to_tests: list[str] = field(
        default_factory=lambda: [
            "test_*.py",
        ]
    )
    remove_patterns: list[str] = field(
        default_factory=lambda: [
            "verify_setup.py",
            "test_installation.py",
            "*.tmp",
            "*.bak",
            ".jupyter_ystore.db",
        ]
    )
    archive_patterns: list[str] = field(
        default_factory=lambda: [
            "info_project.md",
            "*_SUMMARY.md",
            "demo_*",
        ]
    )


@dataclass
class FileItem:
    """Represents a file or directory in the project."""

    path: Path
    is_directory: bool
    size: Optional[int] = None
    modified_time: Optional[float] = None


@dataclass
class CleanupActionItem:
    """Represents a planned cleanup action."""

    action: CleanupAction
    source: Path
    destination: Optional[Path] = None
    status: CleanupStatus = CleanupStatus.PENDING
    reason: Optional[str] = None


class RootDirectoryAnalyzer:
    """Analyzes root directory and identifies items for cleanup."""

    def __init__(self, config: CleanupConfig):
        self.config = config
        self.root_path = Path(".").resolve()

    def scan_root(self) -> tuple[list[FileItem], list[FileItem]]:
        """Scan root directory and separate allowed vs unexpected items.

        Returns:
            Tuple of (allowed_items, unexpected_items)
        """
        allowed_items = []
        unexpected_items = []

        for item in sorted(self.root_path.iterdir()):
            file_item = FileItem(
                path=item,
                is_directory=item.is_dir(),
                size=item.stat().st_size if item.is_file() else None,
                modified_time=item.stat().st_mtime if item.exists() else None,
            )

            # Check if item is allowed
            if self._is_allowed(item):
                allowed_items.append(file_item)
            else:
                unexpected_items.append(file_item)

        return allowed_items, unexpected_items

    def _is_allowed(self, item: Path) -> bool:
        """Check if item is allowed in root directory.

        Args:
            item: Path to check

        Returns:
            True if item is allowed
        """
        name = item.name

        # Check allowed files
        if not item.is_dir():
            if name in self.config.allowed_files:
                return True

        # Check allowed directories
        if item.is_dir():
            if name in self.config.allowed_directories:
                return True
            # Add trailing slash for directory comparison
            if f"{name}/" in self.config.allowed_directories:
                return True

        return False

    def categorize_items(
        self, items: list[FileItem]
    ) -> list[CleanupActionItem]:
        """Categorize items into cleanup actions.

        Args:
            items: List of items to categorize

        Returns:
            List of cleanup actions
        """
        actions = []

        for item in items:
            action = self._categorize_item(item)
            if action:
                actions.append(action)

        return actions

    def _categorize_item(self, item: FileItem) -> Optional[CleanupActionItem]:
        """Categorize a single item.

        Args:
            item: File item to categorize

        Returns:
            CleanupActionItem or None if no action needed
        """
        name = item.path.name

        # Check if should be removed
        for pattern in self.config.remove_patterns:
            if self._matches_pattern(name, pattern):
                return CleanupActionItem(
                    action=CleanupAction.REMOVE,
                    source=item.path,
                    reason=f"Matches remove pattern: {pattern}",
                )

        # Check if should be archived
        for pattern in self.config.archive_patterns:
            if self._matches_pattern(name, pattern):
                return CleanupActionItem(
                    action=CleanupAction.ARCHIVE,
                    source=item.path,
                    destination=Path("docs/archives/") / name,
                    reason=f"Matches archive pattern: {pattern}",
                )

        # Check if should be moved to src/
        for pattern in self.config.move_to_src:
            if self._matches_pattern(name, pattern):
                return CleanupActionItem(
                    action=CleanupAction.MOVE,
                    source=item.path,
                    destination=Path("src/") / name,
                    reason=f"Matches move to src pattern: {pattern}",
                )

        # Check if should be moved to tests/
        for pattern in self.config.move_to_tests:
            if self._matches_pattern(name, pattern):
                return CleanupActionItem(
                    action=CleanupAction.MOVE,
                    source=item.path,
                    destination=Path("tests/") / name,
                    reason=f"Matches move to tests pattern: {pattern}",
                )

        # Default: skip if it's an allowed item
        if self._is_allowed(item.path):
            return CleanupActionItem(
                action=CleanupAction.SKIP,
                source=item.path,
                reason="Allowed in root directory",
            )

        # Unknown item - skip with warning
        logger.warning(f"Unknown item: {item.path}")
        return CleanupActionItem(
            action=CleanupAction.SKIP,
            source=item.path,
            reason="Unknown item - manually review",
        )

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern (supports wildcards).

        Args:
            name: File/directory name
            pattern: Pattern to match (supports * wildcard)

        Returns:
            True if pattern matches name
        """
        import fnmatch

        return fnmatch.fnmatch(name, pattern)


def analyze_root_directory(
    config: Optional[CleanupConfig] = None,
) -> tuple[list[FileItem], list[FileItem], list[CleanupActionItem]]:
    """Analyze root directory for cleanup.

    Args:
        config: Cleanup configuration (uses default if not provided)

    Returns:
        Tuple of (allowed_items, unexpected_items, cleanup_actions)
    """
    if config is None:
        config = CleanupConfig()

    analyzer = RootDirectoryAnalyzer(config)
    allowed, unexpected = analyzer.scan_root()

    # Categorize unexpected items
    cleanup_actions = analyzer.categorize_items(unexpected)

    return allowed, unexpected, cleanup_actions


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)

    print("Analyzing root directory for cleanup...")

    config = CleanupConfig()
    analyzer = RootDirectoryAnalyzer(config)

    allowed, unexpected, actions = analyze_root_directory(config)

    print(f"\n📁 Allowed items ({len(allowed)}):")
    for item in allowed:
        icon = "📁" if item.is_directory else "📄"
        print(f"   {icon} {item.path.name}")

    print(f"\n⚠️  Items to clean up ({len(unexpected)}):")
    for action in actions:
        if action.action == CleanupAction.MOVE:
            print(f"   [→ MOVE] {action.source.name} → {action.destination}")
        elif action.action == CleanupAction.REMOVE:
            print(f"   [✗ REMOVE] {action.source.name}")
        elif action.action == CleanupAction.ARCHIVE:
            print(f"   [📦 ARCHIVE] {action.source.name}")
        else:
            print(f"   [→ SKIP] {action.source.name}")

    print(f"\n✅ Analysis complete!")
