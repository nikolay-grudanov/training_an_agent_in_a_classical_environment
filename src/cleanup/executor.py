"""Cleanup module - Execution engine.

Per contracts/cleanup_system.md:
- Execute file moves, removals, and archives
- Handle edge cases (files in use, permissions)
- Generate ProjectStructure entity JSON
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .core import (
    CleanupConfig,
    CleanupAction,
    CleanupStatus,
    CleanupActionItem,
    RootDirectoryAnalyzer,
)
from .categorizer import FileCategory, FileCategorizer

logger = logging.getLogger(__name__)


@dataclass
class ProjectStructureReport:
    """Report of project structure after cleanup."""

    metadata: dict = field(default_factory=dict)
    root_directory: dict = field(default_factory=dict)
    cleanup_actions: list = field(default_factory=list)

    def __post_init__(self):
        self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata."""
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(Path(".").resolve()),
            "cleanup_status": "completed",
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_structure": {
                "metadata": self.metadata,
                "root_directory": self.root_directory,
                "cleanup_actions": self.cleanup_actions,
            }
        }

    def save(self, filepath: Path) -> None:
        """Save report to JSON file.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Project structure report saved to {filepath}")


class CleanupExecutor:
    """Executes cleanup operations."""

    def __init__(self, config: CleanupConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.report = ProjectStructureReport()
        self.root_path = Path(".").resolve()

        # Initialize categorizer
        self.categorizer = FileCategorizer()

    def execute(self) -> ProjectStructureReport:
        """Execute cleanup operations.

        Returns:
            ProjectStructureReport with results
        """
        logger.info("Starting cleanup execution...")

        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
            self.report.metadata["cleanup_status"] = "dry_run"

        # Analyze root directory
        analyzer = RootDirectoryAnalyzer(self.config)
        allowed, unexpected = analyzer.scan_root()

        # Build root directory report
        self._build_root_report(allowed, unexpected)

        # Process each unexpected item
        for item in unexpected:
            self._process_item(item)

        # Log summary
        moved = sum(
            1 for a in self.report.cleanup_actions if a["action"] == "moved"
        )
        removed = sum(
            1 for a in self.report.cleanup_actions if a["action"] == "removed"
        )
        archived = sum(
            1 for a in self.report.cleanup_actions if a["action"] == "archived"
        )
        skipped = sum(
            1 for a in self.report.cleanup_actions if a["action"] == "skipped"
        )

        logger.info(f"📊 Cleanup Summary:")
        logger.info(f"   Moved: {moved}")
        logger.info(f"   Removed: {removed}")
        logger.info(f"   Archived: {archived}")
        logger.info(f"   Skipped: {skipped}")

        return self.report

    def _build_root_report(
        self, allowed: list, unexpected: list
    ) -> None:
        """Build root directory report section.

        Args:
            allowed: List of allowed items
            unexpected: List of unexpected items
        """
        self.root_directory = {
            "allowed_files": self.config.allowed_files,
            "allowed_directories": self.config.allowed_directories,
            "actual_files": [item.path.name for item in allowed if not item.is_directory],
            "actual_directories": [
                item.path.name for item in allowed if item.is_directory
            ],
            "validation_status": "clean" if len(unexpected) == 0 else "dirty",
        }

    def _process_item(self, item) -> None:
        """Process a single item for cleanup.

        Args:
            item: FileItem to process
        """
        category, destination = self.categorizer.categorize(item.path), None

        if category != FileCategory.KEEP:
            destination = self.categorizer.get_destination(item.path, category)

        # Create action item
        action_item = {
            "action": category.value,
            "source": str(item.path),
            "destination": str(destination) if destination else None,
            "status": "dry_run" if self.dry_run else "pending",
        }

        # Execute action if not dry run
        if not self.dry_run and category != FileCategory.KEEP:
            try:
                self._execute_action(item.path, category, destination)
                action_item["status"] = "completed"
            except Exception as e:
                logger.error(f"Failed to process {item.path}: {e}")
                action_item["status"] = "failed"
                action_item["error"] = str(e)

        self.report.cleanup_actions.append(action_item)

    def _execute_action(
        self,
        source: Path,
        category: FileCategory,
        destination: Optional[Path],
    ) -> None:
        """Execute a cleanup action.

        Args:
            source: Source path
            category: Action category
            destination: Destination path (if applicable)
        """
        if category == FileCategory.REMOVE:
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
            logger.info(f"   [✗ REMOVED] {source.name}")

        elif category in [
            FileCategory.MOVE_TO_SRC,
            FileCategory.MOVE_TO_TESTS,
            FileCategory.MOVE_TO_RESULTS,
            FileCategory.ARCHIVE,
        ]:
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                logger.info(f"   [→ MOVED] {source.name} → {destination}")

        elif category == FileCategory.SKIP:
            logger.info(f"   [→ SKIP] {source.name} (unknown/protected)")

        else:
            logger.warning(f"   [? UNKNOWN] {source.name}")


def execute_cleanup(
    config: Optional[CleanupConfig] = None,
    dry_run: bool = False,
    output_path: Optional[Path] = None,
) -> ProjectStructureReport:
    """Execute project cleanup.

    Args:
        config: Cleanup configuration
        dry_run: If True, preview actions without executing
        output_path: Path to save report

    Returns:
        ProjectStructureReport with results
    """
    if config is None:
        config = CleanupConfig()

    executor = CleanupExecutor(config, dry_run=dry_run)
    report = executor.execute()

    if output_path is None:
        output_path = Path("results/project_structure.json")

    report.save(output_path)

    return report


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()

    print("Testing cleanup execution...")

    # Test dry run
    report = execute_cleanup(dry_run=True)

    print(f"\nDry run complete. See {report.metadata['cleanup_status']}")

    # Print summary
    print(f"\nPlanned actions:")
    for action in report.cleanup_actions[:10]:
        print(f"   [{action['action']}] {action['source']}")

    if len(report.cleanup_actions) > 10:
        print(f"   ... and {len(report.cleanup_actions) - 10} more")

    print("\n✅ Cleanup test complete!")
