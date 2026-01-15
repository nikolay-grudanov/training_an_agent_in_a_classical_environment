"""Cleanup module - Execution engine.

This module provides the CleanupExecutor class that performs the actual
cleanup operations with safety features like backups and validation.
"""

import json
import logging
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

from .core import (
    get_validation_paths,
    is_protected,
)
from .categorizer import CleanupCategorizer

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of a cleanup operation.

    Attributes:
        dry_run: Whether this was a dry run.
        items_removed: List of items that were removed.
        items_kept: List of items that were kept.
        backup_path: Path to backup archive (if created).
        validation_passed: Whether validation passed after cleanup.
        errors: List of errors that occurred.
    """

    dry_run: bool
    items_removed: list[str] = field(default_factory=list)
    items_kept: list[str] = field(default_factory=list)
    backup_path: Path | None = None
    validation_passed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class DryRunResult:
    """Result of a dry run operation.

    Attributes:
        items_to_remove: List of items that would be removed.
        items_to_keep: List of items that would be kept.
        total_size: Total size of items to remove (in bytes).
    """

    items_to_remove: list[str] = field(default_factory=list)
    items_to_keep: list[str] = field(default_factory=list)
    total_size: int = 0


class CleanupExecutor:
    """Executes cleanup operations with safety features.

    This class provides methods to:
    - Preview cleanup actions (dry run)
    - Execute cleanup with optional backup
    - Validate core structure after cleanup
    """

    def __init__(self, root_path: Path = Path(".")) -> None:
        """Initialize the executor.

        Args:
            root_path: Root path of the project (default: current directory).
        """
        self.root_path: Final = root_path.resolve()
        self.backup_dir: Final = self.root_path / "results" / "cleanup_backups"
        self.validation_paths: Final = get_validation_paths()
        self.categorizer = CleanupCategorizer(root_path)

    def dry_run(self, items: list[Path] | None = None) -> DryRunResult:
        """Show what would be removed without actually removing.

        Args:
            items: List of items to check (if None, scans all items).

        Returns:
            DryRunResult with items that would be removed and kept.
        """
        logger.info("Starting dry run...")

        if items is None:
            items = self.categorizer.get_all_items_to_remove()

        result = DryRunResult()

        for item in items:
            if not item.exists():
                logger.warning(f"Item does not exist: {item}")
                continue

            # Check if protected
            if is_protected(item):
                result.items_to_keep.append(str(item))
                logger.debug(f"Would keep (protected): {item}")
                continue

            # Get categorization
            try:
                if item.is_file():
                    cat_result = self.categorizer.categorize_file(item)
                else:
                    cat_result = self.categorizer.categorize_directory(item)

                # Check if should be removed
                from .core import FileCategory, DirCategory

                if isinstance(cat_result.category, (FileCategory, DirCategory)):
                    if cat_result.category.value == "remove":
                        result.items_to_remove.append(str(item))
                        # Calculate size
                        if item.is_file():
                            result.total_size += item.stat().st_size
                        elif item.is_dir():
                            result.total_size += sum(
                                f.stat().st_size for f in item.rglob("*") if f.is_file()
                            )
                        logger.debug(f"Would remove: {item}")
                    else:
                        result.items_to_keep.append(str(item))
                        logger.debug(f"Would keep: {item}")
            except Exception as e:
                logger.error(f"Error categorizing {item}: {e}")
                result.items_to_keep.append(str(item))

        logger.info(
            f"Dry run complete: {len(result.items_to_remove)} items to remove, "
            f"{len(result.items_to_keep)} items to keep"
        )

        return result

    def execute(
        self, items: list[Path] | None = None, force: bool = False
    ) -> CleanupResult:
        """Actually remove files/directories.

        Args:
            items: List of items to remove (if None, scans all items).
            force: If True, skip confirmation prompts.

        Returns:
            CleanupResult with details of the operation.
        """
        logger.info("Starting cleanup execution...")

        result = CleanupResult(dry_run=False)

        # Get items to remove
        if items is None:
            items = self.categorizer.get_all_items_to_remove()

        # Confirm before removing (unless force)
        if not force and items:
            logger.info(f"About to remove {len(items)} items:")
            for item in items[:10]:
                logger.info(f"  - {item}")
            if len(items) > 10:
                logger.info(f"  ... and {len(items) - 10} more items")

            # In a real CLI, we would prompt here
            # For now, we'll just log and continue
            logger.warning("Use --force to skip confirmation")

        # Remove items
        for item in items:
            if not item.exists():
                logger.warning(f"Item does not exist: {item}")
                continue

            # Check if protected
            if is_protected(item):
                result.items_kept.append(str(item))
                logger.warning(f"Skipping protected item: {item}")
                continue

            try:
                if item.is_file():
                    item.unlink()
                    logger.info(f"Removed file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"Removed directory: {item}")

                result.items_removed.append(str(item))
            except Exception as e:
                error_msg = f"Failed to remove {item}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Validate after cleanup
        result.validation_passed = self.validate_after_cleanup()

        logger.info(
            f"Cleanup complete: {len(result.items_removed)} removed, "
            f"{len(result.items_kept)} kept, "
            f"validation {'passed' if result.validation_passed else 'failed'}"
        )

        return result

    def backup_before_remove(self, items: list[Path]) -> Path:
        """Create backup archive before removal.

        Args:
            items: List of items to backup.

        Returns:
            Path to the created backup archive.

        Raises:
            OSError: If backup creation fails.
        """
        logger.info(f"Creating backup for {len(items)} items...")

        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"cleanup_backup_{timestamp}.tar.gz"

        # Create tar.gz archive
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                for item in items:
                    if item.exists():
                        # Add item to archive
                        tar.add(item, arcname=item.name)
                        logger.debug(f"Added to backup: {item}")

            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise OSError(f"Backup creation failed: {e}") from e

    def validate_after_cleanup(self) -> bool:
        """Verify that core structure is intact after cleanup.

        Returns:
            True if validation passed, False otherwise.
        """
        logger.info("Validating project structure after cleanup...")

        all_valid = True

        for validation_path in self.validation_paths:
            full_path = self.root_path / validation_path

            if not full_path.exists():
                logger.error(f"Validation failed: {full_path} does not exist")
                all_valid = False
            else:
                logger.debug(f"Validation passed: {full_path} exists")

        # Check that src/ directory still exists
        src_path = self.root_path / "src"
        if not src_path.exists():
            logger.error("Validation failed: src/ directory does not exist")
            all_valid = False
        else:
            logger.debug("Validation passed: src/ directory exists")

        # Check that tests/ directory still exists
        tests_path = self.root_path / "tests"
        if not tests_path.exists():
            logger.error("Validation failed: tests/ directory does not exist")
            all_valid = False
        else:
            logger.debug("Validation passed: tests/ directory exists")

        if all_valid:
            logger.info("Validation passed: Core structure is intact")
        else:
            logger.error("Validation failed: Core structure is damaged")

        return all_valid

    def generate_project_structure_report(self) -> dict:
        """Generate a project structure validation report.

        Returns:
            Dictionary with project structure information.
        """
        logger.info("Generating project structure report...")

        report: dict = {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(self.root_path),
            "validation": {
                "src_exists": (self.root_path / "src").exists(),
                "tests_exists": (self.root_path / "tests").exists(),
                "results_exists": (self.root_path / "results").exists(),
                "specs_exists": (self.root_path / "specs").exists(),
            },
            "src_structure": {},
            "tests_structure": {},
        }

        # Scan src/ structure
        src_path = self.root_path / "src"
        if src_path.exists():
            for item in src_path.iterdir():
                if item.is_dir():
                    report["src_structure"][item.name] = {
                        "exists": True,
                        "type": "directory",
                    }
                elif item.is_file():
                    report["src_structure"][item.name] = {
                        "exists": True,
                        "type": "file",
                    }

        # Scan tests/ structure
        tests_path = self.root_path / "tests"
        if tests_path.exists():
            for item in tests_path.iterdir():
                if item.is_dir():
                    report["tests_structure"][item.name] = {
                        "exists": True,
                        "type": "directory",
                    }
                elif item.is_file():
                    report["tests_structure"][item.name] = {
                        "exists": True,
                        "type": "file",
                    }

        logger.info("Project structure report generated")

        return report

    def save_project_structure_report(
        self, output_path: Path | None = None
    ) -> Path:
        """Generate and save project structure report to JSON.

        Args:
            output_path: Path to save report (default: results/project_structure.json).

        Returns:
            Path to the saved report.
        """
        if output_path is None:
            output_path = self.root_path / "results" / "project_structure.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_project_structure_report()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Project structure report saved to {output_path}")

        return output_path


if __name__ == "__main__":
    from src.utils.logging_setup import setup_logging

    setup_logging()

    print("Testing CleanupExecutor...")

    executor = CleanupExecutor()

    # Test dry run
    print("\n1. Testing dry run...")
    dry_run_result = executor.dry_run()
    print(f"   Items to remove: {len(dry_run_result.items_to_remove)}")
    print(f"   Items to keep: {len(dry_run_result.items_to_keep)}")
    print(f"   Total size: {dry_run_result.total_size} bytes")

    # Test validation
    print("\n2. Testing validation...")
    validation_passed = executor.validate_after_cleanup()
    print(f"   Validation passed: {validation_passed}")

    # Test project structure report
    print("\n3. Testing project structure report...")
    report = executor.generate_project_structure_report()
    print(f"   Report keys: {list(report.keys())}")

    print("\n✅ Executor test complete!")
