"""Cleanup CLI entry point.

This module provides the command-line interface for the cleanup system.
"""

import argparse
import logging
import sys
from pathlib import Path

from .executor import CleanupExecutor
from .categorizer import CleanupCategorizer
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Project Cleanup System - Clean and organize project structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview cleanup actions (dry run)
  python -m src.cleanup.run --dry-run

  # Preview with verbose output
  python -m src.cleanup.run --dry-run --verbose

  # Execute cleanup with backup
  python -m src.cleanup.run --backup

  # Force cleanup without confirmation
  python -m src.cleanup.run --force

  # Execute with backup and verbose
  python -m src.cleanup.run --backup --verbose
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing (default: True)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup without confirmation prompts",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup archive before removal",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/project_structure.json",
        help="Output file for structure report (default: results/project_structure.json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for cleanup CLI.

    Returns:
        Exit code (0 = success, 1 = warning, 2 = error).
    """
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    logger.info("Starting project cleanup...")

    # Initialize executor
    executor = CleanupExecutor()
    categorizer = CleanupCategorizer()

    # Get removal summary
    summary = categorizer.get_removal_summary()
    total_items = (
        summary["total_count"] if isinstance(summary["total_count"], int) else 0
    )

    logger.info(f"Found {total_items} items to remove")

    if total_items == 0:
        logger.info("✅ Project is already clean!")
        return 0

    # Dry run mode
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be modified")

        dry_run_result = executor.dry_run()

        logger.info("\n📊 Dry Run Summary:")
        logger.info(f"   Items to remove: {len(dry_run_result.items_to_remove)}")
        logger.info(f"   Items to keep: {len(dry_run_result.items_to_keep)}")
        logger.info(f"   Total size: {dry_run_result.total_size:,} bytes")

        if dry_run_result.items_to_remove:
            logger.info("\nItems to remove (first 10):")
            for item in dry_run_result.items_to_remove[:10]:
                logger.info(f"   - {item}")
            if len(dry_run_result.items_to_remove) > 10:
                logger.info(
                    f"   ... and {len(dry_run_result.items_to_remove) - 10} more"
                )

        logger.info("\n⚠️  Run without --dry-run to execute changes")
        return 0

    # Execute mode
    logger.info("⚠️  EXECUTION MODE - Files will be modified")

    # Get items to remove
    items_to_remove = categorizer.get_all_items_to_remove()

    # Create backup if requested
    if args.backup:
        try:
            backup_path = executor.backup_before_remove(items_to_remove)
            logger.info(f"✅ Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return 2

    # Execute cleanup
    try:
        result = executor.execute(items_to_remove, force=args.force)

        logger.info("\n📊 Cleanup Summary:")
        logger.info(f"   Items removed: {len(result.items_removed)}")
        logger.info(f"   Items kept: {len(result.items_kept)}")
        logger.info(
            f"   Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}"
        )

        if result.errors:
            logger.warning(f"\n⚠️  Errors encountered ({len(result.errors)}):")
            for error in result.errors[:5]:
                logger.warning(f"   - {error}")
            if len(result.errors) > 5:
                logger.warning(f"   ... and {len(result.errors) - 5} more")

        # Generate project structure report
        output_path = Path(args.output)
        try:
            report_path = executor.save_project_structure_report(output_path)
            logger.info(f"✅ Project structure report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")

        # Return appropriate exit code
        if not result.validation_passed:
            logger.error("❌ Validation failed - core structure may be damaged")
            return 2
        elif result.errors:
            logger.warning("⚠️  Cleanup completed with errors")
            return 1
        else:
            logger.info("✅ Cleanup completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
