"""Cleanup CLI entry point.

Per contracts/cleanup_system.md:
- CLI interface for project cleanup
- Options: --dry-run, --force, --output, --verbose
"""

import argparse
import sys
from pathlib import Path

from .core import CleanupConfig
from .executor import execute_cleanup
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Project Cleanup System - Clean and organize project structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview cleanup actions (dry run)
  python -m src.cleanup.run --dry-run

  # Execute cleanup
  python -m src.cleanup.run

  # Execute with verbose logging
  python -m src.cleanup.run --verbose

  # Force cleanup of protected files
  python -m src.cleanup.run --force

  # Save report to custom location
  python -m src.cleanup.run --output /tmp/cleanup_report.json
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force removal of protected files",
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
        Exit code (0 = success, 1 = warning, 2 = error)
    """
    args = parse_args()

    # Setup logging
    log_level = 10 if args.verbose else 20  # DEBUG=10, INFO=20
    logger = setup_logging(log_level=log_level)
    logger.info("Starting project cleanup...")

    # Create config
    config = CleanupConfig(
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )

    # Execute cleanup
    output_path = Path(args.output)

    try:
        report = execute_cleanup(
            config=config,
            dry_run=args.dry_run,
            output_path=output_path,
        )

        # Check validation status
        validation_status = report.root_directory.get("validation_status", "unknown")

        if args.dry_run:
            logger.info(f"\n🔍 DRY RUN COMPLETE")
            logger.info(f"   Found {len(report.cleanup_actions)} items to process")
            logger.info(f"   Run without --dry-run to execute changes")

        if validation_status == "clean":
            logger.info(f"\n✅ Project is already clean!")
            return 0
        elif args.dry_run:
            logger.info(f"\n⚠️  Run cleanup to organize project")
            return 0
        else:
            logger.info(f"\n✅ Cleanup completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
