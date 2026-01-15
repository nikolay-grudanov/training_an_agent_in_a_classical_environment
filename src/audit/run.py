"""Audit CLI entry point.

Per contracts/audit_system.md:
- CLI interface for running code audits
- Options: --scope, --output, --format, --verbose, --skip-smoke-tests
"""

import argparse
import sys
from pathlib import Path

from .core import AuditConfig, test_module_import, run_smoke_test, discover_python_files
from .assessor import assess_module
from .report_generator import AuditReportGenerator
from src.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Code Audit System - Scan and assess Python modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit of src/ directory
  python -m src.audit.run

  # Audit tests/ directory with JSON output
  python -m src.audit.run --scope tests/ --output audit_tests.json --format json

  # Verbose audit with both formats
  python -m src.audit.run --verbose --format both

  # Skip smoke tests (faster but less detailed)
  python -m src.audit.run --skip-smoke-tests
        """,
    )

    parser.add_argument(
        "--scope",
        type=str,
        default="src/",
        help="Directory to audit (default: src/)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file prefix (default: auto-generated)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format: markdown, json, or both (default: markdown)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Skip functionality smoke tests",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for audit CLI.

    Returns:
        Exit code (0 = success, 1 = warning, 2 = error)
    """
    args = parse_args()

    # Setup logging
    log_level = 10 if args.verbose else 20  # DEBUG=10, INFO=20
    logger = setup_logging(log_level=log_level)
    logger.info(f"Starting code audit with scope: {args.scope}")

    # Create config
    config = AuditConfig(
        scope=Path(args.scope),
        verbose=args.verbose,
        skip_smoke_tests=args.skip_smoke_tests,
    )

    # Discover Python files
    scope_path = config.scope
    if not scope_path.exists():
        logger.error(f"Scope path does not exist: {scope_path}")
        return 2

    python_files = discover_python_files(scope_path)
    logger.info(f"Found {len(python_files)} Python files to audit")

    if not python_files:
        logger.warning("No Python files found in scope")
        return 1

    # Create report generator
    generator = AuditReportGenerator(
        scope=str(config.scope),
        auditor="Automated Audit System",
        version="1.0.0",
    )

    # Assess each module
    broken_count = 0
    needs_fixing_count = 0
    working_count = 0

    for i, module_path in enumerate(python_files):
        logger.debug(f"[{i + 1}/{len(python_files)}] Auditing: {module_path}")

        # Test import
        import_result = test_module_import(module_path)

        # Run smoke test (if not skipped and import succeeded)
        smoke_result = None
        if not config.skip_smoke_tests and import_result.success:
            smoke_result = run_smoke_test(module_path)

        # Assess module
        assessment = assess_module(module_path, import_result, smoke_result)
        generator.add_assessment(assessment)

        # Log status
        if assessment.status == "broken":
            broken_count += 1
            logger.warning(f"❌ {module_path}: {assessment.notes}")
        elif assessment.status == "needs_fixing":
            needs_fixing_count += 1
            if args.verbose:
                logger.info(f"⚠️ {module_path}: {assessment.notes}")
        else:
            working_count += 1
            if args.verbose:
                logger.info(f"✅ {module_path}: OK")

    # Generate and save reports
    output_dir = Path(".")
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save reports based on format
    if args.format in ["markdown", "both"]:
        md_path = output_dir / "АУДИТ.md"
        generator.save_markdown_report(md_path)

    if args.format in ["json", "both"]:
        json_path = output_dir / "audit_report.json"
        generator.save_json_report(json_path)

    # Print summary
    summary = generator.generate_summary()
    logger.info("\n📊 Audit Complete!")
    logger.info(f"   Total modules: {summary.total_modules}")
    logger.info(f"   Working ✅: {summary.working}")
    logger.info(f"   Broken ❌: {summary.broken}")
    logger.info(f"   Needs Fixing ⚠️: {summary.needs_fixing}")

    # Return appropriate exit code
    if broken_count > 0:
        return 1  # Warnings (some modules broken)
    elif needs_fixing_count > 0:
        return 0  # Success with improvements needed
    else:
        return 0  # All working


if __name__ == "__main__":
    sys.exit(main())
