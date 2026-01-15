"""Audit module - Report generation (Markdown and JSON).

Per FR-002 and research.md requirements:
- Generate Markdown report (АУДИТ.md) with table format
- Generate JSON report (audit_report.json) per data-model.md
- Include summary statistics and metadata
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .assessor import ModuleAssessment, ModuleStatus


@dataclass
class AuditSummary:
    """Summary statistics for audit report."""

    total_modules: int = 0
    working: int = 0
    broken: int = 0
    needs_fixing: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_modules": self.total_modules,
            "working": self.working,
            "broken": self.broken,
            "needs_fixing": self.needs_fixing,
        }

    def calculate_percentages(self) -> dict[str, float]:
        """Calculate percentage breakdown."""
        if self.total_modules == 0:
            return {"working": 0, "broken": 0, "needs_fixing": 0}

        return {
            "working": round(100 * self.working / self.total_modules, 1),
            "broken": round(100 * self.broken / self.total_modules, 1),
            "needs_fixing": round(100 * self.needs_fixing / self.total_modules, 1),
        }


class AuditReportGenerator:
    """Generator for audit reports (Markdown and JSON)."""

    def __init__(
        self,
        scope: str = "src/",
        auditor: str = "Automated Audit System",
        version: str = "1.0.0",
    ):
        self.scope = scope
        self.auditor = auditor
        self.version = version
        self.timestamp = datetime.now().isoformat()
        self.assessments: list[ModuleAssessment] = []

    def add_assessment(self, assessment: ModuleAssessment) -> None:
        """Add a module assessment to the report.

        Args:
            assessment: Module assessment result
        """
        self.assessments.append(assessment)

    def generate_summary(self) -> AuditSummary:
        """Generate summary statistics from assessments.

        Returns:
            AuditSummary with counts
        """
        summary = AuditSummary(total_modules=len(self.assessments))

        for assessment in self.assessments:
            if assessment.status == ModuleStatus.WORKING.value:
                summary.working += 1
            elif assessment.status == ModuleStatus.BROKEN.value:
                summary.broken += 1
            elif assessment.status == ModuleStatus.NEEDS_FIXING.value:
                summary.needs_fixing += 1

        return summary

    def generate_markdown_report(self) -> str:
        """Generate Markdown audit report (АУДИТ.md).

        Returns:
            Markdown formatted report string
        """
        summary = self.generate_summary()
        percentages = summary.calculate_percentages()

        lines = [
            "# Code Audit Report",
            "",
            f"**Date**: {self.timestamp.split('T')[0]}",
            f"**Auditor**: {self.auditor}",
            f"**Scope**: {self.scope}",
            f"**Version**: {self.version}",
            "",
            "## Summary",
            "",
            "| Status | Count | Percentage |",
            "|--------|-------|------------|",
            f"| Working ✅ | {summary.working} | {percentages['working']}% |",
            f"| Broken ❌ | {summary.broken} | {percentages['broken']}% |",
            f"| Needs Fixing ⚠️ | {summary.needs_fixing} | {percentages['needs_fixing']}% |",
            "",
            "## Module Details",
            "",
            "| Module | Path | Import Status | Functionality Test | Status | Notes |",
            "|--------|------|---------------|-------------------|--------|-------|",
        ]

        # Add module rows
        for assessment in self.assessments:
            module_name = assessment.module_name
            file_path = assessment.file_path
            import_status = f"{assessment.status_icon} {assessment.import_status}"
            test_status = f"{assessment.status_icon} {assessment.functionality_test}"
            status = f"{assessment.status_icon} {assessment.status.capitalize()}"
            notes = assessment.notes if assessment.notes else "-"

            lines.append(
                f"| {module_name} | {file_path} | {import_status} | {test_status} | {status} | {notes} |"
            )

        lines.extend(["", "## Recommendations", ""])

        # Generate recommendations based on broken modules
        broken_modules = [
            a for a in self.assessments if a.status == ModuleStatus.BROKEN.value
        ]
        needs_fixing_modules = [
            a for a in self.assessments
            if a.status == ModuleStatus.NEEDS_FIXING.value
        ]

        if broken_modules:
            lines.append("### Critical Issues (Broken Modules)")
            for module in broken_modules:
                lines.append(f"- **{module.module_name}**: {module.notes}")
            lines.append("")

        if needs_fixing_modules:
            lines.append("### Improvements Needed")
            for module in needs_fixing_modules:
                lines.append(f"- **{module.module_name}**: {module.notes}")
            lines.append("")

        if not broken_modules and not needs_fixing_modules:
            lines.append("✅ All modules are working correctly! No fixes needed.")
            lines.append("")

        return "\n".join(lines)

    def generate_json_report(self) -> dict[str, Any]:
        """Generate JSON audit report per data-model.md.

        Returns:
            Dictionary for JSON serialization
        """
        summary = self.generate_summary()

        return {
            "audit_report": {
                "metadata": {
                    "date": self.timestamp,
                    "auditor": self.auditor,
                    "scope": self.scope,
                    "version": self.version,
                },
                "summary": summary.to_dict(),
                "modules": [a.to_dict() for a in self.assessments],
            }
        }

    def save_markdown_report(self, filepath: Path) -> None:
        """Save Markdown report to file.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_markdown_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Markdown report saved: {filepath}")

    def save_json_report(self, filepath: Path) -> None:
        """Save JSON report to file.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_json_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON report saved: {filepath}")


def generate_audit_report(
    assessments: list[ModuleAssessment],
    scope: str = "src/",
    output_dir: Path = Path("."),
) -> tuple[Path, Path]:
    """Generate and save both Markdown and JSON audit reports.

    Args:
        assessments: List of module assessments
        scope: Directory scope that was audited
        output_dir: Output directory for reports

    Returns:
        Tuple of (markdown_path, json_path)
    """
    generator = AuditReportGenerator(scope=scope)
    for assessment in assessments:
        generator.add_assessment(assessment)

    # Save Markdown report (АУДИТ.md)
    markdown_path = output_dir / "АУДИТ.md"
    generator.save_markdown_report(markdown_path)

    # Save JSON report
    json_path = output_dir / "audit_report.json"
    generator.save_json_report(json_path)

    return markdown_path, json_path


if __name__ == "__main__":
    # Test report generation
    from src.audit.core import test_module_import, run_smoke_test, discover_python_files
    from src.audit.assessor import assess_module
    from pathlib import Path

    print("Testing report generation...")

    # Run audit on src/ directory
    scope = Path("src/")
    python_files = discover_python_files(scope)

    print(f"Found {len(python_files)} Python files in {scope}")

    # Create generator
    generator = AuditReportGenerator(scope=str(scope))

    # Assess each module
    for module_path in python_files[:10]:  # Limit to first 10 for testing
        import_result = test_module_import(module_path)
        smoke_result = None
        if import_result.success:
            smoke_result = run_smoke_test(module_path)

        assessment = assess_module(module_path, import_result, smoke_result)
        generator.add_assessment(assessment)

        print(f"  {assessment.status_icon} {assessment.module_name}: {assessment.status}")

    # Generate and save reports
    print("\nGenerating reports...")
    md_path, json_path = generate_audit_report(generator.assessments, str(scope))

    print(f"\n✅ Reports generated:")
    print(f"   Markdown: {md_path}")
    print(f"   JSON: {json_path}")

    # Display summary
    summary = generator.generate_summary()
    print(f"\n📊 Summary:")
    print(f"   Total: {summary.total_modules}")
    print(f"   Working: {summary.working} ✅")
    print(f"   Broken: {summary.broken} ❌")
    print(f"   Needs Fixing: {summary.needs_fixing} ⚠️")
