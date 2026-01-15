"""Audit module - Module assessment and status determination.

Per FR-012 requirements:
- Determine module status (working/broken/needs_fixing)
- Assign status icons (✅/❌/⚠️)
- Generate notes for issues and fixes needed
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .core import ImportTestResult, SmokeTestResult


class ModuleStatus(Enum):
    """Module status categories."""

    WORKING = "working"
    BROKEN = "broken"
    NEEDS_FIXING = "needs_fixing"


# Status icons per research.md and spec
STATUS_ICONS = {
    ModuleStatus.WORKING: "✅",
    ModuleStatus.BROKEN: "❌",
    ModuleStatus.NEEDS_FIXING: "⚠️",
}


@dataclass
class ModuleAssessment:
    """Assessment result for a single module."""

    module_name: str
    file_path: Path
    import_status: str  # "success", "error", "warning"
    functionality_test: str  # "pass", "fail", "skip"
    status: str  # "working", "broken", "needs_fixing"
    status_icon: str  # "✅", "❌", "⚠️"
    notes: str = ""
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_name": self.module_name,
            "file_path": str(self.file_path),
            "import_status": self.import_status,
            "functionality_test": self.functionality_test,
            "status": self.status,
            "status_icon": self.status_icon,
            "notes": self.notes,
            "error_message": self.error_message,
            "error_type": self.error_type,
        }


def assess_module(
    module_path: Path,
    import_result: ImportTestResult,
    smoke_result: Optional[SmokeTestResult] = None,
) -> ModuleAssessment:
    """Assess a module and determine its status.

    Args:
        module_path: Path to the module
        import_result: Result of import test
        smoke_result: Result of smoke test (optional)

    Returns:
        ModuleAssessment with determined status
    """
    module_name = module_path.stem

    # Determine import status
    if import_result.success:
        import_status = "success"
        import_icon = STATUS_ICONS[ModuleStatus.WORKING]
    elif import_result.error_type == "ModuleNotFoundError":
        import_status = "error"
        import_icon = STATUS_ICONS[ModuleStatus.BROKEN]
    else:
        import_status = "error"
        import_icon = STATUS_ICONS[ModuleStatus.BROKEN]

    # Determine functionality test status
    if import_result.success:
        if smoke_result is None:
            functionality_test = "skip"
            test_icon = STATUS_ICONS[ModuleStatus.NEEDS_FIXING]
        elif smoke_result.success:
            functionality_test = "pass"
            test_icon = STATUS_ICONS[ModuleStatus.WORKING]
        else:
            functionality_test = "fail"
            test_icon = STATUS_ICONS[ModuleStatus.NEEDS_FIXING]
    else:
        functionality_test = "skip"
        test_icon = STATUS_ICONS[ModuleStatus.BROKEN]

    # Determine final status
    status, status_icon, notes = determine_final_status(
        module_name, import_status, functionality_test, import_result, smoke_result
    )

    return ModuleAssessment(
        module_name=module_name,
        file_path=module_path,
        import_status=import_status,
        functionality_test=functionality_test,
        status=status.value,
        status_icon=status_icon,
        notes=notes,
        error_message=import_result.error_message,
        error_type=import_result.error_type,
    )


def determine_final_status(
    module_name: str,
    import_status: str,
    functionality_test: str,
    import_result: ImportTestResult,
    smoke_result: Optional[SmokeTestResult],
) -> tuple[ModuleStatus, str, str]:
    """Determine final module status based on import and test results.

    Args:
        module_name: Name of the module
        import_status: Import test status
        functionality_test: Smoke test status
        import_result: Full import test result
        smoke_result: Full smoke test result

    Returns:
        Tuple of (ModuleStatus, status_icon, notes)
    """
    # Case 1: Import successful and smoke test passed → WORKING
    if import_status == "success" and functionality_test == "pass":
        return (
            ModuleStatus.WORKING,
            STATUS_ICONS[ModuleStatus.WORKING],
            "No issues detected",
        )

    # Case 2: Import successful but smoke test failed → NEEDS FIXING
    if import_status == "success" and functionality_test == "fail":
        notes = "Smoke test failed"
        if smoke_result and smoke_result.error_message:
            notes += f": {smoke_result.error_message}"
        return (
            ModuleStatus.NEEDS_FIXING,
            STATUS_ICONS[ModuleStatus.NEEDS_FIXING],
            notes,
        )

    # Case 3: Import failed with ModuleNotFoundError → BROKEN
    if import_result.error_type == "ModuleNotFoundError":
        error_msg = import_result.error_message or "Missing dependency"
        return (
            ModuleStatus.BROKEN,
            STATUS_ICONS[ModuleStatus.BROKEN],
            error_msg,
        )

    # Case 4: Import failed with other error → BROKEN
    if import_status == "error":
        error_msg = import_result.error_message or "Import failed"
        return (
            ModuleStatus.BROKEN,
            STATUS_ICONS[ModuleStatus.BROKEN],
            error_msg,
        )

    # Case 5: Import successful, no smoke test → NEEDS FIXING
    if import_status == "success" and functionality_test == "skip":
        return (
            ModuleStatus.NEEDS_FIXING,
            STATUS_ICONS[ModuleStatus.NEEDS_FIXING],
            "Smoke test not executed",
        )

    # Default fallback
    return (
        ModuleStatus.NEEDS_FIXING,
        STATUS_ICONS[ModuleStatus.NEEDS_FIXING],
        "Unable to determine status",
    )


def generate_fix_suggestion(assessment: ModuleAssessment) -> str:
    """Generate fix suggestion based on assessment.

    Args:
        assessment: Module assessment result

    Returns:
        Suggested fix action
    """
    if assessment.status == ModuleStatus.BROKEN.value:
        if assessment.error_type == "ModuleNotFoundError":
            return f"Install missing dependency: {assessment.error_message}"
        else:
            return f"Fix import error in {assessment.module_name}: {assessment.error_message}"

    if assessment.status == ModuleStatus.NEEDS_FIXING.value:
        if "Smoke test" in assessment.notes:
            return f"Execute smoke test for {assessment.module_name}"
        else:
            return f"Investigate issues in {assessment.module_name}: {assessment.notes}"

    return "No fixes needed"


if __name__ == "__main__":
    # Test module assessment
    from src.audit.core import test_module_import, run_smoke_test

    print("Testing module assessment...")

    # Test on existing modules
    test_modules = [
        Path("src/agents/base.py"),
        Path("src/utils/seeding.py"),
        Path("src/utils/metrics_exporter.py"),
    ]

    for module_path in test_modules:
        if module_path.exists():
            print(f"\n📄 Assessing: {module_path}")
            import_result = test_module_import(module_path)
            smoke_result = (
                run_smoke_test(module_path) if import_result.success else None
            )

            assessment = assess_module(module_path, import_result, smoke_result)

            print(f"   Import: {assessment.import_status} {assessment.status_icon}")
            print(f"   Test: {assessment.functionality_test}")
            print(f"   Status: {assessment.status} {assessment.status_icon}")
            print(f"   Notes: {assessment.notes}")

    print("\n✅ Module assessment test complete!")
