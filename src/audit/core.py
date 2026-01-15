"""Audit module core - Module import testing and smoke tests.

Per research.md decisions:
- Use importlib.util.spec_from_file_location for import testing
- Execute basic smoke tests for main functions
- Handle import errors and capture detailed error messages
"""

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AuditConfig:
    """Configuration for code audit process."""

    scope: Path = Path("src/")
    verbose: bool = False
    skip_smoke_tests: bool = False
    version: str = "1.0.0"


@dataclass
class ImportTestResult:
    """Result of module import test."""

    success: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class SmokeTestResult:
    """Result of smoke test execution."""

    success: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None


def test_module_import(module_path: Path) -> ImportTestResult:
    """Test if module can be imported without executing its full code.

    Args:
        module_path: Path to the Python module file

    Returns:
        ImportTestResult with success status and error details
    """
    try:
        # Get module name from file path
        module_name = module_path.stem

        # Check if spec can be created
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return ImportTestResult(
                success=False,
                error_message="No spec/loader found for module",
                error_type="ImportError",
            )

        # Create module
        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules to handle dependencies
        sys.modules[module_name] = module

        # Try to execute module
        spec.loader.exec_module(module)

        return ImportTestResult(success=True)

    except ImportError as e:
        return ImportTestResult(
            success=False,
            error_message=str(e),
            error_type="ImportError",
        )
    except Exception as e:
        return ImportTestResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
        )


def run_smoke_test(module_path: Path) -> SmokeTestResult:
    """Run basic smoke test for module functionality.

    Args:
        module_path: Path to the Python module file

    Returns:
        SmokeTestResult with success status and error details
    """
    try:
        module_name = module_path.stem

        # Import the module
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return SmokeTestResult(
                    success=False,
                    error_message="No spec/loader found",
                    error_type="ImportError",
                )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_name]

        # Run smoke tests
        smoke_tests_passed = []

        # Test 1: Check if module has __init__ or main function
        if hasattr(module, "__init__") or hasattr(module, "main"):
            smoke_tests_passed.append("has_main_function")

        # Test 2: Check if module has classes (if any)
        classes = [name for name in dir(module) if name.startswith("_") is False and isinstance(getattr(module, name, None), type)]
        if classes:
            smoke_tests_passed.append(f"has_classes:{len(classes)}")

        # Test 3: Check if module has functions
        functions = [name for name in dir(module) if name.startswith("_") is False and callable(getattr(module, name, None))]
        if functions:
            smoke_tests_passed.append(f"has_functions:{len(functions)}")

        # Test 4: Try to instantiate a class if exists
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name, None)
            if isinstance(attr, type):
                try:
                    # Try to instantiate with no args (may fail, that's ok)
                    instance = attr()
                    smoke_tests_passed.append(f"instantiable:{attr_name}")
                    break  # Only instantiate one class
                except (TypeError, ValueError):
                    # Expected for classes requiring arguments
                    pass

        if smoke_tests_passed:
            return SmokeTestResult(success=True)
        else:
            return SmokeTestResult(
                success=False,
                error_message="No smoke test criteria met",
                error_type="SmokeTestError",
            )

    except Exception as e:
        return SmokeTestResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
        )


def discover_python_files(scope: Path) -> list[Path]:
    """Discover all Python files in the given scope.

    Args:
        scope: Directory or file path to audit

    Returns:
        List of Python file paths
    """
    python_files = []

    if scope.is_file() and scope.suffix == ".py":
        return [scope]

    if scope.is_dir():
        for py_file in sorted(scope.rglob("*.py")):
            # Skip __pycache__ directories
            if "__pycache__" in py_file.parts:
                continue
            # Skip .py files starting with test_ (handled separately if needed)
            python_files.append(py_file)

    return python_files


if __name__ == "__main__":
    # Test the audit module
    from src.utils.logging_setup import setup_logging

    logger = setup_logging()
    logger.info("Testing audit module...")

    # Test on a sample file
    test_file = Path("src/agents/base.py")
    if test_file.exists():
        print(f"\nTesting import: {test_file}")
        import_result = test_module_import(test_file)
        print(f"  Import: {'✅ Success' if import_result.success else f'❌ {import_result.error_type}: {import_result.error_message}'}")

        if not import_result.success or True:
            smoke_result = run_smoke_test(test_file)
            print(f"  Smoke test: {'✅ Success' if smoke_result.success else f'❌ {smoke_result.error_type}: {smoke_result.error_message}'}")
    else:
        print(f"Test file not found: {test_file}")

    print("\n✅ Audit module test complete!")
