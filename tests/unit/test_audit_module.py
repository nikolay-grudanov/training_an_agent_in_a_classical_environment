"""Unit tests for audit module components."""

import pytest
from pathlib import Path
import tempfile
import os

from src.audit.core import (
    AuditConfig,
    ImportTestResult,
    SmokeTestResult,
    test_module_import,
    discover_python_files,
)
from src.audit.assessor import (
    ModuleAssessment,
    ModuleStatus,
    assess_module,
    determine_final_status,
    generate_fix_suggestion,
)
from src.audit.report_generator import (
    AuditReportGenerator,
    generate_audit_report,
)


class TestAuditConfig:
    """Tests for AuditConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuditConfig()
        assert config.scope == Path("src/")
        assert config.verbose is False
        assert config.skip_smoke_tests is False
        assert config.version == "1.0.0"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AuditConfig(
            scope=Path("tests/"),
            verbose=True,
            skip_smoke_tests=True,
            version="2.0.0",
        )
        assert config.scope == Path("tests/")
        assert config.verbose is True
        assert config.skip_smoke_tests is True
        assert config.version == "2.0.0"


class TestImportTestResult:
    """Tests for ImportTestResult dataclass."""

    def test_successful_import(self):
        """Test successful import result."""
        result = ImportTestResult(success=True)
        assert result.success is True
        assert result.error_message is None
        assert result.error_type is None

    def test_failed_import(self):
        """Test failed import result."""
        result = ImportTestResult(
            success=False,
            error_message="ModuleNotFoundError: No module named 'nonexistent'",
            error_type="ModuleNotFoundError",
        )
        assert result.success is False
        assert (
            result.error_message == "ModuleNotFoundError: No module named 'nonexistent'"
        )
        assert result.error_type == "ModuleNotFoundError"


class TestSmokeTestResult:
    """Tests for SmokeTestResult dataclass."""

    def test_successful_smoke_test(self):
        """Test successful smoke test result."""
        result = SmokeTestResult(success=True)
        assert result.success is True
        assert result.error_message is None

    def test_failed_smoke_test(self):
        """Test failed smoke test result."""
        result = SmokeTestResult(
            success=False,
            error_message="RuntimeError: some error",
            error_type="RuntimeError",
        )
        assert result.success is False
        assert result.error_message == "RuntimeError: some error"


class TestImportTestingFunctions:
    """Tests for import testing functions."""

    def test_import_existing_module(self):
        """Test importing an existing working module."""
        # Use seeding.py which we know works
        module_path = Path("src/utils/seeding.py")
        if module_path.exists():
            result = test_module_import(module_path)
            assert result.success is True

    def test_import_nonexistent_file(self):
        """Test importing a non-existent file."""
        module_path = Path("/tmp/nonexistent_module_12345.py")
        result = test_module_import(module_path)
        assert result.success is False
        assert result.error_type == "FileNotFoundError"

    def test_import_corrupted_file(self):
        """Test importing a file with syntax errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken_function():\n    syntax error here\n")
            temp_path = f.name

        try:
            result = test_module_import(Path(temp_path))
            assert result.success is False
            assert result.error_type == "SyntaxError"
        finally:
            os.unlink(temp_path)


class TestDiscoverPythonFiles:
    """Tests for discover_python_files function."""

    def test_discover_in_src(self):
        """Test discovering Python files in src directory."""
        files = discover_python_files(Path("src/"))
        assert len(files) > 0
        assert all(f.suffix == ".py" for f in files)
        # Should not include __pycache__
        for f in files:
            assert "__pycache__" not in str(f)

    def test_discover_single_file(self):
        """Test discovering single file."""
        files = discover_python_files(Path("src/utils/seeding.py"))
        assert len(files) == 1
        assert files[0].name == "seeding.py"

    def test_discover_nonexistent(self):
        """Test discovering in non-existent directory."""
        files = discover_python_files(Path("/tmp/nonexistent_12345/"))
        assert len(files) == 0


class TestAssessModule:
    """Tests for assess_module function."""

    def test_assess_working_module(self):
        """Test assessing a working module."""
        module_path = Path("src/utils/seeding.py")
        if not module_path.exists():
            pytest.skip("Module not found")

        import_result = ImportTestResult(success=True)
        smoke_result = SmokeTestResult(success=True)

        assessment = assess_module(module_path, import_result, smoke_result)

        assert assessment.status == "working"
        assert assessment.status_icon == "✅"
        assert assessment.import_status == "success"
        assert assessment.functionality_test == "pass"

    def test_assess_broken_module(self):
        """Test assessing a broken module."""
        module_path = Path("src/utils/seeding.py")
        if not module_path.exists():
            pytest.skip("Module not found")

        import_result = ImportTestResult(
            success=False,
            error_message="ModuleNotFoundError: No module named 'nonexistent'",
            error_type="ModuleNotFoundError",
        )

        assessment = assess_module(module_path, import_result, None)

        assert assessment.status == "broken"
        assert assessment.status_icon == "❌"
        assert assessment.import_status == "error"

    def test_assess_module_with_dep_error(self):
        """Test assessing module with dependency error."""
        module_path = Path("src/agents/base.py")
        if not module_path.exists():
            pytest.skip("Module not found")

        import_result = ImportTestResult(
            success=False,
            error_message="ImportError: some import failed",
            error_type="ImportError",
        )

        assessment = assess_module(module_path, import_result, None)

        assert assessment.status == "broken"
        assert assessment.notes == "ImportError: some import failed"


class TestDetermineFinalStatus:
    """Tests for determine_final_status function."""

    def test_working_module(self):
        """Test status determination for working module."""
        status, icon, notes = determine_final_status(
            "test_module",
            "success",
            "pass",
            ImportTestResult(success=True),
            SmokeTestResult(success=True),
        )
        assert status == ModuleStatus.WORKING
        assert icon == "✅"
        assert notes == "No issues detected"

    def test_broken_module_with_missing_dep(self):
        """Test status for module with missing dependency."""
        import_result = ImportTestResult(
            success=False,
            error_message="ModuleNotFoundError: missing_dep",
            error_type="ModuleNotFoundError",
        )
        status, icon, notes = determine_final_status(
            "test_module",
            "error",
            "skip",
            import_result,
            None,
        )
        assert status == ModuleStatus.BROKEN
        assert icon == "❌"
        assert "missing_dep" in notes

    def test_needs_fixing_module(self):
        """Test status for module needing fixes."""
        import_result = ImportTestResult(success=True)
        smoke_result = SmokeTestResult(
            success=False,
            error_message="RuntimeError",
            error_type="RuntimeError",
        )
        status, icon, notes = determine_final_status(
            "test_module",
            "success",
            "fail",
            import_result,
            smoke_result,
        )
        assert status == ModuleStatus.NEEDS_FIXING
        assert icon == "⚠️"


class TestModuleAssessment:
    """Tests for ModuleAssessment dataclass."""

    def test_to_dict(self):
        """Test ModuleAssessment to_dict method."""
        assessment = ModuleAssessment(
            module_name="test_module",
            file_path=Path("src/test_module.py"),
            import_status="success",
            functionality_test="pass",
            status="working",
            status_icon="✅",
            notes="All good",
        )

        result = assessment.to_dict()

        assert result["module_name"] == "test_module"
        assert result["status"] == "working"
        assert result["status_icon"] == "✅"
        assert result["notes"] == "All good"


class TestAuditReportGenerator:
    """Tests for AuditReportGenerator class."""

    def test_empty_generator(self):
        """Test generator with no assessments."""
        generator = AuditReportGenerator(scope="src/")
        summary = generator.generate_summary()

        assert summary.total_modules == 0
        assert summary.working == 0
        assert summary.broken == 0
        assert summary.needs_fixing == 0

    def test_generator_with_assessments(self):
        """Test generator with sample assessments."""
        generator = AuditReportGenerator(scope="src/")

        # Add working assessment
        generator.add_assessment(
            ModuleAssessment(
                module_name="working_module",
                file_path=Path("src/working.py"),
                import_status="success",
                functionality_test="pass",
                status="working",
                status_icon="✅",
            )
        )

        # Add broken assessment
        generator.add_assessment(
            ModuleAssessment(
                module_name="broken_module",
                file_path=Path("src/broken.py"),
                import_status="error",
                functionality_test="skip",
                status="broken",
                status_icon="❌",
                notes="Missing dep",
            )
        )

        summary = generator.generate_summary()
        assert summary.total_modules == 2
        assert summary.working == 1
        assert summary.broken == 1

    def test_markdown_generation(self):
        """Test Markdown report generation."""
        generator = AuditReportGenerator(scope="src/")
        generator.add_assessment(
            ModuleAssessment(
                module_name="test_module",
                file_path=Path("src/test_module.py"),
                import_status="success",
                functionality_test="pass",
                status="working",
                status_icon="✅",
            )
        )

        report = generator.generate_markdown_report()

        assert "# Code Audit Report" in report
        assert "test_module" in report
        assert "Working ✅" in report

    def test_json_generation(self):
        """Test JSON report generation."""
        generator = AuditReportGenerator(scope="src/")
        generator.add_assessment(
            ModuleAssessment(
                module_name="test_module",
                file_path=Path("src/test_module.py"),
                import_status="success",
                functionality_test="pass",
                status="working",
                status_icon="✅",
            )
        )

        report = generator.generate_json_report()

        assert "audit_report" in report
        assert report["audit_report"]["metadata"]["scope"] == "src/"
        assert len(report["audit_report"]["modules"]) == 1

    def test_percentages(self):
        """Test percentage calculations."""
        generator = AuditReportGenerator(scope="src/")
        for i in range(4):
            generator.add_assessment(
                ModuleAssessment(
                    module_name=f"module_{i}",
                    file_path=Path(f"src/module_{i}.py"),
                    import_status="success",
                    functionality_test="pass",
                    status="working",
                    status_icon="✅",
                )
            )

        summary = generator.generate_summary()
        percentages = summary.calculate_percentages()

        assert percentages["working"] == 100.0


class TestGenerateFixSuggestion:
    """Tests for generate_fix_suggestion function."""

    def test_fix_suggestion_for_broken(self):
        """Test fix suggestion for broken module."""
        assessment = ModuleAssessment(
            module_name="test_module",
            file_path=Path("src/test_module.py"),
            import_status="error",
            functionality_test="skip",
            status="broken",
            status_icon="❌",
            error_message="ModuleNotFoundError: missing_lib",
            error_type="ModuleNotFoundError",
        )

        suggestion = generate_fix_suggestion(assessment)

        assert "Install missing dependency" in suggestion
        assert "missing_lib" in suggestion

    def test_fix_suggestion_for_needs_fixing(self):
        """Test fix suggestion for module needing fixes."""
        assessment = ModuleAssessment(
            module_name="test_module",
            file_path=Path("src/test_module.py"),
            import_status="success",
            functionality_test="fail",
            status="needs_fixing",
            status_icon="⚠️",
            notes="RuntimeError: some issue",
        )

        suggestion = generate_fix_suggestion(assessment)

        assert "Investigate" in suggestion
        assert "some issue" in suggestion


class TestGenerateAuditReport:
    """Tests for generate_audit_report function."""

    def test_generate_audit_report(self):
        """Test complete report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assessments = [
                ModuleAssessment(
                    module_name="test_module",
                    file_path=Path("src/test_module.py"),
                    import_status="success",
                    functionality_test="pass",
                    status="working",
                    status_icon="✅",
                )
            ]

            md_path, json_path = generate_audit_report(
                assessments, "src/", Path(tmpdir)
            )

            assert md_path.exists()
            assert json_path.exists()

            # Check Markdown content
            with open(md_path) as f:
                content = f.read()
            assert "# Code Audit Report" in content
            assert "test_module" in content

            # Check JSON content
            import json

            with open(json_path) as f:
                data = json.load(f)
            assert "audit_report" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
