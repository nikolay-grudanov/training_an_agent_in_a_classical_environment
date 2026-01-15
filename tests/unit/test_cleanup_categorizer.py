"""Unit tests for cleanup categorizer module.

Tests for CleanupCategorizer class.
"""

import tempfile
from pathlib import Path

import pytest

from src.cleanup.categorizer import (
    CleanupCategorizer,
    CategorizationResult,
    categorize_file,
    categorize_directory,
)
from src.cleanup.core import FileCategory, DirCategory


class TestCleanupCategorizer:
    """Tests for CleanupCategorizer class."""

    def test_init(self) -> None:
        """Test CleanupCategorizer initialization."""
        categorizer = CleanupCategorizer()
        assert categorizer.root_path == Path(".").resolve()

    def test_init_with_custom_root(self) -> None:
        """Test CleanupCategorizer initialization with custom root."""
        custom_root = Path("/tmp/test_root")
        categorizer = CleanupCategorizer(root_path=custom_root)
        assert categorizer.root_path == custom_root.resolve()

    def test_categorize_file_keep_readme(self) -> None:
        """Test categorizing README.md as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            readme = tmpdir_path / "README.md"
            readme.write_text("# README")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(readme)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_file_keep_requirements(self) -> None:
        """Test categorizing requirements.txt as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            requirements = tmpdir_path / "requirements.txt"
            requirements.write_text("pytest")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(requirements)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_file_keep_gitignore(self) -> None:
        """Test categorizing .gitignore as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore = tmpdir_path / ".gitignore"
            gitignore.write_text("*.pyc")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(gitignore)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_file_keep_pytest_ini(self) -> None:
        """Test categorizing pytest.ini as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pytest_ini = tmpdir_path / "pytest.ini"
            pytest_ini.write_text("[pytest]")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(pytest_ini)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_file_remove_md_files(self) -> None:
        """Test categorizing .md files (except exceptions) as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_md = tmpdir_path / "test.md"
            test_md.write_text("# test")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(test_md)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_py_files(self) -> None:
        """Test categorizing .py files in root as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            example_py = tmpdir_path / "example.py"
            example_py.write_text("# example")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(example_py)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_environment_yml(self) -> None:
        """Test categorizing environment.yml as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            env_yml = tmpdir_path / "environment.yml"
            env_yml.write_text("name: test")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(env_yml)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_env_example(self) -> None:
        """Test categorizing .env.example as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            env_example = tmpdir_path / ".env.example"
            env_example.write_text("API_KEY=test")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(env_example)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_install_sh(self) -> None:
        """Test categorizing install.sh as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            install_sh = tmpdir_path / "install.sh"
            install_sh.write_text("#!/bin/bash")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(install_sh)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_makefile(self) -> None:
        """Test categorizing Makefile as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            makefile = tmpdir_path / "Makefile"
            makefile.write_text("test:\n\techo test")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(makefile)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_keep_agents_md(self) -> None:
        """Test categorizing AGENTS.md as keep (exception)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            agents_md = tmpdir_path / "AGENTS.md"
            agents_md.write_text("# AGENTS")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(agents_md)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_file_remove_src_logging_py(self) -> None:
        """Test categorizing src/utils/rl_logging.py as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src" / "utils"
            src_dir.mkdir(parents=True)
            logging_py = src_dir / "rl_logging.py"
            logging_py.write_text("# logging")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(logging_py)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_remove_src_logging_config_py(self) -> None:
        """Test categorizing src/utils/logging_setup.py as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src" / "utils"
            src_dir.mkdir(parents=True)
            logging_config_py = src_dir / "logging_setup.py"
            logging_config_py.write_text("# logging config")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(logging_config_py)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.REMOVE

    def test_categorize_file_keep_src_seeding_py(self) -> None:
        """Test categorizing src/utils/seeding.py as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src" / "utils"
            src_dir.mkdir(parents=True)
            seeding_py = src_dir / "seeding.py"
            seeding_py.write_text("# seeding")

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_file(seeding_py)
            assert isinstance(result, CategorizationResult)
            assert result.category == FileCategory.KEEP

    def test_categorize_directory_keep_src(self) -> None:
        """Test categorizing src/ as protected (since it's in protected list)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(src_dir)
            assert isinstance(result, CategorizationResult)
            # src is in PROTECTED_DIRECTORIES, so it should be protected
            assert result.category == DirCategory.PROTECTED

    def test_categorize_directory_keep_tests(self) -> None:
        """Test categorizing tests/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(tests_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_results(self) -> None:
        """Test categorizing results/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            results_dir = tmpdir_path / "results"
            results_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(results_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_specs(self) -> None:
        """Test categorizing specs/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            specs_dir = tmpdir_path / "specs"
            specs_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(specs_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_remove_configs(self) -> None:
        """Test categorizing configs/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(configs_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_data(self) -> None:
        """Test categorizing data/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_dir = tmpdir_path / "data"
            data_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(data_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_demo_checkpoints(self) -> None:
        """Test categorizing demo_checkpoints/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            demo_checkpoints_dir = tmpdir_path / "demo_checkpoints"
            demo_checkpoints_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(demo_checkpoints_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_docs(self) -> None:
        """Test categorizing docs/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            docs_dir = tmpdir_path / "docs"
            docs_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(docs_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_examples(self) -> None:
        """Test categorizing examples/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            examples_dir = tmpdir_path / "examples"
            examples_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(examples_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_logs(self) -> None:
        """Test categorizing logs/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            logs_dir = tmpdir_path / "logs"
            logs_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(logs_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_notebooks(self) -> None:
        """Test categorizing notebooks/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            notebooks_dir = tmpdir_path / "notebooks"
            notebooks_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(notebooks_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_scripts(self) -> None:
        """Test categorizing scripts/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scripts_dir = tmpdir_path / "scripts"
            scripts_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(scripts_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_protected_git(self) -> None:
        """Test categorizing .git/ as protected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            git_dir = tmpdir_path / ".git"
            git_dir.mkdir()

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(git_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.PROTECTED

    def test_categorize_directory_remove_src_api(self) -> None:
        """Test categorizing src/api/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            api_dir = tmpdir_path / "src" / "api"
            api_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(api_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_remove_src_visualization(self) -> None:
        """Test categorizing src/visualization/ as remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            viz_dir = tmpdir_path / "src" / "visualization"
            viz_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(viz_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE

    def test_categorize_directory_keep_src_training(self) -> None:
        """Test categorizing src/training/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            training_dir = tmpdir_path / "src" / "training"
            training_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(training_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_src_agents(self) -> None:
        """Test categorizing src/agents/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            agents_dir = tmpdir_path / "src" / "agents"
            agents_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(agents_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_src_utils(self) -> None:
        """Test categorizing src/utils/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            utils_dir = tmpdir_path / "src" / "utils"
            utils_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(utils_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_src_environments(self) -> None:
        """Test categorizing src/environments/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            env_dir = tmpdir_path / "src" / "environments"
            env_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(env_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_src_audit(self) -> None:
        """Test categorizing src/audit/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audit_dir = tmpdir_path / "src" / "audit"
            audit_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(audit_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_directory_keep_src_cleanup(self) -> None:
        """Test categorizing src/cleanup/ as keep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cleanup_dir = tmpdir_path / "src" / "cleanup"
            cleanup_dir.mkdir(parents=True)

            categorizer = CleanupCategorizer(root_path=tmpdir_path)
            result = categorizer.categorize_directory(cleanup_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.KEEP

    def test_categorize_file_raises_for_directory(self) -> None:
        """Test that categorize_file raises ValueError for directory."""
        categorizer = CleanupCategorizer()
        with pytest.raises(ValueError, match="is a directory, not a file"):
            categorizer.categorize_file(Path("src"))

    def test_categorize_directory_raises_for_file(self) -> None:
        """Test that categorize_directory raises ValueError for file."""
        categorizer = CleanupCategorizer()
        with pytest.raises(ValueError, match="is not a directory"):
            categorizer.categorize_directory(Path("README.md"))

    def test_get_all_items_to_remove(self) -> None:
        """Test get_all_items_to_remove returns list of paths."""
        categorizer = CleanupCategorizer()
        items = categorizer.get_all_items_to_remove()
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, Path)

    def test_get_removal_summary(self) -> None:
        """Test get_removal_summary returns correct format."""
        categorizer = CleanupCategorizer()
        summary = categorizer.get_removal_summary()
        assert isinstance(summary, dict)
        assert "total_count" in summary
        assert "files" in summary
        assert "directories" in summary
        assert "root_items" in summary
        assert "src_items" in summary
        assert isinstance(summary["total_count"], int)
        assert isinstance(summary["files"], list)
        assert isinstance(summary["directories"], list)
        assert isinstance(summary["root_items"], list)
        assert isinstance(summary["src_items"], list)


class TestCategorizeFile:
    """Tests for categorize_file function."""

    def test_categorize_file_readme(self) -> None:
        """Test categorize_file function with README.md."""
        # categorize_file uses default root (current directory), so we can't test with temp files
        # Just test that it returns a CategorizationResult
        result = categorize_file(Path("README.md"))
        assert isinstance(result, CategorizationResult)

    def test_categorize_file_example_py(self) -> None:
        """Test categorize_file function with example.py."""
        # categorize_file uses default root (current directory), so we can't test with temp files
        # Just test that it returns a CategorizationResult
        result = categorize_file(Path("example.py"))
        assert isinstance(result, CategorizationResult)


class TestCategorizeDirectory:
    """Tests for categorize_directory function."""

    def test_categorize_directory_src(self) -> None:
        """Test categorize_directory function with src/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()

            result = categorize_directory(src_dir)
            assert isinstance(result, CategorizationResult)
            # src is in PROTECTED_DIRECTORIES, so it should be protected
            assert result.category == DirCategory.PROTECTED

    def test_categorize_directory_configs(self) -> None:
        """Test categorize_directory function with configs/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()

            result = categorize_directory(configs_dir)
            assert isinstance(result, CategorizationResult)
            assert result.category == DirCategory.REMOVE


class TestCategorizationResult:
    """Tests for CategorizationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a CategorizationResult."""
        result = CategorizationResult(
            path=Path("test.md"),
            category=FileCategory.REMOVE,
            reason="Test reason",
        )
        assert result.path == Path("test.md")
        assert result.category == FileCategory.REMOVE
        assert result.reason == "Test reason"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
