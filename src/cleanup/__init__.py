# Cleanup Module - Project Cleanup and PPO vs A2C Experiments

__version__ = "1.0.0"
__author__ = "Research Team"

from .core import (
    FileCategory,
    DirCategory,
    get_keep_root_items,
    get_remove_root_dirs,
    get_remove_root_files,
    get_keep_root_files_exceptions,
    get_remove_src_dirs,
    get_keep_src_dirs,
    get_remove_src_files,
    get_protected_directories,
    get_protected_files,
    get_backup_dir,
    get_validation_paths,
    is_protected,
    is_keep_root_item,
    is_remove_root_dir,
    is_remove_root_file,
    is_remove_src_dir,
    is_keep_src_dir,
    is_remove_src_file,
)
from .categorizer import (
    CleanupCategorizer,
    CategorizationResult,
    categorize_file,
    categorize_directory,
)
from .executor import CleanupExecutor, CleanupResult, DryRunResult

__all__ = [
    "FileCategory",
    "DirCategory",
    "get_keep_root_items",
    "get_remove_root_dirs",
    "get_remove_root_files",
    "get_keep_root_files_exceptions",
    "get_remove_src_dirs",
    "get_keep_src_dirs",
    "get_remove_src_files",
    "get_protected_directories",
    "get_protected_files",
    "get_backup_dir",
    "get_validation_paths",
    "is_protected",
    "is_keep_root_item",
    "is_remove_root_dir",
    "is_remove_root_file",
    "is_remove_src_dir",
    "is_keep_src_dir",
    "is_remove_src_file",
    "CleanupCategorizer",
    "CategorizationResult",
    "categorize_file",
    "categorize_directory",
    "CleanupExecutor",
    "CleanupResult",
    "DryRunResult",
]
