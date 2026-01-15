"""Logging configuration for Project Cleanup and PPO vs A2C Experiments.

Per NFR-001 and NFR-002 requirements:
- DEBUG-level logging with timestamps
- Output to both console (stdout) and file at results/logs/
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    log_dir: Path = Path("results/logs/"),
    log_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure logging with DEBUG level and file output.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (default: DEBUG)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance
    """
    global _logger

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("rl_experiments")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format with timestamp
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _logger = logger

    # Log startup message
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.debug(f"Log level: {logging.getLevelName(log_level)}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get the configured logger instance.

    Args:
        name: Optional name for sub-logger

    Returns:
        Logger instance
    """
    if _logger is None:
        setup_logging()

    if name:
        return _logger.getChild(name)
    return _logger


class LogContext:
    """Context manager for temporary log level changes."""

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self.original_level: Optional[int] = None
        self.logger = get_logger()

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False


def log_function_entry(logger: logging.Logger):
    """Decorator to log function entry and exit."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}({args}, {kwargs})")
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} with result: {result}")
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # Test logging configuration
    logger = setup_logging()

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print("\n✅ Logging configured successfully!")
    print("Log files in: results/logs/")
