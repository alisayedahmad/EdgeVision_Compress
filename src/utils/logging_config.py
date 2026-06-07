"""Logging configuration for EdgeVision-Compress.

Provides a single entry point for consistent, structured logging
across all modules. Uses Python's standard logging library exclusively —
no print statements anywhere in the project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "edgevision",
) -> logging.Logger:
    """Configure and return a named logger.

    Sets up a logger with a console handler and an optional file handler.
    Uses a consistent timestamp format across all modules. Safe to call
    multiple times — duplicate handlers are prevented.

    Args:
        level: Logging level. Defaults to logging.INFO.
        log_file: Optional path to a log file. Parent directories are
            created automatically. If None, logs only to stdout.
        name: Name of the logger. Defaults to "edgevision".

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> import logging
        >>> logger = setup_logging(level=logging.DEBUG, name="training")
        >>> logger.info("Training started")
        2024-01-01 12:00:00 | INFO     | training | Training started
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers when called multiple times in same process
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always active
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Console handler — always active
    console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optional
    if log_file is not None:
        resolved = Path(log_file)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
