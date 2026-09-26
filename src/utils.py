"""Utility functions for logging, timing, and formatting."""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


def setup_logger(
    name: str = "amazon_er",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and configure a structured logger.

    Args:
        name: Name of the logger.
        log_file: Optional path to a file where logs will be appended.
        level: Logging verbosity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def timer(section_name: str, logger: Optional[logging.Logger] = None) -> Generator[None, None, None]:
    """Context manager for timing execution blocks and logging the duration."""
    start_time = time.perf_counter()
    if logger:
        logger.info(f"Starting {section_name}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        msg = f"Completed {section_name} in {elapsed:.3f} seconds ({elapsed / 60:.2f} minutes)."
        if logger:
            logger.info(msg)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
