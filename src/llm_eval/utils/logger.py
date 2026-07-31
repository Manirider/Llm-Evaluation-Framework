"""
Structured logging utility using Loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Path | str | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure loguru logger format and output streams.
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console logging
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=log_format,
        colorize=True,
    )

    # File logging if path provided
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            level=level.upper(),
            format=log_format,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
