"""Tests for structured logging setup."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from llm_eval.utils.logger import setup_logger


class TestLogger:
    def test_setup_console_only(self) -> None:
        setup_logger(level="DEBUG")
        logger.info("test message")
        assert True  # no exception

    def test_setup_with_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "test.log"
        setup_logger(level="INFO", log_file=log_file)
        logger.info("file log test")
        assert log_file.exists()
        assert log_file.stat().st_size > 0
