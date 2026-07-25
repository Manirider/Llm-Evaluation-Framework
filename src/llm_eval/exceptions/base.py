"""
Custom exception hierarchy for the LLM Evaluation Framework.

Provides structured, domain-specific exceptions with optional diagnostic detail
payloads. Every exception carries a human-readable message and an optional
``details`` dict for machine-consumable context (sample IDs, metric names, file
paths, etc.).
"""

from __future__ import annotations

from typing import Any


class LLMEvalError(Exception):
    """Base exception for all LLM evaluation errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(LLMEvalError):
    """Raised when configuration validation or loading fails."""

    pass


class DatasetValidationError(LLMEvalError):
    """Raised when evaluation dataset schema or content validation fails."""

    pass


class MetricExecutionError(LLMEvalError):
    """Raised when metric computation fails during evaluation."""

    pass


class JudgeExecutionError(LLMEvalError):
    """Raised when LLM-as-a-Judge API or generation fails."""

    pass


class EmbeddingError(LLMEvalError):
    """Raised when embedding generation fails."""

    pass


class PipelineExecutionError(LLMEvalError):
    """Raised when execution pipeline encounters an unrecoverable error."""

    pass


class ReportingError(LLMEvalError):
    """Raised when report generation or serialization fails."""

    pass


class VisualizationError(LLMEvalError):
    """Raised when visualization chart rendering or file output fails."""

    pass
