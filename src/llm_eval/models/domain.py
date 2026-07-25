"""
Domain-driven design entities for the LLM Evaluation Framework.

These models represent core business concepts and are intentionally
decoupled from persistence, CLI, and infrastructure concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    """Metadata describing an evaluation run."""

    run_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationContext:
    """Execution context passed through the evaluation pipeline."""

    run_metadata: RunMetadata
    dataset_path: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricEvaluationRequest:
    """Request to evaluate a single sample with a specific metric."""

    sample_id: str
    metric_name: str
    input_text: str
    actual_output: str
    expected_output: str | None = None
    retrieved_contexts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricEvaluationResponse:
    """Response from a metric evaluation request."""

    sample_id: str
    metric_name: str
    score: float
    passed: bool | None
    reasoning: str | None
    execution_time_ms: float
    raw_details: dict[str, Any] = field(default_factory=dict)
