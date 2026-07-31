"""
Pydantic v2 domain schemas for evaluation dataset inputs, metric outputs, and reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationSample(BaseModel):
    """
    Represents a single evaluation record in a dataset.
    Supports standard LLM, RAG, Agent, and multi-turn evaluation cases.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    sample_id: str = Field(..., description="Unique identifier for the evaluation sample")
    input_text: str = Field(..., description="Prompt or query input")
    actual_output: str = Field(..., description="Generated text response from the model under test")
    expected_output: str | None = Field(default=None, description="Ground truth target output")
    retrieved_contexts: list[str] | None = Field(
        default=None, description="Context passages for RAG evaluation"
    )
    tools_called: list[dict[str, Any]] | None = Field(
        default=None, description="Tool invocation logs"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")

    @model_validator(mode="after")
    def validate_content(self) -> EvaluationSample:
        if not self.input_text.strip():
            raise ValueError("input_text cannot be empty")
        if not self.actual_output.strip():
            raise ValueError("actual_output cannot be empty")
        return self


class MetricResult(BaseModel):
    """
    Represents the output score and diagnostic metadata of a single metric execution on a sample.
    """

    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Unique identifier of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    passed: bool | None = Field(default=None, description="Pass/fail status against threshold")
    reasoning: str | None = Field(
        default=None, description="Explanation for score or judge verdict"
    )
    raw_details: dict[str, Any] = Field(
        default_factory=dict, description="Raw breakdown data (e.g., sub-scores)"
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Time taken to calculate metric in milliseconds"
    )


class SampleEvaluationResult(BaseModel):
    """
    Aggregated collection of metric scores for a single evaluation sample.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    sample: EvaluationSample
    metrics: dict[str, MetricResult]
    errors: list[str] = Field(default_factory=list)


class MetricStatistics(BaseModel):
    """
    Statistical aggregates computed over an entire evaluation run for a metric.
    """

    metric_name: str
    count: int
    mean: float
    std_dev: float
    variance: float
    min: float
    max: float
    median: float
    mode: float
    p10: float
    p25: float
    p75: float
    p90: float
    skewness: float
    kurtosis: float
    ci_95_lower: float
    ci_95_upper: float


class EvaluationRunReport(BaseModel):
    """
    Final summary report artifact containing global statistics, per-sample scores, and metadata.
    """

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_size: int
    configured_metrics: list[str]
    sample_results: list[SampleEvaluationResult]
    metric_summary: dict[str, MetricStatistics]
    execution_duration_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
