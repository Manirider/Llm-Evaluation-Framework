"""Tests for Pydantic domain schemas."""

from __future__ import annotations

import pytest

from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    MetricStatistics,
    SampleEvaluationResult,
)


class TestEvaluationSample:
    def test_valid_sample(self) -> None:
        sample = EvaluationSample(
            sample_id="s1", input_text="Hello", actual_output="World"
        )
        assert sample.sample_id == "s1"

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="input_text cannot be empty"):
            EvaluationSample(sample_id="s1", input_text="  ", actual_output="World")

    def test_empty_output_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_output cannot be empty"):
            EvaluationSample(sample_id="s1", input_text="Hello", actual_output="   ")

    def test_optional_fields(self) -> None:
        sample = EvaluationSample(
            sample_id="s1", input_text="Q", actual_output="A"
        )
        assert sample.expected_output is None
        assert sample.retrieved_contexts is None
        assert sample.tools_called is None

    def test_frozen(self) -> None:
        sample = EvaluationSample(
            sample_id="s1", input_text="Q", actual_output="A"
        )
        with pytest.raises(Exception):
            sample.input_text = "Modified"  # type: ignore[misc]

    def test_extra_fields_allowed(self) -> None:
        sample = EvaluationSample(
            sample_id="s1",
            input_text="Q",
            actual_output="A",
            custom_field="custom_value",
        )
        assert sample.model_extra is not None


class TestMetricResult:
    def test_valid_result(self) -> None:
        r = MetricResult(metric_name="bleu", score=0.85)
        assert r.score == 0.85

    def test_score_clamped(self) -> None:
        with pytest.raises(Exception):
            MetricResult(metric_name="bleu", score=1.5)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(Exception):
            MetricResult(metric_name="bleu", score=-0.1)


class TestMetricStatistics:
    def test_all_fields(self) -> None:
        stats = MetricStatistics(
            metric_name="bleu",
            count=10, mean=0.8, std_dev=0.1, variance=0.01,
            min=0.5, max=1.0, median=0.8, mode=0.8,
            p10=0.6, p25=0.7, p75=0.9, p90=0.95,
            skewness=0.0, kurtosis=0.0,
            ci_95_lower=0.7, ci_95_upper=0.9,
        )
        assert stats.count == 10
        assert stats.mode == 0.8
        assert stats.p10 == 0.6
        assert stats.p90 == 0.95
