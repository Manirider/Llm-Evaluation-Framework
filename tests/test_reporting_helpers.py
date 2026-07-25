"""Tests for reporting helper functions and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.reporting.generator import (
    ReportGenerator,
    _compute_pass_fail,
    _generate_recommendations,
    _metric_quality_tier,
)
from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    MetricStatistics,
    SampleEvaluationResult,
)


def _report_with_stats(stats: dict[str, MetricStatistics]) -> EvaluationRunReport:
    sample = EvaluationSample(sample_id="s1", input_text="Q", actual_output="A")
    return EvaluationRunReport(
        run_id="rec_test",
        dataset_size=1,
        configured_metrics=list(stats.keys()),
        sample_results=[
            SampleEvaluationResult(
                sample_id="s1",
                sample=sample,
                metrics={
                    name: MetricResult(metric_name=name, score=s.mean, passed=s.mean >= 0.5)
                    for name, s in stats.items()
                },
            )
        ],
        metric_summary=stats,
        execution_duration_seconds=1.0,
    )


class TestReportingHelpers:
    def test_quality_tiers(self) -> None:
        assert _metric_quality_tier(0.95) == "Excellent"
        assert _metric_quality_tier(0.80) == "Good"
        assert _metric_quality_tier(0.65) == "Fair"
        assert _metric_quality_tier(0.45) == "Poor"
        assert _metric_quality_tier(0.20) == "Critical"

    def test_pass_fail_counts(self) -> None:
        sample = EvaluationSample(sample_id="s1", input_text="Q", actual_output="A")
        report = EvaluationRunReport(
            run_id="pf",
            dataset_size=1,
            configured_metrics=["bleu"],
            sample_results=[
                SampleEvaluationResult(
                    sample_id="s1",
                    sample=sample,
                    metrics={
                        "bleu": MetricResult(metric_name="bleu", score=0.5, passed=True),
                        "rouge_l": MetricResult(metric_name="rouge_l", score=0.3, passed=False),
                    },
                )
            ],
            metric_summary={},
            execution_duration_seconds=0.1,
        )
        counts = _compute_pass_fail(report)
        assert counts["bleu"]["pass"] == 1
        assert counts["rouge_l"]["fail"] == 1

    def test_recommendations_high_variance(self) -> None:
        stats = MetricStatistics(
            metric_name="bleu",
            count=10,
            mean=0.7,
            std_dev=0.35,
            variance=0.12,
            min=0.1,
            max=0.9,
            median=0.7,
            mode=0.7,
            p10=0.2,
            p25=0.4,
            p75=0.9,
            p90=0.95,
            skewness=-1.5,
            kurtosis=0.5,
            ci_95_lower=0.5,
            ci_95_upper=0.9,
        )
        report = _report_with_stats({"bleu": stats})
        recs = _generate_recommendations(report)
        assert any("variance" in r.lower() for r in recs)
        assert any("skew" in r.lower() for r in recs)

    def test_recommendations_all_good(self) -> None:
        stats = MetricStatistics(
            metric_name="bleu",
            count=5,
            mean=0.92,
            std_dev=0.05,
            variance=0.0025,
            min=0.85,
            max=0.98,
            median=0.92,
            mode=0.92,
            p10=0.86,
            p25=0.88,
            p75=0.95,
            p90=0.97,
            skewness=0.1,
            kurtosis=0.1,
            ci_95_lower=0.88,
            ci_95_upper=0.96,
        )
        report = _report_with_stats({"bleu": stats})
        recs = _generate_recommendations(report)
        assert any("acceptable" in r.lower() for r in recs)

    def test_generate_all_failure_raises(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from llm_eval.exceptions.base import ReportingError

        gen = ReportGenerator(output_dir=tmp_path)
        report = _report_with_stats({})
        with patch.object(gen, "to_json", side_effect=OSError("disk full")):
            with pytest.raises(ReportingError):
                gen.generate_all(report)
