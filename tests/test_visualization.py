"""Tests for the visualization analytics engine."""

from __future__ import annotations

from pathlib import Path

from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    MetricStatistics,
    SampleEvaluationResult,
)
from llm_eval.visualization.engine import VisualAnalyticsEngine


def _make_viz_report() -> EvaluationRunReport:
    """Build a report with multiple samples and metrics for visualization tests."""
    samples_data = [
        ("s1", "Q1", "A1", 0.8, 0.7),
        ("s2", "Q2", "A2", 0.6, 0.9),
        ("s3", "Q3", "A3", 0.9, 0.5),
    ]
    sample_results = []
    for sid, q, a, bleu_score, rouge_score in samples_data:
        sample = EvaluationSample(
            sample_id=sid, input_text=q, actual_output=a, expected_output=a,
        )
        metrics = {
            "bleu": MetricResult(metric_name="bleu", score=bleu_score, passed=bleu_score > 0.5),
            "rouge_l": MetricResult(metric_name="rouge_l", score=rouge_score, passed=rouge_score > 0.5),
        }
        sample_results.append(
            SampleEvaluationResult(sample_id=sid, sample=sample, metrics=metrics)
        )

    def _stats(name: str, scores: list[float]) -> MetricStatistics:
        import numpy as np
        arr = np.array(scores)
        return MetricStatistics(
            metric_name=name, count=len(arr),
            mean=round(float(np.mean(arr)), 4), std_dev=round(float(np.std(arr, ddof=1)), 4),
            variance=round(float(np.var(arr, ddof=1)), 4),
            min=round(float(np.min(arr)), 4), max=round(float(np.max(arr)), 4),
            median=round(float(np.median(arr)), 4), mode=round(float(arr[0]), 4),
            p10=round(float(np.percentile(arr, 10)), 4), p25=round(float(np.percentile(arr, 25)), 4),
            p75=round(float(np.percentile(arr, 75)), 4), p90=round(float(np.percentile(arr, 90)), 4),
            skewness=0.0, kurtosis=0.0, ci_95_lower=0.5, ci_95_upper=0.9,
        )

    return EvaluationRunReport(
        run_id="viz_test", dataset_size=3,
        configured_metrics=["bleu", "rouge_l"],
        sample_results=sample_results,
        metric_summary={
            "bleu": _stats("bleu", [0.8, 0.6, 0.9]),
            "rouge_l": _stats("rouge_l", [0.7, 0.9, 0.5]),
        },
        execution_duration_seconds=1.0,
    )


class TestVisualizationEngine:
    def test_generate_all(self, tmp_path: Path) -> None:
        engine = VisualAnalyticsEngine(output_dir=tmp_path)
        report = _make_viz_report()
        visuals = engine.generate_all_visuals(report)

        expected_keys = [
            "radar", "boxplot", "correlation", "distributions",
            "histogram", "violin", "heatmap", "failure", "comparison",
        ]
        for key in expected_keys:
            assert key in visuals, f"Missing chart: {key}"
            assert visuals[key].exists(), f"Chart file not created: {key}"
            assert visuals[key].stat().st_size > 0, f"Chart file empty: {key}"

    def test_individual_charts(self, tmp_path: Path) -> None:
        import pandas as pd
        engine = VisualAnalyticsEngine(output_dir=tmp_path)
        report = _make_viz_report()
        df = engine._report_to_dataframe(report)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "bleu" in df.columns
