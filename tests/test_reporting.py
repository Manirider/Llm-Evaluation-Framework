"""Tests for the report generator across all output formats."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_eval.reporting.generator import ReportGenerator
from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    MetricStatistics,
    SampleEvaluationResult,
)


def _make_report() -> EvaluationRunReport:
    """Build a minimal report for testing."""
    sample = EvaluationSample(
        sample_id="s1", input_text="What is AI?", actual_output="AI is artificial intelligence.",
        expected_output="Artificial intelligence.", retrieved_contexts=["AI field."],
    )
    metric_result = MetricResult(metric_name="bleu", score=0.85, passed=True, reasoning="Good match")
    sample_result = SampleEvaluationResult(
        sample_id="s1", sample=sample, metrics={"bleu": metric_result},
    )
    stats = MetricStatistics(
        metric_name="bleu", count=1, mean=0.85, std_dev=0.0, variance=0.0,
        min=0.85, max=0.85, median=0.85, mode=0.85,
        p10=0.85, p25=0.85, p75=0.85, p90=0.85,
        skewness=0.0, kurtosis=0.0, ci_95_lower=0.85, ci_95_upper=0.85,
    )
    return EvaluationRunReport(
        run_id="test_report", dataset_size=1, configured_metrics=["bleu"],
        sample_results=[sample_result], metric_summary={"bleu": stats},
        execution_duration_seconds=0.5,
    )


class TestReportGenerator:
    def test_generate_all(self, tmp_path: Path) -> None:
        gen = ReportGenerator(output_dir=tmp_path)
        report = _make_report()
        files = gen.generate_all(report)
        assert "json" in files
        assert "markdown" in files
        assert "html" in files
        assert "csv" in files
        for path in files.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_json_valid(self, tmp_path: Path) -> None:
        import json
        gen = ReportGenerator(output_dir=tmp_path)
        report = _make_report()
        path = gen.to_json(report, tmp_path / "r.json")
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_report"

    def test_markdown_contains_summary(self, tmp_path: Path) -> None:
        gen = ReportGenerator(output_dir=tmp_path)
        report = _make_report()
        path = gen.to_markdown(report, tmp_path / "r.md")
        content = path.read_text()
        assert "Executive Summary" in content
        assert "Metric Ranking" in content
        assert "Recommendations" in content

    def test_html_contains_structure(self, tmp_path: Path) -> None:
        gen = ReportGenerator(output_dir=tmp_path)
        report = _make_report()
        path = gen.to_html(report, tmp_path / "r.html")
        content = path.read_text()
        assert "<html" in content
        assert "test_report" in content

    def test_csv_has_header(self, tmp_path: Path) -> None:
        gen = ReportGenerator(output_dir=tmp_path)
        report = _make_report()
        path = gen.to_csv(report, tmp_path / "r.csv")
        lines = path.read_text().strip().split("\n")
        assert "sample_id" in lines[0]
        assert len(lines) == 2  # header + 1 data row
