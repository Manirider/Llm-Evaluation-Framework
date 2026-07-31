"""Extended CLI integration tests with mocked pipeline execution."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from llm_eval.cli.main import app
from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    MetricStatistics,
    SampleEvaluationResult,
)

runner = CliRunner()


def _minimal_report() -> EvaluationRunReport:
    sample = EvaluationSample(
        sample_id="s1",
        input_text="Q",
        actual_output="A",
        expected_output="A",
    )
    metric_result = MetricResult(metric_name="bleu", score=0.9, passed=True)
    return EvaluationRunReport(
        run_id="cli_test",
        dataset_size=1,
        configured_metrics=["bleu"],
        sample_results=[
            SampleEvaluationResult(sample_id="s1", sample=sample, metrics={"bleu": metric_result})
        ],
        metric_summary={
            "bleu": MetricStatistics(
                metric_name="bleu",
                count=1,
                mean=0.9,
                std_dev=0.0,
                variance=0.0,
                min=0.9,
                max=0.9,
                median=0.9,
                mode=0.9,
                p10=0.9,
                p25=0.9,
                p75=0.9,
                p90=0.9,
                skewness=0.0,
                kurtosis=0.0,
                ci_95_lower=0.9,
                ci_95_upper=0.9,
            )
        },
        execution_duration_seconds=0.1,
    )


def _temp_path_no_space(suffix: str = "") -> Path:
    """Create a temporary path without spaces (Windows temp dir often has spaces)."""
    base = Path(tempfile.gettempdir()) / "llm_eval_test"
    base.mkdir(parents=True, exist_ok=True)
    return base / suffix


class TestCLIIntegration:
    def test_run_command_success(self) -> None:
        dataset = _temp_path_no_space("data.jsonl")
        dataset.write_text(
            json.dumps(
                {"sample_id": "s1", "input_text": "Q", "actual_output": "A", "expected_output": "A"}
            )
            + "\n"
        )
        output_dir = _temp_path_no_space("reports")

        mock_report = _minimal_report()
        with patch("llm_eval.cli.main.EvaluationPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run_batch.return_value = mock_report
            result = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "test_run",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Completed Successfully" in result.output

    def test_run_command_failure(self) -> None:
        dataset = _temp_path_no_space("data.jsonl")
        dataset.write_text(
            json.dumps({"sample_id": "s1", "input_text": "Q", "actual_output": "A"}) + "\n"
        )

        with patch("llm_eval.cli.main.EvaluationPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run_batch.side_effect = RuntimeError("pipeline boom")
            result = runner.invoke(app, ["run", "--dataset", str(dataset)])

        assert result.exit_code != 0
        assert "Pipeline Failure" in result.output

    def test_report_command_success(self) -> None:
        report_path = _temp_path_no_space("report.json")
        report_path.write_text(_minimal_report().model_dump_json())
        output_dir = _temp_path_no_space("out")

        result = runner.invoke(
            app,
            ["report", "--input", str(report_path), "--output-dir", str(output_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "Reports generated" in result.output

    def test_visualize_command_success(self) -> None:
        report_path = _temp_path_no_space("report.json")
        report_path.write_text(_minimal_report().model_dump_json())
        output_dir = _temp_path_no_space("visuals")

        result = runner.invoke(
            app,
            ["visualize", "--input", str(report_path), "--output-dir", str(output_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "Visualizations generated" in result.output

    def test_benchmark_command(self) -> None:
        mock_report = _minimal_report()
        with patch("llm_eval.cli.main.EvaluationPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run_batch.return_value = mock_report
            result = runner.invoke(
                app,
                ["benchmark", "--output-dir", str(_temp_path_no_space("bench"))],
            )
        assert result.exit_code == 0, result.output

    def test_verbose_logging(self) -> None:
        result = runner.invoke(app, ["--verbose", "version"])
        assert result.exit_code == 0

    def test_report_invalid_json(self) -> None:
        bad = _temp_path_no_space("bad.json")
        bad.write_text("{not valid json")
        result = runner.invoke(app, ["report", "--input", str(bad)])
        assert result.exit_code != 0

    def test_visualize_invalid_json(self) -> None:
        bad = _temp_path_no_space("bad.json")
        bad.write_text("{not valid json")
        result = runner.invoke(app, ["visualize", "--input", str(bad)])
        assert result.exit_code != 0

    def test_run_with_config_file(self) -> None:
        dataset = _temp_path_no_space("data.jsonl")
        dataset.write_text(
            json.dumps(
                {"sample_id": "s1", "input_text": "Q", "actual_output": "A", "expected_output": "A"}
            )
            + "\n"
        )
        config = _temp_path_no_space("config.yaml")
        config.write_text("project_name: Test Config Run\n")

        mock_report = _minimal_report()
        with patch("llm_eval.cli.main.EvaluationPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run_batch.return_value = mock_report
            result = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(dataset),
                    "--config",
                    str(config),
                    "--output-dir",
                    str(_temp_path_no_space("out")),
                ],
            )
        assert result.exit_code == 0, result.output
