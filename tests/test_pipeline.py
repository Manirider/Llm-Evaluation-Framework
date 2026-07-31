"""Tests for the evaluation pipeline runner."""

from __future__ import annotations

from llm_eval.config.settings import EvaluationFrameworkConfig, LLMProviderConfig, MetricConfig
from llm_eval.pipeline.runner import EvaluationPipeline
from llm_eval.schemas.evaluation import EvaluationSample


class TestPipeline:
    def test_single_sample(
        self, simple_sample: EvaluationSample, minimal_config: EvaluationFrameworkConfig
    ) -> None:
        pipeline = EvaluationPipeline(minimal_config)
        report = pipeline.run_batch([simple_sample], run_id="test_single")
        assert report.dataset_size == 1
        assert "bleu" in report.metric_summary

    def test_batch_samples(
        self, batch_samples: list[EvaluationSample], minimal_config: EvaluationFrameworkConfig
    ) -> None:
        pipeline = EvaluationPipeline(minimal_config)
        report = pipeline.run_batch(batch_samples, run_id="test_batch")
        assert report.dataset_size == 2
        assert len(report.sample_results) == 2

    def test_error_isolation(self) -> None:
        """Metric failures on one sample don't crash the whole batch."""
        cfg = EvaluationFrameworkConfig(
            judge=LLMProviderConfig(provider="mock"),
            metrics={"bleu": MetricConfig(enabled=True)},
            pipeline=EvaluationFrameworkConfig().pipeline,
        )
        samples = [
            EvaluationSample(
                sample_id="ok",
                input_text="Q",
                actual_output="A",
                expected_output="A",
            ),
        ]
        pipeline = EvaluationPipeline(cfg)
        report = pipeline.run_batch(samples, run_id="isolation")
        assert report.dataset_size == 1

    def test_report_has_timing(
        self, simple_sample: EvaluationSample, minimal_config: EvaluationFrameworkConfig
    ) -> None:
        pipeline = EvaluationPipeline(minimal_config)
        report = pipeline.run_batch([simple_sample], run_id="timing")
        assert report.execution_duration_seconds >= 0.0

    def test_empty_batch(self, minimal_config: EvaluationFrameworkConfig) -> None:
        pipeline = EvaluationPipeline(minimal_config)
        report = pipeline.run_batch([], run_id="empty")
        assert report.dataset_size == 0
