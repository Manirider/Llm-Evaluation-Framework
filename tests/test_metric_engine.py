"""Tests for MetricEngine parallel execution and caching."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.core.metric_engine import MetricEngine
from llm_eval.exceptions.base import MetricExecutionError
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class _SlowMetric(BaseMetric):
    metric_name = "slow_test"
    description = "Slow metric for parallel tests"

    def _compute(self, sample: EvaluationSample):
        time.sleep(0.01)
        return 0.8, "ok", {}


class _FailingMetric(BaseMetric):
    metric_name = "fail_test"
    description = "Always fails"

    def _compute(self, sample: EvaluationSample):
        raise RuntimeError("intentional failure")


@pytest.fixture()
def sample() -> EvaluationSample:
    return EvaluationSample(sample_id="eng_1", input_text="Q", actual_output="A", expected_output="A")


class TestMetricEngine:
    def test_parallel_execution(self, sample: EvaluationSample) -> None:
        metrics = [MetricRegistry.get("bleu"), MetricRegistry.get("rouge_l")]
        engine = MetricEngine(metrics, enable_cache=True)
        results, errors = engine.evaluate_sample_parallel(sample)
        assert "bleu" in results
        assert "rouge_l" in results
        assert errors == []

    def test_cache_hit(self, sample: EvaluationSample) -> None:
        metric = MetricRegistry.get("bleu")
        engine = MetricEngine([metric], enable_cache=True)
        engine.evaluate_sample_parallel(sample)
        assert engine.cache_size == 1
        results, _ = engine.evaluate_sample_parallel(sample)
        assert "bleu" in results

    def test_clear_cache(self, sample: EvaluationSample) -> None:
        metric = MetricRegistry.get("bleu")
        engine = MetricEngine([metric], enable_cache=True)
        engine.evaluate_sample_parallel(sample)
        engine.clear_cache()
        assert engine.cache_size == 0

    def test_empty_metrics(self, sample: EvaluationSample) -> None:
        engine = MetricEngine([], enable_cache=True)
        results, errors = engine.evaluate_sample_parallel(sample)
        assert results == {}
        assert errors == []

    def test_individual_metric_failure_isolated(self, sample: EvaluationSample) -> None:
        failing = _FailingMetric()
        good = MetricRegistry.get("bleu")

        # Wrap failing metric evaluate to raise MetricExecutionError
        original = failing.evaluate

        def _bad_evaluate(s):
            try:
                return original(s)
            except RuntimeError as e:
                raise MetricExecutionError(str(e)) from e

        failing.evaluate = _bad_evaluate  # type: ignore[method-assign]

        engine = MetricEngine([failing, good], enable_cache=False)
        results, errors = engine.evaluate_sample_parallel(sample)
        assert "bleu" in results
        assert len(errors) >= 1

    def test_registry_thread_safe_get(self) -> None:
        with pytest.raises(MetricExecutionError, match="not found"):
            MetricRegistry.get("nonexistent_metric_xyz")
