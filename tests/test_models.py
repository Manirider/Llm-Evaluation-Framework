"""Tests for domain models."""

from __future__ import annotations

from llm_eval.models.domain import (
    EvaluationContext,
    MetricEvaluationRequest,
    MetricEvaluationResponse,
    RunMetadata,
)


class TestDomainModels:
    def test_run_metadata(self) -> None:
        meta = RunMetadata(run_id="run_1", tags={"env": "test"})
        assert meta.run_id == "run_1"
        assert meta.tags["env"] == "test"

    def test_evaluation_context(self) -> None:
        meta = RunMetadata(run_id="run_1")
        ctx = EvaluationContext(run_metadata=meta, dataset_path="/data/eval.jsonl")
        assert ctx.dataset_path == "/data/eval.jsonl"

    def test_metric_request_response(self) -> None:
        req = MetricEvaluationRequest(
            sample_id="s1",
            metric_name="bleu",
            input_text="Q",
            actual_output="A",
            expected_output="A",
        )
        resp = MetricEvaluationResponse(
            sample_id=req.sample_id,
            metric_name=req.metric_name,
            score=0.9,
            passed=True,
            reasoning="Good",
            execution_time_ms=12.5,
        )
        assert resp.score == 0.9
