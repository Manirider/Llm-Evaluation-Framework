"""Tests for the LLM-as-a-Judge registered metric."""

from __future__ import annotations

from llm_eval.metrics.judge_metric import LLMJudgeMetric
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class TestLLMJudgeMetric:
    def test_basic_evaluation(self, simple_sample: EvaluationSample) -> None:
        metric = LLMJudgeMetric(threshold=0.5)
        res = metric.evaluate(simple_sample)
        assert isinstance(res, MetricResult)
        assert 0.0 <= res.score <= 1.0

    def test_metric_name(self) -> None:
        metric = LLMJudgeMetric()
        assert metric.metric_name == "llm_judge"

    def test_custom_criteria(self, simple_sample: EvaluationSample) -> None:
        metric = LLMJudgeMetric(criteria="Evaluate grammar and fluency only.")
        res = metric.evaluate(simple_sample)
        assert res.score >= 0.0

    def test_details_contain_provider(self, simple_sample: EvaluationSample) -> None:
        metric = LLMJudgeMetric()
        res = metric.evaluate(simple_sample)
        assert "judge_provider" in res.raw_details

    def test_prompt_includes_context(self) -> None:
        sample = EvaluationSample(
            sample_id="s1",
            input_text="Q",
            actual_output="A",
            retrieved_contexts=["Context passage."],
        )
        metric = LLMJudgeMetric()
        prompt = metric._build_prompt(sample)
        assert "Context passage" in prompt
        assert "BIAS" in prompt.upper()
