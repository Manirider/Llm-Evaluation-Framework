"""Tests for all RAG metrics (7 total)."""

from __future__ import annotations

from llm_eval.rag.metrics import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevancyMetric,
    FaithfulnessMetric,
    GroundednessMetric,
    HallucinationDetectionMetric,
)
from llm_eval.schemas.evaluation import EvaluationSample


class TestFaithfulness:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = FaithfulnessMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_no_contexts(self, sample_no_contexts: EvaluationSample) -> None:
        m = FaithfulnessMetric()
        r = m.evaluate(sample_no_contexts)
        assert r.score == 1.0  # No contexts = trivially faithful


class TestContextRelevancy:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = ContextRelevancyMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_no_contexts(self, sample_no_contexts: EvaluationSample) -> None:
        m = ContextRelevancyMetric()
        r = m.evaluate(sample_no_contexts)
        assert r.score == 0.0


class TestAnswerRelevancy:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = AnswerRelevancyMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0


class TestContextPrecision:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = ContextPrecisionMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_no_contexts(self, sample_no_contexts: EvaluationSample) -> None:
        m = ContextPrecisionMetric()
        r = m.evaluate(sample_no_contexts)
        assert r.score == 0.0

    def test_details(self, simple_sample: EvaluationSample) -> None:
        m = ContextPrecisionMetric()
        r = m.evaluate(simple_sample)
        assert "passage_scores" in r.raw_details
        assert "relevance_flags" in r.raw_details


class TestContextRecall:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = ContextRecallMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_missing_expected(self, sample_no_expected: EvaluationSample) -> None:
        m = ContextRecallMetric()
        r = m.evaluate(sample_no_expected)
        assert r.score == 0.0


class TestGroundedness:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = GroundednessMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_no_contexts(self, sample_no_contexts: EvaluationSample) -> None:
        m = GroundednessMetric()
        r = m.evaluate(sample_no_contexts)
        assert r.score == 1.0  # Trivially grounded


class TestHallucinationDetection:
    def test_basic(self, simple_sample: EvaluationSample) -> None:
        m = HallucinationDetectionMetric()
        r = m.evaluate(simple_sample)
        assert 0.0 <= r.score <= 1.0

    def test_no_contexts(self, sample_no_contexts: EvaluationSample) -> None:
        m = HallucinationDetectionMetric()
        r = m.evaluate(sample_no_contexts)
        assert r.score == 0.0  # Cannot verify, assumes full hallucination

    def test_details(self, simple_sample: EvaluationSample) -> None:
        m = HallucinationDetectionMetric()
        r = m.evaluate(simple_sample)
        assert "hallucination_ratio" in r.raw_details
        assert "sentence_breakdown" in r.raw_details
