"""Tests for BERTScore and EmbeddingSimilarity semantic metrics."""

from __future__ import annotations

from llm_eval.metrics.semantic import BERTScoreMetric, EmbeddingSimilarityMetric
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class TestEmbeddingSimilarity:
    def test_basic_similarity(self, simple_sample: EvaluationSample) -> None:
        metric = EmbeddingSimilarityMetric(threshold=0.3)
        res = metric.evaluate(simple_sample)
        assert isinstance(res, MetricResult)
        assert 0.0 <= res.score <= 1.0

    def test_no_expected(self, sample_no_expected: EvaluationSample) -> None:
        metric = EmbeddingSimilarityMetric()
        res = metric.evaluate(sample_no_expected)
        assert res.score == 0.0

    def test_details_present(self, simple_sample: EvaluationSample) -> None:
        metric = EmbeddingSimilarityMetric()
        res = metric.evaluate(simple_sample)
        assert "raw_cosine_similarity" in res.raw_details
        assert "normalized_score" in res.raw_details


class TestBERTScore:
    def test_basic_bertscore(self, simple_sample: EvaluationSample) -> None:
        metric = BERTScoreMetric(threshold=0.3)
        res = metric.evaluate(simple_sample)
        assert isinstance(res, MetricResult)
        assert 0.0 <= res.score <= 1.0

    def test_no_expected(self, sample_no_expected: EvaluationSample) -> None:
        metric = BERTScoreMetric()
        res = metric.evaluate(sample_no_expected)
        assert res.score == 0.0

    def test_details_present(self, simple_sample: EvaluationSample) -> None:
        metric = BERTScoreMetric()
        res = metric.evaluate(simple_sample)
        assert "precision" in res.raw_details
        assert "recall" in res.raw_details
        assert "f1" in res.raw_details

    def test_identical_high_score(self) -> None:
        sample = EvaluationSample(
            sample_id="id",
            input_text="Q",
            actual_output="word one two three",
            expected_output="word one two three",
        )
        metric = BERTScoreMetric()
        res = metric.evaluate(sample)
        assert res.score > 0.5
