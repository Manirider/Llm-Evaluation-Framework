"""Tests for BLEU and ROUGE-L classical metrics."""

from __future__ import annotations

from llm_eval.metrics.classical import BLEUMetric, ROUGEMetric
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class TestBLEUMetric:
    def test_basic_bleu(self, simple_sample: EvaluationSample) -> None:
        metric = BLEUMetric(threshold=0.1)
        res = metric.evaluate(simple_sample)
        assert isinstance(res, MetricResult)
        assert 0.0 <= res.score <= 1.0
        assert res.passed is True

    def test_bleu_no_expected(self, sample_no_expected: EvaluationSample) -> None:
        metric = BLEUMetric()
        res = metric.evaluate(sample_no_expected)
        assert res.score == 0.0
        assert "Skipped" in (res.reasoning or "")

    def test_bleu_identical(self) -> None:
        sample = EvaluationSample(
            sample_id="id", input_text="Q", actual_output="exact match sentence here",
            expected_output="exact match sentence here",
        )
        metric = BLEUMetric()
        res = metric.evaluate(sample)
        assert res.score > 0.8

    def test_bleu_configurable_ngrams(self, simple_sample: EvaluationSample) -> None:
        m1 = BLEUMetric(n_grams=1)
        m4 = BLEUMetric(n_grams=4)
        r1 = m1.evaluate(simple_sample)
        r4 = m4.evaluate(simple_sample)
        # Unigram BLEU should generally be >= 4-gram BLEU
        assert r1.score >= r4.score - 0.01

    def test_bleu_ngram_clamped(self) -> None:
        m = BLEUMetric(n_grams=10)
        assert m.n_grams == 4  # clamped to max 4

    def test_bleu_timing(self, simple_sample: EvaluationSample) -> None:
        metric = BLEUMetric()
        res = metric.evaluate(simple_sample)
        assert res.execution_time_ms >= 0.0


class TestROUGEMetric:
    def test_basic_rouge(self, simple_sample: EvaluationSample) -> None:
        metric = ROUGEMetric(threshold=0.3)
        res = metric.evaluate(simple_sample)
        assert 0.0 <= res.score <= 1.0
        assert res.passed is True

    def test_rouge_no_expected(self, sample_no_expected: EvaluationSample) -> None:
        metric = ROUGEMetric()
        res = metric.evaluate(sample_no_expected)
        assert res.score == 0.0

    def test_rouge_identical(self) -> None:
        sample = EvaluationSample(
            sample_id="id", input_text="Q",
            actual_output="identical text here",
            expected_output="identical text here",
        )
        metric = ROUGEMetric()
        res = metric.evaluate(sample)
        assert res.score > 0.95

    def test_rouge_details(self, simple_sample: EvaluationSample) -> None:
        metric = ROUGEMetric()
        res = metric.evaluate(simple_sample)
        assert "precision" in res.raw_details
        assert "recall" in res.raw_details
        assert "f1" in res.raw_details
