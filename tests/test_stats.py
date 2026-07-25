"""Tests for statistical computation utilities."""

from __future__ import annotations

from llm_eval.utils.stats import compute_metric_statistics


class TestStatistics:
    def test_basic_stats(self) -> None:
        scores = [0.8, 0.85, 0.9, 0.95, 1.0]
        stats = compute_metric_statistics("bleu", scores)
        assert stats.count == 5
        assert stats.mean == 0.9
        assert stats.min == 0.8
        assert stats.max == 1.0

    def test_mode_computation(self) -> None:
        scores = [0.5, 0.5, 0.7, 0.8, 0.9]
        stats = compute_metric_statistics("test", scores)
        assert stats.mode == 0.5

    def test_percentiles(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        stats = compute_metric_statistics("test", scores)
        assert stats.p10 <= stats.p25 <= stats.median <= stats.p75 <= stats.p90

    def test_empty_scores(self) -> None:
        stats = compute_metric_statistics("empty", [])
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.std_dev == 0.0

    def test_single_element(self) -> None:
        stats = compute_metric_statistics("single", [0.75])
        assert stats.count == 1
        assert stats.mean == 0.75
        assert stats.std_dev == 0.0
        assert stats.variance == 0.0

    def test_zero_variance(self) -> None:
        stats = compute_metric_statistics("const", [0.5, 0.5, 0.5])
        assert stats.std_dev == 0.0
        assert stats.variance == 0.0

    def test_ci_bounds(self) -> None:
        scores = [0.6, 0.7, 0.8, 0.9, 1.0]
        stats = compute_metric_statistics("ci", scores)
        assert stats.ci_95_lower <= stats.mean
        assert stats.ci_95_upper >= stats.mean
