"""
Metric execution engine with thread-safe caching and parallel metric evaluation.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from llm_eval.core.base_metric import BaseMetric
from llm_eval.exceptions.base import MetricExecutionError
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class MetricEngine:
    """
    Orchestrates parallel metric execution with optional result caching.

    Thread-safe: cache access is guarded by a reentrant lock so concurrent
    pipeline workers can share a single engine instance safely.
    """

    def __init__(self, metrics: Sequence[BaseMetric], *, enable_cache: bool = True) -> None:
        self.metrics = list(metrics)
        self.enable_cache = enable_cache
        self._cache: dict[str, MetricResult] = {}
        self._lock = threading.RLock()

    def _cache_key(self, sample: EvaluationSample, metric_name: str) -> str:
        payload = (
            f"{sample.sample_id}|{metric_name}|"
            f"{sample.input_text}|{sample.actual_output}|"
            f"{sample.expected_output}|{sample.retrieved_contexts}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def evaluate_sample_parallel(
        self, sample: EvaluationSample, max_workers: int | None = None
    ) -> tuple[dict[str, MetricResult], list[str]]:
        """
        Evaluate all configured metrics for a sample in parallel.
        Returns (metric_results, error_messages). Individual metric failures
        do not prevent other metrics from completing.
        """
        if not self.metrics:
            return {}, []

        workers = max_workers or min(len(self.metrics), 4)
        results: dict[str, MetricResult] = {}
        errors: list[str] = []

        def _run(metric: BaseMetric) -> tuple[str, MetricResult | None, str | None]:
            key = self._cache_key(sample, metric.metric_name)
            if self.enable_cache:
                with self._lock:
                    cached = self._cache.get(key)
                if cached is not None:
                    logger.debug(f"Cache hit for {metric.metric_name} on {sample.sample_id}")
                    return metric.metric_name, cached, None

            try:
                result = metric.evaluate(sample)
            except MetricExecutionError as exc:
                return metric.metric_name, None, str(exc)

            if self.enable_cache:
                with self._lock:
                    self._cache[key] = result

            return metric.metric_name, result, None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run, m): m for m in self.metrics}
            for future in as_completed(futures):
                metric_name, result, error = future.result()
                if result is not None:
                    results[metric_name] = result
                elif error:
                    errors.append(error)

        return results, errors

    def clear_cache(self) -> None:
        """Evict all cached metric results."""
        with self._lock:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)
