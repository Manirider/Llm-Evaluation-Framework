"""
Abstract Base Classes for metrics and metric registry pattern.
"""

from abc import ABC, abstractmethod
import threading
import time
from typing import Any, ClassVar
from loguru import logger

from llm_eval.exceptions.base import MetricExecutionError
from llm_eval.schemas.evaluation import EvaluationSample, MetricResult


class BaseMetric(ABC):
    """
    Abstract Base Class for all evaluation metrics.
    Ensures consistent score computation, threshold checks, timing, and error handling.
    """

    metric_name: ClassVar[str] = "base_metric"
    description: ClassVar[str] = "Base evaluation metric interface"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        self.threshold = threshold
        self.config_params = kwargs

    @abstractmethod
    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        """
        Internal metric computation logic.
        Must return a tuple of (score: float [0.0-1.0], reasoning: str | None, raw_details: dict).
        """
        pass

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        """
        Execute metric evaluation on a sample with error boundaries, timing, and score validation.
        """
        start_time = time.perf_counter()
        try:
            score, reasoning, details = self._compute(sample)
            # Clamp score strictly between 0.0 and 1.0
            clamped_score = max(0.0, min(1.0, float(score)))
            passed = (clamped_score >= self.threshold) if self.threshold is not None else None
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            return MetricResult(
                metric_name=self.metric_name,
                score=round(clamped_score, 4),
                passed=passed,
                reasoning=reasoning,
                raw_details=details,
                execution_time_ms=round(execution_time_ms, 2),
            )
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Metric '{self.metric_name}' failed on sample '{sample.sample_id}': {e}")
            raise MetricExecutionError(
                f"Metric calculation error in '{self.metric_name}': {e}",
                details={"sample_id": sample.sample_id, "metric": self.metric_name},
            ) from e


class MetricRegistry:
    """
    Thread-safe central registry for metric discovery, instantiation, and management.
    """

    _registry: ClassVar[dict[str, type[BaseMetric]]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def register(cls, name: str) -> Any:
        """
        Decorator to register a BaseMetric class.
        """
        def decorator(subclass: type[BaseMetric]) -> type[BaseMetric]:
            subclass.metric_name = name
            with cls._lock:
                cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def _ensure_metrics_loaded(cls) -> None:
        try:
            import llm_eval.metrics.classical  # noqa: F401
            import llm_eval.metrics.judge_metric  # noqa: F401
            import llm_eval.metrics.semantic  # noqa: F401
            import llm_eval.rag.metrics  # noqa: F401
        except Exception:
            pass

    @classmethod
    def get(cls, name: str, threshold: float | None = None, **kwargs: Any) -> BaseMetric:
        """
        Instantiate a registered metric by name.
        """
        cls._ensure_metrics_loaded()
        with cls._lock:
            if name not in cls._registry:
                available = list(cls._registry.keys())
                raise MetricExecutionError(
                    f"Metric '{name}' not found in registry. Available metrics: {available}"
                )
            metric_cls = cls._registry[name]
        return metric_cls(threshold=threshold, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all registered metric names.
        """
        cls._ensure_metrics_loaded()
        with cls._lock:
            return list(cls._registry.keys())
