"""
Asynchronous evaluation pipeline with concurrent worker execution and error isolation.
"""

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from llm_eval.config.settings import EvaluationFrameworkConfig
from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.core.metric_engine import MetricEngine
from llm_eval.exceptions.base import PipelineExecutionError
from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    EvaluationSample,
    MetricResult,
    SampleEvaluationResult,
)
from llm_eval.utils.stats import compute_metric_statistics


class EvaluationPipeline:
    """
    Main evaluation pipeline engine orchestrating parallel metric execution,
    sample processing, failure recovery, and report aggregation.
    """

    def __init__(self, config: EvaluationFrameworkConfig) -> None:
        self.config = config
        self.metrics: list[BaseMetric] = []
        self._initialize_metrics()
        self.metric_engine = MetricEngine(
            self.metrics,
            enable_cache=self.config.pipeline.cache_embeddings,
        )

    def _initialize_metrics(self) -> None:
        """
        Instantiate configured active metrics from registry.
        """
        for metric_name, m_cfg in self.config.metrics.items():
            if m_cfg.enabled:
                try:
                    metric_obj = MetricRegistry.get(
                        metric_name,
                        threshold=m_cfg.threshold,
                        **m_cfg.params,
                    )
                    self.metrics.append(metric_obj)
                except Exception as e:
                    logger.warning(f"Could not load metric '{metric_name}': {e}")

        if not self.metrics:
            logger.warning("No active metrics registered in pipeline configuration.")

    def evaluate_sample(self, sample: EvaluationSample) -> SampleEvaluationResult:
        """
        Evaluate a single sample synchronously across all initialized metrics.
        Implements strict sample isolation so metric failures do not crash the sample execution.
        """
        metric_results: dict[str, MetricResult] = {}
        errors: list[str] = []

        metric_results, engine_errors = self.metric_engine.evaluate_sample_parallel(
            sample,
            max_workers=self.config.pipeline.max_workers,
        )
        errors.extend(engine_errors)

        if errors and self.config.pipeline.fail_on_sample_error:
            err_msg = "; ".join(errors)
            raise PipelineExecutionError(err_msg)

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            sample=sample,
            metrics=metric_results,
            errors=errors,
        )

    def run_batch(
        self, samples: Sequence[EvaluationSample], run_id: str = "run_default"
    ) -> EvaluationRunReport:
        """
        Execute evaluation pipeline over a collection of samples using a ThreadPoolExecutor.
        """
        start_time = time.perf_counter()
        logger.info(
            f"Starting evaluation pipeline run '{run_id}' with {len(samples)} samples and {len(self.metrics)} metrics..."
        )

        max_workers = min(self.config.pipeline.max_workers, max(1, len(samples)))
        results: list[SampleEvaluationResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_sample, sample) for sample in samples]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Sample worker thread execution error: {e}")

        duration = time.perf_counter() - start_time
        logger.info(f"Pipeline evaluation completed in {duration:.2f} seconds.")

        # Compute metric aggregates
        metric_scores: dict[str, list[float]] = {m.metric_name: [] for m in self.metrics}
        for sample_res in results:
            for m_name, m_res in sample_res.metrics.items():
                if m_name in metric_scores:
                    metric_scores[m_name].append(m_res.score)

        metric_summary = {
            m_name: compute_metric_statistics(m_name, scores)
            for m_name, scores in metric_scores.items()
        }

        return EvaluationRunReport(
            run_id=run_id,
            dataset_size=len(samples),
            configured_metrics=[m.metric_name for m in self.metrics],
            sample_results=results,
            metric_summary=metric_summary,
            execution_duration_seconds=round(duration, 3),
        )
