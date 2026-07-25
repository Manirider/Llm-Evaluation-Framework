"""
Statistical computation routines for metric aggregates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.stats as sp_stats

from llm_eval.schemas.evaluation import MetricStatistics


def compute_metric_statistics(metric_name: str, scores: Sequence[float]) -> MetricStatistics:
    """
    Compute comprehensive statistical summary for a sequence of metric scores.
    Handles edge cases like single-element or zero-variance distributions safely.
    """
    if not scores:
        return MetricStatistics(
            metric_name=metric_name,
            count=0,
            mean=0.0,
            std_dev=0.0,
            variance=0.0,
            min=0.0,
            max=0.0,
            median=0.0,
            mode=0.0,
            p10=0.0,
            p25=0.0,
            p75=0.0,
            p90=0.0,
            skewness=0.0,
            kurtosis=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
        )

    arr = np.array(scores, dtype=np.float64)
    count = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p10_val = float(np.percentile(arr, 10))
    p25_val = float(np.percentile(arr, 25))
    p75_val = float(np.percentile(arr, 75))
    p90_val = float(np.percentile(arr, 90))

    # Mode computation — use scipy; falls back to minimum on multimodal
    mode_result = sp_stats.mode(arr, keepdims=True)
    mode_val = float(mode_result.mode[0])

    if count > 1:
        var_val = float(np.var(arr, ddof=1))
        std_val = float(np.std(arr, ddof=1))
        skew_val = float(sp_stats.skew(arr))
        kurt_val = float(sp_stats.kurtosis(arr))
        sem = std_val / np.sqrt(count)
        ci_lower, ci_upper = sp_stats.t.interval(0.95, df=count - 1, loc=mean_val, scale=sem)
        ci_lower = max(0.0, float(ci_lower)) if not np.isnan(ci_lower) else mean_val
        ci_upper = min(1.0, float(ci_upper)) if not np.isnan(ci_upper) else mean_val
    else:
        var_val = 0.0
        std_val = 0.0
        skew_val = 0.0
        kurt_val = 0.0
        ci_lower = mean_val
        ci_upper = mean_val

    return MetricStatistics(
        metric_name=metric_name,
        count=count,
        mean=round(mean_val, 4),
        std_dev=round(std_val, 4),
        variance=round(var_val, 4),
        min=round(min_val, 4),
        max=round(max_val, 4),
        median=round(median_val, 4),
        mode=round(mode_val, 4),
        p10=round(p10_val, 4),
        p25=round(p25_val, 4),
        p75=round(p75_val, 4),
        p90=round(p90_val, 4),
        skewness=round(skew_val, 4),
        kurtosis=round(kurt_val, 4),
        ci_95_lower=round(ci_lower, 4),
        ci_95_upper=round(ci_upper, 4),
    )
