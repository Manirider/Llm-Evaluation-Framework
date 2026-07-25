"""
Utils module initialization.
"""

from llm_eval.utils.logger import setup_logger
from llm_eval.utils.stats import compute_metric_statistics

__all__ = ["setup_logger", "compute_metric_statistics"]
