"""
Metrics module initialization.
Ensures automatic metric registration via import-time side effects.
"""

from llm_eval.metrics.classical import BLEUMetric, ROUGEMetric
from llm_eval.metrics.judge_metric import LLMJudgeMetric
from llm_eval.metrics.semantic import BERTScoreMetric, EmbeddingSimilarityMetric

__all__ = [
    "BLEUMetric",
    "ROUGEMetric",
    "BERTScoreMetric",
    "EmbeddingSimilarityMetric",
    "LLMJudgeMetric",
]
