"""
Metrics module initialization.
Ensures automatic metric registration via import-time side effects.
"""

from llm_eval.metrics.classical import BLEUMetric, ROUGEMetric
from llm_eval.metrics.judge_metric import LLMJudgeMetric

# Lazy imports for semantic metrics to avoid torch/sentence_transformers import at module load
BERTScoreMetric = None
EmbeddingSimilarityMetric = None

def _import_semantic():
    global BERTScoreMetric, EmbeddingSimilarityMetric
    if BERTScoreMetric is None:
        from llm_eval.metrics.semantic import BERTScoreMetric as _BERTScoreMetric, EmbeddingSimilarityMetric as _EmbeddingSimilarityMetric
        BERTScoreMetric = _BERTScoreMetric
        EmbeddingSimilarityMetric = _EmbeddingSimilarityMetric
    return BERTScoreMetric, EmbeddingSimilarityMetric

__all__ = [
    "BLEUMetric",
    "ROUGEMetric",
    "BERTScoreMetric",
    "EmbeddingSimilarityMetric",
    "LLMJudgeMetric",
]

def __getattr__(name: str):
    if name in ("BERTScoreMetric", "EmbeddingSimilarityMetric"):
        _import_semantic()
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
