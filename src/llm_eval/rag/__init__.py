"""
RAG module initialization.
"""

FaithfulnessMetric = None
ContextRelevancyMetric = None
AnswerRelevancyMetric = None
ContextPrecisionMetric = None
ContextRecallMetric = None
GroundednessMetric = None
HallucinationDetectionMetric = None

def _import_rag_metrics():
    global FaithfulnessMetric, ContextRelevancyMetric, AnswerRelevancyMetric
    global ContextPrecisionMetric, ContextRecallMetric, GroundednessMetric, HallucinationDetectionMetric
    if FaithfulnessMetric is None:
        from llm_eval.rag.metrics import (
            FaithfulnessMetric as _FaithfulnessMetric,
            ContextRelevancyMetric as _ContextRelevancyMetric,
            AnswerRelevancyMetric as _AnswerRelevancyMetric,
            ContextPrecisionMetric as _ContextPrecisionMetric,
            ContextRecallMetric as _ContextRecallMetric,
            GroundednessMetric as _GroundednessMetric,
            HallucinationDetectionMetric as _HallucinationDetectionMetric,
        )
        FaithfulnessMetric = _FaithfulnessMetric
        ContextRelevancyMetric = _ContextRelevancyMetric
        AnswerRelevancyMetric = _AnswerRelevancyMetric
        ContextPrecisionMetric = _ContextPrecisionMetric
        ContextRecallMetric = _ContextRecallMetric
        GroundednessMetric = _GroundednessMetric
        HallucinationDetectionMetric = _HallucinationDetectionMetric

__all__ = [
    "FaithfulnessMetric",
    "ContextRelevancyMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "GroundednessMetric",
    "HallucinationDetectionMetric",
]

def __getattr__(name: str):
    if name in __all__:
        _import_rag_metrics()
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
