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
    global \
        ContextPrecisionMetric, \
        ContextRecallMetric, \
        GroundednessMetric, \
        HallucinationDetectionMetric
    if FaithfulnessMetric is None:
        from llm_eval.rag.metrics import (
            AnswerRelevancyMetric as _AnswerRelevancyMetric,
        )
        from llm_eval.rag.metrics import (
            ContextPrecisionMetric as _ContextPrecisionMetric,
        )
        from llm_eval.rag.metrics import (
            ContextRecallMetric as _ContextRecallMetric,
        )
        from llm_eval.rag.metrics import (
            ContextRelevancyMetric as _ContextRelevancyMetric,
        )
        from llm_eval.rag.metrics import (
            FaithfulnessMetric as _FaithfulnessMetric,
        )
        from llm_eval.rag.metrics import (
            GroundednessMetric as _GroundednessMetric,
        )
        from llm_eval.rag.metrics import (
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
