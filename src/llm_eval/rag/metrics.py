"""
Retrieval-Augmented Generation (RAG) evaluation metrics:
Faithfulness, Context Relevancy, Answer Relevancy, Context Precision,
Context Recall, Groundedness, and Hallucination Detection.

Supports hybrid embedding-based calculation and LLM-as-a-Judge fallbacks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.judge.providers import MockJudge, create_judge
from llm_eval.schemas.evaluation import EvaluationSample

# Lazy import for EmbeddingService to avoid torch/sentence_transformers at module load
_EmbeddingService = None


def _get_embedding_service():
    """Get or create EmbeddingService instance lazily."""
    global _EmbeddingService
    if _EmbeddingService is None:
        from llm_eval.embeddings.service import EmbeddingService

        _EmbeddingService = EmbeddingService.get_instance()
    return _EmbeddingService


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors with zero-division safety."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _normalize_sim(raw_cosine: float) -> float:
    """Rescale cosine similarity from [-1, 1] to [0, 1]."""
    return max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


@MetricRegistry.register("faithfulness")
class FaithfulnessMetric(BaseMetric):
    """
    Measures whether statements in ``actual_output`` are grounded in the
    ``retrieved_contexts``.  High score indicates low hallucination.
    """

    metric_name: str = "faithfulness"
    description: str = "Groundedness of actual_output against retrieved context passages"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()
        judge_config = kwargs.get("judge_config")
        self.judge = (
            create_judge(judge_config)
            if isinstance(judge_config, LLMProviderConfig)
            else MockJudge()
        )

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.retrieved_contexts:
            return (
                1.0,
                "No retrieved contexts provided; assuming zero context dependency.",
                {},
            )

        combined_context = " ".join(sample.retrieved_contexts)
        sentences = [s.strip() for s in sample.actual_output.split(".") if s.strip()]

        if not sentences:
            return 1.0, "Empty actual output sentences", {}

        ctx_vec = self.embedding_service.embed_single(combined_context)
        sent_vecs = self.embedding_service.embed_texts(sentences)

        supported_count = 0
        sentence_scores: list[float] = []

        for vec in sent_vecs:
            sim = _cosine_similarity(vec, ctx_vec)
            norm_sim = _normalize_sim(sim)
            sentence_scores.append(norm_sim)
            if norm_sim >= 0.55:
                supported_count += 1

        faithfulness_score = float(supported_count / len(sentences))
        reasoning = (
            f"Faithfulness score: {faithfulness_score:.4f} "
            f"({supported_count}/{len(sentences)} sentences supported by retrieved context)."
        )
        details = {
            "total_sentences": len(sentences),
            "supported_sentences": supported_count,
            "sentence_similarity_scores": [round(s, 4) for s in sentence_scores],
        }
        return faithfulness_score, reasoning, details


# ---------------------------------------------------------------------------
# Context Relevancy
# ---------------------------------------------------------------------------


@MetricRegistry.register("context_relevancy")
class ContextRelevancyMetric(BaseMetric):
    """
    Measures the signal-to-noise ratio of ``retrieved_contexts`` relative to the
    prompt ``input_text``.
    """

    metric_name: str = "context_relevancy"
    description: str = "Relevancy of retrieved contexts to the query prompt"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.retrieved_contexts:
            return 0.0, "No retrieved contexts available to evaluate relevancy.", {}

        query_vec = self.embedding_service.embed_single(sample.input_text)
        ctx_vecs = self.embedding_service.embed_texts(sample.retrieved_contexts)

        scores: list[float] = []
        for vec in ctx_vecs:
            sim = _cosine_similarity(query_vec, vec)
            scores.append(_normalize_sim(sim))

        avg_relevancy = float(np.mean(scores))
        reasoning = (
            f"Context Relevancy score: {avg_relevancy:.4f} "
            f"across {len(scores)} retrieved context passages."
        )
        details = {
            "num_passages": len(scores),
            "passage_scores": [round(s, 4) for s in scores],
        }
        return avg_relevancy, reasoning, details


# ---------------------------------------------------------------------------
# Answer Relevancy
# ---------------------------------------------------------------------------


@MetricRegistry.register("answer_relevancy")
class AnswerRelevancyMetric(BaseMetric):
    """
    Measures how directly ``actual_output`` addresses prompt ``input_text``.
    """

    metric_name: str = "answer_relevancy"
    description: str = "Direct relevancy of actual output response to the input query prompt"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        query_vec = self.embedding_service.embed_single(sample.input_text)
        answer_vec = self.embedding_service.embed_single(sample.actual_output)

        sim = _cosine_similarity(query_vec, answer_vec)
        norm_sim = _normalize_sim(sim)

        reasoning = f"Answer Relevancy score: {norm_sim:.4f} against query intent."
        details = {"raw_cosine_similarity": round(sim, 4)}
        return norm_sim, reasoning, details


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------


@MetricRegistry.register("context_precision")
class ContextPrecisionMetric(BaseMetric):
    """
    Measures whether the most relevant context passages appear early in the
    retrieval ranking.  Uses a position-weighted precision@k formulation:
    relevant passages that appear earlier receive higher weight.
    """

    metric_name: str = "context_precision"
    description: str = "Position-weighted precision of retrieved contexts relative to the query"

    def __init__(
        self, threshold: float | None = None, relevance_cutoff: float = 0.55, **kwargs: Any
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()
        self.relevance_cutoff = relevance_cutoff

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.retrieved_contexts:
            return 0.0, "No retrieved contexts to evaluate precision.", {}

        query_vec = self.embedding_service.embed_single(sample.input_text)
        ctx_vecs = self.embedding_service.embed_texts(sample.retrieved_contexts)

        relevance_flags: list[bool] = []
        passage_scores: list[float] = []
        for vec in ctx_vecs:
            sim = _normalize_sim(_cosine_similarity(query_vec, vec))
            passage_scores.append(sim)
            relevance_flags.append(sim >= self.relevance_cutoff)

        # Average precision weighted by position
        weighted_sum = 0.0
        relevant_so_far = 0
        for rank, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                relevant_so_far += 1
                weighted_sum += relevant_so_far / rank

        precision = weighted_sum / len(relevance_flags) if relevance_flags else 0.0
        reasoning = (
            f"Context Precision: {precision:.4f} — "
            f"{sum(relevance_flags)}/{len(relevance_flags)} passages deemed relevant."
        )
        details = {
            "passage_scores": [round(s, 4) for s in passage_scores],
            "relevance_flags": relevance_flags,
        }
        return precision, reasoning, details


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------


@MetricRegistry.register("context_recall")
class ContextRecallMetric(BaseMetric):
    """
    Measures what fraction of the ground-truth expected output claims are
    covered by the retrieved contexts.  High recall means little relevant
    information was missed during retrieval.
    """

    metric_name: str = "context_recall"
    description: str = "Coverage of expected output claims by retrieved contexts"

    def __init__(
        self, threshold: float | None = None, support_cutoff: float = 0.55, **kwargs: Any
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()
        self.support_cutoff = support_cutoff

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.expected_output or not sample.retrieved_contexts:
            return 0.0, "Missing expected_output or retrieved_contexts for recall.", {}

        truth_sentences = [s.strip() for s in sample.expected_output.split(".") if s.strip()]
        if not truth_sentences:
            return 1.0, "No truth sentences extracted.", {}

        combined_ctx = " ".join(sample.retrieved_contexts)
        ctx_vec = self.embedding_service.embed_single(combined_ctx)
        truth_vecs = self.embedding_service.embed_texts(truth_sentences)

        covered = 0
        sentence_scores: list[float] = []
        for vec in truth_vecs:
            sim = _normalize_sim(_cosine_similarity(vec, ctx_vec))
            sentence_scores.append(sim)
            if sim >= self.support_cutoff:
                covered += 1

        recall = float(covered / len(truth_sentences))
        reasoning = (
            f"Context Recall: {recall:.4f} — "
            f"{covered}/{len(truth_sentences)} ground-truth claims covered."
        )
        details = {
            "total_truth_sentences": len(truth_sentences),
            "covered_sentences": covered,
            "sentence_scores": [round(s, 4) for s in sentence_scores],
        }
        return recall, reasoning, details


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


@MetricRegistry.register("groundedness")
class GroundednessMetric(BaseMetric):
    """
    Strict variant of Faithfulness: measures the average semantic similarity of
    each output claim to the best-matching individual context passage (not the
    concatenated context).  Captures whether each statement traces to a specific
    supporting source.
    """

    metric_name: str = "groundedness"
    description: str = "Per-claim traceability to individual context passages"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.retrieved_contexts:
            return 1.0, "No contexts to ground against; trivially grounded.", {}

        sentences = [s.strip() for s in sample.actual_output.split(".") if s.strip()]
        if not sentences:
            return 1.0, "Empty output; trivially grounded.", {}

        sent_vecs = self.embedding_service.embed_texts(sentences)
        ctx_vecs = self.embedding_service.embed_texts(sample.retrieved_contexts)

        claim_best_scores: list[float] = []
        for s_vec in sent_vecs:
            best = max(_normalize_sim(_cosine_similarity(s_vec, c_vec)) for c_vec in ctx_vecs)
            claim_best_scores.append(best)

        groundedness = float(np.mean(claim_best_scores))
        reasoning = (
            f"Groundedness score: {groundedness:.4f} — "
            f"average best-match similarity across {len(sentences)} claims."
        )
        details = {
            "claim_scores": [round(s, 4) for s in claim_best_scores],
            "num_claims": len(sentences),
            "num_contexts": len(sample.retrieved_contexts),
        }
        return groundedness, reasoning, details


# ---------------------------------------------------------------------------
# Hallucination Detection
# ---------------------------------------------------------------------------


@MetricRegistry.register("hallucination_score")
class HallucinationDetectionMetric(BaseMetric):
    """
    Inverse of Faithfulness — computes the ratio of output statements that are
    **not** supported by retrieved contexts.  A high score indicates heavy
    hallucination; a low score (near 0) means the output is well-grounded.

    Note: the score is inverted for the final result so that **higher = better**
    (i.e., higher means *less* hallucination) to remain consistent with the
    framework convention of ``1.0 = best``.
    """

    metric_name: str = "hallucination_score"
    description: str = "Inverse hallucination ratio (1.0 = no hallucination)"

    def __init__(
        self, threshold: float | None = None, support_cutoff: float = 0.55, **kwargs: Any
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_service = _get_embedding_service()
        self.support_cutoff = support_cutoff

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.retrieved_contexts:
            return 0.0, "No contexts; cannot verify output — assuming full hallucination.", {}

        combined_ctx = " ".join(sample.retrieved_contexts)
        sentences = [s.strip() for s in sample.actual_output.split(".") if s.strip()]

        if not sentences:
            return 1.0, "Empty output; no hallucination possible.", {}

        ctx_vec = self.embedding_service.embed_single(combined_ctx)
        sent_vecs = self.embedding_service.embed_texts(sentences)

        unsupported = 0
        sentence_details: list[dict[str, Any]] = []
        for sent, vec in zip(sentences, sent_vecs, strict=False):
            sim = _normalize_sim(_cosine_similarity(vec, ctx_vec))
            is_hallucinated = sim < self.support_cutoff
            if is_hallucinated:
                unsupported += 1
            sentence_details.append(
                {
                    "sentence": sent[:80],
                    "similarity": round(sim, 4),
                    "hallucinated": is_hallucinated,
                }
            )

        hallucination_ratio = unsupported / len(sentences)
        # Invert: 1.0 = no hallucination (best)
        inverted_score = 1.0 - hallucination_ratio

        reasoning = (
            f"Hallucination score: {inverted_score:.4f} "
            f"({unsupported}/{len(sentences)} unsupported sentences)."
        )
        details = {
            "hallucination_ratio": round(hallucination_ratio, 4),
            "unsupported_count": unsupported,
            "total_sentences": len(sentences),
            "sentence_breakdown": sentence_details,
        }
        return inverted_score, reasoning, details
