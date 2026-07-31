"""
Embedding Cosine Similarity and BERTScore semantic evaluation metrics.
"""

from typing import Any

import numpy as np

from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.schemas.evaluation import EvaluationSample


@MetricRegistry.register("embedding_similarity")
class EmbeddingSimilarityMetric(BaseMetric):
    """
    Computes Cosine Similarity between actual_output and expected_output embedding vectors.
    """

    metric_name = "embedding_similarity"
    description = "Cosine similarity between output and expected reference sentence embeddings"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self._embedding_service = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from llm_eval.embeddings.service import EmbeddingService
            self._embedding_service = EmbeddingService.get_instance()
        return self._embedding_service

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.expected_output:
            return 0.0, "Skipped EmbeddingSimilarity calculation: sample missing expected_output", {}

        vecs = self.embedding_service.embed_texts([sample.actual_output, sample.expected_output])
        actual_vec, expected_vec = vecs[0], vecs[1]

        norm_actual = np.linalg.norm(actual_vec)
        norm_expected = np.linalg.norm(expected_vec)

        if norm_actual == 0 or norm_expected == 0:
            return 0.0, "Zero norm vector encountered", {}

        cosine_sim = float(np.dot(actual_vec, expected_vec) / (norm_actual * norm_expected))
        # Rescale [-1, 1] to [0, 1] range safely
        normalized_sim = float((cosine_sim + 1.0) / 2.0)

        reasoning = f"Embedding Cosine Similarity score: {cosine_sim:.4f}"
        details = {
            "raw_cosine_similarity": round(cosine_sim, 4),
            "normalized_score": round(normalized_sim, 4),
        }
        return normalized_sim, reasoning, details


@MetricRegistry.register("bert_score")
class BERTScoreMetric(BaseMetric):
    """
    Semantic similarity metric inspired by BERTScore using dense SentenceTransformer embeddings.
    Calculates precision, recall, and F1 over token-level embedding similarity matrix.
    """

    metric_name = "bert_score"
    description = "Token-level semantic overlap F1 metric via dense embeddings"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self._embedding_service = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from llm_eval.embeddings.service import EmbeddingService
            self._embedding_service = EmbeddingService.get_instance()
        return self._embedding_service

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.expected_output:
            return 0.0, "Skipped BERTScore calculation: sample missing expected_output", {}

        c_tokens = [t for t in sample.actual_output.split() if t.strip()]
        r_tokens = [t for t in sample.expected_output.split() if t.strip()]

        if not c_tokens or not r_tokens:
            return 0.0, "Empty token sequence", {}

        c_vecs = self.embedding_service.embed_texts(c_tokens)
        r_vecs = self.embedding_service.embed_texts(r_tokens)

        # Compute cosine similarity matrix (len(c_tokens) x len(r_tokens))
        c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        r_norms = np.linalg.norm(r_vecs, axis=1, keepdims=True)

        # Avoid division by zero
        c_norms[c_norms == 0] = 1e-8
        r_norms[r_norms == 0] = 1e-8

        c_normalized = c_vecs / c_norms
        r_normalized = r_vecs / r_norms

        sim_matrix = np.dot(c_normalized, r_normalized.T)

        precision = float(np.mean(np.max(sim_matrix, axis=1)))
        recall = float(np.mean(np.max(sim_matrix, axis=0)))

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = float(2 * precision * recall / (precision + recall))

        # Clamp safely
        f1_clamped = max(0.0, min(1.0, f1))
        reasoning = f"BERTScore F1: {f1_clamped:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})"
        details = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1_clamped, 4),
        }
        return f1_clamped, reasoning, details
