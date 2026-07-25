"""Tests for embedding service caching and singleton pattern."""

from __future__ import annotations

import numpy as np

from llm_eval.schemas.evaluation import EvaluationSample


class TestEmbeddingService:
    def test_embed_texts(self, mock_embedding_service) -> None:
        result = mock_embedding_service.embed_texts(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_embed_single(self, mock_embedding_service) -> None:
        result = mock_embedding_service.embed_single("hello")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_deterministic(self, mock_embedding_service) -> None:
        r1 = mock_embedding_service.embed_single("test string")
        r2 = mock_embedding_service.embed_single("test string")
        np.testing.assert_array_equal(r1, r2)

    def test_different_texts_differ(self, mock_embedding_service) -> None:
        r1 = mock_embedding_service.embed_single("text one")
        r2 = mock_embedding_service.embed_single("text two")
        assert not np.array_equal(r1, r2)
