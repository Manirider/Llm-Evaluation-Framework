"""Unit tests for EmbeddingService with mocked SentenceTransformer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_eval.config.settings import EmbeddingConfig
from llm_eval.embeddings.service import EmbeddingService
from llm_eval.exceptions.base import EmbeddingError


@pytest.fixture(autouse=True)
def _clear_singleton():
    EmbeddingService._instances.clear()
    yield
    EmbeddingService._instances.clear()


class TestEmbeddingServiceUnit:
    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_get_instance_singleton(self, mock_st: MagicMock) -> None:
        mock_st.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        svc1 = EmbeddingService.get_instance()
        svc2 = EmbeddingService.get_instance()
        assert svc1 is svc2
        mock_st.assert_called_once()

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_embed_texts_caching(self, mock_st: MagicMock) -> None:
        mock_st.return_value.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        svc = EmbeddingService(EmbeddingConfig(model_name="test-model"))

        r1 = svc.embed_texts(["hello", "world"])
        r2 = svc.embed_texts(["hello"])  # cached

        assert r1.shape[0] == 2
        assert r2.shape[0] == 1
        assert mock_st.return_value.encode.call_count == 1

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_embed_empty_list(self, mock_st: MagicMock) -> None:
        svc = EmbeddingService()
        result = svc.embed_texts([])
        assert result.shape == (0, 384)

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_embed_single(self, mock_st: MagicMock) -> None:
        mock_st.return_value.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        svc = EmbeddingService()
        vec = svc.embed_single("test")
        assert vec.ndim == 1
        assert len(vec) == 2

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_model_load_failure(self, mock_st: MagicMock) -> None:
        mock_st.side_effect = RuntimeError("model not found")
        with pytest.raises(EmbeddingError, match="Failed to load"):
            EmbeddingService()

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_encode_failure(self, mock_st: MagicMock) -> None:
        mock_st.return_value.encode.side_effect = RuntimeError("encode failed")
        svc = EmbeddingService()
        with pytest.raises(EmbeddingError, match="Failed during text embedding"):
            svc.embed_texts(["fail me"])

    @patch("llm_eval.embeddings.service.SentenceTransformer")
    def test_different_config_creates_new_instance(self, mock_st: MagicMock) -> None:
        mock_st.return_value.encode.return_value = np.array([[1.0]])
        cfg_cpu = EmbeddingConfig(model_name="model-a", device="cpu")
        cfg_cuda = EmbeddingConfig(model_name="model-b", device="cuda")
        svc_a = EmbeddingService.get_instance(cfg_cpu)
        svc_b = EmbeddingService.get_instance(cfg_cuda)
        assert svc_a is not svc_b
