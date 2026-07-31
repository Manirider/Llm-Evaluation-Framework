"""
Shared pytest fixtures for the LLM Evaluation Framework test suite.

Provides reusable sample data, configuration objects, and mocked services
so that individual test modules remain focused and DRY.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_eval.config.settings import (
    EvaluationFrameworkConfig,
    LLMProviderConfig,
    MetricConfig,
)
from llm_eval.schemas.evaluation import EvaluationSample

# ------------------------------------------------------------------
# Auto-use fixture to ensure metric registration happens at test start
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_metrics():
    """Ensure all metrics are registered by importing the metrics modules."""
    # Import classical metrics (triggers BLEU, ROUGE-L registration)
    import llm_eval.metrics.classical  # noqa: F401
    import llm_eval.metrics.judge_metric  # noqa: F401

    # Note: semantic and rag metrics are lazy-loaded to avoid torch import
    yield


# ------------------------------------------------------------------
# Evaluation samples
# ------------------------------------------------------------------


@pytest.fixture()
def simple_sample() -> EvaluationSample:
    """Minimal sample with expected output and contexts."""
    return EvaluationSample(
        sample_id="test_simple",
        input_text="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France.",
        retrieved_contexts=["Paris is the capital and most populous city of France."],
    )


@pytest.fixture()
def sample_no_expected() -> EvaluationSample:
    """Sample without expected_output (e.g. open-ended generation)."""
    return EvaluationSample(
        sample_id="test_no_expected",
        input_text="Tell me a joke.",
        actual_output="Why did the chicken cross the road? To get to the other side.",
    )


@pytest.fixture()
def sample_no_contexts() -> EvaluationSample:
    """Sample without retrieved_contexts (non-RAG)."""
    return EvaluationSample(
        sample_id="test_no_ctx",
        input_text="What is AI?",
        actual_output="Artificial Intelligence is intelligence demonstrated by machines.",
        expected_output="AI is intelligence demonstrated by machines.",
    )


@pytest.fixture()
def hallucination_sample() -> EvaluationSample:
    """Sample where the output contradicts the retrieved context."""
    return EvaluationSample(
        sample_id="test_hallucination",
        input_text="What is the population of Mars?",
        actual_output="Mars has a thriving population of 50,000 human colonists.",
        expected_output="Mars has no human population.",
        retrieved_contexts=[
            "Mars is the fourth planet from the Sun with no known inhabitants.",
        ],
    )


@pytest.fixture()
def batch_samples(simple_sample: EvaluationSample) -> list[EvaluationSample]:
    """Small batch of samples for pipeline tests."""
    return [
        simple_sample,
        EvaluationSample(
            sample_id="batch_2",
            input_text="What is Git?",
            actual_output="Git is a version control system.",
            expected_output="Git tracks source code changes.",
            retrieved_contexts=["Git is a free open source VCS."],
        ),
    ]


# ------------------------------------------------------------------
# Configuration fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def default_config() -> EvaluationFrameworkConfig:
    """Default framework configuration with mock judge."""
    return EvaluationFrameworkConfig(
        judge=LLMProviderConfig(provider="mock"),
    )


@pytest.fixture()
def minimal_config() -> EvaluationFrameworkConfig:
    """Config with only BLEU enabled for fast isolated tests."""
    return EvaluationFrameworkConfig(
        judge=LLMProviderConfig(provider="mock"),
        metrics={
            "bleu": MetricConfig(enabled=True, threshold=0.1),
        },
    )


# ------------------------------------------------------------------
# Embedding service mock
# ------------------------------------------------------------------

_EMBED_DIM = 384


def _fake_embed_texts(texts: Any) -> np.ndarray:
    """Deterministic pseudo-embeddings based on text hash for reproducibility."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**31)
        vec = rng.randn(_EMBED_DIM).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)


def _fake_embed_single(text: str) -> np.ndarray:
    return _fake_embed_texts([text])[0]


@pytest.fixture()
def mock_embedding_service():
    """
    Fixture that patches the EmbeddingService singleton so no real
    SentenceTransformer model is downloaded during tests.
    Use this fixture in tests that need embedding functionality.
    """
    mock_service = MagicMock()
    mock_service.embed_texts = MagicMock(side_effect=_fake_embed_texts)
    mock_service.embed_single = MagicMock(side_effect=_fake_embed_single)

    with patch(
        "llm_eval.embeddings.service.EmbeddingService.get_instance",
        return_value=mock_service,
    ):
        yield mock_service
