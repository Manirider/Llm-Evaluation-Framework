"""
Embedding service abstraction with caching, batching, and GPU support.
"""

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from llm_eval.config.settings import EmbeddingConfig
from llm_eval.exceptions.base import EmbeddingError


class EmbeddingService:
    """
    Singleton-style wrapper for SentenceTransformer embedding generation.
    Supports in-memory caching and batch processing.
    """

    _instances: dict[str, "EmbeddingService"] = {}

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._cache: dict[str, np.ndarray] = {}
        try:
            logger.info(f"Loading SentenceTransformer model '{self.config.model_name}' on device '{self.config.device}'...")
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{self.config.model_name}': {e}") from e

    @classmethod
    def get_instance(cls, config: EmbeddingConfig | None = None) -> "EmbeddingService":
        cfg = config or EmbeddingConfig()
        key = f"{cfg.model_name}_{cfg.device}"
        if key not in cls._instances:
            cls._instances[key] = cls(cfg)
        return cls._instances[key]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a list of text strings into numpy embedding vectors with caching.
        """
        if not texts:
            return np.empty((0, 384))

        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        embeddings: list[np.ndarray | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            try:
                encoded = self.model.encode(
                    uncached_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )
                for idx, text, vec in zip(uncached_indices, uncached_texts, encoded):
                    vec_arr = np.array(vec, dtype=np.float32)
                    self._cache[text] = vec_arr
                    embeddings[idx] = vec_arr
            except Exception as e:
                raise EmbeddingError(f"Failed during text embedding generation: {e}") from e

        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Encode a single string into a 1D numpy array.
        """
        res = self.embed_texts([text])
        return res[0]
