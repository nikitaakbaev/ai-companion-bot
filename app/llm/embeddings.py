"""Embedding services."""

import asyncio
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService(ABC):
    """Creates text embeddings."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Create one embedding."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple texts."""


class HashEmbeddingService(EmbeddingService):
    """Small deterministic embedding fallback with no external ML dependencies."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions

    async def embed_text(self, text: str) -> list[float]:
        """Create one embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create normalized bag-of-hashes embeddings."""
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise EmbeddingError("Embedding text must not be empty")
        return [_hash_embedding(text, self.dimensions) for text in clean_texts]


class SentenceTransformerEmbeddingService(EmbeddingService):
    """SentenceTransformers-based local embedding service."""

    def __init__(
        self,
        model_name: str,
        fallback_service: EmbeddingService | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback_service = fallback_service
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._load_error: str | None = None

    async def embed_text(self, text: str) -> list[float]:
        """Create one embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple texts."""
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise EmbeddingError("Embedding text must not be empty")

        try:
            model = await self._get_model()
            vectors = await asyncio.to_thread(
                model.encode,
                clean_texts,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
        except EmbeddingError:
            if self.fallback_service is not None:
                logger.warning("Using fallback embedding service")
                return await self.fallback_service.embed_texts(clean_texts)
            raise
        except Exception as exc:
            logger.exception("Failed to create embeddings")
            raise EmbeddingError("Failed to create embeddings") from exc

        return [[float(value) for value in vector] for vector in vectors]

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise EmbeddingError(self._load_error)

        async with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise EmbeddingError(self._load_error)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
            except ModuleNotFoundError as exc:
                self._load_error = (
                    "sentence-transformers is not installed. Install project dependencies "
                    "or set RAG_ENABLED=false to disable memory retrieval."
                )
                logger.warning("Embedding model is unavailable: %s", self._load_error)
                raise EmbeddingError(self._load_error) from exc
            except Exception as exc:
                self._load_error = (
                    "Failed to load sentence-transformers embedding model "
                    f"({exc.__class__.__name__}: {exc})"
                )
                logger.warning("Embedding model is unavailable: %s", self._load_error)
                raise EmbeddingError(self._load_error) from exc
            return self._model


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    tokens = _tokenize(text)
    if not tokens:
        raise EmbeddingError("Embedding text must not be empty")

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="big", signed=False)
        index = raw % dimensions
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        values[index] += sign

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _tokenize(text: str) -> list[str]:
    normalized = text.casefold()
    words = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        if len(word) >= 4:
            tokens.extend(word[index : index + 4] for index in range(len(word) - 3))
    return tokens
