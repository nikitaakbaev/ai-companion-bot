"""Vector store abstraction and Chroma implementation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector storage is unavailable."""


class VectorSearchResult(BaseModel):
    """One vector search result."""

    id: str
    score: float
    text: str
    metadata: dict = Field(default_factory=dict)


class VectorStore(ABC):
    """Stores and searches embeddings."""

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        """Insert or update a vector."""

    @abstractmethod
    async def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Search similar vectors."""

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete a vector by id."""


class ChromaVectorStore(VectorStore):
    """Persistent Chroma vector store."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collection: Any | None = None
        self._load_lock = asyncio.Lock()
        self._load_error: str | None = None

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        """Insert or update a vector."""
        collection = await self._get_collection()
        await asyncio.to_thread(
            collection.upsert,
            ids=[id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[_sanitize_metadata(metadata)],
        )

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Search similar vectors."""
        if top_k <= 0:
            return []

        collection = await self._get_collection()
        try:
            data = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("Chroma query failed")
            return []

        ids = (data.get("ids") or [[]])[0] if isinstance(data, dict) else []
        documents = (data.get("documents") or [[]])[0] if isinstance(data, dict) else []
        metadatas = (data.get("metadatas") or [[]])[0] if isinstance(data, dict) else []
        distances = (data.get("distances") or [[]])[0] if isinstance(data, dict) else []

        results: list[VectorSearchResult] = []
        for index, item_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = 1.0 - distance
            if score < min_score:
                continue
            results.append(
                VectorSearchResult(
                    id=str(item_id),
                    score=score,
                    text=str(documents[index]) if index < len(documents) else "",
                    metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                )
            )
        return results

    async def delete(self, id: str) -> None:
        """Delete a vector by id."""
        collection = await self._get_collection()
        await asyncio.to_thread(collection.delete, ids=[id])

    async def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._load_error is not None:
            raise VectorStoreError(self._load_error)

        async with self._load_lock:
            if self._collection is not None:
                return self._collection
            if self._load_error is not None:
                raise VectorStoreError(self._load_error)
            try:
                import chromadb

                client = await asyncio.to_thread(
                    chromadb.PersistentClient,
                    path=self.persist_directory,
                )
                self._collection = await asyncio.to_thread(
                    client.get_or_create_collection,
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except ModuleNotFoundError as exc:
                self._load_error = (
                    "chromadb is not installed. Install project dependencies "
                    "or set RAG_ENABLED=false to disable memory retrieval."
                )
                logger.warning("Vector store is unavailable: %s", self._load_error)
                raise VectorStoreError(self._load_error) from exc
            except Exception:
                logger.exception("Failed to initialize Chroma vector store")
                raise
            return self._collection


def _sanitize_metadata(metadata: dict) -> dict:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            sanitized[str(key)] = value
        elif isinstance(value, list):
            sanitized[str(key)] = ", ".join(str(item) for item in value)
        else:
            sanitized[str(key)] = str(value)
    return sanitized
