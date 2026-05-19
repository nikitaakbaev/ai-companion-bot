from app.memory.vector_store import VectorSearchResult, VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.items: dict[str, tuple[list[float], str, dict]] = {}

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        self.items[id] = (embedding, text, metadata)

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        results = [
            VectorSearchResult(id=id, score=item[0][0], text=item[1], metadata=item[2])
            for id, item in self.items.items()
            if item[0][0] >= min_score
        ]
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self.items.pop(id, None)


async def test_vector_store_upsert_query_and_delete() -> None:
    store = FakeVectorStore()

    await store.upsert("a", [0.8], "alpha", {"user_id": 1})
    await store.upsert("b", [0.4], "beta", {"user_id": 1})

    results = await store.query([1.0], top_k=5, min_score=0.5)
    assert [result.id for result in results] == ["a"]

    await store.delete("a")
    assert await store.query([1.0], top_k=5, min_score=0.1) == [
        VectorSearchResult(id="b", score=0.4, text="beta", metadata={"user_id": 1})
    ]
