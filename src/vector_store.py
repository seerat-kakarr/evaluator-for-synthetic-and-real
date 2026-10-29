"""Persistent vector index so grant embeddings are computed once, not every run."""

import numpy as np


class VectorStore:
    def __init__(self, path: str = "vector_store", name: str = "grants"):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.name = name
        self.collection = self._create()

    def _create(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def is_current(self, ids: list[str], documents: list[str], embedder: str) -> bool:
        """True when the stored documents and embedder match, so embedding can be skipped."""
        if self.collection.count() != len(ids):
            return False
        if not ids:
            return True
        stored = self.collection.get(ids=ids)
        saved = {
            id_: (document, (metadata or {}).get("embedder"))
            for id_, document, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        return saved == {id_: (document, embedder) for id_, document in zip(ids, documents)}

    def replace(self, ids: list[str], documents: list[str], vectors: np.ndarray, embedder: str):
        # Recreate rather than clear: a collection locks in its vector size on first insert.
        self.client.delete_collection(self.name)
        self.collection = self._create()
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=np.asarray(vectors).tolist(),
                metadatas=[{"embedder": embedder}] * len(ids),
            )

    def similarities(self, query: np.ndarray) -> dict[str, float]:
        """Cosine similarity of the query to every stored document, keyed by id."""
        result = self.collection.query(
            query_embeddings=np.asarray(query).tolist(),
            n_results=self.collection.count(),
        )
        return {
            id_: 1 - distance
            for id_, distance in zip(result["ids"][0], result["distances"][0])
        }
