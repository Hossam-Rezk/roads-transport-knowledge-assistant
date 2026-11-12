"""Chroma vector store for semantic search.

Chroma keeps the embedding of each chunk plus the metadata fields used for
filtering. Chunk text and full metadata are read back from the ChunkStore.
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.models import Chunk

COLLECTION_NAME = "legislation_chunks"
FILTERABLE_FIELDS = ("document_id", "primary_category", "year", "legislation_type", "status")


class VectorStore:
    def __init__(self, chroma_dir: Path, embeddings: Embeddings):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._chroma = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def replace_document(self, document_id: str, chunks: list[Chunk]) -> None:
        """Remove the document's old vectors, then embed and store the new chunks."""
        self.delete_document(document_id)
        if not chunks:
            return
        self._chroma.add_texts(
            texts=[chunk.search_text for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def delete_document(self, document_id: str) -> None:
        self._chroma._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._chroma._collection.count()

    def search(self, query: str, where: dict, limit: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_distance) pairs, closest first. `where` is a
        Chroma metadata filter applied inside the vector search."""
        matches = self._chroma.similarity_search_with_score(query, k=limit, filter=where)
        return [(document.id, float(distance)) for document, distance in matches]

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        # Chroma does not accept None values, so missing fields are left out.
        values = {field: getattr(chunk, field) for field in FILTERABLE_FIELDS}
        return {field: value for field, value in values.items() if value is not None}
