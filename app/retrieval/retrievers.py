"""The two first-stage retrievers: lexical (BM25) and semantic (embeddings).

Both take the same inputs and return `RetrievedChunk` lists ranked best-first,
so the fusion step can treat them identically.
"""
import re

from app.core.models import RetrievedChunk, SearchFilters
from app.retrieval.filters import to_chroma_where, to_sql_where
from app.storage.chunk_store import ChunkStore
from app.storage.vector_store import VectorStore

# Very common words carry no signal for BM25 and only slow the query down.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or "
    "that the this to was what when where which who will with".split())


def build_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined with OR.
    Quoting stops user input being interpreted as FTS5 syntax."""
    terms = [term for term in re.findall(r"[A-Za-z0-9]+", query.lower())
             if term not in STOPWORDS]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))


class LexicalRetriever:
    def __init__(self, chunk_store: ChunkStore, primary_category: str):
        self._chunk_store = chunk_store
        self._primary_category = primary_category

    def search(self, query: str, filters: SearchFilters, limit: int) -> list[RetrievedChunk]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        where_sql, where_params = to_sql_where(filters, self._primary_category)
        matches = self._chunk_store.search_bm25(fts_query, where_sql, where_params, limit)
        return [RetrievedChunk(chunk=chunk, scores={"bm25": score}, ranks={"lexical": rank})
                for rank, (chunk, score) in enumerate(matches, start=1)]


class SemanticRetriever:
    def __init__(self, vector_store: VectorStore, chunk_store: ChunkStore, primary_category: str):
        self._vector_store = vector_store
        self._chunk_store = chunk_store
        self._primary_category = primary_category

    def search(self, query: str, filters: SearchFilters, limit: int) -> list[RetrievedChunk]:
        where = to_chroma_where(filters, self._primary_category)
        matches = self._vector_store.search(query, where, limit)
        distance_by_id = dict(matches)
        chunks = self._chunk_store.get_chunks([chunk_id for chunk_id, _ in matches])
        return [RetrievedChunk(chunk=chunk,
                               scores={"cosine_distance": distance_by_id[chunk.chunk_id]},
                               ranks={"semantic": rank})
                for rank, chunk in enumerate(chunks, start=1)]
