"""Search use case: run the retrieval subgraph on its own (no answer generation)."""
import logging
from dataclasses import dataclass

from app.config import Settings
from app.core.errors import IndexUnavailableError
from app.core.logging_setup import log_event
from app.core.models import RetrievedChunk, SearchFilters, SearchMode
from app.graph.retrieval_graph import from_state
from app.storage.chunk_store import ChunkStore

logger = logging.getLogger(__name__)


@dataclass
class SearchOutcome:
    results: list[RetrievedChunk]          # final ranked list (top_k)
    candidate_counts: dict[str, int]       # how many candidates each stage produced
    reranked: bool


class SearchService:
    def __init__(self, retrieval_graph, chunk_store: ChunkStore, settings: Settings):
        self._retrieval_graph = retrieval_graph
        self._chunk_store = chunk_store
        self._settings = settings

    async def search(self, query: str, mode: SearchMode, filters: SearchFilters,
                     top_k: int, rerank: bool = True) -> SearchOutcome:
        if self._chunk_store.count_chunks() == 0:
            raise IndexUnavailableError("the index is empty; call POST /ingest first")

        state = await self._retrieval_graph.ainvoke({
            "standalone_query": query,
            "filters": filters.model_dump(exclude_none=True),
            "search_mode": mode.value,
            "top_k": top_k,
            "rerank_enabled": rerank,
        })
        candidate_counts = {
            "lexical": len(state.get("lexical_candidates", [])),
            "semantic": len(state.get("semantic_candidates", [])),
            "fused": len(state.get("retrieval_candidates", [])),
            "final": len(state.get("final_evidence", [])),
        }
        log_event(logger, "search_finished", mode=mode.value, reranked=rerank, **candidate_counts)
        return SearchOutcome(results=from_state(state.get("final_evidence")),
                             candidate_counts=candidate_counts, reranked=rerank)
