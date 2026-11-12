"""Second-stage reranking with a local cross-encoder (FlashRank).

A cross-encoder reads the query and one chunk together and outputs a relevance
score for the pair. It is more accurate than first-stage retrieval but runs
once per candidate, so it is only applied to the short fused candidate list.
The model file is downloaded once into `cache_dir` and then runs on the CPU.
"""
import threading
from pathlib import Path

from flashrank import Ranker, RerankRequest

from app.core.models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker: Ranker | None = None
        self._load_lock = threading.Lock()

    def _get_ranker(self) -> Ranker:
        # Loaded on first use so the API starts quickly and tests need no model.
        with self._load_lock:
            if self._ranker is None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._ranker = Ranker(model_name=self._model_name, cache_dir=str(self._cache_dir))
        return self._ranker

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        passages = [{"id": candidate.chunk.chunk_id, "text": candidate.chunk.search_text}
                    for candidate in candidates]
        scored_passages = self._get_ranker().rerank(RerankRequest(query=query, passages=passages))
        rerank_score_by_id = {passage["id"]: float(passage["score"]) for passage in scored_passages}

        reranked = sorted(candidates, key=lambda item: rerank_score_by_id[item.chunk.chunk_id],
                          reverse=True)[:top_k]
        for position, candidate in enumerate(reranked, start=1):
            candidate.scores["rerank"] = rerank_score_by_id[candidate.chunk.chunk_id]
            candidate.ranks["final"] = position
        return reranked
