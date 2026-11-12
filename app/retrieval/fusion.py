"""Reciprocal rank fusion (RRF).

BM25 scores and cosine distances are on different scales and cannot be added.
RRF ignores the raw scores and uses only each chunk's rank position:

    rrf_score(chunk) = sum over lists of  1 / (k + rank_in_that_list)

A chunk found by both retrievers appears once in the output (deduplicated by
chunk_id) with both contributions added, so agreement between lists is rewarded.
"""
from app.core.models import RetrievedChunk


def reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    fused_by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for position, candidate in enumerate(ranked_list, start=1):
            chunk_id = candidate.chunk.chunk_id
            if chunk_id not in fused_by_id:
                fused_by_id[chunk_id] = RetrievedChunk(chunk=candidate.chunk)
            fused = fused_by_id[chunk_id]
            fused.scores.update(candidate.scores)     # keep bm25 / cosine_distance for inspection
            fused.ranks.update(candidate.ranks)       # keep lexical / semantic rank
            fused.scores["rrf"] = fused.scores.get("rrf", 0.0) + 1.0 / (k + position)

    fused_list = sorted(fused_by_id.values(), key=lambda item: item.scores["rrf"], reverse=True)
    for position, fused in enumerate(fused_list, start=1):
        fused.ranks["fused"] = position
    return fused_list
