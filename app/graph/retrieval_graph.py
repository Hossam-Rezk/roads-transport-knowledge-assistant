"""Reusable retrieval subgraph.

            +--> lexical_search ---+
  START --> |                      +--> fuse_candidates --> rerank_candidates --> END
            +--> semantic_search --+

* `route_by_search_mode` starts one branch (lexical / semantic) or both (hybrid).
* In hybrid mode both branches run in the same step, in parallel. LangGraph runs
  `fuse_candidates` once, in the next step, after every branch started in the
  previous step has finished - that is the join.
* `/search` invokes this graph directly; `/chat` uses it as a node of the chat graph.
"""
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.core.models import RetrievedChunk, SearchFilters, SearchMode
from app.graph.node_logging import logged_node
from app.graph.state import RetrievalState
from app.retrieval.fusion import reciprocal_rank_fusion


def to_state(candidates: list[RetrievedChunk]) -> list[dict]:
    return [candidate.model_dump() for candidate in candidates]


def from_state(candidates: list[dict] | None) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(candidate) for candidate in candidates or []]


def route_by_search_mode(state: RetrievalState) -> list[str]:
    search_mode = state.get("search_mode", SearchMode.HYBRID.value)
    if search_mode == SearchMode.LEXICAL.value:
        return ["lexical_search"]
    if search_mode == SearchMode.SEMANTIC.value:
        return ["semantic_search"]
    return ["lexical_search", "semantic_search"]     # hybrid: fan out to both


def build_retrieval_graph(lexical_retriever, semantic_retriever, reranker, settings: Settings):
    """The three collaborators only need `.search(query, filters, limit)` and
    `.rerank(query, candidates, top_k)`, so tests can pass simple fakes."""

    @logged_node("lexical_search")
    def lexical_search(state: RetrievalState) -> dict:
        results = lexical_retriever.search(
            state["standalone_query"], SearchFilters(**state.get("filters", {})),
            settings.candidates_per_branch)
        return {"lexical_candidates": to_state(results)}

    @logged_node("semantic_search")
    def semantic_search(state: RetrievalState) -> dict:
        results = semantic_retriever.search(
            state["standalone_query"], SearchFilters(**state.get("filters", {})),
            settings.candidates_per_branch)
        return {"semantic_candidates": to_state(results)}

    @logged_node("fuse_candidates")
    def fuse_candidates(state: RetrievalState) -> dict:
        ranked_lists = [from_state(state.get("lexical_candidates")),
                        from_state(state.get("semantic_candidates"))]
        fused = reciprocal_rank_fusion([ranked for ranked in ranked_lists if ranked],
                                       k=settings.rrf_k)
        return {"retrieval_candidates": to_state(fused)}

    @logged_node("rerank_candidates")
    def rerank_candidates(state: RetrievalState) -> dict:
        candidates = from_state(state.get("retrieval_candidates"))
        top_k = state.get("top_k", settings.default_top_k)
        if state.get("rerank_enabled", True):
            final = reranker.rerank(state["standalone_query"], candidates, top_k)
        else:
            final = candidates[:top_k]            # inspection mode: fused order only
        return {"final_evidence": to_state(final)}

    builder = StateGraph(RetrievalState)
    builder.add_node("lexical_search", lexical_search)
    builder.add_node("semantic_search", semantic_search)
    builder.add_node("fuse_candidates", fuse_candidates)
    builder.add_node("rerank_candidates", rerank_candidates)

    builder.add_conditional_edges(START, route_by_search_mode,
                                  ["lexical_search", "semantic_search"])
    builder.add_edge("lexical_search", "fuse_candidates")
    builder.add_edge("semantic_search", "fuse_candidates")
    builder.add_edge("fuse_candidates", "rerank_candidates")
    builder.add_edge("rerank_candidates", END)
    return builder.compile()
