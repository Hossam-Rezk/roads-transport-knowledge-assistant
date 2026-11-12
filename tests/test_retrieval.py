"""Filters, lexical search, fusion/deduplication and the retrieval subgraph."""
from app.core.models import RetrievedChunk, SearchFilters
from app.graph.retrieval_graph import build_retrieval_graph, route_by_search_mode
from app.retrieval.filters import to_chroma_where, to_sql_where
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.retrievers import LexicalRetriever, build_fts_query
from tests.conftest import CATEGORY, FakeReranker, FakeRetriever, make_chunk


# ------------------------------------------------------------------- filters

def test_category_restriction_is_always_present():
    assert to_sql_where(SearchFilters(), CATEGORY) == ("chunks.primary_category = ?", [CATEGORY])
    assert to_chroma_where(SearchFilters(), CATEGORY) == {"primary_category": CATEGORY}


def test_filters_are_combined():
    filters = SearchFilters(year=2023, legislation_type="Law")
    where_sql, params = to_sql_where(filters, CATEGORY)
    assert where_sql == "chunks.primary_category = ? AND chunks.year = ? AND chunks.legislation_type = ?"
    assert params == [CATEGORY, 2023, "Law"]
    assert to_chroma_where(filters, CATEGORY) == {"$and": [
        {"primary_category": CATEGORY}, {"year": 2023}, {"legislation_type": "Law"}]}


def test_fts_query_is_quoted_and_drops_stopwords():
    assert build_fts_query('What is the "toll" fee? OR NEAR(') == '"toll" OR "fee" OR "near"'
    assert build_fts_query("what is the") == ""


# ------------------------------------------------------------ lexical search

def test_lexical_search_respects_each_filter_and_their_combination(chunk_store, indexed_chunks):
    retriever = LexicalRetriever(chunk_store, CATEGORY)
    query = "toll licence vehicles"

    everything = retriever.search(query, SearchFilters(), 10)
    by_year = retriever.search(query, SearchFilters(year=2023), 10)
    by_type = retriever.search(query, SearchFilters(legislation_type="Law"), 10)
    combined = retriever.search(query, SearchFilters(year=2023, legislation_type="Resolution"), 10)
    no_match = retriever.search(query, SearchFilters(year=2023, legislation_type="Law"), 10)

    assert {item.chunk.chunk_id for item in everything} == {"c1", "c2", "c3"}
    assert [item.chunk.chunk_id for item in by_year] == ["c3"]
    assert {item.chunk.chunk_id for item in by_type} == {"c1", "c2"}
    assert [item.chunk.chunk_id for item in combined] == ["c3"]
    assert no_match == []
    assert all("bm25" in item.scores and item.ranks["lexical"] >= 1 for item in everything)


def test_lexical_search_never_returns_another_category(chunk_store, indexed_chunks):
    outsider = make_chunk("x1", document_id="dlp-99", primary_category="Traffic",
                          text="The toll for traffic offences.")
    chunk_store.replace_document("dlp-99", "Traffic law", "hash", [outsider])
    results = LexicalRetriever(chunk_store, CATEGORY).search("toll traffic", SearchFilters(), 10)
    assert "x1" not in {item.chunk.chunk_id for item in results}


# -------------------------------------------------------------------- fusion

def ranked(*chunk_ids, rank_name):
    return [RetrievedChunk(chunk=make_chunk(chunk_id), ranks={rank_name: rank})
            for rank, chunk_id in enumerate(chunk_ids, start=1)]


def test_fusion_deduplicates_and_rewards_agreement():
    lexical = ranked("a", "b", "c", rank_name="lexical")
    semantic = ranked("d", "b", "a", rank_name="semantic")

    fused = reciprocal_rank_fusion([lexical, semantic], k=60)
    fused_ids = [item.chunk.chunk_id for item in fused]

    assert sorted(fused_ids) == ["a", "b", "c", "d"]            # each chunk once
    assert fused_ids[:2] == ["a", "b"]                           # found by both lists -> on top
    chunk_b = fused[1]
    assert chunk_b.ranks == {"lexical": 2, "semantic": 2, "fused": 2}
    assert abs(chunk_b.scores["rrf"] - (1 / 62 + 1 / 62)) < 1e-9


def test_fusion_uses_ranks_not_raw_scores():
    lexical = ranked("a", "b", rank_name="lexical")
    lexical[1].scores["bm25"] = 9999.0                           # a huge raw score must not matter
    fused = reciprocal_rank_fusion([lexical], k=60)
    assert [item.chunk.chunk_id for item in fused] == ["a", "b"]


# ---------------------------------------------------------- retrieval graph

def test_route_by_search_mode():
    assert route_by_search_mode({"search_mode": "lexical"}) == ["lexical_search"]
    assert route_by_search_mode({"search_mode": "semantic"}) == ["semantic_search"]
    assert route_by_search_mode({"search_mode": "hybrid"}) == ["lexical_search", "semantic_search"]


async def test_hybrid_runs_both_branches_then_joins_once(settings):
    lexical = FakeRetriever([make_chunk("a"), make_chunk("b")], "lexical")
    semantic = FakeRetriever([make_chunk("b"), make_chunk("c")], "semantic")
    graph = build_retrieval_graph(lexical, semantic, FakeReranker(), settings)

    state = await graph.ainvoke({"standalone_query": "toll", "filters": {"year": 2006},
                                 "search_mode": "hybrid", "top_k": 2})

    assert len(lexical.calls) == len(semantic.calls) == 1
    assert lexical.calls[0]["filters"] == semantic.calls[0]["filters"] == SearchFilters(year=2006)
    fused_ids = [item["chunk"]["chunk_id"] for item in state["retrieval_candidates"]]
    assert fused_ids == ["b", "a", "c"]                          # deduplicated, joined once
    final_ids = [item["chunk"]["chunk_id"] for item in state["final_evidence"]]
    assert final_ids == ["c", "a"]                               # reranker changed the order, top_k=2
    assert state["final_evidence"][0]["ranks"]["final"] == 1


async def test_single_mode_runs_only_one_branch(settings):
    lexical = FakeRetriever([make_chunk("a")], "lexical")
    semantic = FakeRetriever([make_chunk("b")], "semantic")
    graph = build_retrieval_graph(lexical, semantic, FakeReranker(), settings)

    state = await graph.ainvoke({"standalone_query": "toll", "search_mode": "lexical", "top_k": 5})

    assert len(lexical.calls) == 1 and semantic.calls == []
    assert [item["chunk"]["chunk_id"] for item in state["final_evidence"]] == ["a"]
