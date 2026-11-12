"""Routing, interrupt/resume, citation validation, memory and thread isolation,
using the real chat graph with mocked model responses."""
from langgraph.types import Command

from app.core.models import SearchFilters
from app.graph.query_planner import QueryPlan
from tests.conftest import FakeAnswerGenerator, FakePlanner


def legal_plan(query="toll fee", **overrides) -> QueryPlan:
    return QueryPlan(intent="legal_question", standalone_query=query, **overrides)


def ambiguous_plan() -> QueryPlan:
    return legal_plan(needs_clarification=True, clarification_question="Which fee do you mean?")


def thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def question(text: str, **extra) -> dict:
    return {"user_message": text, "search_mode": "hybrid", "request_filters": {}, "top_k": 3, **extra}


async def test_supported_question_is_answered_with_resolvable_citations(build_graph):
    graph = build_graph(FakePlanner([legal_plan()]))
    result = await graph.ainvoke(question("What is the toll fee?"), thread("t1"))

    answer = result["answer"]
    assert answer["status"] == "answered"
    citation = answer["citations"][0]
    assert citation["chunk_id"] == "c1" and citation["document_id"] == "dlp-1"
    assert citation["locator_type"] == "pdf_page" and citation["locator"] == "2"
    assert citation["source_url"].endswith("dlp-1.pdf")


async def test_out_of_scope_skips_retrieval_and_answering(build_graph):
    answer_generator = FakeAnswerGenerator()
    graph = build_graph(FakePlanner([QueryPlan(intent="out_of_scope", standalone_query="weather")]),
                        answer_generator)
    result = await graph.ainvoke(question("What is the weather?"), thread("t1"))

    assert result["answer"]["status"] == "out_of_scope"
    assert result["final_evidence"] == [] and answer_generator.calls == 0


async def test_no_evidence_abstains_without_calling_the_model(build_graph):
    answer_generator = FakeAnswerGenerator()
    graph = build_graph(FakePlanner([legal_plan()]), answer_generator,
                        lexical_chunks=[], semantic_chunks=[])
    result = await graph.ainvoke(question("Speeding fine?"), thread("t1"))

    assert result["answer"]["status"] == "insufficient_evidence"
    assert answer_generator.calls == 0


async def test_fabricated_citation_is_rejected(build_graph):
    invented = FakeAnswerGenerator(answer="The fine is 500 [made-up-1].", cited_chunk_ids=["made-up-1"])
    graph = build_graph(FakePlanner([legal_plan()]), invented)
    result = await graph.ainvoke(question("What is the fine?"), thread("t1"))

    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["answer"]["citations"] == []


async def test_only_chunks_shown_to_the_model_can_be_cited(build_graph):
    # c3 exists in the index but was neither in the evidence nor read through the tool.
    generator = FakeAnswerGenerator(answer="Fee is 4 [c1]. Licence needed [c3].",
                                    cited_chunk_ids=["c1", "c3"])
    graph = build_graph(FakePlanner([legal_plan()]), generator, semantic_chunks=[])
    result = await graph.ainvoke(question("fee?"), thread("t1"))

    assert [c["chunk_id"] for c in result["answer"]["citations"]] == ["c1"]
    assert "[c3]" not in result["answer"]["answer"]


async def test_ambiguous_question_interrupts_then_resumes_same_run(build_graph):
    planner = FakePlanner([ambiguous_plan(), legal_plan("Salik toll fee")])
    answer_generator = FakeAnswerGenerator()
    graph = build_graph(planner, answer_generator)

    paused = await graph.ainvoke(question("What is the fee?"), thread("t1"))
    pending = paused["__interrupt__"][0]
    assert pending.value == {"question": "Which fee do you mean?", "attempt": 1, "max_attempts": 2}
    assert answer_generator.calls == 0
    snapshot = await graph.aget_state(thread("t1"))
    assert snapshot.next == ("ask_clarification",)
    assert snapshot.values["clarification"]["status"] == "pending"

    resumed = await graph.ainvoke(Command(resume="The Salik toll"), thread("t1"))

    assert resumed["answer"]["status"] == "answered"
    assert resumed["standalone_query"] == "Salik toll fee"
    assert resumed["clarification"] == {"status": "resolved", "attempts": 1,
                                        "question": "Which fee do you mean?"}
    assert [m.content for m in resumed["messages"]][:3] == [
        "What is the fee?", "Which fee do you mean?", "The Salik toll"]
    assert answer_generator.calls == 1                       # no duplicate answer generation


async def test_at_most_two_clarification_attempts(build_graph):
    planner = FakePlanner([ambiguous_plan(), ambiguous_plan(), ambiguous_plan()])
    graph = build_graph(planner)

    first = await graph.ainvoke(question("fee?"), thread("t1"))
    second = await graph.ainvoke(Command(resume="not sure"), thread("t1"))
    third = await graph.ainvoke(Command(resume="still not sure"), thread("t1"))

    assert first["__interrupt__"][0].value["attempt"] == 1
    assert second["__interrupt__"][0].value["attempt"] == 2
    assert "__interrupt__" not in third                      # forced to proceed
    assert third["answer"]["status"] == "answered"
    assert [call["may_ask"] for call in planner.calls] == [True, True, False]


async def test_follow_up_sees_history_and_per_question_state_is_reset(build_graph):
    planner = FakePlanner([legal_plan("toll fee", filters=SearchFilters(year=2006)),
                           legal_plan("toll fee exemptions")])
    graph = build_graph(planner)

    first = await graph.ainvoke(question("What is the toll fee?"), thread("t1"))
    second = await graph.ainvoke(question("And who is exempt from it?"), thread("t1"))

    second_call_history = [m.content for m in planner.calls[1]["messages"]]
    assert second_call_history[0] == "What is the toll fee?"            # memory persisted
    assert second_call_history[-1] == "And who is exempt from it?"
    assert len(second["messages"]) == 4
    assert first["filters"] == {"year": 2006} and second["filters"] == {}   # reset per question


async def test_threads_are_isolated(build_graph):
    planner = FakePlanner([legal_plan(), legal_plan()])
    graph = build_graph(planner)

    await graph.ainvoke(question("Thread A secret question"), thread("A"))
    result_b = await graph.ainvoke(question("Thread B question"), thread("B"))

    assert [m.content for m in planner.calls[1]["messages"]] == ["Thread B question"]
    assert len(result_b["messages"]) == 2


async def test_request_filters_override_model_filters(build_graph):
    planner = FakePlanner([legal_plan(filters=SearchFilters(year=2006, legislation_type="Law"))])
    graph = build_graph(planner)
    result = await graph.ainvoke(question("fee?", request_filters={"year": 2023}), thread("t1"))
    assert result["filters"] == {"year": 2023, "legislation_type": "Law"}


async def test_missing_parts_of_a_split_article_are_added_as_evidence(build_graph, chunk_store):
    from tests.conftest import make_chunk
    parts = [make_chunk(f"dlp-7-004-{index}", document_id="dlp-7", part_index=index, part_count=3,
                        text=f"Obligation list part {index + 1}.") for index in range(3)]
    chunk_store.replace_document("dlp-7", "Law of dlp-7", "hash-7", parts)
    generator = FakeAnswerGenerator(answer="All obligations [dlp-7-004-0].",
                                    cited_chunk_ids=["dlp-7-004-0"])
    graph = build_graph(FakePlanner([legal_plan()]), generator,
                        lexical_chunks=[parts[1]], semantic_chunks=[])     # only part 2 retrieved

    result = await graph.ainvoke(question("List the obligations"), thread("t1"))

    evidence_ids = [item["chunk"]["chunk_id"] for item in result["final_evidence"]]
    assert sorted(evidence_ids) == ["dlp-7-004-0", "dlp-7-004-1", "dlp-7-004-2"]
    assert result["answer"]["status"] == "answered"          # the added part is citable


def test_query_plan_accepts_null_filters_from_the_model():
    plan = QueryPlan.model_validate(
        {"intent": "legal_question", "standalone_query": "toll fee", "filters": None})
    assert plan.filters == SearchFilters()
