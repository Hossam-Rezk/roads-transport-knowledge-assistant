"""API contract: schema validation and controlled errors (no real model)."""
import pytest
from fastapi.testclient import TestClient

from app.api.schemas import ChatResponse, SearchResponse
from app.graph.query_planner import QueryPlan
from app.main import create_app
from tests.conftest import FakePlanner


def legal_plan(**overrides) -> QueryPlan:
    return QueryPlan(intent="legal_question", standalone_query="toll fee", **overrides)


@pytest.fixture
def client_factory(build_test_container):
    def factory(plans=()):
        return TestClient(create_app(build_test_container(FakePlanner(plans))))
    return factory


def test_health_reports_index_without_secrets(client_factory):
    body = client_factory().get("/health").json()
    assert body["index_available"] is True and body["chunks"] == 3
    assert "key" not in str(body).lower()


@pytest.mark.parametrize("payload", [
    {"message": ""},                                             # empty
    {"message": "   "},                                          # whitespace only
    {"message": "x" * 2001},                                     # too long
    {"message": "hello", "search_mode": "fuzzy"},                # unsupported mode
    {"message": "hello", "top_k": 0},
    {"message": "hello", "top_k": 21},
    {"message": "hello", "filters": {"year": "last year"}},      # wrong filter type
    {"message": "hello", "filters": {"legislation_type": "Poem"}},
    {"message": "hello", "filters": {"colour": "red"}},          # unknown filter
])
def test_invalid_chat_requests_are_rejected(client_factory, payload):
    response = client_factory().post("/chat", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "toll", "top_k": 50}, {"query": "toll", "mode": "magic"}])
def test_invalid_search_requests_are_rejected(client_factory, payload):
    assert client_factory().post("/search", json=payload).status_code == 422


def test_search_response_labels_mode_stage_and_scores(client_factory):
    response = client_factory().post("/search", json={"query": "toll", "mode": "hybrid", "top_k": 2})
    body = SearchResponse.model_validate(response.json())        # conforms to the schema
    assert response.status_code == 200
    assert body.mode.value == "hybrid" and body.ranking_stage == "reranked"
    assert body.candidate_counts == {"lexical": 2, "semantic": 2, "fused": 3, "final": 2}
    assert {score.type for score in body.results[0].scores} == {"rrf", "rerank"}
    assert response.headers["X-Request-ID"] == body.request_id


def test_chat_response_conforms_to_contract(client_factory):
    response = client_factory([legal_plan()]).post("/chat", json={"message": "What is the toll fee?"})
    body = ChatResponse.model_validate(response.json())
    assert body.status == "answered" and body.thread_id and body.request_id
    assert body.citations[0].chunk_id == "c1" and body.clarification is None


def test_clarification_then_resume_through_the_api(client_factory):
    client = client_factory([
        legal_plan(needs_clarification=True, clarification_question="Which fee?"), legal_plan()])

    paused = client.post("/chat", json={"thread_id": "t1", "message": "What is the fee?"}).json()
    assert paused["status"] == "needs_clarification" and paused["answer"] is None
    assert paused["clarification"]["question"] == "Which fee?"

    # A new /chat on a paused thread is refused; the caller must resume.
    blocked = client.post("/chat", json={"thread_id": "t1", "message": "Another question"})
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "clarification_pending"

    wrong_id = client.post("/chat/resume", json={
        "thread_id": "t1", "interrupt_id": "wrong", "clarification": "The toll"})
    assert wrong_id.status_code == 409 and wrong_id.json()["error"]["code"] == "invalid_resume"

    resumed = client.post("/chat/resume", json={
        "thread_id": "t1", "interrupt_id": paused["clarification"]["interrupt_id"],
        "clarification": "The toll"}).json()
    assert resumed["status"] == "answered" and resumed["thread_id"] == "t1"


def test_resume_without_pending_clarification_is_rejected(client_factory):
    response = client_factory().post("/chat/resume", json={
        "thread_id": "never-used", "clarification": "The toll"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_resume"


def test_empty_index_gives_controlled_error(client_factory, chunk_store):
    client = client_factory()
    for document_id in chunk_store.list_document_ids():
        chunk_store.delete_document(document_id)

    for path, payload in (("/search", {"query": "toll"}), ("/chat", {"message": "toll?"})):
        response = client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "index_unavailable"


def read_events(response) -> list[dict]:
    import json
    return [json.loads(block[len("data: "):]) for block in response.text.split("\n\n") if block]


def test_chat_stream_sends_progress_then_final_response(client_factory):
    response = client_factory([legal_plan()]).post(
        "/chat/stream", json={"thread_id": "s1", "message": "What is the toll fee?"})
    events = read_events(response)

    assert response.headers["content-type"].startswith("text/event-stream")
    stage_keys = [event["key"] for event in events if event["type"] == "stage"]
    assert stage_keys == ["plan_query", "lexical_search", "rerank_candidates",
                          "generate_answer", "validate_citations"]
    assert any(event["type"] == "sources" for event in events)
    final = ChatResponse.model_validate(events[-1]["response"])
    assert events[-1]["type"] == "final" and final.status == "answered"


def test_stream_reports_errors_as_events(client_factory):
    response = client_factory().post(
        "/chat/resume/stream", json={"thread_id": "never-used", "clarification": "The toll"})
    assert read_events(response) == [{"type": "error", "code": "invalid_resume",
                                      "message": "this thread has no pending clarification"}]
