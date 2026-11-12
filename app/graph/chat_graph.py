"""The main chat graph.

  START
    -> start_turn            reset per-question state, record the user message
    -> plan_query            LLM -> validated QueryPlan
         |- out of scope ----------> scope_response -----------------> END
         |- needs clarification ---> ask_clarification (interrupt) --> plan_query
         '- otherwise -------------> retrieve (retrieval subgraph)
                                       |- no evidence -> insufficient_evidence_response -> END
                                       '- evidence ---> complete_split_articles
                                                        -> generate_answer (agent + tool + middleware)
                                                        -> validate_citations -> END

Collaborators (planner, retrieval graph, answer generator) are passed in, so
tests can build the same graph with fakes and no network access.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.config import Settings
from app.core.logging_setup import log_event
from app.core.models import Citation, RetrievedChunk
from app.graph.node_logging import logged_node
from app.graph.prompts import DISCLAIMER, INSUFFICIENT_EVIDENCE_ANSWER, OUT_OF_SCOPE_ANSWER
from app.graph.retrieval_graph import from_state, to_state
from app.graph.state import ChatState
from app.storage.chunk_store import ChunkStore

logger = logging.getLogger("app.graph")


def build_answer(status: str, text: str, citations: list[dict] | None = None) -> dict:
    """State update that finishes a turn: the API answer plus the AI message
    that goes into the thread's memory."""
    return {"answer": {"status": status, "answer": text, "citations": citations or []},
            "messages": [AIMessage(text)]}


def build_chat_graph(planner, retrieval_graph, answer_generator, chunk_store: ChunkStore,
                     settings: Settings, checkpointer):

    # ------------------------------------------------------------------ nodes

    @logged_node("start_turn")
    def start_turn(state: ChatState) -> dict:
        """Everything except `messages` describes one question, so reset it here."""
        return {
            "messages": [HumanMessage(state["user_message"])],
            "query_plan": {}, "standalone_query": "", "filters": {},
            "clarification": {"status": "none", "attempts": 0, "question": None},
            "lexical_candidates": [], "semantic_candidates": [],
            "retrieval_candidates": [], "final_evidence": [],
            "answer_draft": {}, "answer": {},
        }

    @logged_node("plan_query")
    async def plan_query(state: ChatState) -> dict:
        clarification = state["clarification"]
        may_ask = clarification["attempts"] < settings.max_clarification_attempts
        plan = await planner.plan(state["messages"], chunk_store.list_document_titles(), may_ask)

        # Filters sent explicitly by the caller win over filters the model inferred.
        filters = {**plan.filters.model_dump(exclude_none=True),
                   **{key: value for key, value in state.get("request_filters", {}).items()
                      if value is not None}}

        asks_clarification = (plan.intent == "legal_question" and plan.needs_clarification
                              and may_ask)
        if asks_clarification:
            clarification = {**clarification, "status": "pending",
                             "question": plan.clarification_question
                             or "Could you clarify what you would like to know?"}
        log_event(logger, "query_planned", intent=plan.intent, filters=filters,
                  needs_clarification=asks_clarification)
        return {"query_plan": plan.model_dump(), "standalone_query": plan.standalone_query,
                "filters": filters, "clarification": clarification}

    @logged_node("ask_clarification")
    def ask_clarification(state: ChatState) -> dict:
        """Pause the graph and wait for the user.

        `interrupt()` saves the state and stops the run. When the API resumes the
        thread, this node runs again FROM ITS FIRST LINE and `interrupt()` returns
        the user's reply. Nothing with side effects may come before it."""
        clarification = state["clarification"]
        user_reply = interrupt({"question": clarification["question"],
                                "attempt": clarification["attempts"] + 1,
                                "max_attempts": settings.max_clarification_attempts})
        return {
            "messages": [AIMessage(clarification["question"]), HumanMessage(str(user_reply))],
            "clarification": {"status": "resolved", "attempts": clarification["attempts"] + 1,
                              "question": clarification["question"]},
        }

    @logged_node("scope_response")
    def scope_response(state: ChatState) -> dict:
        return build_answer("out_of_scope", OUT_OF_SCOPE_ANSWER)

    @logged_node("insufficient_evidence_response")
    def insufficient_evidence_response(state: ChatState) -> dict:
        return build_answer("insufficient_evidence", INSUFFICIENT_EVIDENCE_ANSWER)

    @logged_node("complete_split_articles")
    def complete_split_articles(state: ChatState) -> dict:
        """A long article is indexed as several parts and retrieval may select only
        one of them. Add the missing parts so the model never answers from half an
        article (e.g. items 8-13 of a 13-item list). Bounded by a setting."""
        evidence = from_state(state["final_evidence"])
        present_ids = {item.chunk.chunk_id for item in evidence}
        completed, added_count = [], 0
        for item in evidence:
            completed.append(item)
            if item.chunk.part_count == 1:
                continue
            for part in chunk_store.get_all_parts(item.chunk):
                if (part.chunk_id not in present_ids
                        and added_count < settings.max_added_article_parts):
                    completed.append(RetrievedChunk(chunk=part))
                    present_ids.add(part.chunk_id)
                    added_count += 1
        log_event(logger, "split_articles_completed", added_parts=added_count)
        return {"final_evidence": to_state(completed)}

    @logged_node("generate_answer")
    async def generate_answer(state: ChatState) -> dict:
        generated = await answer_generator.generate(
            state["standalone_query"], from_state(state["final_evidence"]))
        return {"answer_draft": {**generated.draft.model_dump(),
                                 "tool_chunk_ids": generated.tool_chunk_ids}}

    @logged_node("validate_citations")
    def validate_citations(state: ChatState) -> dict:
        """A citation is accepted only if the chunk was actually shown to the
        model (final evidence or a tool result) AND exists in the index."""
        draft = state["answer_draft"]
        if draft["status"] != "answered":
            return build_answer("insufficient_evidence", INSUFFICIENT_EVIDENCE_ANSWER)

        shown_chunk_ids = {item["chunk"]["chunk_id"] for item in state["final_evidence"]}
        shown_chunk_ids.update(draft["tool_chunk_ids"])
        cited_chunk_ids = list(dict.fromkeys(draft["cited_chunk_ids"]))
        accepted_ids = [chunk_id for chunk_id in cited_chunk_ids if chunk_id in shown_chunk_ids]
        rejected_ids = [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in shown_chunk_ids]

        cited_chunks = chunk_store.get_chunks(accepted_ids)
        log_event(logger, "citations_validated", accepted=len(cited_chunks), rejected=rejected_ids)
        if not cited_chunks:
            # An "answer" without a single verifiable citation is not grounded.
            return build_answer("insufficient_evidence", INSUFFICIENT_EVIDENCE_ANSWER)

        answer_text = draft["answer"]
        for chunk_id in rejected_ids:
            answer_text = answer_text.replace(f"[{chunk_id}]", "")
        citations = [Citation.from_chunk(chunk).model_dump() for chunk in cited_chunks]
        return build_answer("answered", f"{answer_text.strip()}\n\n{DISCLAIMER}", citations)

    # ----------------------------------------------------------------- routing

    def route_after_plan(state: ChatState) -> str:
        if state["query_plan"]["intent"] == "out_of_scope":
            return "scope_response"
        if state["clarification"]["status"] == "pending":
            return "ask_clarification"
        return "retrieve"

    def route_after_retrieval(state: ChatState) -> str:
        if not state["final_evidence"]:
            return "insufficient_evidence_response"
        return "complete_split_articles"

    # ------------------------------------------------------------------- graph

    builder = StateGraph(ChatState)
    builder.add_node("start_turn", start_turn)
    builder.add_node("plan_query", plan_query)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("scope_response", scope_response)
    builder.add_node("retrieve", retrieval_graph)                 # subgraph as a node
    builder.add_node("insufficient_evidence_response", insufficient_evidence_response)
    builder.add_node("complete_split_articles", complete_split_articles)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("validate_citations", validate_citations)

    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "plan_query")
    builder.add_conditional_edges("plan_query", route_after_plan,
                                  ["scope_response", "ask_clarification", "retrieve"])
    builder.add_edge("ask_clarification", "plan_query")
    builder.add_conditional_edges("retrieve", route_after_retrieval,
                                  ["complete_split_articles", "insufficient_evidence_response"])
    builder.add_edge("complete_split_articles", "generate_answer")
    builder.add_edge("generate_answer", "validate_citations")
    builder.add_edge("scope_response", END)
    builder.add_edge("insufficient_evidence_response", END)
    builder.add_edge("validate_citations", END)
    return builder.compile(checkpointer=checkpointer)
