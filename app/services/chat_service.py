"""Chat use case: run or resume the chat graph for one thread.

Responsibilities kept out of the graph itself:
  * one run per thread at a time (a second concurrent request gets HTTP 409)
  * deciding whether a thread is waiting for a clarification
  * an overall timeout and translating model errors into controlled errors
  * shaping the graph result into the API's chat outcome

Every run is streamed internally. `chat_stream` / `resume_stream` yield progress
events while the graph works and one final event with the outcome; `chat` /
`resume` consume the same stream and return only the outcome.

Event shapes
  {"type": "stage",   "key": "plan_query", "label": "Understanding your question"}
  {"type": "sources", "documents": ["Law No. (22) of 2006 ...", ...]}
  {"type": "final",   "response": {thread_id, status, answer, citations, clarification}}
"""
import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from langgraph.types import Command

from app.config import Settings
from app.core.errors import (
    IndexUnavailableError,
    InvalidResumeError,
    ModelFailureError,
    PendingClarificationError,
    RequestTimeoutError,
    ThreadBusyError,
)
from app.core.logging_setup import log_event, thread_id_var
from app.core.models import SearchFilters, SearchMode
from app.storage.chunk_store import ChunkStore

logger = logging.getLogger(__name__)

# Graph step -> text shown to the user while that step runs. Steps that are not
# listed are too quick or too technical to be worth showing.
STAGE_LABELS = {
    "plan_query": "Understanding your question",
    "lexical_search": "Searching the legislation",
    "rerank_candidates": "Ranking the most relevant passages",
    "generate_answer": "Writing an answer from the evidence",
    "tools": "Reading a referenced article in full",      # the agent's tool call
    "validate_citations": "Checking every citation",
}


class ChatService:
    def __init__(self, chat_graph, chunk_store: ChunkStore, settings: Settings):
        self._chat_graph = chat_graph
        self._chunk_store = chunk_store
        self._settings = settings
        self._thread_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------ public API

    async def chat_stream(self, thread_id: str | None, message: str, search_mode: SearchMode,
                          filters: SearchFilters, top_k: int) -> AsyncIterator[dict]:
        thread_id = thread_id or uuid.uuid4().hex
        async with self._exclusive_run(thread_id):
            self._require_index()
            config = self._config(thread_id)
            if await self._get_pending_interrupt(config):
                raise PendingClarificationError(
                    "this thread is waiting for a clarification; use POST /chat/resume")
            graph_input = {
                "user_message": message,
                "search_mode": search_mode.value,
                "request_filters": filters.model_dump(exclude_none=True),
                "top_k": top_k,
            }
            async for event in self._run(graph_input, config, thread_id):
                yield event

    async def resume_stream(self, thread_id: str, interrupt_id: str | None,
                            clarification: str) -> AsyncIterator[dict]:
        async with self._exclusive_run(thread_id):
            config = self._config(thread_id)
            pending = await self._get_pending_interrupt(config)
            if pending is None:
                raise InvalidResumeError("this thread has no pending clarification")
            if interrupt_id and interrupt_id != pending.id:
                raise InvalidResumeError("interrupt_id does not match the pending clarification")
            # Command(resume=...) continues the paused run; the value becomes the
            # return value of interrupt() inside the ask_clarification node.
            async for event in self._run(Command(resume=clarification), config, thread_id):
                yield event

    async def chat(self, thread_id: str | None, message: str, search_mode: SearchMode,
                   filters: SearchFilters, top_k: int) -> dict:
        return await self._final_outcome(
            self.chat_stream(thread_id, message, search_mode, filters, top_k))

    async def resume(self, thread_id: str, interrupt_id: str | None, clarification: str) -> dict:
        return await self._final_outcome(self.resume_stream(thread_id, interrupt_id, clarification))

    async def get_thread(self, thread_id: str) -> dict:
        """Saved state of a thread, for inspection and demos."""
        snapshot = await self._chat_graph.aget_state(self._config(thread_id))
        pending = snapshot.interrupts[0] if snapshot.interrupts else None
        values = snapshot.values
        return {
            "thread_id": thread_id,
            "exists": bool(values),
            "next_nodes": list(snapshot.next),
            "pending_interrupt": ({"interrupt_id": pending.id, **pending.value}
                                  if pending else None),
            "clarification": values.get("clarification"),
            "standalone_query": values.get("standalone_query"),
            "filters": values.get("filters"),
            "search_mode": values.get("search_mode"),
            "messages": [{"role": message.type, "content": message.content}
                         for message in values.get("messages", [])],
        }

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    async def _final_outcome(events: AsyncIterator[dict]) -> dict:
        outcome = {}
        async for event in events:
            if event["type"] == "final":
                outcome = event["response"]
        return outcome

    @asynccontextmanager
    async def _exclusive_run(self, thread_id: str):
        lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise ThreadBusyError("another request is already running on this thread")
        async with lock:
            thread_id_var.set(thread_id)
            yield

    def _require_index(self) -> None:
        if self._chunk_store.count_chunks() == 0:
            raise IndexUnavailableError("the index is empty; call POST /ingest first")

    async def _get_pending_interrupt(self, config: dict):
        snapshot = await self._chat_graph.aget_state(config)
        return snapshot.interrupts[0] if snapshot.interrupts else None

    async def _run(self, graph_input, config: dict, thread_id: str) -> AsyncIterator[dict]:
        """Run the graph, yielding progress events and then the final outcome."""
        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                # "tasks" reports each node as it starts; "updates" carries what a
                # node wrote; subgraphs=True includes the retrieval subgraph and the agent.
                async for _namespace, stream_mode, data in self._chat_graph.astream(
                        graph_input, config, stream_mode=["tasks", "updates"], subgraphs=True):
                    progress_event = self._to_progress_event(stream_mode, data)
                    if progress_event:
                        yield progress_event
        except TimeoutError as error:
            raise RequestTimeoutError("the request took too long and was stopped") from error
        except openai.OpenAIError as error:
            log_event(logger, "model_failure", error=type(error).__name__)
            raise ModelFailureError(
                f"the language model call failed ({type(error).__name__})") from error

        # The saved state is the source of truth for how the run ended.
        snapshot = await self._chat_graph.aget_state(config)
        if snapshot.interrupts:
            pending = snapshot.interrupts[0]
            log_event(logger, "chat_paused_for_clarification", interrupt_id=pending.id)
            outcome = {"thread_id": thread_id, "status": "needs_clarification", "answer": None,
                       "citations": [],
                       "clarification": {"interrupt_id": pending.id, **pending.value}}
        else:
            answer = snapshot.values["answer"]
            log_event(logger, "chat_finished", status=answer["status"],
                      citation_count=len(answer["citations"]))
            outcome = {"thread_id": thread_id, "status": answer["status"],
                       "answer": answer["answer"], "citations": answer["citations"],
                       "clarification": None}
        yield {"type": "final", "response": outcome}

    @staticmethod
    def _to_progress_event(stream_mode: str, data: dict) -> dict | None:
        if stream_mode == "tasks":
            is_task_start = "input" in data
            if is_task_start and data["name"] in STAGE_LABELS:
                return {"type": "stage", "key": data["name"], "label": STAGE_LABELS[data["name"]]}
        elif stream_mode == "updates" and "rerank_candidates" in data:
            evidence = data["rerank_candidates"].get("final_evidence", [])
            titles = list(dict.fromkeys(item["chunk"]["title"] for item in evidence))
            if titles:
                return {"type": "sources", "documents": titles}
        return None
