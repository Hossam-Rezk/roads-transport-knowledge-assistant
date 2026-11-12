"""HTTP endpoints. Routes only validate input, call a service and shape the
response; they contain no business logic."""
import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.container import Container
from app.core.errors import AppError, ThreadBusyError
from app.core.logging_setup import log_event, request_id_var
from app.core.models import SearchMode

logger = logging.getLogger("app.http")
router = APIRouter()
ingest_lock = asyncio.Lock()

ERROR_RESPONSES = {status: {"model": ErrorResponse} for status in (409, 502, 503, 504)}


def get_container(request: Request) -> Container:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(container: Container = Depends(get_container)):
    """Readiness and index availability. Never returns secrets."""
    chunks = container.chunk_store.count_chunks()
    vectors = container.vector_store.count()
    return HealthResponse(
        index_available=chunks > 0 and vectors > 0,
        documents=container.chunk_store.count_documents(),
        chunks=chunks,
        vectors=vectors,
        chat_model=container.settings.chat_deployment,
        embedding_model=container.settings.embedding_deployment,
    )


@router.post("/ingest", response_model=IngestResponse, responses=ERROR_RESPONSES, tags=["system"])
async def ingest(body: IngestRequest | None = None,
                 container: Container = Depends(get_container)):
    """Ingest the approved local corpus (synchronous). Unchanged files are skipped."""
    if ingest_lock.locked():
        raise ThreadBusyError("an ingestion run is already in progress")
    async with ingest_lock:
        force = body.force if body else False
        report = await run_in_threadpool(container.ingestion_pipeline.run, force)
    return IngestResponse(**report.model_dump(), request_id=request_id_var.get())


@router.post("/search", response_model=SearchResponse, responses=ERROR_RESPONSES, tags=["search"])
async def search(body: SearchRequest, container: Container = Depends(get_container)):
    """Retrieval only: inspect ranked chunks without generating an answer."""
    outcome = await container.search_service.search(
        body.query, body.mode, body.filters, body.top_k, body.rerank)
    if outcome.reranked:
        ranking_stage = "reranked"
    else:
        ranking_stage = "fused" if body.mode == SearchMode.HYBRID else "retrieved"
    return SearchResponse(
        request_id=request_id_var.get(),
        mode=body.mode,
        ranking_stage=ranking_stage,
        filters=body.filters,
        candidate_counts=outcome.candidate_counts,
        results=[SearchResult.from_retrieved(rank, retrieved)
                 for rank, retrieved in enumerate(outcome.results, start=1)],
    )


@router.post("/chat", response_model=ChatResponse, responses=ERROR_RESPONSES, tags=["chat"])
async def chat(body: ChatRequest, container: Container = Depends(get_container)):
    """Ask a question. Returns an answer, an abstention, an out-of-scope result
    or a clarification question (then call /chat/resume)."""
    outcome = await container.chat_service.chat(
        body.thread_id, body.message, body.search_mode, body.filters, body.top_k)
    return ChatResponse(**outcome, request_id=request_id_var.get())


@router.post("/chat/resume", response_model=ChatResponse, responses=ERROR_RESPONSES, tags=["chat"])
async def resume_chat(body: ChatResumeRequest, container: Container = Depends(get_container)):
    """Answer a pending clarification question and continue the paused run."""
    outcome = await container.chat_service.resume(
        body.thread_id, body.interrupt_id, body.clarification)
    return ChatResponse(**outcome, request_id=request_id_var.get())


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(body: ChatRequest, container: Container = Depends(get_container)):
    """Same as /chat, but streams progress as Server-Sent Events: `stage` and
    `sources` events while the graph runs, then one `final` event holding the
    ChatResponse. Errors arrive as an `error` event (the HTTP status is 200)."""
    events = container.chat_service.chat_stream(
        body.thread_id, body.message, body.search_mode, body.filters, body.top_k)
    return server_sent_events(events)


@router.post("/chat/resume/stream", tags=["chat"])
async def resume_chat_stream(body: ChatResumeRequest,
                             container: Container = Depends(get_container)):
    """Same as /chat/resume, streamed as Server-Sent Events."""
    events = container.chat_service.resume_stream(
        body.thread_id, body.interrupt_id, body.clarification)
    return server_sent_events(events)


def server_sent_events(events: AsyncIterator[dict]) -> StreamingResponse:
    request_id = request_id_var.get()

    def format_event(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async def event_stream():
        request_id_var.set(request_id)       # the stream runs after the route has returned
        try:
            async for event in events:
                if event["type"] == "final":
                    response = ChatResponse(**event["response"], request_id=request_id)
                    event = {"type": "final", "response": response.model_dump()}
                yield format_event(event)
        except AppError as error:
            log_event(logger, "controlled_error", code=error.code, message=error.message)
            yield format_event({"type": "error", "code": error.code, "message": error.message})
        except Exception:
            logger.exception("unexpected_error")
            yield format_event({"type": "error", "code": "internal_error",
                                "message": "an unexpected error occurred"})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/threads/{thread_id}", tags=["chat"])
async def get_thread(thread_id: str, container: Container = Depends(get_container)):
    """Inspect a thread's saved state (messages, pending clarification)."""
    return await container.chat_service.get_thread(thread_id)
