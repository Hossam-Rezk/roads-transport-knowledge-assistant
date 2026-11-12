"""Request and response models of the HTTP API (validated by Pydantic)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.models import Citation, RetrievedChunk, SearchFilters, SearchMode
from app.ingestion.pipeline import IngestionReport

MAX_MESSAGE_LENGTH = 2000      # characters; documented maximum for queries and messages
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def _require_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty or whitespace")
    return value


# ---------------------------------------------------------------- health / ingest

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    index_available: bool
    documents: int
    chunks: int
    vectors: int
    chat_model: str
    embedding_model: str


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Re-index files even if unchanged.")


class IngestResponse(IngestionReport):
    request_id: str


# ------------------------------------------------------------------------ search

class SearchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "query": "Who is exempt from the traffic toll?", "mode": "hybrid",
        "filters": {"legislation_type": "Law"}, "top_k": 5}})

    query: str = Field(max_length=MAX_MESSAGE_LENGTH)
    mode: SearchMode = SearchMode.HYBRID
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True, description="False returns the pre-rerank order.")

    _clean_query = field_validator("query")(_require_text)


class LabelledScore(BaseModel):
    type: Literal["bm25", "cosine_distance", "rrf", "rerank"]
    value: float
    note: str


SCORE_NOTES = {
    "bm25": "lexical match strength; higher is better; unbounded",
    "cosine_distance": "embedding distance; lower is better",
    "rrf": "reciprocal rank fusion of the lexical and semantic ranks; higher is better",
    "rerank": "cross-encoder relevance; higher is better",
}


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str
    source_url: str | None
    primary_category: str
    locator_type: str
    locator: str
    segment_type: str
    article_number: str | None
    article_title: str | None
    legislation_type: str | None
    number: int | None
    year: int | None
    authority: str | None
    status: str | None
    text: str
    scores: list[LabelledScore]
    stage_ranks: dict[str, int] = Field(
        description="Position of this chunk at each stage: lexical, semantic, fused, final.")

    @classmethod
    def from_retrieved(cls, rank: int, retrieved: RetrievedChunk) -> "SearchResult":
        chunk = retrieved.chunk
        return cls(
            rank=rank,
            **chunk.model_dump(include={
                "chunk_id", "document_id", "title", "source_url", "primary_category",
                "locator_type", "locator", "segment_type", "article_number", "article_title",
                "legislation_type", "number", "year", "authority", "status", "text"}),
            scores=[LabelledScore(type=score_type, value=value, note=SCORE_NOTES[score_type])
                    for score_type, value in retrieved.scores.items()],
            stage_ranks=retrieved.ranks,
        )


class SearchResponse(BaseModel):
    request_id: str
    mode: SearchMode
    ranking_stage: Literal["reranked", "fused", "retrieved"]
    filters: SearchFilters
    candidate_counts: dict[str, int]
    score_warning: str = ("Scores of different types are not comparable with each other "
                          "and are not confidence values.")
    results: list[SearchResult]


# -------------------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": "demo-thread-1",
        "message": "What does the law say about licensing autonomous vehicles?",
        "search_mode": "hybrid", "filters": {}}})

    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN,
                                  description="Omit to start a new thread.")
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)
    search_mode: SearchMode = SearchMode.HYBRID
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=5, ge=1, le=20)

    _clean_message = field_validator("message")(_require_text)


class ChatResumeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "thread_id": "demo-thread-1", "clarification": "I mean the Salik traffic toll fee."}})

    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    interrupt_id: str | None = Field(
        default=None, description="Optional check that you are answering the pending question.")
    clarification: str = Field(max_length=MAX_MESSAGE_LENGTH)

    _clean_clarification = field_validator("clarification")(_require_text)


class ClarificationPayload(BaseModel):
    interrupt_id: str
    question: str
    attempt: int
    max_attempts: int


class ChatResponse(BaseModel):
    thread_id: str
    request_id: str
    status: Literal["answered", "needs_clarification", "insufficient_evidence", "out_of_scope"]
    answer: str | None
    citations: list[Citation]
    clarification: ClarificationPayload | None


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None
