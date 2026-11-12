"""Domain models shared by every layer (ingestion, storage, retrieval, graph, API)."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LegislationType = Literal["Law", "Resolution", "Regulation"]
LegislationStatus = Literal["In Force", "In Force as Amended", "Not in Force"]


class SearchMode(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchFilters(BaseModel):
    """Optional metadata filters. The primary-category restriction is always
    applied in addition to these and cannot be switched off."""
    model_config = ConfigDict(extra="forbid")

    year: int | None = Field(default=None, ge=1900, le=2100)
    legislation_type: LegislationType | None = None
    status: LegislationStatus | None = None


class Chunk(BaseModel):
    """One indexed piece of a legislation document."""
    chunk_id: str
    document_id: str
    title: str
    source_url: str | None
    primary_category: str

    # Where the chunk sits in the source document.
    locator_type: str = "pdf_page"
    locator: str                      # "6" or "6-7"
    page_start: int
    page_end: int
    segment_type: str                 # "preamble" | "article" | "schedule"
    article_number: str | None = None
    article_title: str | None = None
    part_index: int = 0               # position when a long segment is split
    part_count: int = 1

    text: str                         # the chunk's own text
    search_text: str                  # text with a title/article prefix, used for indexing

    # Legislation metadata from the manifest (None when the portal has no value).
    legislation_type: str | None = None
    number: int | None = None
    year: int | None = None
    authority: str | None = None
    status: str | None = None


class Citation(BaseModel):
    """A reference from an answer to one indexed chunk and its place in the source."""
    document_id: str
    chunk_id: str
    title: str
    source_url: str | None
    locator_type: str
    locator: str
    article: str | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Citation":
        return cls(document_id=chunk.document_id, chunk_id=chunk.chunk_id, title=chunk.title,
                   source_url=chunk.source_url, locator_type=chunk.locator_type,
                   locator=chunk.locator, article=chunk.article_number)


class RetrievedChunk(BaseModel):
    """A chunk plus how it scored. Scores are labelled by type and are NOT
    comparable with each other or usable as a confidence value."""
    chunk: Chunk
    scores: dict[str, float] = Field(default_factory=dict)   # bm25 / cosine_distance / rrf / rerank
    ranks: dict[str, int] = Field(default_factory=dict)      # lexical / semantic / fused / final
