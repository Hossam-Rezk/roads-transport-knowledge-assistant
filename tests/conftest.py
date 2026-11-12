"""Shared test helpers. All tests run offline: the model, the embeddings and
the reranker are replaced by the small fakes below."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.container import Container
from app.core.models import Chunk, RetrievedChunk
from app.graph.answer_agent import AnswerDraft, GeneratedAnswer
from app.graph.chat_graph import build_chat_graph
from app.graph.retrieval_graph import build_retrieval_graph
from app.services.chat_service import ChatService
from app.services.search_service import SearchService
from app.storage.chunk_store import ChunkStore

CATEGORY = "Roads and Transport"


def make_chunk(chunk_id: str, document_id: str = "dlp-1", text: str = "The toll fee is four dirhams.",
               **overrides) -> Chunk:
    values = dict(
        chunk_id=chunk_id, document_id=document_id, title=f"Law of {document_id}",
        source_url=f"https://example.test/{document_id}.pdf", primary_category=CATEGORY,
        locator="2", page_start=2, page_end=2, segment_type="article", article_number="3",
        text=text, search_text=text, legislation_type="Law", year=2006, status="In Force")
    values.update(overrides)
    return Chunk(**values)


class FakeRetriever:
    """Returns a fixed ranked list and remembers the filters it was called with."""
    def __init__(self, chunks: list[Chunk], rank_name: str):
        self.chunks, self.rank_name, self.calls = chunks, rank_name, []

    def search(self, query, filters, limit):
        self.calls.append({"query": query, "filters": filters})
        return [RetrievedChunk(chunk=chunk, ranks={self.rank_name: rank})
                for rank, chunk in enumerate(self.chunks[:limit], start=1)]


class FakeReranker:
    """Reverses the fused order so tests can see that reranking is a separate stage."""
    def rerank(self, query, candidates, top_k):
        reranked = list(reversed(candidates))[:top_k]
        for position, candidate in enumerate(reranked, start=1):
            candidate.scores["rerank"] = 1.0 / position
            candidate.ranks["final"] = position
        return reranked


class FakePlanner:
    """Returns the queued plans one by one and records every call."""
    def __init__(self, plans):
        self.plans, self.calls = list(plans), []

    async def plan(self, messages, document_titles, may_ask_clarification):
        self.calls.append({"messages": list(messages), "may_ask": may_ask_clarification})
        return self.plans.pop(0)


class FakeAnswerGenerator:
    def __init__(self, status="answered", answer="The fee is four dirhams [c1].",
                 cited_chunk_ids=("c1",), tool_chunk_ids=()):
        self.draft = AnswerDraft(status=status, answer=answer, cited_chunk_ids=list(cited_chunk_ids))
        self.tool_chunk_ids, self.calls = list(tool_chunk_ids), 0

    async def generate(self, question, evidence):
        self.calls += 1
        return GeneratedAnswer(draft=self.draft, tool_chunk_ids=self.tool_chunk_ids)


class FakeVectorStore:
    def __init__(self):
        self.chunks_by_document: dict[str, list[Chunk]] = {}

    def replace_document(self, document_id, chunks):
        self.chunks_by_document[document_id] = chunks

    def delete_document(self, document_id):
        self.chunks_by_document.pop(document_id, None)

    def count(self):
        return sum(len(chunks) for chunks in self.chunks_by_document.values())


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        manifest_path=tmp_path / "manifest.json", pdf_dir=tmp_path / "pdf",
        sqlite_path=tmp_path / "index" / "chunks.sqlite", chroma_dir=tmp_path / "index" / "chroma",
        checkpoint_path=tmp_path / "index" / "checkpoints.sqlite",
        ingestion_report_path=tmp_path / "index" / "report.json",
        chunk_size=300, chunk_overlap=40)


@pytest.fixture
def chunk_store(settings) -> ChunkStore:
    return ChunkStore(settings.sqlite_path)


@pytest.fixture
def indexed_chunks(chunk_store) -> list[Chunk]:
    """Three chunks in the store: c1 and c2 from a 2006 law, c3 from a 2023 resolution."""
    law_chunks = [make_chunk("c1"), make_chunk("c2", text="Ambulances are exempt from the toll.")]
    resolution_chunks = [make_chunk("c3", document_id="dlp-2", legislation_type="Resolution",
                                    year=2023, status="Not in Force",
                                    text="Autonomous vehicles need a licence.")]
    chunk_store.replace_document("dlp-1", "Law of dlp-1", "hash-1", law_chunks)
    chunk_store.replace_document("dlp-2", "Law of dlp-2", "hash-2", resolution_chunks)
    return law_chunks + resolution_chunks


@pytest.fixture
def build_graph(settings, chunk_store, indexed_chunks):
    """Factory: build the real chat graph around fake collaborators."""
    def factory(planner, answer_generator=None, lexical_chunks=None, semantic_chunks=None):
        lexical = FakeRetriever(indexed_chunks[:2] if lexical_chunks is None else lexical_chunks,
                                "lexical")
        semantic = FakeRetriever(indexed_chunks[1:] if semantic_chunks is None else semantic_chunks,
                                 "semantic")
        retrieval_graph = build_retrieval_graph(lexical, semantic, FakeReranker(), settings)
        return build_chat_graph(planner, retrieval_graph, answer_generator or FakeAnswerGenerator(),
                                chunk_store, settings, InMemorySaver())
    return factory


@pytest.fixture
def build_test_container(settings, chunk_store, indexed_chunks, build_graph):
    def factory(planner):
        lexical = FakeRetriever(indexed_chunks[:2], "lexical")
        semantic = FakeRetriever(indexed_chunks[1:], "semantic")
        retrieval_graph = build_retrieval_graph(lexical, semantic, FakeReranker(), settings)
        chat_graph = build_graph(planner)
        vector_store = FakeVectorStore()
        vector_store.replace_document("all", indexed_chunks)
        return Container(
            settings=settings, chunk_store=chunk_store, vector_store=vector_store,
            ingestion_pipeline=None,
            search_service=SearchService(retrieval_graph, chunk_store, settings),
            chat_service=ChatService(chat_graph, chunk_store, settings),
            chat_graph=chat_graph, retrieval_graph=retrieval_graph)
    return factory
