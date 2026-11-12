"""Wires the layers together. This is the only place that knows how every
object is constructed; everything else receives its collaborators as arguments.

    API routes -> services -> graphs -> retrieval -> storage
                                     -> ingestion -> storage
"""
from dataclasses import dataclass

from app.config import Settings
from app.graph.answer_agent import AnswerGenerator
from app.graph.chat_graph import build_chat_graph
from app.graph.query_planner import QueryPlanner
from app.graph.retrieval_graph import build_retrieval_graph
from app.ingestion.pipeline import IngestionPipeline
from app.llm import create_chat_model, create_embeddings
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retrievers import LexicalRetriever, SemanticRetriever
from app.services.chat_service import ChatService
from app.services.search_service import SearchService
from app.storage.chunk_store import ChunkStore
from app.storage.vector_store import VectorStore


@dataclass
class Container:
    settings: Settings
    chunk_store: ChunkStore
    vector_store: VectorStore
    ingestion_pipeline: IngestionPipeline
    search_service: SearchService
    chat_service: ChatService
    chat_graph: object
    retrieval_graph: object


def build_container(settings: Settings, checkpointer) -> Container:
    # storage
    chunk_store = ChunkStore(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, create_embeddings(settings))

    # ingestion
    ingestion_pipeline = IngestionPipeline(settings, chunk_store, vector_store)

    # retrieval
    lexical_retriever = LexicalRetriever(chunk_store, settings.primary_category)
    semantic_retriever = SemanticRetriever(vector_store, chunk_store, settings.primary_category)
    reranker = CrossEncoderReranker(settings.reranker_model, settings.reranker_cache_dir)
    retrieval_graph = build_retrieval_graph(lexical_retriever, semantic_retriever, reranker, settings)

    # chat graph
    chat_model = create_chat_model(settings)
    planner = QueryPlanner(chat_model, settings.planner_history_messages)
    answer_generator = AnswerGenerator(chat_model, chunk_store, settings)
    chat_graph = build_chat_graph(planner, retrieval_graph, answer_generator,
                                  chunk_store, settings, checkpointer)

    return Container(
        settings=settings,
        chunk_store=chunk_store,
        vector_store=vector_store,
        ingestion_pipeline=ingestion_pipeline,
        search_service=SearchService(retrieval_graph, chunk_store, settings),
        chat_service=ChatService(chat_graph, chunk_store, settings),
        chat_graph=chat_graph,
        retrieval_graph=retrieval_graph,
    )
