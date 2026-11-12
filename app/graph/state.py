"""Typed state for the two graphs.

Lists of candidates are stored as plain dicts (`RetrievedChunk.model_dump()`)
so the SQLite checkpointer can serialise them without custom types.

What persists and what resets
-----------------------------
* `messages` is the only field that accumulates across turns of a thread
  (the `add_messages` reducer appends). It is the thread's short-term memory.
* Every other field describes the CURRENT question. The `start_turn` node
  overwrites them all at the beginning of each new question, so evidence from
  an earlier question can never leak into a later answer.

Parallel writes
---------------
In hybrid mode the lexical and semantic nodes run in the same step. Each writes
its own key (`lexical_candidates` / `semantic_candidates`), so there is no
conflicting write and no reducer is needed for them.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RetrievalState(TypedDict, total=False):
    # inputs
    standalone_query: str
    filters: dict                     # SearchFilters.model_dump()
    search_mode: str                  # "lexical" | "semantic" | "hybrid"
    top_k: int
    rerank_enabled: bool
    # outputs
    lexical_candidates: list[dict]    # written only by lexical_search
    semantic_candidates: list[dict]   # written only by semantic_search
    retrieval_candidates: list[dict]  # fused + deduplicated, before reranking
    final_evidence: list[dict]        # after reranking, cut to top_k


class ChatState(RetrievalState, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # persists across turns
    user_message: str
    request_filters: dict             # filters sent explicitly by the API caller
    query_plan: dict                  # QueryPlan.model_dump()
    clarification: dict               # {"status": none|pending|resolved, "attempts": int, "question": str|None}
    answer_draft: dict                # model output before citation validation
    answer: dict                      # {"status", "answer", "citations"} returned to the API
