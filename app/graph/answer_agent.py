"""The answer agent: a LangChain `create_agent` component that receives the
selected evidence, may call `get_document_excerpt`, and returns a validated
`AnswerDraft`. The chat graph embeds it as one node.
"""
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.config import Settings
from app.core.errors import ModelFailureError
from app.core.models import RetrievedChunk
from app.graph.middleware import ExecutionLoggingMiddleware
from app.graph.prompts import ANSWER_SYSTEM_PROMPT
from app.graph.tools import build_excerpt_tool
from app.storage.chunk_store import ChunkStore


class AnswerDraft(BaseModel):
    """What the model must return. Citations are validated afterwards."""
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(description="The answer, with a [chunk_id] after every legal statement.")
    cited_chunk_ids: list[str] = Field(
        default_factory=list, description="Every chunk id cited in the answer.")


@dataclass
class GeneratedAnswer:
    draft: AnswerDraft
    tool_chunk_ids: list[str]     # chunks the model read through the tool


def format_evidence(evidence: list[RetrievedChunk]) -> str:
    """Wrap each chunk in a tagged block so the model can tell quoted document
    text (data) apart from its instructions."""
    blocks = []
    for item in evidence:
        chunk = item.chunk
        part_note = (f' part="{chunk.part_index + 1} of {chunk.part_count}"'
                     if chunk.part_count > 1 else "")
        blocks.append(
            f'<evidence chunk_id="{chunk.chunk_id}" document_id="{chunk.document_id}" '
            f'title="{chunk.title}" section="{chunk.segment_type} {chunk.article_number or ""}" '
            f'page="{chunk.locator}" status="{chunk.status}"{part_note}>\n'
            f"{chunk.text}\n</evidence>")
    return "\n\n".join(blocks)


class AnswerGenerator:
    def __init__(self, chat_model: BaseChatModel, chunk_store: ChunkStore, settings: Settings):
        self._agent = create_agent(
            model=chat_model,
            tools=[build_excerpt_tool(chunk_store)],
            system_prompt=ANSWER_SYSTEM_PROMPT,
            response_format=AnswerDraft,
            # Middleware order: the first entry is the outermost wrapper.
            middleware=[
                ModelCallLimitMiddleware(run_limit=settings.max_model_calls_per_answer,
                                         exit_behavior="end"),
                ToolCallLimitMiddleware(run_limit=settings.max_tool_calls_per_answer),
                ModelRetryMiddleware(max_retries=settings.model_max_retries, on_failure="error"),
                ExecutionLoggingMiddleware(),     # innermost: logs every single attempt
            ],
        )

    async def generate(self, question: str, evidence: list[RetrievedChunk]) -> GeneratedAnswer:
        prompt = f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}"
        result = await self._agent.ainvoke({"messages": [HumanMessage(prompt)]})

        draft = result.get("structured_response")
        if draft is None:
            raise ModelFailureError("the answer agent stopped before producing an answer "
                                    "(model-call limit reached)")
        tool_chunk_ids = [chunk_id
                          for message in result["messages"]
                          if isinstance(message, ToolMessage) and message.artifact
                          for chunk_id in message.artifact]
        return GeneratedAnswer(draft=draft, tool_chunk_ids=tool_chunk_ids)
