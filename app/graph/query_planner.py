"""Query planning: turn the latest user message (plus thread context) into a
validated `QueryPlan` using the chat model's structured output."""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from app.core.models import SearchFilters
from app.graph.prompts import CLARIFICATION_LIMIT_NOTE, PLANNER_SYSTEM_PROMPT


class QueryPlan(BaseModel):
    intent: Literal["legal_question", "out_of_scope"] = Field(
        description="Whether the latest message is about Dubai roads and transport legislation.")
    standalone_query: str = Field(
        description="The latest message rewritten as a self-contained search query.")
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Metadata filters the user explicitly asked for; null fields otherwise.")
    needs_clarification: bool = Field(
        default=False, description="True if the question is too ambiguous to search.")
    clarification_question: str | None = Field(
        default=None, description="One short question to ask the user when clarification is needed.")

    @field_validator("filters", mode="before")
    @classmethod
    def treat_missing_filters_as_empty(cls, value):
        # The model sometimes sends `"filters": null` instead of an empty object.
        return {} if value is None else value


class QueryPlanner:
    def __init__(self, chat_model: BaseChatModel, history_messages: int):
        self._structured_model = chat_model.with_structured_output(
            QueryPlan, method="function_calling")
        self._history_messages = history_messages

    async def plan(self, messages: list[AnyMessage], document_titles: list[str],
                   may_ask_clarification: bool) -> QueryPlan:
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            clarification_limit_note="" if may_ask_clarification else CLARIFICATION_LIMIT_NOTE,
            document_titles="\n".join(f"- {title}" for title in document_titles),
        )
        recent_messages = messages[-self._history_messages:]
        return await self._structured_model.ainvoke(
            [SystemMessage(system_prompt), *recent_messages])
