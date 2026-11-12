"""The tool the answer model may choose to call.

`get_document_excerpt(document_id, locator)` returns the full text of one
article of an indexed document. Arguments are validated by Pydantic, and the
lookup goes through the ChunkStore, so only the approved corpus is reachable.
"""
import re

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.storage.chunk_store import ChunkStore

MAX_EXCERPT_CHARACTERS = 4000


class ExcerptArguments(BaseModel):
    document_id: str = Field(
        pattern=r"^[A-Za-z0-9-]{1,40}$",
        description="Document id exactly as shown in the evidence, e.g. dlp-1525.")
    locator: str = Field(
        max_length=30,
        description='Article number to read, e.g. "6" or "Article (6)".')


def build_excerpt_tool(chunk_store: ChunkStore) -> BaseTool:

    @tool("get_document_excerpt", args_schema=ExcerptArguments,
          response_format="content_and_artifact")
    def get_document_excerpt(document_id: str, locator: str) -> tuple[str, list[str]]:
        """Read the full text of one article of an indexed document. Use it when an
        evidence chunk is only part of an article or refers to another article."""
        article_number = re.search(r"\d+", locator)
        if not article_number:
            raise ValueError(f"locator '{locator}' does not contain an article number")
        if document_id not in chunk_store.list_document_ids():
            raise ValueError(f"document '{document_id}' is not in the indexed corpus")

        chunks = chunk_store.get_article_chunks(document_id, article_number.group())
        if not chunks:
            raise ValueError(f"document '{document_id}' has no Article ({article_number.group()})")

        excerpt_parts, returned_chunk_ids, used_characters = [], [], 0
        for chunk in chunks:
            if used_characters + len(chunk.text) > MAX_EXCERPT_CHARACTERS and excerpt_parts:
                break
            excerpt_parts.append(f"[chunk_id: {chunk.chunk_id} | page {chunk.locator}]\n{chunk.text}")
            returned_chunk_ids.append(chunk.chunk_id)
            used_characters += len(chunk.text)
        # The artifact (chunk ids) is used later to validate citations.
        return "\n\n".join(excerpt_parts), returned_chunk_ids

    return get_document_excerpt
