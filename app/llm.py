"""Factories for the chat model and the embedding model.

Both use the Azure AI Foundry OpenAI-compatible (`/openai/v1`) endpoint, so the
standard LangChain OpenAI classes work with `base_url` + the deployment name.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
    )


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        # Send plain text; the endpoint does the tokenising.
        check_embedding_ctx_length=False,
    )
