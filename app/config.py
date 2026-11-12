"""Application settings.

Every tunable value lives here and can be overridden with an environment
variable of the same name (upper-cased) or a line in the `.env` file.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Models (Azure AI Foundry, OpenAI-compatible endpoint) ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr = SecretStr("")
    chat_deployment: str = "gpt-4.1-mini"
    embedding_deployment: str = "text-embedding-3-small"
    model_timeout_seconds: float = 60
    model_max_retries: int = 2

    # --- Corpus ---
    primary_category: str = "Roads and Transport"
    manifest_path: Path = Path("data/manifest.json")
    pdf_dir: Path = Path("data/raw/pdf")

    # --- Index files (all local) ---
    sqlite_path: Path = Path("data/index/chunks.sqlite")
    chroma_dir: Path = Path("data/index/chroma")
    checkpoint_path: Path = Path("data/index/checkpoints.sqlite")
    ingestion_report_path: Path = Path("data/index/ingestion_report.json")
    reranker_cache_dir: Path = Path("data/model_cache")
    log_file: Path = Path("logs/app.log")

    # --- Chunking ---
    chunk_size: int = 1200      # characters
    chunk_overlap: int = 150    # characters

    # --- Retrieval ---
    candidates_per_branch: int = 20
    rrf_k: int = 60
    default_top_k: int = 5
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    max_added_article_parts: int = 6   # cap when completing split articles for the answer

    # --- Execution bounds ---
    max_clarification_attempts: int = 2
    max_model_calls_per_answer: int = 6
    max_tool_calls_per_answer: int = 3
    request_timeout_seconds: float = 120
    planner_history_messages: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
