"""The corpus manifest: the approved list of documents and their metadata.

The manifest is produced by `scripts/fetch_corpus.py` from the portal's
filtered search results. Ingestion trusts metadata from here only; it never
guesses type, year or status from the document text.
"""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ManifestDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)   # accept "type" or "legislation_type"

    document_id: str
    title: str
    primary_category: str
    language: str = "en"
    source_url: str | None = None
    content_url_pdf: str | None = None
    content_url_html: str | None = None
    retrieval_date: str | None = None
    legislation_type: str | None = Field(default=None, alias="type")
    number: int | None = None
    year: int | None = None
    issuing_authority: str | None = None
    status: str | None = None


def load_manifest(manifest_path: Path) -> list[ManifestDocument]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [ManifestDocument.model_validate(entry) for entry in manifest["documents"]]
