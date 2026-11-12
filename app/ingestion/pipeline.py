"""Ingestion pipeline: manifest -> parse -> chunk -> SQLite + Chroma.

Outcome per document:
  indexed   parsed and written to both stores
  skipped   file hash unchanged since the last run (nothing rewritten)
  excluded  not allowed in the corpus (wrong category, or a PDF not in the manifest)
  failed    missing or unreadable file; the error is reported, never hidden
"""
import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.logging_setup import log_event
from app.ingestion.chunker import chunk_document
from app.ingestion.manifest import ManifestDocument, load_manifest
from app.ingestion.pdf_parser import PdfParseError, extract_lines
from app.storage.chunk_store import ChunkStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionIssue(BaseModel):
    document_id: str
    reason: str


class IngestionReport(BaseModel):
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    excluded: int = 0
    total_documents_in_index: int = 0
    total_chunks_in_index: int = 0
    errors: list[IngestionIssue] = Field(default_factory=list)
    exclusions: list[IngestionIssue] = Field(default_factory=list)


class IngestionPipeline:
    def __init__(self, settings: Settings, chunk_store: ChunkStore, vector_store: VectorStore):
        self._settings = settings
        self._chunk_store = chunk_store
        self._vector_store = vector_store

    def run(self, force: bool = False) -> IngestionReport:
        """Ingest the approved corpus. `force=True` re-indexes unchanged files too."""
        report = IngestionReport()
        documents = load_manifest(self._settings.manifest_path)

        for document in documents:
            self._ingest_one(document, report, force)

        self._report_files_missing_from_manifest(documents, report)
        self._remove_documents_no_longer_approved(documents, report)

        report.total_documents_in_index = self._chunk_store.count_documents()
        report.total_chunks_in_index = self._chunk_store.count_chunks()
        self._settings.ingestion_report_path.write_text(
            report.model_dump_json(indent=2), encoding="utf-8")
        log_event(logger, "ingestion_finished", indexed=report.indexed, skipped=report.skipped,
                  failed=report.failed, excluded=report.excluded)
        return report

    def _ingest_one(self, document: ManifestDocument, report: IngestionReport, force: bool) -> None:
        if document.primary_category != self._settings.primary_category:
            self._exclude(report, document.document_id,
                          f"primary category is '{document.primary_category}', "
                          f"not '{self._settings.primary_category}'")
            return

        pdf_path = self._settings.pdf_dir / f"{document.document_id}.pdf"
        if not pdf_path.exists():
            self._fail(report, document.document_id, f"file not found: {pdf_path.name}")
            return

        file_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if not force and self._chunk_store.get_document_hash(document.document_id) == file_sha256:
            report.skipped += 1
            return

        try:
            lines = extract_lines(pdf_path)
            chunks = chunk_document(document, lines,
                                    self._settings.chunk_size, self._settings.chunk_overlap)
            # Vectors first: if embedding fails, SQLite keeps no record of the
            # document, so the next run retries it instead of skipping it.
            self._vector_store.replace_document(document.document_id, chunks)
            self._chunk_store.replace_document(
                document.document_id, document.title, file_sha256, chunks)
        except PdfParseError as error:
            self._fail(report, document.document_id, str(error))
            return
        except Exception as error:  # e.g. embedding API failure
            self._fail(report, document.document_id, f"indexing failed: {error}")
            return

        report.indexed += 1
        log_event(logger, "document_indexed", document_id=document.document_id,
                  chunk_count=len(chunks))

    def _report_files_missing_from_manifest(self, documents: list[ManifestDocument],
                                            report: IngestionReport) -> None:
        known_ids = {document.document_id for document in documents}
        for pdf_path in sorted(self._settings.pdf_dir.glob("*.pdf")):
            if pdf_path.stem not in known_ids:
                self._exclude(report, pdf_path.stem,
                              "file is not listed in the manifest, so its category is unverified")

    def _remove_documents_no_longer_approved(self, documents: list[ManifestDocument],
                                             report: IngestionReport) -> None:
        approved_ids = {document.document_id for document in documents
                        if document.primary_category == self._settings.primary_category}
        for document_id in self._chunk_store.list_document_ids():
            if document_id not in approved_ids:
                self._vector_store.delete_document(document_id)
                self._chunk_store.delete_document(document_id)
                log_event(logger, "document_removed", document_id=document_id)

    @staticmethod
    def _fail(report: IngestionReport, document_id: str, reason: str) -> None:
        report.failed += 1
        report.errors.append(IngestionIssue(document_id=document_id, reason=reason))
        log_event(logger, "document_failed", document_id=document_id, reason=reason)

    @staticmethod
    def _exclude(report: IngestionReport, document_id: str, reason: str) -> None:
        report.excluded += 1
        report.exclusions.append(IngestionIssue(document_id=document_id, reason=reason))
        log_event(logger, "document_excluded", document_id=document_id, reason=reason)


def load_last_report(report_path: Path) -> dict | None:
    return json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
