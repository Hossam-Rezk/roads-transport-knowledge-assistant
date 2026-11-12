"""SQLite store for documents and chunks, plus the FTS5 (BM25) lexical index.

This is the source of truth for chunk text and metadata. The vector store only
keeps embeddings and the few metadata fields needed for filtering.
A new connection is opened per operation, which keeps the class thread-safe.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.models import Chunk

CHUNK_COLUMNS = list(Chunk.model_fields)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    file_sha256  TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks ({", ".join(CHUNK_COLUMNS)}, PRIMARY KEY (chunk_id));
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(chunk_id UNINDEXED, search_text, tokenize = 'porter unicode61');
"""


class ChunkStore:
    def __init__(self, sqlite_path: Path):
        self._path = sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    # ---------- writes ----------

    def replace_document(self, document_id: str, title: str, file_sha256: str,
                         chunks: list[Chunk]) -> None:
        """Delete any previous version of the document and store the new chunks.
        Running this twice with the same input leaves exactly one copy."""
        placeholders = ", ".join("?" for _ in CHUNK_COLUMNS)
        with self._connect() as connection:
            self._delete_document(connection, document_id)
            connection.executemany(
                f"INSERT INTO chunks VALUES ({placeholders})",
                [[getattr(chunk, column) for column in CHUNK_COLUMNS] for chunk in chunks],
            )
            connection.executemany(
                "INSERT INTO chunks_fts (chunk_id, search_text) VALUES (?, ?)",
                [(chunk.chunk_id, chunk.search_text) for chunk in chunks],
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (document_id, title, file_sha256, len(chunks),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            self._delete_document(connection, document_id)

    @staticmethod
    def _delete_document(connection: sqlite3.Connection, document_id: str) -> None:
        connection.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE document_id = ?)", (document_id,))
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    # ---------- reads ----------

    def get_document_hash(self, document_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_sha256 FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return row["file_sha256"] if row else None

    def list_document_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT document_id FROM documents").fetchall()
        return [row["document_id"] for row in rows]

    def list_document_titles(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document_id, title FROM documents ORDER BY document_id").fetchall()
        return [f"{row['document_id']}: {row['title']}" for row in rows]

    def count_documents(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def count_chunks(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Return chunks in the same order as `chunk_ids`; unknown ids are skipped."""
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids).fetchall()
        chunks_by_id = {row["chunk_id"]: Chunk(**dict(row)) for row in rows}
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]

    def get_article_chunks(self, document_id: str, article_number: str) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND article_number = ? "
                "ORDER BY chunk_id", (document_id, article_number)).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def get_all_parts(self, chunk: Chunk) -> list[Chunk]:
        """All parts of the article (segment) that `chunk` belongs to, in order.
        Chunk ids look like "<document>-<segment>-<part>", so parts share a prefix."""
        segment_prefix = chunk.chunk_id.rsplit("-", 1)[0]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND chunk_id LIKE ? "
                "ORDER BY part_index", (chunk.document_id, f"{segment_prefix}-%")).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def search_bm25(self, fts_query: str, where_sql: str, where_params: list,
                    limit: int) -> list[tuple[Chunk, float]]:
        """Full-text search. `where_sql` is a metadata filter applied inside the
        query, i.e. before ranking and before the LIMIT. Returns (chunk, score)
        with higher score = better match."""
        sql = f"""
            SELECT chunks.*, bm25(chunks_fts) AS raw_bm25
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ? AND {where_sql}
            ORDER BY raw_bm25
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, [fts_query, *where_params, limit]).fetchall()
        results = []
        for row in rows:
            values = dict(row)
            raw_bm25 = values.pop("raw_bm25")     # SQLite returns "lower is better"
            results.append((Chunk(**values), -raw_bm25))
        return results
