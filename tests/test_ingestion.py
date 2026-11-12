"""Parsing, chunk metadata, repeat ingestion and the corpus boundary."""
import json

import pymupdf

from app.ingestion.chunker import chunk_document, split_into_segments
from app.ingestion.manifest import ManifestDocument
from app.ingestion.pdf_parser import Line, clean_page_lines
from app.ingestion.pipeline import IngestionPipeline
from tests.conftest import CATEGORY, FakeVectorStore

DOCUMENT = ManifestDocument(
    document_id="dlp-9", title="Law No. (9) of 2023 Test Law", primary_category=CATEGORY,
    content_url_pdf="https://example.test/dlp-9.pdf", legislation_type="Law",
    number=9, year=2023, issuing_authority="H.H. The Ruler", status="In Force")

LINES = [
    Line("We, the Ruler, do hereby issue this Law.", 1),
    Line("Definitions", 1),
    Line("Article (1)", 1),
    Line("Road: any public way.", 1),
    Line("Article (2)", 2),
    Line("No vehicle may use the road unless it is licensed by the", 2),
    Line("authority in accordance with this Law.", 3),
    Line("Schedule (1)", 3),
    Line("Fee 4.00", 3),
]


def test_clean_page_lines_removes_header_footer_and_separator():
    raw_lines = ["", "Law No. (9) of 2023 Test Law ", "Page 2 of 13 ", "ـــــــــــــــ",
                 "Article (1) ", "Body text. ",
                 "© 2023 The Supreme Legislation Committee in the Emirate of Dubai",
                 "1Every effort has been made to produce an accurate English version."]
    assert clean_page_lines(raw_lines) == ["Article (1)", "Body text."]


def test_segments_follow_article_and_schedule_boundaries():
    segments = split_into_segments(LINES)
    assert [(s.segment_type, s.number, s.title) for s in segments] == [
        ("preamble", None, None), ("article", "1", "Definitions"),
        ("article", "2", None), ("schedule", "1", None)]


def test_chunk_metadata_and_locators():
    chunks = chunk_document(DOCUMENT, LINES, chunk_size=300, chunk_overlap=40)
    article_two = next(chunk for chunk in chunks if chunk.article_number == "2")

    assert article_two.locator_type == "pdf_page"
    assert article_two.locator == "2-3"                       # the article spans two pages
    assert article_two.text.startswith("No vehicle may use the road unless it is licensed by the authority")
    assert article_two.search_text.startswith("Law No. (9) of 2023 Test Law\nArticle (2)")
    assert (article_two.document_id, article_two.year, article_two.legislation_type,
            article_two.status, article_two.primary_category) == (
            "dlp-9", 2023, "Law", "In Force", CATEGORY)
    assert article_two.source_url == "https://example.test/dlp-9.pdf"


def test_missing_metadata_is_stored_as_null():
    document = ManifestDocument(document_id="dlp-8", title="Untyped", primary_category=CATEGORY)
    chunk = chunk_document(document, LINES, 300, 40)[0]
    assert chunk.legislation_type is None and chunk.year is None and chunk.status is None


def test_chunk_ids_are_stable_and_unique():
    first = chunk_document(DOCUMENT, LINES, 300, 40)
    second = chunk_document(DOCUMENT, LINES, 300, 40)
    ids = [chunk.chunk_id for chunk in first]
    assert ids == [chunk.chunk_id for chunk in second]
    assert len(ids) == len(set(ids))


def test_long_article_is_split_with_overlap():
    long_lines = [Line("Article (1)", 1)] + [
        Line(f"Clause {number} requires the operator to keep full records of every trip.", 1)
        for number in range(1, 15)]
    chunks = chunk_document(DOCUMENT, long_lines, chunk_size=300, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert all(chunk.part_count == len(chunks) for chunk in chunks)
    assert chunks[0].text.split("\n")[-1] in chunks[1].text      # overlap repeats the last clause


# ------------------------------------------------------------------ pipeline

def write_pdf(path, article_text: str):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Test Law\nPage 1 of 1\nDefinitions\nArticle (1)\n" + article_text * 6)
    pdf.save(path)


def write_manifest(settings, entries):
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps({"documents": entries}), encoding="utf-8")


def manifest_entry(document_id, category=CATEGORY):
    return {"document_id": document_id, "title": f"Law {document_id}", "primary_category": category,
            "type": "Law", "year": 2020, "status": "In Force"}


def test_reingesting_unchanged_files_creates_no_duplicates(settings, chunk_store):
    write_manifest(settings, [manifest_entry("dlp-1"), manifest_entry("dlp-2")])
    write_pdf(settings.pdf_dir / "dlp-1.pdf", "Vehicles must be licensed by the authority.\n")
    write_pdf(settings.pdf_dir / "dlp-2.pdf", "Drivers must hold a valid permit.\n")
    vector_store = FakeVectorStore()
    pipeline = IngestionPipeline(settings, chunk_store, vector_store)

    first = pipeline.run()
    chunk_count = chunk_store.count_chunks()
    second = pipeline.run()

    assert (first.indexed, first.skipped, first.failed) == (2, 0, 0)
    assert (second.indexed, second.skipped) == (0, 2)
    assert chunk_store.count_chunks() == chunk_count == vector_store.count()
    assert chunk_store.count_documents() == 2


def test_changed_file_is_reindexed_without_duplicates(settings, chunk_store):
    write_manifest(settings, [manifest_entry("dlp-1")])
    write_pdf(settings.pdf_dir / "dlp-1.pdf", "Vehicles must be licensed.\n")
    pipeline = IngestionPipeline(settings, chunk_store, FakeVectorStore())
    pipeline.run()

    write_pdf(settings.pdf_dir / "dlp-1.pdf", "Vehicles must be inspected every year.\n")
    report = pipeline.run()

    assert report.indexed == 1 and chunk_store.count_documents() == 1
    assert "inspected" in chunk_store.get_article_chunks("dlp-1", "1")[0].text


def test_out_of_category_and_unlisted_files_are_excluded_and_failures_reported(settings, chunk_store):
    write_manifest(settings, [manifest_entry("dlp-1"),
                              manifest_entry("dlp-50", category="Traffic"),
                              manifest_entry("dlp-404")])
    for name in ("dlp-1", "dlp-50", "not-in-manifest"):
        write_pdf(settings.pdf_dir / f"{name}.pdf", "Some provision text for the test.\n")

    report = IngestionPipeline(settings, chunk_store, FakeVectorStore()).run()

    assert chunk_store.list_document_ids() == ["dlp-1"]
    assert {issue.document_id for issue in report.exclusions} == {"dlp-50", "not-in-manifest"}
    assert [issue.document_id for issue in report.errors] == ["dlp-404"]      # missing file
    assert (report.indexed, report.excluded, report.failed) == (1, 2, 1)
