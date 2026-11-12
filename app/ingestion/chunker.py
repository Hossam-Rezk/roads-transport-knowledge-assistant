"""Article-aware chunking.

Step 1  split the document into segments at its own boundaries:
        preamble (before Article 1), one segment per article, one per schedule.
Step 2  a segment that fits in `chunk_size` becomes one chunk; a longer one is
        split on line/clause boundaries with `chunk_overlap` characters repeated.
Step 3  each chunk gets a stable id, its page locator, and a `search_text`
        that prefixes the document title and article heading for context.
"""
import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.models import Chunk
from app.ingestion.manifest import ManifestDocument
from app.ingestion.pdf_parser import Line

ARTICLE_HEADING = re.compile(r"^Article\s*\((\d+)\)$", re.IGNORECASE)
SCHEDULE_HEADING = re.compile(r"^(?:Schedule|Table|Annex|Appendix)\s*\((\d+)\)$", re.IGNORECASE)
SENTENCE_END = (".", ";", ":")
MAX_TITLE_LENGTH = 120


@dataclass
class Segment:
    segment_type: str                    # "preamble" | "article" | "schedule"
    number: str | None = None            # article or schedule number
    title: str | None = None             # the article's title line, if it has one
    lines: list[Line] = field(default_factory=list)

    @property
    def heading(self) -> str:
        if self.segment_type == "preamble":
            return "Preamble"
        label = "Article" if self.segment_type == "article" else "Schedule"
        heading = f"{label} ({self.number})"
        return f"{heading}: {self.title}" if self.title else heading


def looks_like_title(text: str) -> bool:
    """Article titles are short lines such as "Definitions" with no end punctuation."""
    return (len(text) <= MAX_TITLE_LENGTH and text[0].isupper()
            and not text.endswith((*SENTENCE_END, ",")))


def split_into_segments(lines: list[Line]) -> list[Segment]:
    segments = [Segment(segment_type="preamble")]
    for line in lines:
        article = ARTICLE_HEADING.match(line.text)
        schedule = SCHEDULE_HEADING.match(line.text)
        if article:
            # In this corpus an article's title is the line just above "Article (n)".
            previous_lines = segments[-1].lines
            title = None
            if previous_lines and looks_like_title(previous_lines[-1].text):
                title = previous_lines.pop().text
            segments.append(Segment("article", number=article.group(1), title=title))
        elif schedule:
            segments.append(Segment("schedule", number=schedule.group(1)))
        else:
            segments[-1].lines.append(line)
    return [segment for segment in segments if segment.lines]


def join_wrapped_lines(lines: list[Line]) -> list[Line]:
    """PDF text is hard-wrapped. Join lines into paragraphs, starting a new
    paragraph after a line that ends a sentence or clause."""
    paragraphs: list[Line] = []
    start_new = True
    for line in lines:
        if start_new:
            paragraphs.append(Line(text=line.text, page=line.page))
        else:
            paragraphs[-1].text += " " + line.text
        start_new = line.text.endswith(SENTENCE_END)
    return paragraphs


def chunk_document(document: ManifestDocument, lines: list[Line],
                   chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n", "; ", ". ", " ", ""], add_start_index=True,
    )
    chunks: list[Chunk] = []
    for segment_index, segment in enumerate(split_into_segments(lines)):
        paragraphs = join_wrapped_lines(segment.lines)
        segment_text = "\n".join(paragraph.text for paragraph in paragraphs)
        last_page = segment.lines[-1].page

        # Remember at which character offset each paragraph (and its page) starts.
        paragraph_starts: list[tuple[int, int]] = []
        offset = 0
        for paragraph in paragraphs:
            paragraph_starts.append((offset, paragraph.page))
            offset += len(paragraph.text) + 1

        def page_at(character_offset: int) -> int:
            page = paragraph_starts[0][1]
            for start, paragraph_page in paragraph_starts:
                if start > character_offset:
                    break
                page = paragraph_page
            return page

        parts = splitter.create_documents([segment_text])
        for part_index, part in enumerate(parts):
            start = part.metadata["start_index"]
            page_start = page_at(start)
            is_last_part = part_index == len(parts) - 1
            page_end = last_page if is_last_part else page_at(start + len(part.page_content))
            locator = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
            chunks.append(Chunk(
                chunk_id=f"{document.document_id}-{segment_index:03d}-{part_index}",
                document_id=document.document_id,
                title=document.title,
                source_url=document.content_url_pdf or document.source_url,
                primary_category=document.primary_category,
                locator=locator,
                page_start=page_start,
                page_end=page_end,
                segment_type=segment.segment_type,
                article_number=segment.number if segment.segment_type == "article" else None,
                article_title=segment.title,
                part_index=part_index,
                part_count=len(parts),
                text=part.page_content,
                search_text=f"{document.title}\n{segment.heading}\n{part.page_content}",
                legislation_type=document.legislation_type,
                number=document.number,
                year=document.year,
                authority=document.issuing_authority,
                status=document.status,
            ))
    return chunks
