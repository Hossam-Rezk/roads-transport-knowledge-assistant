"""Extract clean text lines from a text-based PDF, keeping page numbers.

Every page of a portal PDF has the same noise:
  * a running header: the document title (1-3 lines) then "Page X of N"
  * on some pages a footer: a copyright line and a translation disclaimer
  * a decorative separator line under the title
`clean_page_lines` removes these so they do not pollute the indexes.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PAGE_MARKER = re.compile(r"^Page \d+ of \d+$")
COPYRIGHT_LINE = re.compile(r"^©\s*\d{4}\s+The Supreme Legislation Committee")
SEPARATOR_LINE = re.compile(r"^[ـ_\-–—]{5,}$")
HEADER_SEARCH_LINES = 6      # the "Page X of N" marker is always near the top
MIN_DOCUMENT_CHARACTERS = 200


class PdfParseError(Exception):
    """The PDF could not be turned into usable text."""


@dataclass
class Line:
    text: str
    page: int   # 1-based PDF page number


def clean_page_lines(raw_lines: list[str]) -> list[str]:
    lines = [line.strip() for line in raw_lines if line.strip()]

    # Drop the running header: everything up to and including "Page X of N".
    for index, line in enumerate(lines[:HEADER_SEARCH_LINES]):
        if PAGE_MARKER.match(line):
            lines = lines[index + 1:]
            break

    # Drop the footer: the copyright line and everything after it on the page.
    for index, line in enumerate(lines):
        if COPYRIGHT_LINE.match(line):
            lines = lines[:index]
            break

    return [line for line in lines if not SEPARATOR_LINE.match(line)]


def extract_lines(pdf_path: Path) -> list[Line]:
    """Return every cleaned text line of the PDF with its page number."""
    lines: list[Line] = []
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                for text in clean_page_lines(page.get_text().split("\n")):
                    lines.append(Line(text=text, page=page_number))
    except Exception as error:
        raise PdfParseError(f"cannot read PDF: {error}") from error

    if sum(len(line.text) for line in lines) < MIN_DOCUMENT_CHARACTERS:
        raise PdfParseError("no extractable text (scanned PDF?)")
    return lines
