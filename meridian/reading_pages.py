from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


def _ocr_top(page, dpi: int = 110) -> str:
    scale = dpi / 72.0
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        alpha=False,
    )
    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )
    top = image.crop(
        (0, 0, image.width, int(image.height * 0.30))
    )
    text = pytesseract.image_to_string(
        top,
        config="--oem 1 --psm 6",
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def discover_reading_pages(pdf_path: str | Path) -> list[int]:
    """Return every page whose header carries a reading-table marker.

    The previous detector forced a *consecutive pair* of pages, which broke
    documents where the reading table lives on a single page (e.g. the 11-page
    Performance Evaluation document whose only reading table is page 11).  We
    instead return all pages whose top OCR contains a reading-table marker,
    excluding the cover page (page 1), so callers can apply a template-aware
    strategy to however many pages the source actually uses.
    """
    pdf_path = Path(pdf_path)
    with pymupdf.open(str(pdf_path)) as doc:
        hits: list[int] = []
        for idx, page in enumerate(doc):
            text = _ocr_top(page)
            if re.search(r"\breading\b|\brecording\b", text):
                hits.append(idx + 1)
    # The cover page ("... readings included ...") also matches the word
    # "reading", so drop page 1 from the reading-table candidates.
    hits = [h for h in hits if h != 1]
    return hits


def detect_reading_pages(pdf_path: str | Path) -> list[int]:
    """Template-A compatible reading page detection (a consecutive pair).

    Kept for backward compatibility with the verified CE crop pipeline, which
    consumes exactly two reading pages.  The first two discovered reading
    pages are returned as themselves (they are already consecutive in the CE
    family); falling back to the last two pages only when none are found.
    """
    pdf_path = Path(pdf_path)
    with pymupdf.open(str(pdf_path)) as doc:
        n = len(doc)

    hits = discover_reading_pages(pdf_path)

    if len(hits) >= 2:
        # The CE family's reading pages are consecutive and in order.
        return hits[0:2]

    if hits:
        page_no = hits[0]
        if page_no < n:
            return [page_no, page_no + 1]

    if n >= 2:
        return [n - 1, n]

    raise ValueError(
        f"Could not identify a pair of Reading pages in {pdf_path}"
    )
