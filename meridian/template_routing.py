"""Template routing for the ME performance-report parser.

The parser supports two document templates:

* **Template A (CE)** - the monthly ``ME_PERFORMANCE_REPORT`` family, whose
  Reading section is a *calibrated two-page* layout consumed by the crop
  pipeline through a fixed 14-region geometry.
* **Template B** - other ``Performance Report / Evaluation`` documents whose
  Reading section is a *dense whole-page* layout (one or more pages) consumed
  by ``template_b.process_tb``.

The discriminator is document structure only; it never inspects the filename
and never hardcodes a page number or a page count for a given document.

Signals
-------
* Every page whose top OCR carries a Reading-table marker (``reading`` /
  ``recording``) is a candidate Reading page (``discover_reading_pages``).
* The **CE family is a genuine two-page calibrated Reading layout**: it has
  exactly two Reading pages and the *second* Reading page is the right-hand
  half of that layout, whose top section is the fuel-oil / consumption /
  LCV / density / price block.  (Template B places the fuel/consumption
  content differently - a single dense page, or a second page that is a
  cylinder-condition / bearing page rather than the CE fuel half.)
* Any Reading set that is not that calibrated two-page CE layout is routed to
  Template B.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from .reading_pages import (
    detect_reading_pages,
    discover_reading_pages,
)

TEMPLATE_A = "A"
TEMPLATE_B = "B"

# Markers that identify the CE two-page Reading layout's right-hand half (the
# fuel-oil / consumption page).  These are the report's own section labels.
_CE_FUEL_MARKERS = (
    "fuel oil",
    "absolute consumption",
    "specific foc",
    "viscosit",
    "density",
    "lcv",
    "price",
    "sulfur",
    "lubricat",
    "water cont",
)


def _ocr_top_band(pdf_path: str | Path, page_no: int, top_frac: float = 0.35,
                  dpi: int = 130) -> str:
    """Return the lower-cased top-band OCR text of one 1-based page."""
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_no - 1]
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    band = image.crop((0, 0, image.width, int(image.height * top_frac)))
    text = pytesseract.image_to_string(band, config="--oem 1 --psm 6")
    return re.sub(r"\s+", " ", text).lower()


def _second_reading_page_is_ce_fuel_half(pdf_path, second_page_no) -> bool:
    """True when the second Reading page's top is the CE fuel/consumption half."""
    band = _ocr_top_band(pdf_path, second_page_no)
    return any(marker in band for marker in _CE_FUEL_MARKERS)


def detect_template(pdf_path: str | Path) -> str:
    """Return TEMPLATE_A (CE crop pipeline) or TEMPLATE_B (whole-page).

    The classification is structural: a document is Template A only when its
    Reading section is the calibrated two-page CE layout (exactly two Reading
    pages whose second page is the fuel/consumption half).  Every other
    arrangement - a single dense Reading page, or a multi-page Reading section
    that is not the CE two-page spread - is Template B.
    """
    reading = discover_reading_pages(pdf_path)

    if len(reading) == 2 and _second_reading_page_is_ce_fuel_half(
        pdf_path, reading[1]
    ):
        return TEMPLATE_A

    return TEMPLATE_B


def reading_pages_for(pdf_path: str | Path, template: str) -> list[int]:
    """Return the Reading page numbers appropriate to a template.

    Template A keeps the backward-compatible consecutive-pair behaviour the
    verified CE crop pipeline depends on.  Template B returns every page whose
    OCR marks it as a Reading page (the document's true Reading section, which
    may be one page or several).
    """
    if template == TEMPLATE_A:
        return detect_reading_pages(pdf_path)
    return discover_reading_pages(pdf_path)
