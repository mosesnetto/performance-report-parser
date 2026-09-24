"""Generic geometry fallback recovery pass (Phase 6).

Runs as a SECOND pass over a :class:`ReconstructionResult`, AFTER the engine's
own row/column assignment. It fills ONLY blank value cells that the primary
geometry assignment left empty, using unclaimed value-bearing OCR tokens whose
geometry falls inside the cell's column window on the same physical row.

Hard guarantees:
  * Never overwrites a populated cell (only ``cell.is_blank`` cells are touched).
  * Fills a cell only when there is EXACTLY ONE strong value candidate; with
    zero or multiple candidates the cell stays blank (no guessing).
  * Preserves token geometry/confidence/raw_text; provenance is marked via the
    cell's ``recovered`` flag so the adapter tags it ``RECOVERED_BY_GEOMETRY_FALLBACK``.
  * Purely geometric / field-agnostic: no field-ID, filename, month or
    coordinate-specific branches.
"""

from __future__ import annotations

import re
from typing import List

from .table_reconstruction import (
    ReconstructionResult,
    _looks_like_value_token,
)

# Column-window tolerances (fractions of the cell column width / row height).
_COL_TOL = 0.35
_ROW_TOL = 0.6

# Value-like tokens that are not plain integers/floats but still carry a datum
# (dates, times, mixed like '1.0/0.6'). Kept generic -- no field knowledge.
_DATE_RE = re.compile(r"^\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_SLASH_RE = re.compile(r"^\s*\d+([.,]\d+)?\s*[/÷]\s*\d+([.,]\d+)?\s*$")


def _is_value_candidate(tok) -> bool:
    """A token is a strong value candidate if it is number-like or date/time-
    like. Pure letters / OCR artefacts are excluded."""
    if _looks_like_value_token(tok):
        return True
    text = (getattr(tok, "text", "") or "").strip()
    return bool(
        _DATE_RE.match(text)
        or _TIME_RE.match(text)
        or _SLASH_RE.match(text)
    )


def recover_dense_empty_cells(result: ReconstructionResult) -> int:
    """Fill blank value cells from unclaimed value-bearing tokens.

    Returns the number of cells filled. Mutates ``result`` in place.
    """
    if result is None:
        return 0
    tokens = list(result.tokens or [])
    filled = 0

    # Tokens already attributed to a populated value cell ANYWHERE are claimed
    # and must never be double-assigned by the fallback.  A token that has been
    # absorbed into one row's populated cell (e.g. the -10 deviation in its own
    # CYL1 cell) must not be recycled into a blank cell of a neighbouring row:
    # doing so leaks a value into the wrong field.  Building the set globally
    # (not per-row) prevents that cross-row contamination while still allowing
    # genuinely unclaimed stray tokens (present in `tokens` but in no cell) to
    # be recovered into a single blank cell.
    claimed: set = set()
    for pr in result.physical_rows:
        row = pr.model
        if row is None:
            continue
        for c in row.value_cells:
            if not c.is_blank:
                claimed.update(id(t) for t in c.tokens)

    for pr in result.physical_rows:
        row = pr.model
        if row is None:
            continue
        ylo, yhi = row.bbox[1], row.bbox[3]
        yh = max(yhi - ylo, 1.0)
        ywin_lo = ylo - _ROW_TOL * yh
        ywin_hi = yhi + _ROW_TOL * yh

        for cell in row.value_cells:
            if not cell.is_blank:
                continue
            col = cell.column
            if col is None:
                continue
            xmin, xmax = col.x_min, col.x_max
            w = max(xmax - xmin, 1.0)
            win_lo = xmin - _COL_TOL * w
            win_hi = xmax + _COL_TOL * w

            cands: List = []
            for t in tokens:
                if id(t) in claimed:
                    continue
                if t.right < win_lo or t.left > win_hi:
                    continue
                if t.bottom < ywin_lo or t.top > ywin_hi:
                    continue
                if not _is_value_candidate(t):
                    continue
                cands.append(t)

            if len(cands) == 1:
                t = cands[0]
                cell.tokens = [t]
                cell.is_blank = False
                cell.recovered = True
                claimed.add(id(t))
                filled += 1

    return filled
