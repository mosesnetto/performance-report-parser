"""
V4 Generalized Table Reconstruction Engine
==========================================

Standalone geometry engine that takes raw OCR tokens and reconstructs
the physical table structure without any page-specific rules, field IDs,
month names, filenames, or hardcoded header vocabularies.

Hardened data model
-------------------
- explicit row model: label / unit / qualifier / value regions
- cell model: column/header path, value bbox, status indicator metadata
- blank cell preservation (no left-shift of later cells)
- generic compound header reconstruction (multi-token, e.g. "CYL" + "10")

Input:  list of OCR token dicts (text, left, top, width, height, conf, page, region)
Output: ReconstructionResult with physical rows, header bands, hierarchical
        headers, leaf columns, cell geometry, blank cells, status indicators,
        and debug reconstruction.

Headers are discovered dynamically from OCR text content and geometry.
No header names are hard-coded.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

# ───────────────────────────────────────────────────────────
# Grid-detection thresholds (generic, geometry-only)
# ───────────────────────────────────────────────────────────
_GRID_MAX_COL_WIDTH = 130.0
_GRID_MAX_COL_WIDTH_SPARSE = 420.0
_GRID_MAX_PITCH_RATIO = 2.5
# A non-numeric gap at least this many times the band's median word gap is a
# column break separating side-by-side label-only fields; the floor guards
# against a tiny band where all gaps are small.
_FIELD_GAP_RATIO = 3.0
_FIELD_GAP_MIN_PX = 40.0
# Max vertical gap (px) between a value run and the genuine label band below it
# for forward-binding (label printed slightly below its values on the same
# visual row due to OCR baseline variation).
_CONTINUATION_GAP_MAX = 40.0


# ───────────────────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────────────────


class RegionType(str, Enum):
    """Classification of a horizontal strip / token within a data row."""

    LABEL = "label"                      # row's text identifier
    UNIT = "unit"                        # unit of measure (barG, %, kW, ...)
    QUALIFIER = "qualifier"              # ALL / MOP SETTING: / (e) annotations
    VALUE = "value"                      # a numeric value in a cell
    OTHER = "other"


class HeaderKind(str, Enum):
    """Kind of a header node in the reconstructed header tree."""

    TITLE = "title"                      # document/meta title band (not a column)
    GROUP = "group"                      # parent/group header spanning children
    LEAF = "leaf"                        # a value column header (leaf)
    LABEL_ZONE = "label_zone"            # header above the row-label area


class StatusIndicator:
    """
    A status indicator found inside a cell (stored as metadata).

    Detected generically as small non-value tokens that sit inside or
    immediately adjacent to a cell (e.g. OCR 'status dot' artifacts).
    """

    __slots__ = ("kind", "text", "bbox")

    def __init__(self, kind: str, text: str, bbox: Optional[Tuple[float, float, float, float]]):
        self.kind = kind
        self.text = text
        self.bbox = bbox

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
        }


# ───────────────────────────────────────────────────────────
# Data classes
# ───────────────────────────────────────────────────────────


@dataclass
class Token:
    """A single OCR word with bounding-box geometry."""

    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    page: int
    region: str

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def cx(self) -> float:
        return self.left + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.top + self.height / 2.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "right": self.right,
            "bottom": self.bottom,
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "conf": self.conf,
            "page": self.page,
            "region": self.region,
        }


@dataclass
class Band:
    """A contiguous horizontal band of tokens (header or data)."""

    y_min: float
    y_max: float
    tokens: List[Token]
    row_type: str = "data"  # "header", "title", "data", "separator"

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def text(self) -> str:
        return " ".join(t.text for t in sorted(self.tokens, key=lambda t: t.left))


@dataclass
class Column:
    """A reconstructed vertical value-column region."""

    col_id: str
    label: Optional[str]          # merged leaf header label (e.g. "CYL 10")
    path: List[str]               # header path from group root to this leaf
    x_min: float
    x_max: float
    header: Optional["HeaderNode"] = None

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    def to_dict(self) -> dict:
        return {
            "col_id": self.col_id,
            "label": self.label,
            "path": self.path,
            "x_min": round(self.x_min, 1),
            "x_max": round(self.x_max, 1),
            "cx": round(self.cx, 1),
            "width": round(self.width, 1),
        }


@dataclass
class HeaderNode:
    """A node in the reconstructed header tree."""

    label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    level: int
    kind: HeaderKind = HeaderKind.LEAF
    children: List["HeaderNode"] = field(default_factory=list)
    column: Optional[Column] = None

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "label": self.label,
            "kind": self.kind.value,
            "level": self.level,
            "cx": round(self.cx, 1),
            "x_min": round(self.x_min, 1),
            "x_max": round(self.x_max, 1),
            "is_leaf": self.is_leaf,
        }
        if self.column:
            d["column"] = self.column.col_id
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Region:
    """A labeled horizontal region inside a data row."""

    region_type: RegionType
    tokens: List[Token] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in sorted(self.tokens, key=lambda t: t.left))

    @property
    def normalized_text(self) -> str:
        """Cleaned text (raw OCR/punctuation preserved via `text`/tokens)."""
        return normalize_text(self.text)

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.tokens:
            return None
        left = min(t.left for t in self.tokens)
        top = min(t.top for t in self.tokens)
        right = max(t.right for t in self.tokens)
        bottom = max(t.bottom for t in self.tokens)
        return (float(left), float(top), float(right), float(bottom))

    def to_dict(self) -> dict:
        return {
            "region_type": self.region_type.value,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class Cell:
    """A single logical cell in the value grid."""

    row_idx: int
    column: Column
    tokens: List[Token]
    is_blank: bool
    status: List[StatusIndicator] = field(default_factory=list)
    recovered: bool = False

    @property
    def col_id(self) -> str:
        return self.column.col_id

    @property
    def header_path(self) -> List[str]:
        return self.column.path

    @property
    def header_label(self) -> Optional[str]:
        return self.column.label

    @property
    def raw_text(self) -> str:
        """Reconstructed text exactly as OCR produced it (raw tokens)."""
        return " ".join(t.text for t in sorted(self.tokens, key=lambda t: t.left))

    @property
    def text(self) -> str:
        """Normalized reconstructed text (raw OCR preserved via raw_text/tokens)."""
        return normalize_text(self.raw_text)

    @property
    def cell_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Physical cell box: column x-range clamped to row y-range."""
        if self.column is None:
            return None
        left = self.column.x_min
        right = self.column.x_max
        ys = [t.top for t in self.tokens] + [t.bottom for t in self.tokens]
        if not ys:
            return None
        return (left, float(min(ys)), right, float(max(ys)))

    @property
    def value_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Bounding box of the value tokens only (None if blank)."""
        if not self.tokens:
            return None
        left = min(t.left for t in self.tokens)
        top = min(t.top for t in self.tokens)
        right = max(t.right for t in self.tokens)
        bottom = max(t.bottom for t in self.tokens)
        return (float(left), float(top), float(right), float(bottom))

    @property
    def confidence(self) -> float:
        if not self.tokens:
            return 0.0
        return statistics.mean(t.conf for t in self.tokens)

    def to_dict(self) -> dict:
        return {
            "col_id": self.col_id,
            "header_path": self.header_path,
            "header_label": self.header_label,
            "text": self.text,
            "raw_text": self.raw_text,
            "is_blank": self.is_blank,
            "raw_ocr": [t.to_dict() for t in self.tokens],
            "cell_bbox": list(self.cell_bbox) if self.cell_bbox else None,
            "value_bbox": list(self.value_bbox) if self.value_bbox else None,
            "confidence": round(self.confidence, 2),
            "status": [s.to_dict() for s in self.status],
        }


@dataclass
class RowModel:
    """
    Explicit physical row model for a data row.

    The row label text is NOT represented as a table column; it lives in
    dedicated label/unit/qualifier regions. Only the value grid yields cells.
    """

    row_idx: int
    bbox: Tuple[float, float, float, float]          # full row box
    label_region: Region = field(default_factory=lambda: Region(RegionType.LABEL))
    unit_region: Region = field(default_factory=lambda: Region(RegionType.UNIT))
    qualifier_regions: List[Region] = field(default_factory=list)
    cells: List[Cell] = field(default_factory=list)
    status_indicators: List[StatusIndicator] = field(default_factory=list)
    status_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value_cells(self) -> List[Cell]:
        return [c for c in self.cells]

    @property
    def y_min(self) -> float:
        return self.bbox[1]

    @property
    def y_max(self) -> float:
        return self.bbox[3]

    @property
    def num_blank_cells(self) -> int:
        return sum(1 for c in self.cells if c.is_blank)

    def to_dict(self) -> dict:
        return {
            "row_idx": self.row_idx,
            "bbox": list(self.bbox),
            "label_region": self.label_region.to_dict(),
            "unit_region": self.unit_region.to_dict(),
            "qualifier_regions": [r.to_dict() for r in self.qualifier_regions],
            "cells": [c.to_dict() for c in self.cells],
            "status_indicators": [s.to_dict() for s in self.status_indicators],
            "status_metadata": self.status_metadata,
        }


@dataclass
class PhysicalRow:
    """A physical row: header bands and data rows both appear here."""

    row_idx: int
    y_min: float
    y_max: float
    row_type: str  # "header", "title", "data", "separator"
    model: Optional[RowModel] = None

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def text(self) -> str:
        if self.model is not None:
            return self.model.label_region.text
        return ""

    @property
    def num_tokens(self) -> int:
        if self.model is None:
            return 0
        return (
            len(self.model.label_region.tokens)
            + len(self.model.unit_region.tokens)
            + sum(len(r.tokens) for r in self.model.qualifier_regions)
            + sum(len(c.tokens) for c in self.model.cells)
        )

    @property
    def num_blank_cells(self) -> int:
        if self.model is None:
            return 0
        return self.model.num_blank_cells


@dataclass
class ReconstructionStatus:
    """Status indicators for the reconstruction."""

    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.warnings.append(msg)


@dataclass
class ReconstructionResult:
    """Complete output of the table reconstruction engine."""

    tokens: List[Token]
    bands: List[Band]
    physical_rows: List[PhysicalRow]
    header_bands: List[Band]
    header_hierarchy: List[HeaderNode]
    leaf_columns: List[Column]
    status: ReconstructionStatus
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_tokens": len(self.tokens),
            "num_bands": len(self.bands),
            "num_physical_rows": len(self.physical_rows),
            "num_header_bands": len(self.header_bands),
            "num_columns": len(self.leaf_columns),
            "columns": [c.to_dict() for c in self.leaf_columns],
            "header_tree": [h.to_dict() for h in self.header_hierarchy],
            "physical_rows": [
                r.model.to_dict() if r.model is not None
                else {"row_idx": r.row_idx, "row_type": r.row_type}
                for r in self.physical_rows
            ],
            "status": _to_dict(self.status),
            "debug": self.debug,
        }


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses/enums to plain dicts for serialization."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    return obj


# ───────────────────────────────────────────────────────────
# Step 1: Ingest raw token dicts -> Token objects
# ───────────────────────────────────────────────────────────


def ingest_tokens(raw_tokens: List[dict]) -> List[Token]:
    """Convert raw OCR token dicts to Token objects."""
    result: List[Token] = []
    for d in raw_tokens:
        try:
            t = Token(
                text=str(d.get("text", "")),
                left=int(d["left"]),
                top=int(d["top"]),
                width=int(d["width"]),
                height=int(d["height"]),
                conf=float(d.get("conf", 0)),
                page=int(d.get("page", 0)),
                region=str(d.get("region", "")),
            )
            result.append(t)
        except (KeyError, ValueError, TypeError):
            continue
    return result


# ───────────────────────────────────────────────────────────
# Step 2: Vertical clustering into bands
# ───────────────────────────────────────────────────────────


def _median_token_height(tokens: List[Token]) -> float:
    heights = [t.height for t in tokens if t.height > 0]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def _band_tolerance(tokens: List[Token]) -> float:
    med_h = _median_token_height(tokens)
    return max(4.0, med_h * 0.70)


def cluster_bands(tokens: List[Token], tolerance: Optional[float] = None) -> List[Band]:
    """Group tokens into horizontal bands by vertical (y) proximity."""
    if not tokens:
        return []

    tol = tolerance or _band_tolerance(tokens)
    sorted_tokens = sorted(tokens, key=lambda t: (t.cy, t.left))

    bands: List[Band] = []
    for tok in sorted_tokens:
        best_band: Optional[Band] = None
        best_dist = float("inf")
        for band in bands:
            d = abs(tok.cy - band.cy)
            if d <= tol and d < best_dist:
                best_band = band
                best_dist = d
        if best_band is None:
            bands.append(Band(y_min=float(tok.top), y_max=float(tok.bottom), tokens=[tok]))
        else:
            best_band.tokens.append(tok)
            best_band.y_min = min(best_band.y_min, float(tok.top))
            best_band.y_max = max(best_band.y_max, float(tok.bottom))

    bands.sort(key=lambda b: b.cy)
    return bands


# ───────────────────────────────────────────────────────────
# Step 3: Classify bands as header vs data
# ───────────────────────────────────────────────────────────

_OCR_NUM_NOISE = set("|®©/@\\,()[]{}'`\"°% ")


# Glyphs that are non-semantic OCR artifacts (stars, filler boxes, scanner
# speckle, marking stamps) and can safely be removed from RECONSTRUCTED text
# without losing meaning. Note this is intentionally more conservative than
# _OCR_NUM_NOISE: we keep '.', '-', ',', ':', '%', '°', '/' and parentheses,
# which carry semantic content in reconstructed labels/values.
_OCR_ARTIFACT_GLYPHS = frozenset("|®©@\\`") | frozenset("{}[]")


def _clean_numeric(text: str) -> str:
    return "".join(ch for ch in text.strip() if ch not in _OCR_NUM_NOISE)


def normalize_text(text: str) -> str:
    """
    Normalize reconstructed text by stripping non-semantic OCR artifacts
    while preserving the numerals, letters and meaningful punctuation.

    This is a PURE view over the input string: it never mutates the source
    tokens, so raw OCR text and per-token bounding boxes are always preserved
    (see Cell.raw_text / Cell.tokens).
    """
    out = "".join(ch for ch in text if ch not in _OCR_ARTIFACT_GLYPHS)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([,.:;])", r"\1", out)
    return out


# Generic trailing/leading punctuation glued onto a numeric value token
# (e.g. '29.0;', '0.20)').  Such a token is still a value and must not be
# mistaken for part of a field label during dense side-by-side form
# segmentation.  Only strips edge PUNCTUATION (not letters/digits, so 'C1' /
# '1A' stay non-numeric) and leaves internal punctuation (dates/times) intact.
_NUM_EDGE_PUNCT = re.compile(
    r"^[^\w.,+\-]+|[^\w.,+\-]+$"
)


def _looks_like_number(text: str) -> bool:
    cleaned = _clean_numeric(text)
    if not cleaned:
        return False
    cleaned_edge = _NUM_EDGE_PUNCT.sub("", cleaned)
    if not cleaned_edge:
        return False
    # Strip common unit suffixes after numbers (e.g. "15C" -> "15", "40C" -> "40")
    # Only when the suffix is a single letter C/F/K and the rest parses as a number.
    if len(cleaned_edge) > 1 and cleaned_edge[-1].upper() in ("C", "F", "K"):
        numeric_part = cleaned_edge[:-1]
        try:
            float(numeric_part)
            return True
        except ValueError:
            pass
    try:
        float(cleaned_edge)
        return True
    except ValueError:
        return False


def _looks_like_value_token(tok: Token) -> bool:
    return _looks_like_number(tok.text)


def _band_label_ratio(band: Band, *, merge_compound: bool = True) -> float:
    """
    Fraction of a band that is label-like (vs numeric value).

    Numeric tokens that are CONTICUOUS preview of a compound header prefix
    (e.g. the "10" in "CYL 10", the "5" in "SER NO 5") are counted as
    labels, not values, so that header bands with numeric continuations are
    not mistaken for data rows.

    When *merge_compound* is False the compound-header merge is suppressed and
    each token counts independently; this is used for bands below the header
    vertical level where an alpha+numeric pair is ordinary label+value, not a
    compound column label.
    """
    if not band.tokens:
        return 0.0
    sorted_toks = sorted(band.tokens, key=lambda t: t.left)
    if merge_compound:
        merged = _merge_compound_headers(sorted_toks)
    else:
        merged = [[t] for t in sorted_toks]
    label_units = 0
    for group in merged:
        if len(group) == 1 and _looks_like_number(group[0].text):
            continue  # isolated numeric = value
        label_units += 1
    return label_units / len(merged) if merged else 0.0


def _band_horizontal_coverage(band: Band, page_width: float) -> float:
    if not band.tokens or page_width <= 0:
        return 0.0
    leftmost = min(t.left for t in band.tokens)
    rightmost = max(t.right for t in band.tokens)
    return (rightmost - leftmost) / page_width


def classify_bands(bands: List[Band]) -> List[Band]:
    """
    Classify each band as 'header', 'title', 'data', or 'separator'.

    Heuristics (applied in order):
    1. Separator: very thin or sparse (< 2 tokens).
    2. Has numeric values in the value zone (right portion) -> data.
       This catches sparse tables where rows have many label tokens but few values.
    3. Not text-dominant -> data. A true header row is dominated by column
       labels (label ratio >= 0.80). Data rows carry numeric values and OCR
       artifacts, so their label ratio stays below this.
    4. Title: text-dominant but with low horizontal coverage (< 30% of page)
       OR too few tokens.
    5. Header: text-dominant with high horizontal coverage and enough tokens.
    """
    if not bands:
        return []

    all_rights: List[int] = []
    all_lefts: List[int] = []
    for band in bands:
        for tok in band.tokens:
            all_rights.append(tok.right)
            all_lefts.append(tok.left)
    page_width = float(max(all_rights) - min(all_lefts)) if all_rights and all_lefts else 800.0
    page_left = float(min(all_lefts)) if all_lefts else 0.0

    COVERAGE_THRESHOLD = 0.30
    MIN_HEADER_TOKENS = 2
    # Value zone starts at ~30% from left edge (skip label zone)
    VALUE_ZONE_LEFT = page_left + page_width * 0.30

    def _is_compound_header_continuation(tok: Token, band: Band) -> bool:
        """Check if a numeric token is a compound header continuation (e.g. 'CYL' + '4')."""
        if not _looks_like_number(tok.text):
            return False
        # Find previous token in reading order (non-overlapping)
        prev_toks = [t for t in band.tokens if t.right <= tok.left]
        # Also check for overlapping alphabetic prefix (OCR artifact where prefix overlaps number)
        overlap_toks = [t for t in band.tokens if t.left < tok.right and t.right > tok.left and t is not tok]
        
        candidates = []
        if prev_toks:
            candidates.append(max(prev_toks, key=lambda t: t.right))
        # Check overlapping tokens that are alphabetic and mostly to the left
        for ot in overlap_toks:
            if _is_alphabetic(ot.text) and ot.left < tok.left:
                candidates.append(ot)
        
        if not candidates:
            return False
        
        # Check each candidate
        widths = [t.width for t in band.tokens if t.width > 0]
        med_w = statistics.median(widths) if widths else 0.0
        auto_gap = max(med_w * 0.6, 6.0) if med_w else 12.0
        
        for prev in candidates:
            gap = tok.left - prev.right
            # For overlapping tokens, gap is negative; treat as tight
            if gap < 0:
                gap = 0
            same_baseline = abs(tok.top - prev.top) <= max(2.0, tok.height * 0.35)
            if gap <= auto_gap and same_baseline and _is_alphabetic(prev.text):
                return True
        return False

    for band in bands:
        if len(band.tokens) < 2:
            # A sparse data row can legitimately hold a single value token
            # (many columns blank). Preserve it as a data row so its value is
            # not dropped/shifted; only non-numeric single tokens that are NOT
            # genuine field labels are treated as strays (separators).
            if len(band.tokens) == 1 and (
                _looks_like_number(band.tokens[0].text)
                or _is_genuine_label(list(band.tokens))
            ):
                band.row_type = "data"
            else:
                band.row_type = "separator"
            continue

        # Check for numeric tokens in the value zone (right portion of page).
        # Sparse form/table rows have few numeric values but they appear in the
        # value zone, not the label zone. Header bands are text-dominant across
        # the full width.
        # Exclude numeric tokens that are compound header continuations (e.g. CYL + 4).
        has_value_zone_numeric = any(
            _looks_like_number(tok.text) and tok.cx >= VALUE_ZONE_LEFT
            and not _is_compound_header_continuation(tok, band)
            for tok in band.tokens
        )
        if has_value_zone_numeric:
            band.row_type = "data"
            continue

        if _band_label_ratio(band) < 0.80:
            band.row_type = "data"
            continue
        coverage = _band_horizontal_coverage(band, page_width)
        if coverage >= COVERAGE_THRESHOLD and len(band.tokens) >= MIN_HEADER_TOKENS:
            band.row_type = "header"
        else:
            band.row_type = "title"

    return bands


# ───────────────────────────────────────────────────────────
# Step 4: Compound header reconstruction + hierarchy
# ───────────────────────────────────────────────────────────


def _is_alphabetic(token_text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z._%°]+", token_text.strip()))


def _is_mergeable(token_text: str) -> bool:
    """Token is mergeable in compound headers if it contains letters and is not purely numeric."""
    t = token_text.strip()
    if not t:
        return False
    if _is_pure_numeric(t):
        return False
    return bool(re.search(r"[A-Za-z]", t))


def _is_pure_numeric(token_text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", _clean_numeric(token_text)))


def _merge_compound_headers(
    sorted_toks: List[Token],
    gap_threshold: Optional[float] = None,
) -> List[List[Token]]:
    """
    Merge adjacent header tokens into a single logical header where the
    geometry and lexical pattern indicate one compound label.

    Generic rules (no hardcoded names):
    1. Prefix + numeric continuation: an alphabetic token immediately followed
       by a purely numeric token at a small gap is a compound column label
       (e.g. "CYL" + "10" -> "CYL 10").
    2. Very close adjacency (gap well below the median header inter-token gap)
       with lexical continuation merges an arbitrary leading word + trailing
       word even when both are alphabetic.

    The horizontal-gap threshold is derived from the token geometry itself.
    """
    if len(sorted_toks) < 2:
        return [sorted_toks] if sorted_toks else []

    widths = [t.width for t in sorted_toks if t.width > 0]
    med_w = statistics.median(widths) if widths else 0.0
    # Tight-adjacency threshold derived from token geometry (roughly one
    # character width). Continuation tokens are near-touching; real column
    # boundaries have wider inter-token gaps.
    auto_gap = max(med_w * 0.6, 6.0) if med_w else 12.0

    # Compound continuations share one text baseline. Group/legend headers
    # and value-column leaves live on different vertical lines and must NOT be
    # merged (e.g. a spanning GROUP label sitting above the first value column).
    heights = [t.height for t in sorted_toks if t.height > 0]
    med_h = statistics.median(heights) if heights else 16.0
    baseline_tol = max(2.0, med_h * 0.35)

    groups: List[List[Token]] = [[sorted_toks[0]]]
    for i in range(1, len(sorted_toks)):
        prev = sorted_toks[i - 1]
        curr = sorted_toks[i]
        gap = curr.left - prev.right
        same_baseline = abs(curr.top - prev.top) <= baseline_tol

        # Merge when the gap is a tight continuation, the tokens share a
        # baseline, AND the pair forms a lexical unit: prefix+number, or
        # multiple mergeable words that are near-touching (one continued label).
        if gap <= auto_gap and same_baseline:
            prev_mergeable = _is_mergeable(prev.text)
            curr_mergeable = _is_mergeable(curr.text)
            _is_pure_numeric(prev.text)
            curr_num = _is_pure_numeric(curr.text)
            if (prev_mergeable and curr_num) or (prev_mergeable and curr_mergeable):
                groups[-1].append(curr)
                continue

        groups.append([curr])

    return groups


def _build_header_hierarchy(level_groups: Dict[int, List[HeaderNode]]) -> List[HeaderNode]:
    """Wire parent-child relationships by x-overlap across levels."""
    if not level_groups:
        return []

    sorted_levels = sorted(level_groups.keys())
    if len(sorted_levels) <= 1:
        return level_groups[sorted_levels[0]]

    for li in range(len(sorted_levels) - 1):
        parents = level_groups[sorted_levels[li]]
        children = level_groups[sorted_levels[li + 1]]
        for child in children:
            best_parent: Optional[HeaderNode] = None
            best_overlap = 0.0
            for parent in parents:
                overlap = _x_overlap(parent, child)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent = parent
            if best_parent is not None and best_overlap > 0:
                best_parent.children.append(child)

    return level_groups[sorted_levels[0]]


def _x_overlap(a: HeaderNode, b: HeaderNode) -> float:
    overlap_start = max(a.x_min, b.x_min)
    overlap_end = min(a.x_max, b.x_max)
    return max(0.0, overlap_end - overlap_start)


def extract_header_labels(
    header_bands: List[Band],
    data_bands: Optional[List[Band]] = None,
) -> List[HeaderNode]:
    """
    Build the header tree from classified header bands.

    The PRIMARY column scheme is the header band whose compound-merged labels
    yield the most leaf columns within the value grid (geometry-derived, so a
    text-only row label misclassified as a header can never overshadow a real
    multi-column header). Its merged labels become leaf headers. Other bands
    above/beside it become title/group headers.

    A 'label_zone' header is created when the primary band has a token that
    falls left of the value grid - it annotates the row-label area rather
    than a value column. Callers refine via discover_columns.
    """
    if not header_bands:
        return []

    grid_left = _compute_grid_left(data_bands or [])
    min_col_width = 10.0

    def _band_value_leaf_count(band: Band) -> int:
        """How many compound-merged label groups of this band fall inside the
        value grid (at/right of grid_left). The real column scheme has many;
        a pure row-label/unit band has one or none."""
        ss = sorted(band.tokens, key=lambda t: t.left)
        groups = _merge_compound_headers(ss)
        cnt = 0
        for g in groups:
            cx = float(sum(t.cx for t in g)) / len(g)
            if grid_left is None or cx >= grid_left - min_col_width:
                cnt += 1
        return cnt

    # Prefer the band with the most value-grid leaf columns; a header with
    # many merged column labels but no value positions scores lower. Tie-break
    # by token count, then by higher position (visual header row above data).
    primary = max(
        header_bands,
        key=lambda b: (_band_value_leaf_count(b), len(b.tokens), -b.cy),
    )
    primary_sorted = sorted(primary.tokens, key=lambda t: t.left)
    merged = _merge_compound_headers(primary_sorted)

    leaf_nodes: List[HeaderNode] = []
    for group in merged:
        label_text = " ".join(t.text for t in group)
        node = HeaderNode(
            label=label_text,
            x_min=float(min(t.left for t in group)),
            x_max=float(max(t.right for t in group)),
            y_min=float(min(t.top for t in group)),
            y_max=float(max(t.bottom for t in group)),
            level=0,
            kind=HeaderKind.LEAF,
        )
        leaf_nodes.append(node)

    # Order leaves by x.
    leaf_nodes.sort(key=lambda n: n.x_min)

    # Add title/meta bands above the primary as title/group headers.
    # These are document metadata, NOT parents of the value columns.
    group_nodes: List[HeaderNode] = []
    title_bands = sorted([b for b in header_bands if b is not primary], key=lambda b: b.cy)
    for band in title_bands:
        gtoks = sorted(band.tokens, key=lambda t: t.left)
        gnode = HeaderNode(
            label=" ".join(t.text for t in gtoks),
            x_min=float(min(t.left for t in gtoks)),
            x_max=float(max(t.right for t in gtoks)),
            y_min=float(band.y_min),
            y_max=float(band.y_max),
            level=-1,
            kind=HeaderKind.TITLE,
        )
        group_nodes.append(gnode)

    # Return title bands and primary leaves all as roots (leaves are not
    # duplicated beneath the title). Group/leaf typing is refined later by
    # discover_columns (label_zone vs value column).
    roots = group_nodes + leaf_nodes
    roots.sort(key=lambda n: (n.level, n.x_min))
    return roots


# ───────────────────────────────────────────────────────────
# Step 5: Discover value columns + label-zone boundary
# ───────────────────────────────────────────────────────────


def _compute_grid_left(data_bands: List[Band]) -> Optional[float]:
    """
    Leftmost x-center among numeric (value) tokens across data bands.
    Defines the left boundary of the value grid.
    """
    value_cx: List[float] = []
    for band in data_bands:
        for tok in band.tokens:
            if _looks_like_number(tok.text):
                value_cx.append(tok.cx)
    if not value_cx:
        return None
    return min(value_cx)


def discover_columns(
    header_bands: List[Band],
    data_bands: List[Band],
    min_col_width: float = 10.0,
) -> Tuple[List[Column], List[HeaderNode]]:
    """
    Discover value columns plus a label-zone header list.

    Anchors on the PRIMARY header band (most tokens). Its compound-merged
    labels whose centers fall at/right of the value-grid left edge become
    leaf value columns. Labels left of that edge are label-zone headers
    (QUALITY / Imported) and are excluded from the value grid.

    Returns (leaf_columns, header_hierarchy) where the hierarchy's leaf
    nodes carry their resolved Column.
    """
    hierarchy = extract_header_labels(header_bands, data_bands)
    if not hierarchy:
        return [], []

    # Collect LEAF-precursor nodes (value-column or label-zone candidates).
    # Title/group metadata bands are NOT column candidates.
    def _leaves(nodes: List[HeaderNode]) -> List[HeaderNode]:
        out: List[HeaderNode] = []
        for n in nodes:
            if n.kind == HeaderKind.TITLE or n.kind == HeaderKind.GROUP:
                out.extend(_leaves(n.children))
            elif n.is_leaf:
                out.append(n)
            else:
                out.extend(_leaves(n.children))
        return out

    leaf_nodes = _leaves(hierarchy)

    grid_left = _compute_grid_left(data_bands)

    value_leaves: List[HeaderNode] = []
    label_zone_leaves: List[HeaderNode] = []
    for node in leaf_nodes:
        if grid_left is None or node.cx >= grid_left - min_col_width:
            value_leaves.append(node)
        else:
            node.kind = HeaderKind.LABEL_ZONE
            label_zone_leaves.append(node)

    value_leaves.sort(key=lambda n: n.cx)
    if len(value_leaves) < 2:
        return [], hierarchy

    # Recompute boundaries using each leaf node's own x-extent for stability.
    columns: List[Column] = []
    for i, node in enumerate(value_leaves):
        if i == 0 and len(value_leaves) > 1:
            x_min = node.x_min
        elif i == 0:
            x_min = node.x_min - min_col_width / 2.0
        else:
            x_min = (value_leaves[i - 1].x_max + node.x_min) / 2.0

        if i == len(value_leaves) - 1 and len(value_leaves) > 1:
            x_max = node.x_max
        elif i == len(value_leaves) - 1:
            x_max = node.x_max + min_col_width / 2.0
        else:
            x_max = (node.x_max + value_leaves[i + 1].x_min) / 2.0

        col = Column(col_id=f"col_{i}", label=node.label, path=[node.label],
                     x_min=x_min, x_max=x_max, header=node)
        node.column = col
        columns.append(col)

    # Rebuild hierarchy so group/title parents reference value-leaf columns.
    _attach_columns_to_groups(hierarchy, columns)
    return columns, hierarchy


def _attach_columns_to_groups(hierarchy: List[HeaderNode], columns: List[Column]) -> None:
    """Attach resolved columns to leaf nodes anywhere in the hierarchy."""
    col_by_node: Dict[int, Column] = {}
    for c in columns:
        if c.header is not None:
            col_by_node[id(c.header)] = c

    def _walk(nodes: List[HeaderNode]) -> None:
        for n in nodes:
            if id(n) in col_by_node:
                n.column = col_by_node[id(n)]
            _walk(n.children)

    _walk(hierarchy)


def _discover_columns_from_gaps(tokens: List[Token], min_col_width: float) -> List[Column]:
    if not tokens:
        return []
    all_cx = sorted(set(t.cx for t in tokens))
    if len(all_cx) < 2:
        return []
    gaps: List[Tuple[float, float]] = []
    for i in range(1, len(all_cx)):
        gaps.append((all_cx[i] - all_cx[i - 1], (all_cx[i] + all_cx[i - 1]) / 2.0))
    if not gaps:
        return []
    med_gap = statistics.median([g[0] for g in gaps])
    threshold = max(med_gap * 2.0, min_col_width)
    break_points: List[float] = [all_cx[0] - min_col_width]
    for gap_size, mid in gaps:
        if gap_size > threshold:
            break_points.append(mid)
    break_points.append(all_cx[-1] + min_col_width)
    columns: List[Column] = []
    for i in range(len(break_points) - 1):
        x_min = break_points[i]
        x_max = break_points[i + 1]
        label = _find_dominant_label(tokens, x_min, x_max)
        columns.append(Column(col_id=f"col_{i}", label=label, path=[label or f"col_{i}"],
                              x_min=x_min, x_max=x_max))
    return columns


def _find_dominant_label(tokens: Sequence[Token], x_min: float, x_max: float) -> Optional[str]:
    candidates = [t for t in tokens if x_min <= t.cx <= x_max and not _looks_like_number(t.text)]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t.top, t.left))
    return candidates[0].text


# ───────────────────────────────────────────────────────────
# Step 6: Build explicit row models (label/unit/qualifier/cells)
# ───────────────────────────────────────────────────────────


def _is_unit_token(tok: Token, unit_cx_set: Optional[set]) -> bool:
    """
    Is a label-zone token a unit?

    Primary signal is GEOMETRY: the token sits on a horizontally-aligned
    column of short tokens that repeats across data rows (unit_cx_set).
    We deliberately do NOT use short all-letter shape alone, because ordinary
    label words ("Firing", "pmax", "MEAN") are also short alphabetic strings
    and would wrongly be consumed from the row label.
    """
    t = tok.text.strip()
    if not t:
        return False
    # Punctuation-annotations are qualifiers, not units.
    if t.startswith("(") or t.endswith(")") or t.endswith(":"):
        return False
    if _looks_like_number(t):
        return False
    # Geometry: horizontally-aligned unit column spanning multiple rows.
    if unit_cx_set is not None and round(tok.cx, 0) in unit_cx_set:
        return True
    # Unit-ish symbol ending (e.g. a degree or percent symbol) as weak
    # evidence, disambiguated later by trailing position in the row.
    if _unit_shape_matches(t):
        return True
    return False


# Generic engineering/SI unit lexemes (international, not document specific).
# Used only as WEAK evidence combined with geometry/position; never alone.
_UNIT_LEXEMES = frozenset({
    "bar", "barg", "mbarg", "mbar", "pa", "kpa", "mpa",
    "kw", "mw", "w", "rpm", "mm", "cm", "m", "km",
    "s", "sec", "min", "h", "hr", "hz", "khz", "mhz",
    "c", "celsius", "r", "k", "kg", "t", "kn", "l", "l/m",
    "n", "nm", "microns",
    # engineering / fuel units (generic vocabulary, no field knowledge)
    "cst", "mmwg", "mmh2o", "mmhg", "g/kwh", "kg/m2", "kg/m3", "kg/m4",
    "kj/kg", "m3/h", "m3", "l/h", "l/min", "m/h", "m/s", "m3/min",
    "usd/t", "usd", "eur", "%/h", "%/min", "jd", "mm/d", "mm/h",
})


def _unit_shape_matches(text: str) -> bool:
    """Generic unit-shape evidence (no document rules / field names).

    Recognises a bounded vocabulary of common engineering/fuel units, plus
    degree/percent forms and ratio units. Deliberately NOT a broad short-word
    match: label words like 'mean', 'temp', 'oil', 'in' must never be treated
    as units. Also accepts units with trailing OCR punctuation ('.', ',', ')')
    and a unit immediately preceding a parenthesis.
    """
    t = text.strip().strip(".,;:)").strip()
    if not t:
        return False
    low = t.lower()
    norm = low.replace("°", "deg").replace("º", "deg")
    if norm in _UNIT_LEXEMES:
        return True
    # degree/temperature: 'degc', 'cdeg', 'degf', 'fdeg'
    if norm in {"degc", "cdeg", "degf", "fdeg", "deg", "cdegf"}:
        return True
    if norm.endswith("%"):
        return True
    # ratio / composite units with a known numerator/denominator vocabulary.
    if re.fullmatch(r"(?:[a-z]{2,4}/[a-z]{2,4}|[a-z]{1,3}/m\d|[a-z]{1,3}/m\?|bar[ag]|[a-z]{2,4}\s?/\s?[a-z]{2,4})", norm):
        return True
    return False


def _detect_unit_cx(data_bands: List[Band], label_zone_x_min: float, grid_left: float) -> set:
    """
    Detect unit columns via inter-row horizontal alignment of short
    label-zone tokens that repeat across multiple data rows.
    
    Units are expected at the RIGHT edge of the label zone, adjacent to
    the value grid. Tokens on the left side of the label zone are label
    words, not units.
    """
    cx_counts: Dict[int, int] = defaultdict(int)
    # Only consider the rightmost portion of the label zone for unit detection
    # (e.g., rightmost 30% of label zone width, or fixed distance from grid_left)
    label_zone_width = grid_left - label_zone_x_min
    unit_zone_left = grid_left - max(0.3 * label_zone_width, 80.0)
    
    for band in data_bands:
        seen: set = set()
        for tok in band.tokens:
            # Only consider tokens in the unit zone (right portion of label zone)
            if unit_zone_left <= tok.cx < grid_left:
                t = tok.text.strip()
                if not t or _looks_like_number(t):
                    continue
                if t.startswith("(") or t.endswith(")"):
                    continue
                if len(t) > 6:
                    continue
                rcx = round(tok.cx, 0)
                if rcx in seen:
                    continue
                seen.add(rcx)
                cx_counts[rcx] += 1
    if len(data_bands) == 0:
        return set()
    # A unit column must repeat in a meaningful fraction of data rows.
    threshold = max(2, int(len(data_bands) * 0.4))
    return {cx for cx, cnt in cx_counts.items() if cnt >= threshold}


def build_row_models(
    data_bands: List[Band],
    columns: List[Column],
    label_zone_headers: List[HeaderNode],
    grid_left: Optional[float],
) -> List[RowModel]:
    """
    Build an explicit RowModel for each data band.

    Splits each row's tokens into:
      - value cells (tokens assigned to value columns)
      - a label region (remaining text left of the value grid)
      - a unit region (aligned short tokens repeating across rows)
      - qualifier regions (parenthesized / colon annotations)
    """
    if not columns:
        return []

    # Value-grid left edge from columns.
    grid_left_eff = min(c.x_min for c in columns) if columns else (grid_left or 0.0)

    # Determine label-zone x range.
    if label_zone_headers:
        lz_x_min = min(n.x_min for n in label_zone_headers)
        max(n.x_max for n in label_zone_headers)
    else:
        lz_x_min = 0.0

    unit_cx = _detect_unit_cx(data_bands, lz_x_min, grid_left_eff)

    rows: List[RowModel] = []
    for ri, band in enumerate(data_bands):
        value_col_tokens = _assign_row_tokens_to_columns(band, columns)
        label_zone_tokens: List[Token] = []
        for tok in band.tokens:
            if any(tok is vt for vt in value_col_tokens):
                continue
            if tok.cx < grid_left_eff:
                label_zone_tokens.append(tok)

        # Build value cells, preserving blanks (positional).
        cells: List[Cell] = []
        col_tokens: Dict[str, List[Token]] = defaultdict(list)
        for tok, col_id in value_col_tokens:
            col_tokens[col_id].append(tok)
        for col in columns:
            toks = sorted(col_tokens.get(col.col_id, []), key=lambda t: t.left)
            cell = Cell(row_idx=ri, column=col, tokens=toks, is_blank=len(toks) == 0)
            _detect_cell_status(cell)
            cells.append(cell)

        row_model = _build_label_zone_model(band, label_zone_tokens, unit_cx, grid_left_eff)

        # Gather status indicators from all cells.
        row_status: List[StatusIndicator] = []
        for c in cells:
            row_status.extend(c.status)

        model = RowModel(
            row_idx=ri,
            bbox=(lz_x_min, band.y_min, grid_left_eff, band.y_max),
            label_region=row_model[0],
            unit_region=row_model[1],
            qualifier_regions=row_model[2],
            cells=cells,
            status_indicators=row_status,
            status_metadata={
                "num_cells": len(cells),
                "num_blank_cells": sum(1 for c in cells if c.is_blank),
            },
        )
        rows.append(model)

    return rows


def _adopt_grid_title_labels(rows: List[PhysicalRow], bands: List[Band]) -> None:
    """Give a grid data row a real field label when its own label is blank or
    junk, by adopting the label of a vertically-adjacent title/label band.

    Many grids print a data row's descriptive label on a nearby title-like band
    (e.g. 'ENGINE power estimated') that sits slightly above or below the value
    row, rather than inline. A data row with an empty or non-genuine label and
    a vertically-adjacent title band carrying a genuine label inherits that
    label, so the row maps to its schema field instead of a blank/dropped one.
    Only when the row's own label is unusable, so no real inline label is
    overwritten.
    """
    adopted_ids: set = set()
    for pr in rows:
        m = pr.model
        if m is None:
            continue
        if _has_alpha_word(m.label_region.tokens, 2):
            continue
        cand = None
        best_gap = float("inf")
        for b in bands:
            if b.row_type not in ("title", "label"):
                continue
            if id(b) in adopted_ids:
                continue
            if not _is_genuine_label(b.tokens):
                continue
            gap = max(b.y_min - m.bbox[3], m.bbox[1] - b.y_max)
            if gap <= _CONTINUATION_GAP_MAX and gap < best_gap:
                # adopt label tokens replacing the empty/junk label region
                cand = b
                best_gap = gap
        if cand is not None:
            m.label_region = Region(RegionType.LABEL, tokens=list(cand.tokens))
            y0 = min(m.bbox[1], cand.y_min)
            y1 = max(m.bbox[3], cand.y_max)
            m.bbox = (m.bbox[0], y0, m.bbox[2], y1)
            adopted_ids.add(id(cand))


def _band_label_zone_tokens(band: Band, grid_left: float) -> List[Token]:
    """Tokens of a band that fall in the row-label zone (left of the value
    grid). Column-header labels sit at/inside the grid, so only genuine field
    text ends up here."""
    gl = float(grid_left)
    return [t for t in band.tokens if t.cx < gl]


def _is_label_band_field(band: Band, grid_left: float) -> bool:
    """Is this header/title band a label-only FIELD row (not a real column
    scheme / parent / leaf header, and not a footnote)?

    A real parent or leaf header prints its column labels at/inside the value
    grid, so its tokens are never concentrated in the row-label zone. A field
    label (e.g. 'Turbocharger efficiency', 'Water press. SAC in', 'LINER WALL
    temp. aft') sits left of the grid and forms a genuine label. Downstream a
    promoted band that matches no schema field is simply dropped, so the only
    risk is distortion of the header tree -- which this predicate prevents by
    only ever ADDING row models, never reclassifying the band.
    """
    if band.row_type not in ("header", "title"):
        return False
    lz = _band_label_zone_tokens(band, grid_left)
    if not lz:
        return False
    ordered = sorted(lz, key=lambda t: t.left)
    first = ordered[0]
    # Footnote / annotation lines ("*measured at flange") are not fields.
    if first.text.strip().startswith("*"):
        return False
    if not _is_genuine_label(lz):
        return False
    # A value-bearing band is a data row, not a label-only field band.
    text_toks = [t for t in band.tokens if t.text.strip()]
    if not text_toks:
        return False
    val_count = sum(1 for t in band.tokens
                    if _looks_like_number(t.text) or _is_date_like(t.text))
    if val_count * 2 >= len(text_toks):
        return False
    return True


def _build_label_band_model(band: Band, columns: List[Column], grid_left: float,
                            row_idx: int) -> RowModel:
    """Build a label-only RowModel for a promoted header/title field band.

    The label-zone tokens are split into label/unit/qualifier. The band's
    value-position tokens are column-header labels or OCR junk -- never this
    field's values -- so every grid cell is left physically blank
    (SOURCE_BLANK) with no fabricated value. ``status_metadata['label_band']``
    tells the adapter this is a promoted label-only row so it may extract a
    trailing text value for ``allow_text_values`` fields.
    """
    lz = sorted(_band_label_zone_tokens(band, grid_left), key=lambda t: t.left)
    kept, unit_toks, quals = _split_label_unit(lz, None)
    if not kept and unit_toks:
        kept = unit_toks
        unit_toks = []
    label_region = Region(RegionType.LABEL, tokens=kept)
    unit_region = Region(RegionType.UNIT, tokens=unit_toks)
    cells: List[Cell] = []
    for col in columns:
        cells.append(Cell(row_idx=row_idx, column=col, tokens=[], is_blank=True))
    ys = [t.top for t in lz] + [t.bottom for t in lz]
    y_min = min(ys) if ys else float(band.y_min)
    y_max = max(ys) if ys else float(band.y_max)
    xs = [t.left for t in lz] + [t.right for t in lz]
    x_min = min(xs) if xs else 0.0
    gl = float(grid_left)
    model = RowModel(
        row_idx=row_idx,
        bbox=(float(x_min), y_min, gl, y_max),
        label_region=label_region,
        unit_region=unit_region,
        qualifier_regions=quals,
        cells=cells,
        status_metadata={
            "label_band": True,
            "num_cells": len(cells),
            "num_blank_cells": len(cells),
        },
    )
    return model


def _assign_row_tokens_to_columns(band: Band, columns: List[Column]) -> List[Tuple[Token, str]]:
    """Assign tokens at/right of the value grid to the nearest column."""
    result: List[Tuple[Token, str]] = []
    grid_left = min(c.x_min for c in columns)
    for tok in band.tokens:
        if tok.cx < grid_left:
            continue
        col_id = _assign_token_to_column(tok, columns)
        if col_id is not None:
            result.append((tok, col_id))
    return result


def _assign_token_to_column(tok: Token, columns: List[Column]) -> Optional[str]:
    for col in columns:
        if col.x_min <= tok.cx <= col.x_max:
            return col.col_id
    best_col: Optional[Column] = None
    best_dist = float("inf")
    for col in columns:
        dist = abs(tok.cx - col.cx)
        if dist < best_dist:
            best_dist = dist
            best_col = col
    return best_col.col_id if best_col else None


def _detect_cell_status(cell: Cell) -> None:
    """
    Detect a status indicator inside a cell.

    Generic approach: a short non-numeric token within the cell that is not
    part of the value (e.g. an isolated status dot / symbol). If the cell has
    one numeric value and stray small symbol tokens, tag them as status.
    """
    if cell.is_blank:
        return
    [_clean_numeric(t.text) for t in cell.tokens]
    [t for t in cell.tokens if _looks_like_number(t.text)]
    status_tokens = [
        t for t in cell.tokens
        if not _looks_like_number(t.text) and t.text.strip()
    ]
    # Only treat as status if there is at least one real value too, and the
    # status token is a standalone symbol (not a word).
    for st in status_tokens:
        txt = st.text
        if re.fullmatch(r"[^A-Za-z0-9]+", txt.strip()):
            cell.status.append(
                StatusIndicator(
                    kind="symbol",
                    text=txt,
                    bbox=(float(st.left), float(st.top), float(st.right), float(st.bottom)),
                )
            )


def _build_label_zone_model(
    band: Band,
    label_zone_tokens: List[Token],
    unit_cx: set,
    grid_left: float,
) -> Tuple[Region, Region, List[Region]]:
    """
    Classify the label-zone tokens into label / unit / qualifier regions.

    Row-label segmentation rule: the full connected run of words in each row
    is the LABEL. A unit may only be split out when geometry supports it:
      - the token is on a horizontally-aligned unit column (unit_cx), OR
      - the token is the trailing token directly adjacent to the value grid
        AND matches generic unit-shape evidence.
    This keeps mid-label words (e.g. "Firing", "pmax", "MEAN") in the label.
    """
    lz = sorted(label_zone_tokens, key=lambda t: t.left)
    label_region = Region(RegionType.LABEL)
    unit_region = Region(RegionType.UNIT)
    qualifiers: List[Region] = []

    if not lz:
        return label_region, unit_region, qualifiers

    # Trailing label-zone token (rightmost) is the position where a row-level
    # unit could legitimately sit, adjacent to the value grid.
    trailing = lz[-1]

    for tok in lz:
        txt = tok.text.strip()
        if not txt:
            continue
        # Qualifiers first: punctuation annotations, colon/set clauses.
        if txt.startswith("(") or txt.endswith(")") or txt.endswith(":") or txt in ("ALL", "SUM"):
            qualifiers.append(Region(RegionType.QUALIFIER, [tok]))
            continue
        # Unit by geometry (aligned column) ...
        is_unit = False
        if _is_unit_token(tok, unit_cx):
            is_unit = True
        elif tok is trailing and len(lz) > 1 and _unit_shape_matches(txt):
            # ... or by shape evidence, but ONLY for the trailing token so we
            # never consume label words.
            is_unit = True
        if is_unit:
            unit_region.tokens.append(tok)
        else:
            label_region.tokens.append(tok)

    return label_region, unit_region, qualifiers


# ───────────────────────────────────────────────────────────
# Step 7: Status and debug diagnostics
# ───────────────────────────────────────────────────────────


def _build_status(
    tokens: List[Token],
    bands: List[Band],
    columns: List[Column],
    header_bands: List[Band],
    row_models: List[RowModel],
) -> ReconstructionStatus:
    status = ReconstructionStatus()

    if not tokens:
        status.fail("No input tokens")
        return status
    if not bands:
        status.fail("No bands produced from tokens")
        return status
    if not header_bands:
        status.warn("No header bands detected; column mapping may be imprecise")
    if not columns:
        status.warn("No columns discovered; all cells will be unmapped")
    if not row_models:
        status.warn("No data row models produced")

    total_blank = sum(r.num_blank_cells for r in row_models)
    total_cells = sum(len(r.cells) for r in row_models)
    if total_cells > 0:
        blank_pct = total_blank / total_cells * 100
        if blank_pct > 50:
            status.warn(f"High blank cell rate: {blank_pct:.0f}% ({total_blank}/{total_cells})")
        else:
            status.add_info(f"Blank cell rate: {blank_pct:.0f}% ({total_blank}/{total_cells})")

    pages = set(t.page for t in tokens)
    regions = set(t.region for t in tokens)
    status.add_info(f"Pages: {sorted(pages)}")
    status.add_info(f"Regions: {sorted(regions)}")
    status.add_info(
        f"Tokens: {len(tokens)}, Bands: {len(bands)}, Headers: {len(header_bands)}, "
        f"Columns: {len(columns)}, DataRows: {len(row_models)}"
    )
    return status


def _build_debug(
    tokens: List[Token],
    bands: List[Band],
    columns: List[Column],
    header_bands: List[Band],
    row_models: List[RowModel],
) -> Dict[str, Any]:
    return {
        "band_summary": [
            {
                "idx": i,
                "row_type": band.row_type,
                "cy": round(band.cy, 1),
                "height": round(band.height, 1),
                "num_tokens": len(band.tokens),
                "text_preview": band.text[:120],
            }
            for i, band in enumerate(bands)
        ],
        "column_summary": [c.to_dict() for c in columns],
        "gap_analysis": [
            {
                "left_token": sorted(band.tokens, key=lambda t: t.left)[i - 1].text,
                "right_token": sorted(band.tokens, key=lambda t: t.left)[i].text,
                "gap_px": round(
                    sorted(band.tokens, key=lambda t: t.left)[i].left
                    - sorted(band.tokens, key=lambda t: t.left)[i - 1].right, 1,
                ),
            }
            for band in header_bands
            for i in range(1, len(band.tokens))
        ],
        "token_height_stats": _height_stats(tokens),
        "page_dims": _page_dims(tokens),
    }


def _height_stats(tokens: List[Token]) -> dict:
    heights = [t.height for t in tokens if t.height > 0]
    if not heights:
        return {}
    heights.sort()
    return {
        "min": heights[0],
        "max": heights[-1],
        "median": heights[len(heights) // 2],
        "mean": round(statistics.mean(heights), 1),
        "count": len(heights),
    }


def _page_dims(tokens: List[Token]) -> dict:
    if not tokens:
        return {}
    return {"width": max(t.right for t in tokens), "height": max(t.bottom for t in tokens)}


# ───────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────


def reconstruct(
    raw_tokens: List[dict],
    band_tolerance: Optional[float] = None,
    min_col_width: float = 10.0,
) -> ReconstructionResult:
    """
    Main entry point: reconstruct the report geometry from raw OCR tokens.

    This is a SINGLE common geometry engine (no table/pair modes). It:
      1. clusters tokens into horizontal bands,
      2. locates a value-column grid when one is physically present,
      3. composes complete logical rows (one per printed field) via
         field-run segmentation and continuation merging,
      4. maps each logical row's value tokens to the located columns.

    The schema declares WHAT (field, parent/leaf columns, unit, value
    structure); this engine locates WHERE those structures sit in the PDF and
    produces RowModels the adapter can map into the schema.
    """
    tokens = ingest_tokens(raw_tokens)
    bands = cluster_bands(tokens, tolerance=band_tolerance)
    bands = classify_bands(bands)

    header_bands = [b for b in bands if b.row_type == "header"]
    data_bands = [b for b in bands if b.row_type == "data"]

    header_hierarchy = extract_header_labels(header_bands)
    columns, header_hierarchy2 = discover_columns(header_bands, data_bands, min_col_width)
    header_hierarchy = header_hierarchy2 if header_hierarchy2 else header_hierarchy

    grid_left = _compute_grid_left(data_bands)
    label_zone_headers = [
        n for n in _flatten_header_nodes(header_hierarchy) if n.kind == HeaderKind.LABEL_ZONE
    ]
    _ = (grid_left, label_zone_headers)  # retained for the common column solver

    # Decide whether a real value grid is physically present. A grid needs
    # several value columns that are (a) narrow, (b) populated in enough
    # distinct data rows, and (c) evenly spaced. Pair/form regions (one value
    # per field, no regular table) fail one or more of these and fall back to
    # a single VALUE column. Purely geometric: schema-independent.
    is_grid = _is_grid(bands, columns)

    if is_grid:
        used_columns = columns
    else:
        used_columns = [
            Column(
                col_id="VALUE", label="VALUE", path=["VALUE"],
                x_min=0.0, x_max=max((t.right for t in tokens), default=1.0),
            )
        ]

    # Two row-composition strategies, selected purely by geometry (_is_grid):
    #  - grid regions have a uniform value grid whose rows carry a label plus
    #    values across many columns (per-column mapping via build_row_models),
    #  - pair/form regions hold side-by-side single-value fields on one strip
    #    (per-field composition via _field_run_rows).
    if is_grid:
        row_models = build_row_models(data_bands, used_columns, label_zone_headers, grid_left)
        grid_left_eff = float(min(c.x_min for c in used_columns))
        physical_rows: List[PhysicalRow] = []
        data_index = {id(b): i for i, b in enumerate(data_bands)}
        for band in bands:
            model = None
            if band.row_type == "data":
                di = data_index.get(id(band))
                if di is not None and di < len(row_models):
                    model = row_models[di]
            elif _is_label_band_field(band, grid_left_eff):
                # A label-only field row whose label is printed as a
                # header/title band (e.g. 'Turbocharger efficiency', 'Water
                # press. SAC in'). Give the adapter a blank semantic row so it
                # is emitted as SOURCE_BLANK / text instead of being silently
                # dropped as SOURCE_ABSENT. Real column-scheme/parent/leaf
                # header bands are excluded by _is_label_band_field.
                model = _build_label_band_model(band, used_columns, grid_left_eff,
                                                len(physical_rows))
            physical_rows.append(
                PhysicalRow(row_idx=len(physical_rows), y_min=band.y_min,
                            y_max=band.y_max, row_type=band.row_type, model=model)
            )
        _adopt_grid_title_labels(physical_rows, bands)
        rows = physical_rows
    else:
        rows = _field_run_rows(bands, used_columns, is_grid)
        for pr in rows:
            if pr.model is not None:
                for c in pr.model.cells:
                    _detect_cell_status(c)
        physical_rows = rows

    status = _build_status(tokens, bands, used_columns, header_bands,
                           [r.model for r in rows if r.model is not None])
    debug = _build_debug(tokens, bands, used_columns, header_bands,
                         [r.model for r in rows if r.model is not None])

    return ReconstructionResult(
        tokens=tokens,
        bands=bands,
        physical_rows=physical_rows,
        header_bands=header_bands,
        header_hierarchy=header_hierarchy,
        leaf_columns=used_columns,
        status=status,
        debug=debug,
    )


def _flatten_header_nodes(nodes: List[HeaderNode]) -> List[HeaderNode]:
    out: List[HeaderNode] = []
    for n in nodes:
        out.append(n)
        out.extend(_flatten_header_nodes(n.children))
    return out


# ───────────────────────────────────────────────────────────
# Step 8: Pair / form layout reconstruction
#
# Many crop regions are NOT wide columnar tables. The same horizontal strip
# can hold several side-by-side "label <value>" field blocks (e.g. 'Lub oil
# temp ENGINE in °C | 44.0' and 'Fuel oil press. ENGINE in bar | 7.10' on one
# line), or a stacked form (label above value). The columnar engine mis-handles
# these by folding the whole line into one row with garbled "columns".
#
# This generic path segments every data band into horizontal LABEL:VALUE
# blocks and rebuilds each block as its own RowModel with a complete composed
# label + unit + qualifier and a single VALUE cell. It is layout-agnostic
# (geometry / gap / X alignment driven; no per-field, per-file or per-PDF
# rules).
# ───────────────────────────────────────────────────────────


def _is_value_artifact(text: str) -> bool:
    """Generic value-adjacent artifacts kept with the value (not the label)."""
    t = text.strip()
    if not t:
        return False
    if _looks_like_number(t):
        return True
    # One or more artifact glyphs, so a multi-glyph OCR token made only of
    # table rules / filler boxes / scanner speckle (e.g. '|®', '®©', '{}')
    # is kept with the value rather than misread as a field label.
    return bool(re.fullmatch(r"[|/\\®©@`'{}[\]\"°%]+|%\b", t))


def _field_runs_in_band(tokens: List[Token]) -> List[Tuple[List[Token], List[Token]]]:
    """Split a band's tokens into (label, value) FIELD RUNS.

    A label:value line has the form ``<label words...> <unit?> <value>``
    repeated left-to-right, one field per value. We walk the band left-to-right:
      - accumulate non-numeric tokens as the current label,
      - when a run of numeric tokens is hit, close the field and take those
        numerics as its value,
      - when a non-numeric token sits far to the right of the current label's
        tokens (a column break much larger than the band's normal word
        spacing), close the field even without a value so side-by-side
        label-only fields ('Lub oil press. COOLING OIL | Fuel oil FILTER out')
        separate cleanly.
    The next non-numeric token after a numeric run starts the next field, so
    side-by-side fields ('Lub oil ... 44.0 | Fuel press ... 7.10') separate
    cleanly without any per-line gap tuning.
    """
    ordered = sorted(tokens, key=lambda t: t.left)
    if not ordered:
        return []
    # Reference spacing = median of consecutive token gaps; a column break is
    # much larger (>= _FIELD_GAP_RATIO * reference) and far larger than a word bar.
    gaps = [b.left - a.right for a, b in zip(ordered, ordered[1:]) if (b.left - a.right) > 0]
    gaps_sorted = sorted(gaps) if gaps else []
    ref = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else float("inf")
    gap_threshold = max(ref * _FIELD_GAP_RATIO, _FIELD_GAP_MIN_PX) if gaps_sorted else float("inf")

    fields: List[Tuple[List[Token], List[Token]]] = []
    cur_label: List[Token] = []
    cur_value: List[Token] = []
    seen_value = False
    for prev, tok in zip([None] + ordered[:-1], ordered):
        is_num = _is_value_token(tok) or _is_value_artifact(tok.text)
        gap_break = (
            prev is not None
            and not is_num
            and (tok.left - prev.right) > gap_threshold
            and len(cur_label) > 0
        )
        if is_num:
            cur_value.append(tok)
            seen_value = True
        else:
            if seen_value or gap_break:
                # A unit token printed immediately after this field's value
                # (e.g. '°C' after '29.0;' in a dense side-by-side form) belongs
                # to THIS field, not the next. Absorb it into the label so that
                # _split_label_unit later surfaces it as the unit region; keep
                # scanning until a genuinely new label word (the next field)
                # closes the field and starts a fresh run.
                if (
                    seen_value
                    and cur_value
                    and _unit_shape_matches(tok.text)
                    and (tok.left - cur_value[-1].right) <= gap_threshold
                    and not gap_break
                ):
                    cur_label.append(tok)
                    continue
                # close the previous field; start a fresh label.
                fields.append((cur_label, cur_value))
                cur_label = [tok]
                cur_value = []
                seen_value = False
            else:
                cur_label.append(tok)
    # trailing field / a label with no value (e.g. form label above its value)
    if cur_label or cur_value:
        fields.append((cur_label, cur_value))
    return fields


def _split_label_unit(label_tokens: List[Token], unit_cx: Optional[float]) -> Tuple[List[Token], List[Token], List[Region]]:
    """Split a field's label tokens into (label, unit, qualifiers).

    The trailing token(s) that carry generic unit evidence are moved to the
    unit region; parenthesised / colon clauses become qualifiers. Generic and
    geometry-driven, never field-specific.
    """
    ordered = sorted(label_tokens, key=lambda t: t.left)
    unit_region_toks: List[Token] = []
    qual_regions: List[Region] = []
    kept: List[Token] = []

    # Collect every trailing unit-shaped token (rode from the right).
    tail = list(reversed(ordered))
    unit_idx = 0
    for idx, tok in enumerate(tail):
        txt = tok.text.strip()
        if txt.startswith("(") or txt.endswith(")") or txt.endswith(":"):
            break
        if txt in ("ALL", "SUM"):
            break
        is_unit = _unit_shape_matches(txt)
        if unit_cx is not None and abs(tok.cx - unit_cx) / max(tok.width, 1.0) < 1.5:
            is_unit = True
        if is_unit and (len(ordered) > 1 or len(unit_region_toks) == 0):
            unit_region_toks.append(tok)
            unit_idx = idx + 1
        else:
            break

    for idx, tok in enumerate(reversed(ordered)):
        if idx < unit_idx:
            continue
        txt = tok.text.strip()
        if txt.startswith("(") or txt.endswith(")") or txt.endswith(":"):
            qual_regions.append(Region(RegionType.QUALIFIER, [tok]))
            continue
        if txt in ("ALL", "SUM"):
            qual_regions.append(Region(RegionType.QUALIFIER, [tok]))
            continue
        kept.append(tok)

    kept.sort(key=lambda t: t.left)
    unit_region_toks.sort(key=lambda t: t.left)
    return kept, unit_region_toks, qual_regions


@dataclass
class _FieldRun:
    """One logical field found by segmenting a band into label:value blocks."""
    label_toks: List[Token]
    value_toks: List[Token]
    is_header: bool = False
    band_key: int = -1

    @property
    def y_min(self) -> int:
        return min((t.top for t in self.label_toks + self.value_toks), default=0)

    @property
    def y_max(self) -> int:
        return max((t.bottom for t in self.label_toks + self.value_toks), default=0)


def _is_genuine_label(label_toks: List[Token]) -> bool:
    """Is this a real descriptive field label (has a word of >=3 letters)?

    OCR gutter/table artifacts such as '(', '|®' or single letters produce
    label-like tokens that must not act as field owners. A genuine label
    carries at least one three-plus-letter alphabetic word ('ENGINE power',
    'Cooling water', ...)."""
    if not label_toks:
        return False
    for t in label_toks:
        for w in normalize_text(t.text).split():
            w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
            if re.search(r"[A-Za-z]", w) and len(w) >= 3:
                return True
    return False


def _has_alpha_word(label_toks: List[Token], min_len: int = 2) -> bool:
    """Does the label hold an alphabetic word of at least `min_len` letters?

    Used for the title-adoption guard so that short but REAL field labels
    (quadrant markers such as 'fp', 'fwd', 'aft', 'exh') are preserved instead
    of being mistaken for OCR junk; only a label genuinely lacking an
    alphabetic word (e.g. '(', '|®') is eligible to adopt a nearby title."""
    if not label_toks:
        return False
    for t in label_toks:
        for w in normalize_text(t.text).split():
            w = re.sub(r"[^A-Za-z]", "", w)
            if len(w) >= min_len:
                return True
    return False


def _is_date_like(text: str) -> bool:
    """Generic date/time recognition (e.g. 4/15/2025, 21:40, 15-DEC-26)."""
    t = text.strip()
    if not t:
        return False
    if re.match(r"^\d{1,4}[/\-:]\d{1,4}([/\-:]\d{1,4})?$", t):
        return True
    if re.match(r"^\d{1,4}[-/][A-Za-z]{3,}[-/]\d{1,4}$", t):
        return True
    return False


def _is_value_token(tok: Token) -> bool:
    """Is this token value-bearing (number, date, time)?"""
    t = tok.text.strip()
    if _looks_like_number(t):
        return True
    if _is_date_like(t):
        return True
    return False


def _is_data_bearing(band: Band) -> bool:
    """Does this band carry any field value regardless of its row_type label?"""
    return any(_is_value_token(t) for t in band.tokens)


def _is_grid(bands: List[Band], columns: List[Column]) -> bool:
    """Is a real, evenly-spaced value grid physically present?

    A single data row is insufficient evidence either way, so a region with
    one data-bearing band trusts the discovered header columns. Otherwise the
    region is a grid only if it has enough value columns that are each
    (a) narrow, (b) populated in several distinct rows, and (c) evenly spaced.
    Garbled pair/form regions (one value per field at field-specific positions)
    routinely fail one or more of these checks.

    Purely geometric; the schema does not influence this decision.
    """
    rows = [b for b in bands if _is_data_bearing(b)]
    if len(rows) < 2:
        return len(columns) >= 2

    # Fewer data rows give weaker evidence; scale the support thresholds down so
    # a real grid is still recognized when the region has few populated rows.
    # Wide column ranges are also normal for sparse few-column tables, so the
    # per-column width cap is relaxed when there are few rows.
    min_poly = 2 if len(rows) >= 3 else 1
    min_strong = 3 if len(rows) >= 4 else 2
    width_cap = _GRID_MAX_COL_WIDTH if len(rows) >= 4 else _GRID_MAX_COL_WIDTH_SPARSE

    strong: List[float] = []
    for col in columns:
        width = col.x_max - col.x_min
        if width > width_cap:
            continue
        half = width / 2.0
        col_cx = (col.x_min + col.x_max) / 2.0
        populated_rows: set = set()
        for r_i, b in enumerate(rows):
            vals = [t for t in b.tokens if _is_value_token(t)]
            if not vals:
                continue
            if min(abs(t.cx - col_cx) for t in vals) <= max(15.0, half + 15.0):
                populated_rows.add(r_i)
        if len(populated_rows) >= min_poly:
            strong.append(col_cx)

    if len(strong) < min_strong:
        return False
    strong.sort()
    gaps = [strong[i + 1] - strong[i] for i in range(len(strong) - 1)]
    if len(gaps) >= 2:
        mx = max(gaps)
        mn = min(gaps)
        if mn > 0 and mx / mn > _GRID_MAX_PITCH_RATIO:
            return False
    return True


def _row_bbox(run: _FieldRun) -> Tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) for a field run."""
    all_toks = run.label_toks + run.value_toks
    if not all_toks:
        return (0.0, float(run.y_min), 0.0, float(run.y_max))
    return (
        float(min(t.left for t in all_toks)),
        float(min(t.top for t in all_toks)),
        float(max(t.right for t in all_toks)),
        float(max(t.bottom for t in all_toks)),
    )


def _geometry_association_score(
    value_run: _FieldRun,
    label_run: _FieldRun,
    label_is_above: bool,
) -> float:
    """Score how well a value fragment associates with a candidate label row.

    Purely geometric: vertical overlap/proximity, horizontal alignment, and
    the label run's vertical extent relative to the value tokens.  No schema
    knowledge, no field IDs, no expected values.
    """
    if not value_run.value_toks or not label_run.label_toks:
        return 0.0

    vy_min = float(value_run.y_min)
    vy_max = float(value_run.y_max)
    ly_min = float(label_run.y_min)
    ly_max = float(label_run.y_max)

    # --- vertical overlap / proximity ---
    overlap_start = max(vy_min, ly_min)
    overlap_end = min(vy_max, ly_max)
    overlap_px = max(0.0, overlap_end - overlap_start)

    if overlap_px > 0:
        # Score proportional to how much of the value band is covered by the
        # label band's vertical extent.
        band_height = max(vy_max - vy_min, 1.0)
        vscore = min(1.0, overlap_px / band_height)
    else:
        # No overlap: penalise by distance.  0 px distance → 0.6, decaying to
        # 0 at _CONTINUATION_GAP_MAX.
        vdist = min(abs(vy_min - ly_max), abs(vy_max - ly_min))
        if vdist > _CONTINUATION_GAP_MAX:
            vscore = 0.0
        else:
            vscore = 0.6 * (1.0 - vdist / _CONTINUATION_GAP_MAX)

    # --- horizontal alignment ---
    # The value tokens should sit to the right of (or within) the label row's
    # horizontal extent.  Values left of the label are unlikely to belong to it.
    val_left = min(t.left for t in value_run.value_toks)
    lab_left = min(t.left for t in label_run.label_toks)
    lab_right = max(t.right for t in label_run.label_toks)

    if val_left >= lab_left - 30:
        hscore = 1.0
    elif val_left >= lab_right:
        hscore = 0.8
    else:
        hscore = 0.2

    return vscore * 0.7 + hscore * 0.3


def _is_value_dominated_run(run: _FieldRun) -> bool:
    """Is this field run value-dominated (no real descriptive label)?

    A run is value-dominated when its label tokens are garbled OCR artifacts
    (primarily non-alphabetic) rather than real descriptive words.  This is
    used to decide whether the run's values should be reassociated with a
    nearby genuine label row via 2D geometry.  Runs with genuine descriptive
    labels keep their own values — they are NOT orphans."""
    if not run.label_toks:
        return True
    for t in run.label_toks:
        for w in normalize_text(t.text).split():
            w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
            if not w:
                continue
            alpha_count = sum(1 for c in w if c.isalpha())
            if alpha_count >= 3 and alpha_count * 2 > len(w):
                return False
    return True


def _run_essentially_blank(run: _FieldRun) -> bool:
    """Does this run own NO real value yet?

    A run is effectively blank when it carries no measured value token at
    all -- no numbers, dates or times -- only decorative glyph artifacts
    ('|', '®, '()').  Such a run is a dangling field label still waiting for
    its value, so the value tokens of a closely following qualifier run
    belong to it.  A run holding a real value is an OWNED field row and never
    receives join values."""
    return not any(_is_value_token(t) for t in run.value_toks)


def _is_short_qualifier_label(run: _FieldRun) -> bool:
    """Is this run's label a short qualifier/positional tag, not a field name?

    Qualifier-style labels are at most two tokens and carry no word of five
    or more letters ('rel v', 'Eng. room', 'aft', 'Trim', 'STW', '2/E').  A
    word of five-plus letters ('Density', 'Fuel', 'pressure', 'Seawater')
    marks a genuine descriptive field label that MUST keep its own row; only
    labels failing that test may donate label + value to the field before
    them."""
    if not run.label_toks or len(run.label_toks) > 2:
        return False
    for t in run.label_toks:
        for w in normalize_text(t.text).split():
            w = re.sub(r"[^A-Za-z]", "", w)
            if len(w) >= 5:
                return False
    return True


def _runs_on_same_line(a: _FieldRun, b: _FieldRun) -> bool:
    """Do two runs share the same printed text line (vertical band overlap)?"""
    ys = [t.top for t in a.label_toks] + [t.bottom for t in a.label_toks]
    zs = [t.top for t in b.label_toks] + [t.bottom for t in b.label_toks]
    return bool(ys and zs and (min(zs) <= max(ys) and min(ys) <= max(zs)))


def _run_label_span(run: _FieldRun) -> float:
    xs = [t.left for t in run.label_toks]
    xe = [t.right for t in run.label_toks]
    if not xs:
        return 0.0
    return max(xe) - min(xs)


def _merge_side_by_side_form_values(runs: List[_FieldRun]) -> List[_FieldRun]:
    """Generic dense side-by-side form segmentation / value association.

    In a non-grid form line the gap-based field-run splitter can cut one
    logical field into a label-only run followed by a tiny qualifier-style
    run that happens to carry the value::

        'Ambient humidity'  |  'rel v 32.0%'
        'Barom. pr. |'      |  'Eng. room 1.010 @ 1.013'

    When a run whose label is a SHORT QUALIFIER (<=2 tokens, no 5+-letter
    word) carries value tokens and the field run to its LEFT on the same
    printed line holds no real value, the field name and the value belong
    together: the qualifier label and its value tokens are absorbed back into
    the left run.

    Genuine side-by-side fields never merge: a run label with a 5+-letter
    word ('Density FLOWM. kg/m?') keeps its own row, and a left neighbour
    that already holds a real value ('Specific FOC 166.2 | Sulfur content
    0.50') is not a dangling label.  A large horizontal gap does not by
    itself force a merge.  Purely structural/geometric: no field IDs, no
    vocabulary, no schema.
    """
    if len(runs) < 2:
        return runs
    # In bands where multiple genuine labels share one visual line (a multi-field
    # layout like 'Absolute consumption | Target feed rate@MCR | BN 40'), only
    # the first genuine label is the real field owner; later blank genuine labels
    # are column headers / qualifiers for values already present.  Limiting the
    # qualifier-merge to the first genuine label per band prevents stealing a
    # value from its correct orphan-association owner.
    first_genuine_in_band: dict = {}
    for r in runs:
        bk = r.band_key
        if bk not in first_genuine_in_band and r.label_toks and _is_genuine_label(r.label_toks):
            first_genuine_in_band[bk] = id(r)
    compact: List[_FieldRun] = [runs[0]]
    for run in runs[1:]:
        owner = compact[-1]
        # A trailing unit fragment printed on the same line ('barA', '°C',
        # 'm', 'kn') is not a field; it qualifies the field to its left and is
        # folded into it (surfaced later as the row's unit region).
        unit_fold = (
            owner.band_key == run.band_key
            and owner.label_toks
            and not run.value_toks
            and bool(run.label_toks)
            and all(_unit_shape_matches(t.text) for t in run.label_toks)
            and _runs_on_same_line(owner, run)
        )
        if unit_fold:
            owner.label_toks = sorted(
                owner.label_toks + run.label_toks, key=lambda t: t.left
            )
            continue
        merge = (
            owner.band_key == run.band_key
            and owner.label_toks
            and _is_genuine_label(owner.label_toks)
            and _run_essentially_blank(owner)
            and run.value_toks
            and _is_short_qualifier_label(run)
            and _runs_on_same_line(owner, run)
            and id(owner) == first_genuine_in_band.get(owner.band_key)
        )
        if merge:
            gap = (
                min(t.left for t in run.label_toks)
                - max(t.right for t in owner.label_toks)
            )
            if gap <= max(150.0, 1.5 * _run_label_span(owner)):
                owner.label_toks = sorted(
                    owner.label_toks + run.label_toks, key=lambda t: t.left
                )
                owner.value_toks = owner.value_toks + run.value_toks
                continue
        compact.append(run)
    return compact


def _associate_orphan_values(runs: List[_FieldRun]) -> List[bool]:
    """2D-geometry-based orphan value association.

    For every value-dominated run (garbled or absent label), evaluate the
    nearest compatible label row ABOVE and the nearest compatible row BELOW.
    Reassign the values only when one candidate scores significantly higher
    than the other.  If the evidence is ambiguous, preserve the values as-is
    (OCR ambiguity).

    Runs that already carry a genuine descriptive label keep their own values;
    they are not orphans and must NOT be reassigned.

    Returns a parallel list of booleans: True means the run's values were
    reassigned and the run should be skipped when building PhysicalRows.
    """
    bound = [False] * len(runs)

    # Pre-compute which runs are genuine-label rows (candidates for receiving
    # values).  A run with a genuine label is a valid target; a run without
    # one is an orphan that should not donate its own values to itself.
    label_indices = [i for i, r in enumerate(runs) if _is_genuine_label(r.label_toks)]

    for i, run in enumerate(runs):
        if not run.value_toks:
            continue

        # Safety: runs with a genuine descriptive label already own their values.
        # Only value-dominated runs (garbled/no label) are orphans eligible for
        # reassociation.  This prevents moving a value away from its real label.
        if not _is_value_dominated_run(run):
            continue

        best_above_idx: Optional[int] = None
        best_above_score = 0.0
        best_below_idx: Optional[int] = None
        best_below_score = 0.0

        for li in label_indices:
            if li == i:
                continue
            lrun = runs[li]
            score = _geometry_association_score(run, lrun, li < i)
            if li < i:
                if score > best_above_score:
                    best_above_score = score
                    best_above_idx = li
            else:
                if score > best_below_score:
                    best_below_score = score
                    best_below_idx = li

        # Safety rule: choose only when one candidate is clearly superior.
        chosen: Optional[int] = None
        if best_above_idx is not None and best_below_idx is not None:
            if best_above_score > best_below_score * 1.5 and best_above_score >= 0.5:
                chosen = best_above_idx
            elif best_below_score > best_above_score * 1.5 and best_below_score >= 0.5:
                chosen = best_below_idx
            # else: ambiguous — do NOT reassign
        elif best_above_idx is not None and best_above_score >= 0.5:
            chosen = best_above_idx
        elif best_below_idx is not None and best_below_score >= 0.5:
            chosen = best_below_idx

        if chosen is not None and chosen != i:
            runs[chosen].value_toks.extend(run.value_toks)
            run.value_toks = []
            bound[i] = True

    return bound


def _field_run_rows(bands: List[Band], columns: List[Column], is_grid: bool) -> List[PhysicalRow]:
    """
    Segment report bands into label:value field runs with vertical
    continuation, then lift each run into a RowModel.

    A run is created when a label block is present; a pure-value band (or run)
    directly after a labelled run continues that field (its values are appended
    to the pending row). This yields one RowModel per printed logical field and
    correctly re-joins a label printed on its own line above the value line.
    """
    # All non-decorative bands participate so a label-only band (the field
    # name on its own line) can still own the value band beneath it. 'title'
    # bands are included too: many reports print a field's full label (e.g.
    # 'LINER WALL temp. °C') as a title-like line above the line that carries
    # its numeric values, so a title band is a legitimate continuation owner.
    
    # Compute grid_left from data bands for header band filtering
    data_bands = [b for b in bands if b.row_type == "data"]
    grid_left = _compute_grid_left(data_bands)
    
    # Include non-separator bands, but exclude pure column header bands
    # (header bands with no label-zone tokens) unless they are label-only
    # field bands detected by _is_label_band_field.
    def _include_band(band: Band) -> bool:
        if band.row_type == "separator":
            return False
        if band.row_type != "header":
            return True
        # Header band: use _is_label_band_field to decide if it's a genuine
        # field label band. This function checks for genuine descriptive labels
        # in the label zone, not just column headers.
        if grid_left is not None and _is_label_band_field(band, grid_left):
            return True
        # Also include title bands that might be field labels
        if band.row_type == "title" and grid_left is not None:
            label_zone_tokens = [t for t in band.tokens if t.cx < grid_left]
            if label_zone_tokens and _is_genuine_label(label_zone_tokens):
                return True
        # If no data bands (grid_left is None), include header bands -
        # they might be the only content (e.g., header-only pages).
        if grid_left is None:
            return True
        return False
    
    source_bands = sorted(
        (b for b in bands if _include_band(b)),
        key=lambda b: b.y_min,
    )
    has_data_bands = any(b.row_type == "data" for b in source_bands)

    # Pre-scan header bands to identify repeating column-header label texts.
    # A genuine column header (e.g. "REF", "CALC", "CYL1") appears as a
    # label-only run in 2+ header bands.  Field labels are unique — they
    # appear in exactly one band.  This structural signal replaces the old
    # crude all-alphabetic heuristic that incorrectly dropped real field
    # labels (e.g. "Lub oil temp. TURB. OIL in") in side-by-side form
    # regions like engine_media_plant.
    _header_band_label_counts: Dict[str, int] = {}
    for _hb in source_bands:
        if _hb.row_type != "header":
            continue
        for _ltoks, _vtoks in _field_runs_in_band(_hb.tokens):
            if not _vtoks and _ltoks:
                _key = ' '.join(
                    t.text.strip().lower()
                    for t in sorted(_ltoks, key=lambda t: t.left)
                )
                _header_band_label_counts[_key] = _header_band_label_counts.get(_key, 0) + 1
    # Only labels that repeat in 2+ header bands are column headers.
    _repeating_header_labels = {
        text for text, cnt in _header_band_label_counts.items() if cnt >= 2
    }

    runs: List[_FieldRun] = []
    pending: Optional[_FieldRun] = None
    for band_key, band in enumerate(source_bands):
        band_is_header = band.row_type == "header"
        for label_toks, value_toks in _field_runs_in_band(band.tokens):
            # Drop a label-only run in a header band only if it is a
            # *repeating column header*: the same short label text appears
            # as a label-only run in 2+ different header bands.  Genuine
            # field labels (unique to one band) and section headers that
            # happen to be all-alpha are preserved so the field matcher
            # can pair them with their values.
            if band_is_header and has_data_bands and not value_toks and label_toks:
                label_text = ' '.join(
                    t.text.strip().lower()
                    for t in sorted(label_toks, key=lambda t: t.left)
                )
                if label_text in _repeating_header_labels:
                    continue
            if label_toks:
                pending = _FieldRun(label_toks=label_toks, value_toks=list(value_toks),
                                    is_header=band_is_header, band_key=band_key)
                runs.append(pending)
            elif value_toks:
                # A value run without its own label: vertical continuation of the
                # immediately-preceding real field label (label on its own line
                # above the value line). Only a genuine field label owns a
                # continuation, never a column-header band or a well-separated row.
                if pending is not None and not pending.is_header:
                    pending.value_toks.extend(value_toks)
                else:
                    runs.append(_FieldRun(label_toks=[], value_toks=list(value_toks),
                                          band_key=band_key))

    # In a form/non-grid region a horizontal form row's label and its trailing
    # text value are frequently split by the gap-based run segmentation into
    # consecutive label-only runs of the SAME printed line (e.g. 'ENGINE state'
    # followed by 'stable').  Rejoin such short label-only tails onto the
    # preceding run so the field's text datum stays with its label.  Columnar
    # regions are untouched: value-carrying runs never merge, and grid rows
    # keep their exact label/qualifier split.
    if not is_grid and len(runs) > 1:
        compact: List[_FieldRun] = [runs[0]]
        for run in runs[1:]:
            prev = compact[-1]
            prev_ys = [t.top for t in prev.label_toks] + [t.bottom for t in prev.label_toks]
            run_ys = [t.top for t in run.label_toks] + [t.bottom for t in run.label_toks]
            same_line = prev_ys and run_ys and (
                min(run_ys) <= max(prev_ys) and min(prev_ys) <= max(run_ys)
            )
            if (
                not is_grid
                and prev.band_key == run.band_key
                and not prev.value_toks
                and not run.value_toks
                and prev.label_toks
                and 0 < len(run.label_toks) <= 3
                and same_line
            ):
                prev.label_toks.extend(run.label_toks)
            else:
                compact.append(run)
        runs = compact

    # Dense side-by-side value association: a dangling field label followed on
    # the same printed line by a tiny qualifier run that carries the value
    # ('Ambient humidity' | 'rel v 32.0%') regains its value.  Grid regions
    # keep their exact per-column row composition.
    if not is_grid and len(runs) > 1:
        runs = _merge_side_by_side_form_values(runs)

    # 2D geometry-based orphan value association.  For every run carrying value
    # tokens, evaluate the best compatible label row above AND below using
    # vertical overlap/proximity and horizontal alignment.  Reassign only when
    # one candidate is clearly superior; otherwise preserve as OCR ambiguity.
    bound = _associate_orphan_values(runs)

    out: List[PhysicalRow] = []
    for ri, run in enumerate(runs):
        if bound[ri]:
            continue
        # A header band label-only run is normally a column header ("REF CALC
        # AVG CYL1..CYL10"), not a data field. But it can also be a real field
        # whose value row is blank/OCR-missing (e.g. 'Lub oil press. COOLING
        # OIL' with no value). Keeping it as a blank row lets the adapter emit
        # an honest SOURCE_BLANK; a true column header matches no schema field
        # and is dropped downstream, so no guard is needed here.
        kept, unit_toks, qualifiers = _split_label_unit(run.label_toks, None)
        if not kept and unit_toks:
            kept = unit_toks
            unit_toks = []
        if not kept and not run.value_toks:
            continue
        label_region = Region(RegionType.LABEL, tokens=kept)
        unit_region = Region(RegionType.UNIT, tokens=unit_toks)
        allb = list(kept) + unit_toks + run.value_toks
        ys = [t.top for t in allb] + [t.bottom for t in allb]
        y_min = min(ys) if ys else run.y_min
        y_max = max(ys) if ys else run.y_max
        xmin = min((t.left for t in allb), default=0)
        xmax = max((t.right for t in allb), default=0)

        if is_grid:
            cells = _grid_cells(ri, run.value_toks, columns)
        else:
            cells = [_cell_for_value(ri, columns[0], run.value_toks)]

        # A label-only run (no value tokens) carries its field name but no
        # value.  The adapter may extract a text value from the label zone
        # when the schema field allows text values (e.g. "Running mode
        # Economy" -> label="Running mode", text_value="Economy").
        is_label_band = len(run.value_toks) == 0 and _is_genuine_label(run.label_toks)

        status_meta: Dict[str, Any] = {}
        if is_label_band:
            status_meta["label_band"] = True

        model = RowModel(
            row_idx=ri,
            bbox=(float(xmin), float(y_min), float(xmax), float(y_max)),
            label_region=label_region,
            unit_region=unit_region,
            qualifier_regions=qualifiers,
            cells=cells,
            status_metadata=status_meta,
        )
        out.append(PhysicalRow(ri, float(y_min), float(y_max), "data", model))
    return out


def _cell_for_value(row_idx: int, col: Column, value_toks: List[Token]) -> Cell:
    """Build a populated (or blank) cell for a value token set under a column."""
    return Cell(row_idx=row_idx, column=col, tokens=list(value_toks),
                is_blank=not value_toks)


def _grid_cells(row_idx: int, value_toks: List[Token], columns: List[Column]) -> List[Cell]:
    """Assign each value token to its nearest grid column (blank preserved).

    Nearest-column-center assignment keeps every value accounted for even when
    an OCR token drifts slightly off its column's strict x-range.
    """
    if not columns:
        return [] if not columns else [_cell_for_value(row_idx, columns[0], value_toks)]
    centers = [((c.x_min + c.x_max) / 2.0, c) for c in columns]
    placed: List[List[Token]] = [[] for _ in columns]
    for t in value_toks:
        idx = min(range(len(centers)), key=lambda i: abs(centers[i][0] - t.cx))
        placed[idx].append(t)
    return [_cell_for_value(row_idx, c, placed[i]) for i, (_, c) in enumerate(centers)]


# ───────────────────────────────────────────────────────────
# Whole-page multi-zone decomposition
# ───────────────────────────────────────────────────────────
#
# A whole-page OCR token stream may contain multiple structural zones
# (e.g. a pair/form section, a cylinder-pressure grid, a scavenge-air
# grid).  ``discover_zones`` splits the token list at zone boundaries
# so that each zone can be independently reconstructed.  Zone detection
# uses only geometric evidence: vertical gaps between token bands,
# changes in horizontal token extent, and the presence of grid-column
# header keywords (REF / CALC / AVG / …) that mark the start of a
# new columnar section.
#
# ``reconstruct_zones`` orchestrates the per-zone reconstruction and
# aggregates the results into a single ``ReconstructionResult`` whose
# physical rows carry the correct page coordinates.

# Grid-column header keywords – used to detect the top of a new grid
# section.  The check is case-insensitive and tolerates trailing dots.
_GRID_HEADER_RE = re.compile(
    r'^(?:REF\.?|CALC\.?|AVG\.?|MEASURED\.?|TC[0-9])$', re.IGNORECASE
)
_EXTENT_RATIO = 0.60


def _gap_threshold(tokens: List[Token]) -> float:
    """Scale-relative gap threshold for zone boundary detection.

    Returns 1.95× the band-clustering tolerance, which is derived from the
    median token height.  At 250 DPI (median height ≈ 16 px, tolerance ≈ 11 px)
    this yields ≈ 22 px — above typical intra-section row gaps (0-20 px) and
    at/below section-boundary gaps (≥22 px) on the validated Template-B pages.

    Because the threshold is proportional to median token height it scales
    automatically with DPI / page size.  The 1.95× multiplier was validated
    against both Template-B pages to separate section boundaries from
    ordinary intra-section row spacing.

    Section boundaries are only triggered when a structural gap exceeds this
    threshold AND is confirmed by grid-column header keywords (lookahead or
    direct) or a horizontal extent change — preventing ordinary intra-section
    gaps from creating spurious zone splits.
    """
    return _band_tolerance(tokens) * 1.95


def _band_x_extent(band: Band) -> float:
    """Rightmost x-coordinate of a band's tokens."""
    return max(t.right for t in band.tokens) if band.tokens else 0.0


def _grid_keyword_count(band: Band) -> int:
    """Count of grid-column header keywords in a band."""
    count = 0
    for t in band.tokens:
        if _GRID_HEADER_RE.match(t.text.strip()):
            count += 1
    return count


def _is_grid_header_band(band: Band) -> bool:
    """True when a band contains ≥2 grid-column header keywords.

    A real column-header row (REF. CALC. AVG. CYL1 …) always carries
    multiple grid keywords in the same band.  This stricter check is used
    for lookahead propagation so that metadata title rows (single "MEASURED")
    are reached only through proximity to the real header.
    """
    return _grid_keyword_count(band) >= 2


def _has_grid_keyword(band: Band) -> bool:
    """True when a band contains ≥1 grid-column header keyword.

    Used for direct zone-boundary detection after a structural gap.
    A single keyword such as "MEASURED" in a metadata title row can
    participate in a zone boundary when it follows a significant gap;
    the gap itself is the primary structural evidence and the keyword
    merely confirms the boundary is a section transition.
    """
    return _grid_keyword_count(band) >= 1


def discover_zones(tokens: List[Token]) -> List[List[Token]]:
    """Split a whole-page token stream into spatial structural zones.

    Returns a list of token lists, one per zone.  Each zone contains
    tokens from a contiguous vertical span of the page.  Zone boundaries
    are determined by:

    * vertical gaps between token bands that exceed a scale-relative
      threshold (≈ 2× band tolerance),
    * changes in horizontal token extent (a shrink to < 60 % of the
      previous wide band's extent after a gap signals a narrower zone),
    * proximity to a multi-keyword grid-column header band (REF+CALC,
      REF+CALC+AVG, …) that appears within a few bands after the gap,
      marking the top of a new columnar section.

    When no boundaries are detected the entire token list is returned as
    a single zone.  The function is purely geometric and does not depend
    on field names, filenames, page numbers, or expected values.
    """
    if not tokens:
        return []

    tol = _band_tolerance(tokens)
    gap_thr = _gap_threshold(tokens)
    bands = cluster_bands(tokens, tolerance=tol)

    n = len(bands)

    # Pre-compute: for each band index, does it or any of the next 2 bands
    # contain a MULTI-keyword grid header (≥2 keywords)?  Propagation
    # reaches metadata title rows (single "MEASURED") that sit immediately
    # above the real column-header row, so those rows participate in zone
    # boundaries through proximity rather than through their own keyword
    # count.
    lookahead_grid: List[bool] = [False] * n
    for i in range(n - 1, -1, -1):
        if _is_grid_header_band(bands[i]):
            lookahead_grid[i] = True
        elif i + 1 < n and lookahead_grid[i + 1]:
            # Only propagate if bands[i+1] is close (no large gap between them).
            gap_next = bands[i + 1].y_min - bands[i].y_max
            if gap_next < gap_thr:
                lookahead_grid[i] = True

    zones: List[List[Token]] = []
    current_tokens: List[Token] = []
    prev_x_max = 0.0
    prev_y_max = 0.0

    for bi, band in enumerate(bands):
        if not band.tokens:
            continue

        x_max = _band_x_extent(band)
        gap = band.y_min - prev_y_max if prev_y_max > 0 else 0.0
        lookahead = lookahead_grid[bi]
        direct_header = _has_grid_keyword(band)
        extent_drop = (
            prev_x_max > 0 and x_max < prev_x_max * _EXTENT_RATIO
        )

        # Decide whether to start a new zone.
        start_new_zone = False
        if current_tokens and gap >= gap_thr:
            if lookahead or direct_header or extent_drop:
                start_new_zone = True
            elif gap >= gap_thr * 2:
                # Very large gap without header – also split.
                start_new_zone = True

        if start_new_zone and current_tokens:
            zones.append(current_tokens)
            current_tokens = []

        current_tokens.extend(band.tokens)
        prev_x_max = x_max
        prev_y_max = band.y_max

    if current_tokens:
        zones.append(current_tokens)

    return zones if len(zones) > 1 else [tokens]


class ZoneResult:
    """Result of reconstructing a single spatial zone."""

    __slots__ = ('zone_idx', 'page', 'y_min', 'y_max',
                 'physical_rows', 'header_bands', 'leaf_columns',
                 'header_hierarchy', 'tokens')

    def __init__(self, zone_idx: int, page: int, y_min: float, y_max: float,
                 physical_rows: List[PhysicalRow],
                 header_bands: List[Band],
                 leaf_columns: List[Column],
                 header_hierarchy: List[HeaderNode],
                 tokens: List[Token]):
        self.zone_idx = zone_idx
        self.page = page
        self.y_min = y_min
        self.y_max = y_max
        self.physical_rows = physical_rows
        self.header_bands = header_bands
        self.leaf_columns = leaf_columns
        self.header_hierarchy = header_hierarchy
        self.tokens = tokens


def reconstruct_zones(
    raw_tokens: List[dict],
    band_tolerance: Optional[float] = None,
    min_col_width: float = 10.0,
) -> Tuple[ReconstructionResult, List[ZoneResult]]:
    """Multi-zone reconstruction for whole-page OCR token streams.

    Splits the input tokens into spatial zones via ``discover_zones``,
    runs ``reconstruct`` independently on each zone, and aggregates the
    results into a single ``ReconstructionResult``.

    Returns ``(merged_result, zone_results)`` where *zone_results* carries
    per-zone detail (rows, columns, bboxes) for adapter integration.
    """
    all_tokens = ingest_tokens(raw_tokens)
    zones = discover_zones(all_tokens)

    zone_results: List[ZoneResult] = []
    all_rows: List[PhysicalRow] = []
    all_header_bands: List[Band] = []
    all_columns: List[Column] = []
    all_hierarchy: List[HeaderNode] = []
    all_bands: List[Band] = []

    row_offset = 0
    for zi, zone_tokens in enumerate(zones):
        zone_dicts = [
            {
                'text': t.text, 'left': t.left, 'top': t.top,
                'width': t.width, 'height': t.height,
                'conf': t.conf, 'page': t.page, 'region': t.region,
            }
            for t in zone_tokens
        ]
        zone_result = reconstruct(zone_dicts, band_tolerance=band_tolerance,
                                  min_col_width=min_col_width)

        # Adjust row indices to be contiguous across zones.
        for pr in zone_result.physical_rows:
            pr.row_idx += row_offset
            if pr.model is not None:
                pr.model.row_idx += row_offset

        zone_page = zone_tokens[0].page if zone_tokens else 0
        zone_y_min = min(t.top for t in zone_tokens) if zone_tokens else 0
        zone_y_max = max(t.bottom for t in zone_tokens) if zone_tokens else 0

        zone_results.append(ZoneResult(
            zone_idx=zi,
            page=zone_page,
            y_min=zone_y_min,
            y_max=zone_y_max,
            physical_rows=zone_result.physical_rows,
            header_bands=zone_result.header_bands,
            leaf_columns=zone_result.leaf_columns,
            header_hierarchy=zone_result.header_hierarchy,
            tokens=zone_tokens,
        ))

        all_rows.extend(zone_result.physical_rows)
        all_header_bands.extend(zone_result.header_bands)
        all_bands.extend(zone_result.bands)
        all_hierarchy.extend(zone_result.header_hierarchy)

        # Columns: merge by col_id, keeping unique.
        existing_ids = {c.col_id for c in all_columns}
        for c in zone_result.leaf_columns:
            if c.col_id not in existing_ids:
                all_columns.append(c)
                existing_ids.add(c.col_id)

        row_offset += len(zone_result.physical_rows)

    merged = ReconstructionResult(
        tokens=all_tokens,
        bands=all_bands,
        physical_rows=all_rows,
        header_bands=all_header_bands,
        header_hierarchy=all_hierarchy,
        leaf_columns=all_columns,
        status=ReconstructionStatus(ok=True, warnings=[]),
        debug={"zone_count": len(zones)},
    )
    return merged, zone_results
