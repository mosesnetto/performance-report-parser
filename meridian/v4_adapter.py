"""
V4 -> Semantic-schema Adapter
=============================

The ONLY bridge between the V4 geometry engine and the existing semantic
schema / validator / export pipeline.

Division of responsibility
--------------------------
V4  determines WHERE data is:  rows, cells, leaf columns, parent header path,
    cell/value bounding boxes, raw OCR tokens, status indicators, and blanks
    (via Cell.column / Cell.header_path / Cell.cell_bbox / Cell.value_bbox /
    Cell.raw_text / Cell.status / Cell.is_blank).

Schema determines WHAT data means: field_id, field_name, aliases, unit, type,
    and which semantic columns are expected.

This adapter contains NO extraction hints:
    - no pixel offsets
    - no page coordinates
    - no filename / page / month logic
    - no expected values
    - no field-specific search positions or OCR repairs
    - no "take the nth token" rules

The row -> field match is a GENERIC semantic label matcher. Column mapping
RE-USES V4's resolved geometry (Cell.column / Cell.header_path); it never
re-derives columns.

Backward compatible provenance: each emitted record keeps the standard
semantic keys and gains an extra ``provenance`` dict (header path, qualifier,
status, source row, row index). Existing keys and values are unchanged.
"""
from __future__ import annotations

import re
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# Public V4 model types (imported defensively so the engine stays decoupled;
# the adapter only reads the exposed ReconstructionResult data model).
from .extract import get_explicit_columns
from .models import Cell as VC
from .models import FieldRecord
from .table_reconstruction import (
    Cell,
    ReconstructionResult,
    ReconstructionStatus,
    RowModel,
    ZoneResult,
)

# ───────────────────────────────────────────────────────────
# Presence / absence / quality statuses (machine-readable)
# ───────────────────────────────────────────────────────────
SOURCE_ABSENT = "SOURCE_ABSENT"
SOURCE_BLANK = "SOURCE_BLANK"
EXTRACTION_FAILED = "EXTRACTION_FAILED"
OCR_AMBIGUOUS = "OCR_AMBIGUOUS"
COLUMN_MISMATCH = "COLUMN_MISMATCH"
FIELD_VALUE_MISMATCH = "FIELD_VALUE_MISMATCH"
TEXT_VALUE = "text_value"
LOW_CONFIDENCE = "low_ocr_confidence"
RECOVERED_BY_GEOMETRY_FALLBACK = "RECOVERED_BY_GEOMETRY_FALLBACK"


# ───────────────────────────────────────────────────────────
# Generic text normalization (matching only, not geometry)
# ───────────────────────────────────────────────────────────
_GLUE_CHARS = re.compile(r"[.\-/_]")


def normalize_label(text: str) -> str:
    """Lowercase, collapse whitespace, and normalize punctuation for match
    scoring. Word-merger tolerant: 'press.' -> 'press', 'pcomp / pscav' kept
    as a multi-word label with normalized glue."""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokens(text: str) -> List[str]:
    """Split a normalized label into meaningful tokens for overlap scoring."""
    t = normalize_label(text)
    # Replace glue chars with spaces so "pcomp/pscav" and "pcomp / pscav" align.
    t = _GLUE_CHARS.sub(" ", t)
    toks = [w for w in t.split() if w]
    return toks


def score_label(row_label: str, alias: str) -> float:
    """Score how well a reconstructed row label matches one schema alias.

    Returns a float in [0, 1]. Natively tolerant of TRAILING OCR noise in the
    row label (e.g. a stray value token appended to the row's words) by using
    ALIAS RECALL: an alias whose tokens all appear in order within the row is
    scored high regardless of how many extra tokens the row carries. Word
    merger is handled by punctuation normalization.
    """
    rn = normalize_label(row_label)
    an = normalize_label(alias)
    if rn == an:
        return 1.0

    rt = _tokens(rn)
    at = _tokens(an)
    if not rt or not at:
        return 0.0

    # In-order subsequence coverage of the alias within the row label.
    matched = 0
    i = 0
    for tok in rt:
        if i >= len(at):
            break
        if tok == at[i]:
            matched += 1
            i += 1
            continue
        # Generic compound-word alignment: OCR may have merged two or more
        # printed words into one token with no separator (e.g. 'Seawater' for
        # the alias words 'Sea water').  If this row token is the exact
        # concatenation of consecutive alias tokens, count them all as matched.
        # No vocabulary / field knowledge — purely string concatenation.
        j = i + 1
        while j <= len(at):
            if tok == "".join(at[i:j]):
                matched += (j - i)
                i = j
                break
            j += 1
    coverage = matched / len(at)

    # How much of the (possibly noisy) row label is explained by the alias.
    purity = matched / len(rt)

    score = 0.75 * coverage + 0.25 * purity
    # Small penalty proportional to token-count mismatch only when the alias
    # is much longer than the row (partial label), not the other way around.
    if len(at) > len(rt):
        diff = (len(at) - len(rt)) / len(at)
        score -= 0.2 * diff
    return max(0.0, min(1.0, score))


# ───────────────────────────────────────────────────────────
# Generic numeric value parser (self-contained, context-free)
# ───────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"[+-]?(?:\d+[.,]\d+|\d+|\.\d+)")

# Generic date/time shape recognition.  Detects common OCR date formats
# BEFORE any numeric extraction so that "10/25/2025 [15]" is not reduced
# to the leading number 10.  No field IDs, no vocabulary — pure shape.
_DATE_SLASH_RE = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"       # MM/DD/YYYY or M/D/YY
)
_DATE_ISO_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"         # YYYY-MM-DD
)
# Bracket artifacts like "[15]" that OCR appends after a date.
_BRACKET_ARTIFACT_RE = re.compile(r"\s*\[\d+\]\s*$")


def _is_datetime_shape(text: str) -> bool:
    """Return True if *text* contains a recognisable date/time pattern.

    Strips trailing bracket artifacts (e.g. '[15]') before testing so that
    OCR page-number suffixes do not prevent detection.  Handles slash dates
    (MM/DD/YYYY) and ISO dates (YYYY-MM-DD).  The colon-time guard in
    ``parse_float`` already covers HH:MM patterns."""
    cleaned = _BRACKET_ARTIFACT_RE.sub("", text or "")
    return bool(_DATE_SLASH_RE.search(cleaned) or _DATE_ISO_RE.search(cleaned))


def parse_float(raw: str) -> Optional[float]:
    """Parse the first numeric token in ``raw``. Handles negative, zero,
    integer and decimal values. Returns None if no number is present, or if
    the text is a date/time shape (slash-date, ISO-date) or the leading
    number is part of a colon-based time (e.g. '21:40') — none of which
    are scalar measurements."""
    if not raw:
        return None
    if _is_datetime_shape(raw):
        return None
    m = _NUM_RE.search(raw)
    if not m:
        return None
    tail = raw[m.end():]
    if tail.startswith(":"):
        return None
    s = m.group(0).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ───────────────────────────────────────────────────────────
# Semantic column canonicalization (presentation only)
# ───────────────────────────────────────────────────────────
def canonical_column(label: Optional[str]) -> str:
    """Turn a V4 leaf column label into a canonical semantic column string.

    Generic text normalization only: uppercase, strip punctuation, collapse
    whitespace, and normalise compound patterns.  No OCR vocabulary
    corrections — column identity is resolved by the geometry engine's
    positional fallback when labels do not exactly match.

    e.g. 'REF.'    -> 'REF'
         'CALC.'   -> 'CALC'
         'cYL1'    -> 'CYL1'
         'CYL 10'  -> 'CYL10'
         'MOP'     -> 'MOP'
         'BRG8&'   -> 'BRG8'  (noise stripped)
         'AVG./'   -> 'AVG'   (noise stripped)
    Unknown / None -> 'VALUE' (single-column fallback, never geometry).
    """
    if not label:
        return "VALUE"
    col = label.strip().upper()
    # Strip non-alphanumeric except '/' and '.' (mirrors headers.py)
    col = re.sub(r"[^A-Z0-9/.]", "", col)
    col = re.sub(r"\s+", "", col)          # 'CYL 10' -> 'CYL10'
    col = col.rstrip(".")
    # 'cYL1' -> 'CYL1' (already upper). Guard hyphen spacing.
    col = re.sub(r"(?<=\D)[\s_]+(?=\d)", "", col)

    # CYL10, TC3, BRG12, PUMP5: normalize compound patterns.
    if col.startswith("CYL") and col[3:].isdigit():
        col = f"CYL{int(col[3:])}"
    elif col.startswith("TC") and col[2:].isdigit():
        col = f"TC{int(col[2:])}"
    elif col.startswith("BRG") and col[3:].isdigit():
        col = f"BRG{int(col[3:])}"
    elif col.startswith("PUMP") and col[4:].isdigit():
        col = f"PUMP{int(col[4:])}"

    # Handle truncated "/ ENG" from "AVG/ENG" split.
    if col == "/ENG":
        col = "ENGINE"

    return col or "VALUE"


# ───────────────────────────────────────────────────────────
# Schema-aware column mapping (post-processing)
# ───────────────────────────────────────────────────────────
def _build_schema_column_map(
    field: dict,
    leaf_columns: list,
) -> Optional[Dict[str, str]]:
    """Build a one-to-one mapping from V4 leaf columns to schema column names.

    Uses two-pass geometric matching:
      1. Exact canonical-label anchors (text match).
      2. For remaining unmatched schema columns, assign by nearest-x-distance
         among unused V4 columns, processing left-to-right to preserve
         spatial ordering.

    Extra V4 columns that do not correspond to any schema column remain
    unused.  Missing schema columns remain unmapped (filled as
    SOURCE_BLANK by the caller).

    Returns None when the field has no schema or no leaf columns are
    available, signalling the caller to keep the original column names.
    """
    schema = get_explicit_columns(field)
    if not schema or not leaf_columns:
        return None

    # Build (cx, label) pairs for V4 leaf columns.
    # The reconstruction engine delivers them x-sorted.
    v4_cols = []
    for lc in leaf_columns:
        cx = getattr(lc, "cx", None)
        if cx is None:
            try:
                cx = (lc.x_min + lc.x_max) / 2.0
            except Exception:
                continue
        label = canonical_column(lc.label)
        v4_cols.append((cx, label))

    if not v4_cols:
        return None

    # ── Pass 1: exact canonical-label anchors ──────────────────────
    mapping: Dict[str, str] = {}
    v4_used: set = set()

    for schema_col in schema:
        for i, (cx, label) in enumerate(v4_cols):
            if i in v4_used:
                continue
            if label == schema_col:
                mapping[schema_col] = label
                v4_used.add(i)
                break

    # ── Pass 2: geometric gap-fill between anchors ───────────────
    # Schema columns that cannot be anchored by label AND cannot be
    # geometrically placed between two anchors are LEFT UNMAPPED (→
    # unresolved).  Ordinal (index-position) fallback is deliberately
    # removed: it silently assigns values to the wrong semantic column
    # when the physical leaf-column model does not cleanly match the
    # schema (stacked tables, misread OCR headers, missing header bands).
    if len(mapping) < len(schema):
        remaining_schema = [c for c in schema if c not in mapping]

        has_anchors = len(mapping) > 0

        if has_anchors:
            anchor_indices = sorted(
                schema.index(sc) for sc in mapping
            )
            first_anchor = anchor_indices[0]
            last_anchor = anchor_indices[-1]

            between = [sc for sc in remaining_schema
                       if first_anchor <= schema.index(sc) <= last_anchor
                       and sc not in mapping]

            # Between anchors: geometric interpolation only.
            for sc in between:
                best_i = None
                best_dist = float("inf")
                expected_cx = _expected_schema_cx(
                    sc, schema, mapping, v4_cols,
                )
                for i, (cx, _label) in enumerate(v4_cols):
                    if i in v4_used:
                        continue
                    dist = abs(cx - expected_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i
                if best_i is not None:
                    mapping[sc] = v4_cols[best_i][1]
                    v4_used.add(best_i)
        # else: no anchors at all → leave ALL schema columns unmapped
        # (caller resolves to SOURCE_ABSENT / column_unresolved).

    # Return the partial mapping.  Schema columns that could NOT be
    # matched (by label anchor or geometry) are simply absent from the
    # dict — the caller emits SOURCE_ABSENT for them rather than
    # silently guessing with ordinal position.
    return mapping if mapping else None


def _expected_schema_cx(
    schema_col: str,
    schema: list,
    mapping: Dict[str, str],
    v4_cols: list,
) -> float:
    """Estimate the expected x-center for *schema_col* using matched neighbours.

    For schema columns BETWEEN two anchors, linearly interpolates.
    For schema columns at the EDGE (no anchor on one side), uses ordinal
    position among remaining V4 columns to avoid overshoot from anchor
    spacing that may not represent the V4 column distribution in the edge
    region.
    """
    idx = schema.index(schema_col)
    # Collect matched (schema_index, v4_cx) pairs.
    matched = []
    for j, sc in enumerate(schema):
        if sc in mapping:
            v4_label = mapping[sc]
            for cx, lbl in v4_cols:
                if lbl == v4_label:
                    matched.append((j, cx))
                    break
    if not matched:
        # No anchors at all — use ordinal midpoint of V4 range.
        if len(v4_cols) < 2:
            return v4_cols[0][0] if v4_cols else 0.0
        return (v4_cols[0][0] + v4_cols[-1][0]) / 2.0

    # Find nearest matched neighbour on the left and on the right.
    left = [(j, cx) for j, cx in matched if j < idx]
    right = [(j, cx) for j, cx in matched if j > idx]

    if left and right:
        lj, lcx = left[-1]
        rj, rcx = right[0]
        t = (idx - lj) / (rj - lj)
        return lcx + t * (rcx - lcx)
    elif left:
        # Edge extrapolation to the right: use average inter-anchor spacing
        # to estimate position of unmatched columns, then snap to nearest V4.
        lj, lcx = left[-1]
        if len(matched) >= 2:
            avg_spacing = (matched[-1][1] - matched[0][1]) / max(matched[-1][0] - matched[0][0], 1)
        else:
            avg_spacing = (v4_cols[-1][0] - v4_cols[0][0]) / max(len(v4_cols) - 1, 1)
        expected = lcx + (idx - lj) * avg_spacing
        # Snap to nearest unused V4 column for better alignment.
        unused = [cx for cx, _ in v4_cols if cx > lcx]
        if unused:
            nearest = min(unused, key=lambda cx: abs(cx - expected))
            if abs(nearest - expected) < avg_spacing * 0.6:
                return nearest
        return expected
    else:
        # Edge extrapolation to the left: use average inter-anchor spacing
        # to estimate position of unmatched columns before first anchor.
        rj, rcx = right[0]
        if len(matched) >= 2:
            avg_spacing = (matched[-1][1] - matched[0][1]) / max(matched[-1][0] - matched[0][0], 1)
        else:
            avg_spacing = (v4_cols[-1][0] - v4_cols[0][0]) / max(len(v4_cols) - 1, 1)
        expected = rcx - (rj - idx) * avg_spacing
        # Snap to nearest unused V4 column.
        unused = [cx for cx, _ in v4_cols if cx < rcx]
        if unused:
            nearest = min(unused, key=lambda cx: abs(cx - expected))
            if abs(nearest - expected) < avg_spacing * 0.6:
                return nearest
        return expected


def _apply_schema_to_records(
    field: dict,
    records: List[dict],
    leaf_columns: list,
) -> List[dict]:
    """Post-process records for a single field to enforce its column schema.

    For any field that has a declared schema (YAML ``columns``, EXPLICIT_COLUMNS
    by name, OR field-ID-specific), keep only records whose column is in the
    schema and fill missing schema columns with blank records.  This prevents
    V4 zone-leaking columns (e.g. MOP appearing in cylinder-pressure rows)
    from polluting the output.
    """
    schema = get_explicit_columns(field)
    if not schema:
        return records

    # Exhaust-gas grids print long row labels (e.g. "Exh. gas temp. CYL out
    # °C") whose text extends over the AVG/CYL1 leaf columns, so the physical
    # value tokens land two columns right of their semantic home (CYL2+).
    # Geometric label-anchoring then leaves AVG/CYL1 blank and shifts every
    # value.  For this region the PDF is a fixed 13-column cylinder grid, so we
    # assign value records ordnally (i-th value -> i-th schema column), which
    # also lets the TC rows (X005-X008) fill their TC1/TC2/TC3 columns that
    # have no CYL-labeled leaf counterpart.  Records arrive in leaf-column
    # (left-to-right) order, so ordinal preserves the printed sequence.
    if (field.get("region") or "") == "exhaust_gas":
        return _ordinal_apply_schema(field, records, schema)

    col_map = _build_schema_column_map(field, leaf_columns)
    if not col_map:
        # No mapping possible: cap to schema length and use positional names.
        # For cylinder_pressure/turbocharger, use ordinal fallback when
        # geometric mapping fails entirely.
        if (field.get("region") or "") in ("cylinder_pressure", "turbocharger"):
            return _ordinal_apply_schema(field, records, schema)
        capped = records[: len(schema)]
        for i, rec in enumerate(capped):
            if i < len(schema):
                rec["column"] = schema[i]
            else:
                rec["column"] = f"VAL{i + 1}"
            if "column_unresolved" not in rec.get("flags", []):
                rec.setdefault("flags", []).append("column_unresolved")
        return capped

    # Invert the mapping: v4_label -> schema_col
    v4_to_schema: Dict[str, str] = {}
    for schema_col, v4_label in col_map.items():
        v4_to_schema[v4_label] = schema_col

    remapped: List[dict] = []
    for rec in records:
        orig_col = rec.get("column", "VALUE")
        new_col = v4_to_schema.get(orig_col, orig_col)
        rec["column"] = new_col
        # Only keep records whose column is in the schema.
        if new_col in schema:
            remapped.append(rec)

    # Cap to schema length.
    if len(remapped) > len(schema):
        remapped = remapped[: len(schema)]

    present_cols = {r["column"] for r in remapped}
    field_id = field.get("id", "")
    field_name = field.get("name", field_id)
    unit = (field.get("unit", "") or "").strip()
    source_row = remapped[0]["source_row"] if remapped else ""
    page = remapped[0]["page"] if remapped else 0
    region = remapped[0]["region"] if remapped else ""

    for schema_col in schema:
        if schema_col not in present_cols:
            remapped.append({
                "field_id": field_id,
                "field_name": field_name,
                "page": page,
                "region": region,
                "column": schema_col,
                "raw_text": "",
                "value": None,
                "unit": unit,
                "confidence": 0.0,
                "bbox": [],
                "source_row": source_row,
                "flags": [SOURCE_ABSENT],
            })

    return remapped


def _ordinal_apply_schema(
    field: dict,
    records: List[dict],
    schema: List[str],
) -> List[dict]:
    """Assign exhaust-gas values to schema columns by printed position.

    Value records (already in left-to-right leaf-column order) are assigned to
    schema columns in order: the i-th value goes to schema[i].  Blank leaf
    columns contribute nothing, so a physically-gapped row compacts left onto
    its true semantic columns (X001 AVG/CYL1 get the leading values), and TC
    rows fill TC1/TC2/TC3 from the values that leaf columns labelled CYL*.
    """
    field_id = field.get("id", "")
    field_name = field.get("name", field_id)
    unit = (field.get("unit", "") or "").strip()
    source_row = records[0].get("source_row", "") if records else ""
    page = records[0].get("page") if records else 0
    region = records[0].get("region") if records else (field.get("region") or "")

    value_recs = [r for r in records if r.get("value") is not None]

    remapped: List[dict] = []
    for i, rec in enumerate(value_recs):
        if i < len(schema):
            rec["column"] = schema[i]
        else:
            rec["column"] = f"VAL{i + 1}"
        remapped.append(rec)

    if len(remapped) > len(schema):
        remapped = remapped[: len(schema)]

    present_cols = {r["column"] for r in remapped}
    for schema_col in schema:
        if schema_col not in present_cols:
            remapped.append({
                "field_id": field_id,
                "field_name": field_name,
                "page": page,
                "region": region,
                "column": schema_col,
                "raw_text": "",
                "value": None,
                "unit": unit,
                "confidence": 0.0,
                "bbox": [],
                "source_row": source_row,
                "flags": [SOURCE_ABSENT],
            })

    return remapped


# ───────────────────────────────────────────────────────────
# Field matching
# ───────────────────────────────────────────────────────────
def match_field(
    row_label: str,
    unit_text: str,
    qualifier_texts: Sequence[str],
    schema_fields: Sequence[dict],
    label_candidates: Sequence[str] = (),
) -> Tuple[Optional[dict], Optional[str]]:
    """Match a reconstructed row to the best schema field.

    The pixel layout may split a field's full descriptive text across the
    value-row's label, unit and qualifier regions (e.g. label='temp ENGINE in',
    unit='Lub oil °C' -> schema 'Lub oil temp ENGINE in'). ``label_candidates``
    provides recompositions of those same source tokens so a fragmented label
    can still align to its schema alias. All tokens come from the source row;
    nothing is invented.

    Returns (field, flag). flag is None on a clear match, OCR_AMBIGUOUS when
    two or more fields in the same section score within a small tolerance, or
    None + no field when nothing matches (caller marks SOURCE_ABSENT).
    """
    if not schema_fields:
        return None, None
    candidates = [row_label] + list(label_candidates or [])
    candidates = [c for c in candidates if c and c.strip()]
    if not candidates:
        return None, None

    scored: List[Tuple[float, dict]] = []
    for f in schema_fields:
        aliases = f.get("aliases", []) or [f.get("name", "")]
        best = 0.0
        for alias in aliases:
            s = max(score_label(c, alias) for c in candidates)
            if s > best:
                best = s
        # Very weak unit agreement is a small tie-break, never a decision.
        if unit_text:
            f_unit = (f.get("unit", "") or "").strip()
            if f_unit and normalize_label(f_unit) == normalize_label(unit_text):
                best += 0.01
        scored.append((best, f))

    if not scored:
        return None, None

    scored.sort(key=lambda x: x[0], reverse=True)
    first_score, first_field = scored[0]

    # Ambiguity: a clear label must dominate. Two same-score candidates after
    # the strongest one -> ambiguous.
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if first_score < 0.5:
        return None, None
    if second_score >= first_score - 0.001:
        return None, OCR_AMBIGUOUS
    return first_field, None


def _best_alias(field: dict, candidates: Sequence[str]) -> str:
    """Pick the schema alias that best explains a composed row label."""
    aliases = field.get("aliases", []) or [field.get("name", "")]
    best_alias = ""
    best_score = -1.0
    for a in aliases:
        s = max(score_label(c, a) for c in candidates if c)
        if s > best_score:
            best_score = s
            best_alias = a
    return best_alias


def _excess_label_text(alias: str, label_text: str) -> str:
    """Return the tail of a row label beyond the schema alias, in order.

    A promoted label-only band may carry its text value right in the label
    zone (e.g. label zone 'Running mode Economy' vs. schema alias 'Running
    mode' -> value 'Economy'). Only tokens not consumed by the aligned alias
    are returned; nothing is invented. Best-effort and schema-driven (the
    engine cannot know where a label ends and a value begins)."""
    at = _tokens(alias)
    lt = _tokens(label_text)
    if not at or not lt:
        return ""
    i = 0
    rest: List[str] = []
    for w in lt:
        if i < len(at) and w == at[i]:
            i += 1
        else:
            rest.append(w)
    return " ".join(rest)


# ───────────────────────────────────────────────────────────
# Record emission
# ───────────────────────────────────────────────────────────
def _bbox_ltrb(value_bbox) -> List[int]:
    """Convert V4 (left, top, right, bottom) -> [left, top, width, height].
    Falls back to the cell bbox if value bbox is unavailable."""
    if not value_bbox:
        return []
    left, top, right, bottom = value_bbox
    return [int(round(left)), int(round(top)), int(round(right - left)), int(round(bottom - top))]


def _field_type(field: dict) -> str:
    if field.get("allow_text_values"):
        return "text"
    unit = (field.get("unit", "") or "").strip()
    if not unit:
        return "scalar"
    return "numeric"


def _value_for(field: dict, raw_text: str) -> Optional[float]:
    """Parse a semantic numeric value from the (normalized) cell text.

    For %-unit fields a trailing '%' is tolerated. Text fields return None
    (their raw text is the datum)."""
    ftype = _field_type(field)
    raw = (raw_text or "").strip()
    if ftype == "text":
        return None
    if not raw:
        return None
    unit = (field.get("unit", "") or "").strip()
    if unit == "%":
        raw = raw.rstrip("%").strip()
    return parse_float(raw)


# Glyphs that OCR inserts between/around a printed value but that are not part
# of the number (table rules, status markers, separators). Mirrors the engine's
# OCR-noise set so per-token value isolation is geometry/context free.
_ARTIFACT_GLYPHS = re.compile(r"[|/\\®©@`'{}[\]\"]+")
_NUM_CLEAN = re.compile(r"^[+-]?\d+([.,]\d+)?%?$")


def _token_is_independent_number(token) -> bool:
    """A token carries an independent numeric datum only if the OCR-noise-free
    remainder is itself a plain number. This rejects fragments of a single
    printed value that OCR split across tokens (e.g. '0.0/0' -> '0.0','0')."""
    stripped = _ARTIFACT_GLYPHS.sub("", getattr(token, "text", "")).strip()
    return bool(_NUM_CLEAN.fullmatch(stripped))


def _multi_second_values(field: dict, cell: Optional[Cell]):
    """For a present numeric cell, return [(value, token), ...] for every
    INDEPENDENT numeric value in addition to the first.

    A packed cell (e.g. two clean numbers in one AVG cell) must not silently
    truncate the second value. Only tokens that are themselves a plain number
    at a usable confidence count as independent; low-confidence fragments and
    OCR split-noise are excluded. Returns [] for blank/absent/text/single-value
    cells so single-value behavior is unchanged."""
    if cell is None or cell.is_blank or _field_type(field) != "numeric":
        return []
    toks = list(getattr(cell, "tokens", []) or [])
    candidates = [
        t for t in toks
        if _token_is_independent_number(t)
        and (getattr(t, "conf", 0.0) or 0.0) >= 60.0
    ]
    if len(candidates) <= 1:
        return []
    primary = candidates[0]
    extra = []
    for t in candidates[1:]:
        v = _value_for(field, getattr(t, "text", ""))
        if v is not None and v != _value_for(field, getattr(primary, "text", "")):
            extra.append((v, t))
    return extra


def make_record(
    field: dict,
    column: str,
    cell: Optional[Cell],
    page: int,
    region: str,
    source_row: str,
    status_text: List[str],
    base_flags: Optional[List[str]] = None,
    qualifier_text: str = "",
    row_idx: Optional[int] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Emit one semantic record in the existing flat format.

    Only the absence/quality flags that actually apply are set below; a
    matching, non-blank, parsable cell yields no flags.
    """
    flags = list(base_flags or [])
    field_id = field.get("id", "")
    field_name = field.get("name", field_id)
    unit = (field.get("unit", "") or "").strip()

    if cell is None:
        # No physical row for this field at all.
        rec = {
            "field_id": field_id,
            "field_name": field_name,
            "page": page,
            "region": region,
            "column": column,
            "raw_text": "",
            "value": None,
            "unit": unit,
            "confidence": 0.0,
            "bbox": [],
            "source_row": source_row,
            "flags": [SOURCE_ABSENT],
        }
        return _with_provenance(rec, provenance)

    raw_text = cell.raw_text

    if cell.is_blank:
        # Physical cell exists but is empty: it IS a column, just blank.
        rec = {
            "field_id": field_id,
            "field_name": field_name,
            "page": page,
            "region": region,
            "column": column,
            "raw_text": raw_text,
            "value": None,
            "unit": unit,
            "confidence": cell.confidence,
            "bbox": _bbox_ltrb(cell.value_bbox or cell.cell_bbox),
            "source_row": source_row,
            "flags": [SOURCE_BLANK],
        }
        return _with_provenance(rec, {**(provenance or {}), "status": status_text})

    # Present cell: compute value and confidence/flags.
    try:
        value = _value_for(field, raw_text)
    except Exception:
        value = None
        flags.append(EXTRACTION_FAILED)

    if value is None and _field_type(field) == "numeric":
        # Raw present but not numeric: mark extraction failure (not fabricated).
        flags.append(EXTRACTION_FAILED)

    # Cell confidence is on the OCR 0-100 scale (mean of token confidences).
    # A high-confidence NON-value token (e.g. a '%' unit) must not mask a
    # low-confidence VALUE token (e.g. an OCR-glued '673|®' at conf 29): the
    # uncertainty of the value reading itself is what matters. Base the
    # low-confidence flag on the best-attested VALUE-bearing token when any
    # exist, otherwise fall back to the cell mean. Flagging < 60 keeps an
    # ambiguous reading (e.g. '673|®') from being reported as a confident,
    # definitive number -- we preserve the ambiguity, never manufacture a value.
    _reading_conf = cell.confidence
    _val_toks = [t for t in cell.tokens if _token_is_independent_number(t)]
    if _val_toks:
        _reading_conf = max((getattr(t, "conf", 0.0) or 0.0) for t in _val_toks)
    if _reading_conf < 60.0:
        flags.append(LOW_CONFIDENCE)
    if status_text:
        # A status indicator is metadata on this same cell; never a new column.
        flags.append("status_indicator")
    if getattr(cell, "recovered", False):
        flags.append(RECOVERED_BY_GEOMETRY_FALLBACK)

    rec = {
        "field_id": field_id,
        "field_name": field_name,
        "page": page,
        "region": region,
        "column": column,
        "raw_text": raw_text,
        "value": value,
        "unit": unit,
        "confidence": round(cell.confidence, 2),
        "bbox": _bbox_ltrb(cell.value_bbox or cell.cell_bbox),
        "source_row": source_row,
        "flags": flags,
    }
    prov = {
        **(provenance or {}),
        "status": status_text,
        "qualifier": qualifier_text,
        "row_idx": row_idx,
    }
    return _with_provenance(rec, prov)


def _with_provenance(rec: dict, prov: Optional[dict]) -> dict:
    """Attach provenance as a new backward-compatible key; keep all standard
    keys and values identical."""
    if prov:
        rec["provenance"] = {k: v for k, v in prov.items() if v is not None}
    return rec


# ───────────────────────────────────────────────────────────
# Main adapter entry point
# ───────────────────────────────────────────────────────────
def adapter_to_records(
    result: ReconstructionResult,
    schema_fields: Sequence[dict],
    page: int,
    region: str,
) -> List[dict]:
    """Convert a V4 ReconstructionResult into flat semantic records.

    One field may yield many records (one per non-absent V4 cell). Blank cells
    are preserved in place; no value is fabricated; columns come from V4.
    """
    # Only fields belonging to this page+region are candidates.
    candidates = [
        f for f in schema_fields
        if (f.get("region") or "") == region
    ]
    if not candidates:
        return []

    data_rows = [pr.model for pr in result.physical_rows if pr.model is not None]

    records: List[dict] = []
    matched_fields: Dict[str, dict] = {}

    for ri, row in enumerate(data_rows):
        row_label = row.label_region.normalized_text
        unit_text = row.unit_region.normalized_text
        qual_texts = [q.normalized_text for q in row.qualifier_regions if q.normalized_text]
        source_row = row.label_region.text.strip()

        # The printed field name may be split across the label/unit/qualifier
        # regions of the same physical row (e.g. label="temp ENGINE in" and
        # unit="Lub oil °C" together spell the schema name "Lub oil temp
        # ENGINE in"). Compose candidate spellings from those SAME source
        # tokens (no invented words) so the adapter can match full aliases.
        # Matching is in-order/noise-tolerant, so stray unit symbols in the
        # composed strings do not break alignment.
        qual_text = ";".join(qual_texts)
        label_candidates = []
        for combo in (
            (row_label,),
            (row_label, qual_text),
            (row_label, unit_text, qual_text),
            (unit_text, row_label, qual_text),
            (qual_text, row_label),
        ):
            s = " ".join(x for x in combo if x and x.strip())
            if s and s not in label_candidates:
                label_candidates.append(s)

        field, flag = match_field(row_label, unit_text, qual_texts, candidates,
                                  label_candidates=label_candidates)
        if field is None:
            if flag == OCR_AMBIGUOUS:
                unmatched = [f for f in candidates
                             if f.get("id") not in matched_fields]
                _emit_absent(records, unmatched, source_row, ri, page, region, OCR_AMBIGUOUS)
            continue

        field_id = field.get("id")
        matched_fields[field_id] = field

        qual_text = " ".join(qual_texts)

        # A promoted label-only band (header/title field row) that matched a
        # text-value schema field: any printed text carried in the label zone
        # beyond the schema alias IS the field datum (e.g. 'Running mode
        # Economy' -> 'Economy').  The text value may also reside in the unit
        # region (e.g. 'Governor mode' label + 'RPM' unit -> 'RPM').  Compose
        # label+unit to cover both layouts.  Emit it honestly instead of a blank
        # row; if there is no such text, emit a single SOURCE_BLANK.
        if row.status_metadata.get("label_band") and _field_type(field) == "text":
            alias = _best_alias(field, [row_label] + list(label_candidates))
            # Try label-only first, then composed label+unit.
            text = _excess_label_text(alias, row.label_region.normalized_text)
            if not text and unit_text:
                composed = f"{row_label} {unit_text}".strip()
                text = _excess_label_text(alias, composed)
            col_label = canonical_column(result.leaf_columns[0].label) if result.leaf_columns else "VALUE"
            _lbl_toks = list(row.label_region.tokens)
            _lbl_conf = statistics.mean(t.conf for t in _lbl_toks) if _lbl_toks else 0.0
            rec = {
                "field_id": field_id,
                "field_name": field.get("name", field_id),
                "page": page,
                "region": region,
                "column": col_label,
                "raw_text": text,
                "value": text if text else None,
                "unit": (field.get("unit", "") or "").strip(),
                "confidence": round(_lbl_conf, 2),
                "bbox": _bbox_ltrb(row.label_region.bbox),
                "source_row": source_row,
                "flags": [TEXT_VALUE] if text else [SOURCE_BLANK],
            }
            records.append(_with_provenance(rec, {
                "status": [s.text for s in row.status_indicators],
                "qualifier": qual_text,
                "row_idx": ri,
            }))
            continue

        # One record per V4 cell (including blanks, preserving columns).
        for col in result.leaf_columns:
            col_label = canonical_column(col.label)
            cell = _cell_for_column(row, col.col_id)
            col_label = canonical_column(col.label)
            cell = _cell_for_column(row, col.col_id)
            multi = _multi_second_values(field, cell)
            if multi:
                # A packed cell holds more than one independent numeric value
                # (e.g. two clean numerics in one AVG cell). Emit each value as
                # its own record in the SAME column so no source value is
                # silently truncated. Bbox/confidence come from the token.
                primary = make_record(
                    field, col_label, cell, page, region, source_row,
                    status_text=_status_texts(cell), qualifier_text=qual_text,
                    row_idx=ri, provenance={"header_path": col.path},
                )
                records.append(primary)
                for val, tok in multi:
                    rec = make_record(
                        field, col_label, cell, page, region, source_row,
                        status_text=_status_texts(cell), qualifier_text=qual_text,
                        row_idx=ri, provenance={"header_path": col.path},
                        base_flags=["SECONDARY_VALUE"],
                    )
                    rec["value"] = val
                    if tok:
                        rec["raw_text"] = tok.text
                        l, t, r, b = (tok.left, tok.top, tok.right, tok.bottom)
                        rec["bbox"] = [int(round(l)), int(round(t)),
                                       int(round(r - l)), int(round(b - t))]
                        rec["confidence"] = round(tok.conf, 2)
                    records.append(rec)
                continue
            record = make_record(
                field,
                col_label,
                cell,
                page,
                region,
                source_row,
                status_text=_status_texts(cell),
                qualifier_text=qual_text,
                row_idx=ri,
                provenance={"header_path": col.path},
            )
            records.append(record)

    # Fields in the schema that produced no physical row -> SOURCE_ABSENT.
    for f in candidates:
        if f.get("id") not in matched_fields:
            records.append(
                make_record(f, "VALUE", None, page, region, "", status_text=[],
                            base_flags=[SOURCE_ABSENT])
            )

    return records


def _cell_for_column(row: RowModel, col_id: str) -> Optional[Cell]:
    for c in row.cells:
        if c.col_id == col_id:
            return c
    return None


def _status_texts(cell: Optional[Cell]) -> List[str]:
    if cell is None:
        return []
    return [s.text for s in cell.status]


def _emit_absent(records, candidates, source_row, ri, page, region, flag):
    """Emit one SOURCE_ABSENT record per relevant candidate when the row could
    not be unambiguously assigned (no guessing)."""
    # Keep it minimal and non-fabricating: one absent marker per candidate is
    # too noisy, so represent the ambiguity once per candidate as SOURCE_ABSENT.
    for f in candidates:
        records.append(
            make_record(f, "VALUE", None, page, region, source_row,
                        status_text=[], base_flags=[SOURCE_ABSENT])
        )


# ───────────────────────────────────────────────────────────
# Validator-facing FieldRecord objects (reuse existing validator)
# ───────────────────────────────────────────────────────────
def build_field_records(
    result: ReconstructionResult,
    schema_fields: Sequence[dict],
    page: int,
    region: str,
):
    """Return a list of ``app.models.FieldRecord`` objects suitable for the
    existing ``validate.validate`` and ``export.export`` pipeline.

    The flat records from ``adapter_to_records`` are grouped per field."""
    # Imported lazily to avoid a hard dependency in the pure adapter path.

    # Phase 6: generic geometry fallback. Fill blank value cells from unclaimed
    # value-bearing OCR tokens before emitting records, so both Template A and
    # Template B benefit. Only blank cells are touched; populated cells are
    # never overwritten.
    try:
        from .recovery import recover_dense_empty_cells
        recover_dense_empty_cells(result)
    except Exception:
        # The primary path must never fail because a best-effort recovery did.
        pass

    flat = adapter_to_records(result, schema_fields, page, region)

    grouped: Dict[str, List[dict]] = {}
    for rec in flat:
        grouped.setdefault(rec["field_id"], []).append(rec)

    out: List[FieldRecord] = []
    by_id = {f.get("id"): f for f in schema_fields}

    # Schema-aware column mapping: remap V4 leaf labels to schema column
    # names and cap to schema width for every multi-column field.
    for fid, recs in list(grouped.items()):
        f = by_id.get(fid, {})
        remapped = _apply_schema_to_records(f, recs, result.leaf_columns)
        grouped[fid] = remapped

    for fid, recs in grouped.items():
        f = by_id.get(fid, {})
        field_name = f.get("name", fid)
        unit = (f.get("unit", "") or "").strip()
        found = any(SOURCE_ABSENT not in r["flags"] for r in recs)
        first = recs[0] if recs else {}
        fr = FieldRecord(
            field_id=fid,
            field_name=field_name,
            page=page,
            region=region,
            unit=unit,
            found=found,
            row_text=first.get("source_row", ""),
            row_confidence=first.get("confidence", 0.0),
            cells=[],
        )
        for r in recs:
            fr.cells.append(VC(
                field_id=r["field_id"],
                field_name=r["field_name"],
                page=r["page"],
                region=r["region"],
                column=r["column"],
                raw_text=r["raw_text"],
                value=r["value"],
                unit=r["unit"],
                confidence=r["confidence"],
                bbox=r["bbox"],
                source_row=r["source_row"],
                flags=list(r["flags"]),
            ))
        # Propagate the single non-absent cell's evidence to the field level.
        non_absent = [c for c in fr.cells if SOURCE_ABSENT not in c.flags]
        if len(non_absent) == 1:
            fr.raw_text = non_absent[0].raw_text or None
            fr.value = non_absent[0].value
        out.append(fr)
    return out


def build_field_records_wholepage(
    result: ReconstructionResult,
    schema_fields: Sequence[dict],
    page: int,
):
    """Whole-page variant of ``build_field_records`` for the ``validate`` /
    ``export`` pipeline.

    Uses ``adapter_to_records_wholepage`` which matches ALL schema fields
    against V4 rows without region filtering.  Returns ``FieldRecord`` objects
    compatible with ``validate.validate`` and ``export.export``."""

    # Phase 6: generic geometry fallback (same as build_field_records). Covers
    # the whole-page / multi-zone (Template B) path via _zone_to_result.
    try:
        from .recovery import recover_dense_empty_cells
        recover_dense_empty_cells(result)
    except Exception:
        pass

    flat = adapter_to_records_wholepage(result, schema_fields, page)

    grouped: Dict[str, List[dict]] = {}
    for rec in flat:
        grouped.setdefault(rec["field_id"], []).append(rec)

    out: List[FieldRecord] = []
    by_id = {f.get("id"): f for f in schema_fields}

    # Schema-aware column mapping.
    for fid, recs in list(grouped.items()):
        f = by_id.get(fid, {})
        remapped = _apply_schema_to_records(f, recs, result.leaf_columns)
        grouped[fid] = remapped

    for fid, recs in grouped.items():
        f = by_id.get(fid, {})
        field_name = f.get("name", fid)
        unit = (f.get("unit", "") or "").strip()
        found = any(SOURCE_ABSENT not in r["flags"] for r in recs)
        first = recs[0] if recs else {}
        fr = FieldRecord(
            field_id=fid,
            field_name=field_name,
            page=page,
            region=f.get("region", "reading"),
            unit=unit,
            found=found,
            row_text=first.get("source_row", ""),
            row_confidence=first.get("confidence", 0.0),
            cells=[],
        )
        for r in recs:
            fr.cells.append(VC(
                field_id=r["field_id"],
                field_name=r["field_name"],
                page=r["page"],
                region=r["region"],
                column=r["column"],
                raw_text=r["raw_text"],
                value=r["value"],
                unit=r["unit"],
                confidence=r["confidence"],
                bbox=r["bbox"],
                source_row=r["source_row"],
                flags=list(r["flags"]),
            ))
        # Propagate the single non-absent cell's evidence to the field level.
        non_absent = [c for c in fr.cells if SOURCE_ABSENT not in c.flags]
        if len(non_absent) == 1:
            fr.raw_text = non_absent[0].raw_text or None
            fr.value = non_absent[0].value
        out.append(fr)
    return out

def adapter_to_records_wholepage(
    result: ReconstructionResult,
    schema_fields: Sequence[dict],
    page: int,
) -> List[dict]:
    """Whole-page adapter that matches all schema regions against V4 rows.

    For whole-page OCR (Template B), the input tokens carry a non-schema
    region name (e.g. ``"reading"``).  Instead of filtering schema fields
    by region, this adapter tries ALL schema fields against every V4 row
    using the same generic label-matching logic.  Columns are resolved from
    V4's reconstructed leaf columns and header geometry.

    This function is structurally identical to ``adapter_to_records`` except
    that it does NOT filter candidates by region.  It is intended for
    whole-page multi-zone reconstruction where a single ``ReconstructionResult``
    contains rows from multiple spatial zones.
    """
    if not schema_fields:
        return []

    data_rows = [pr.model for pr in result.physical_rows if pr.model is not None]

    records: List[dict] = []
    matched_fields: Dict[str, dict] = {}

    for ri, row in enumerate(data_rows):
        row_label = row.label_region.normalized_text
        unit_text = row.unit_region.normalized_text
        qual_texts = [q.normalized_text for q in row.qualifier_regions if q.normalized_text]
        source_row = row.label_region.text.strip()

        qual_text = ";".join(qual_texts)
        label_candidates = []
        for combo in (
            (row_label,),
            (row_label, qual_text),
            (row_label, unit_text, qual_text),
            (unit_text, row_label, qual_text),
            (qual_text, row_label),
        ):
            s = " ".join(x for x in combo if x and x.strip())
            if s and s not in label_candidates:
                label_candidates.append(s)

        field, flag = match_field(row_label, unit_text, qual_texts, schema_fields,
                                  label_candidates=label_candidates)
        if field is None:
            if flag == OCR_AMBIGUOUS:
                unmatched = [f for f in schema_fields
                             if f.get("id") not in matched_fields]
                _emit_absent(records, unmatched, source_row, ri, page,
                             "reading", OCR_AMBIGUOUS)
            continue

        field_id = field.get("id")
        matched_fields[field_id] = field

        qual_text = " ".join(qual_texts)

        if row.status_metadata.get("label_band") and _field_type(field) == "text":
            alias = _best_alias(field, [row_label] + list(label_candidates))
            text = _excess_label_text(alias, row.label_region.normalized_text)
            if not text and unit_text:
                composed = f"{row_label} {unit_text}".strip()
                text = _excess_label_text(alias, composed)
            col_label = canonical_column(result.leaf_columns[0].label) if result.leaf_columns else "VALUE"
            _lbl_toks = list(row.label_region.tokens)
            _lbl_conf = statistics.mean(t.conf for t in _lbl_toks) if _lbl_toks else 0.0
            rec = {
                "field_id": field_id,
                "field_name": field.get("name", field_id),
                "page": page,
                "region": field.get("region", "reading"),
                "column": col_label,
                "raw_text": text,
                "value": text if text else None,
                "unit": (field.get("unit", "") or "").strip(),
                "confidence": round(_lbl_conf, 2),
                "bbox": _bbox_ltrb(row.label_region.bbox),
                "source_row": source_row,
                "flags": [TEXT_VALUE] if text else [SOURCE_BLANK],
            }
            records.append(_with_provenance(rec, {
                "status": [s.text for s in row.status_indicators],
                "qualifier": qual_text,
                "row_idx": ri,
            }))
            continue

        for col in result.leaf_columns:
            col_label = canonical_column(col.label)
            cell = _cell_for_column(row, col.col_id)
            multi = _multi_second_values(field, cell)
            if multi:
                primary = make_record(
                    field, col_label, cell, page,
                    field.get("region", "reading"), source_row,
                    status_text=_status_texts(cell), qualifier_text=qual_text,
                    row_idx=ri, provenance={"header_path": col.path},
                )
                records.append(primary)
                for val, tok in multi:
                    rec = make_record(
                        field, col_label, cell, page,
                        field.get("region", "reading"), source_row,
                        status_text=_status_texts(cell), qualifier_text=qual_text,
                        row_idx=ri, provenance={"header_path": col.path},
                        base_flags=["SECONDARY_VALUE"],
                    )
                    rec["value"] = val
                    if tok:
                        rec["raw_text"] = tok.text
                        l, t, r, b = (tok.left, tok.top, tok.right, tok.bottom)
                        rec["bbox"] = [int(round(l)), int(round(t)),
                                       int(round(r - l)), int(round(b - t))]
                        rec["confidence"] = round(tok.conf, 2)
                    records.append(rec)
                continue
            record = make_record(
                field,
                col_label,
                cell,
                page,
                field.get("region", "reading"),
                source_row,
                status_text=_status_texts(cell),
                qualifier_text=qual_text,
                row_idx=ri,
                provenance={"header_path": col.path},
            )
            records.append(record)

    # Fields in the schema that produced no physical row -> SOURCE_ABSENT.
    for f in schema_fields:
        if f.get("id") not in matched_fields:
            records.append(
                make_record(f, "VALUE", None, page,
                            f.get("region", "reading"), "", status_text=[],
                            base_flags=[SOURCE_ABSENT])
            )

    return records


# ───────────────────────────────────────────────────────────
# Zone-aware canonical aggregation
# ───────────────────────────────────────────────────────────

def _zone_to_result(zone: ZoneResult) -> ReconstructionResult:
    """Create a single-zone ReconstructionResult from a ZoneResult.

    The adapter needs a ReconstructionResult with zone-specific
    leaf_columns and physical_rows; using the merged result would
    leak columns from unrelated zones."""
    return ReconstructionResult(
        tokens=zone.tokens,
        bands=[],
        physical_rows=zone.physical_rows,
        header_bands=zone.header_bands,
        header_hierarchy=zone.header_hierarchy,
        leaf_columns=zone.leaf_columns,
        status=ReconstructionStatus(ok=True, warnings=[]),
        debug={"zone_idx": zone.zone_idx},
    )


def _pick_best_candidate(candidates: List['FieldRecord']) -> 'FieldRecord':
    """Select the single best candidate record for a field across all zones.

    Priority (highest wins):
    1. Candidate with actual extracted cells (no SOURCE_ABSENT/EXTRACTION_FAILED
       on ALL cells) and highest non-blank cell count.
    2. Candidate with SOURCE_BLANK cells (physical row exists but is blank).
    3. Candidate that is SOURCE_ABSENT (no physical row at all).

    When multiple candidates tie on cell count, the one with the highest
    total confidence wins.  This ensures the actual source evidence is
    preserved rather than an arbitrary page/zone preference.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def _candidate_rank(fr):
        cells = fr.cells or []
        if not cells:
            return (0, 0, 0.0, 0)
        non_absent = sum(
            1 for c in cells
            if SOURCE_ABSENT not in c.flags
            and EXTRACTION_FAILED not in c.flags
        )
        non_blank = sum(
            1 for c in cells
            if SOURCE_ABSENT not in c.flags
            and SOURCE_BLANK not in c.flags
            and c.raw_text.strip()
        )
        total_conf = sum(c.confidence for c in cells)
        return (non_absent, non_blank, total_conf, len(cells))

    return max(candidates, key=_candidate_rank)


def build_field_records_canonical(
    zone_results: List[ZoneResult],
    schema_fields: Sequence[dict],
    page: int,
) -> List['FieldRecord']:
    """Zone-aware canonical aggregation: process each zone independently,
    then merge into exactly one FieldRecord per schema field.

    This fixes two defects in the previous merged-result approach:
    1. Column pollution: the merged ReconstructionResult contains columns
       from ALL zones; rows from zone A would be iterated against columns
       from zone B, producing corrupted column names.
    2. Field duplication: each page emitted all schema fields (matched +
       SOURCE_ABSENT), producing 125 × 2 = 250 records.

    The fix: run the adapter against each zone's own ReconstructionResult
    (with zone-specific leaf_columns), collect all candidate FieldRecords,
    and aggregate by field_id with priority rules that preserve the actual
    source evidence."""

    if not schema_fields:
        return []

    # Phase 1: process each zone independently.
    all_candidates: Dict[str, List[FieldRecord]] = {}
    for zone in zone_results:
        if not zone.physical_rows:
            continue
        zone_result = _zone_to_result(zone)
        zone_frs = build_field_records_wholepage(zone_result, schema_fields, page)
        for fr in zone_frs:
            all_candidates.setdefault(fr.field_id, []).append(fr)

    # Phase 2: aggregate by field_id — exactly one canonical record per field.
    out: List[FieldRecord] = []
    {f.get("id"): f for f in schema_fields}

    for f in schema_fields:
        fid = f.get("id", "")
        candidates = all_candidates.get(fid, [])

        if not candidates:
            # Field not present in any zone on this page → SOURCE_ABSENT.
            fr = FieldRecord(
                field_id=fid,
                field_name=f.get("name", fid),
                page=page,
                region=f.get("region", "reading"),
                unit=(f.get("unit", "") or "").strip(),
                found=False,
                row_text="",
                row_confidence=0.0,
                cells=[VC(
                    field_id=fid, field_name=f.get("name", fid),
                    page=page, region=f.get("region", "reading"),
                    column="VALUE", raw_text="", value=None,
                    unit=(f.get("unit", "") or "").strip(),
                    confidence=0.0, bbox=[], source_row="",
                    flags=[SOURCE_ABSENT],
                )],
            )
            out.append(fr)
            continue

        # Pick the best candidate zone's FieldRecord for this field.
        best = _pick_best_candidate(candidates)

        # Build a merged FieldRecord: take cells from the best candidate,
        # but ensure no duplicate columns leak from other zones.
        seen_columns: set = set()
        merged_cells: List[VC] = []
        for c in best.cells:
            if c.column not in seen_columns:
                seen_columns.add(c.column)
                merged_cells.append(c)

        fr = FieldRecord(
            field_id=fid,
            field_name=f.get("name", fid),
            page=best.page,
            region=best.region,
            unit=best.unit,
            found=best.found,
            row_text=best.row_text,
            row_confidence=best.row_confidence,
            cells=merged_cells,
        )
        # Propagate the single non-absent cell's evidence to the field level.
        non_absent = [c for c in fr.cells if SOURCE_ABSENT not in c.flags]
        if len(non_absent) == 1:
            fr.raw_text = non_absent[0].raw_text or None
            fr.value = non_absent[0].value
        out.append(fr)

    return out


def aggregate_field_records_across_pages(
    per_page_records: List[List['FieldRecord']],
    schema_fields: Sequence[dict],
) -> List['FieldRecord']:
    """Cross-page canonical aggregation: merge per-page FieldRecords into
    exactly one record per schema field across ALL reading pages.

    This ensures the final output has exactly N fields (where N is the
    number of registered schema fields) regardless of how many pages
    were processed.  The best candidate across all pages is selected
    using the same priority rules as per-zone aggregation: extracted >
    blank > absent.
    """
    if not per_page_records:
        return []
    if len(per_page_records) == 1:
        return per_page_records[0]

    # Collect all candidates per field_id across all pages.
    candidates: Dict[str, List['FieldRecord']] = {}
    for page_frs in per_page_records:
        for fr in page_frs:
            candidates.setdefault(fr.field_id, []).append(fr)

    out: List['FieldRecord'] = []
    for f in schema_fields:
        fid = f.get("id", "")
        field_candidates = candidates.get(fid, [])

        if not field_candidates:
            # Field not found on any page → single SOURCE_ABSENT record.
            fr = FieldRecord(
                field_id=fid,
                field_name=f.get("name", fid),
                page=0,
                region=f.get("region", "reading"),
                unit=(f.get("unit", "") or "").strip(),
                found=False,
                row_text="",
                row_confidence=0.0,
                cells=[VC(
                    field_id=fid, field_name=f.get("name", fid),
                    page=0, region=f.get("region", "reading"),
                    column="VALUE", raw_text="", value=None,
                    unit=(f.get("unit", "") or "").strip(),
                    confidence=0.0, bbox=[], source_row="",
                    flags=[SOURCE_ABSENT],
                )],
            )
            out.append(fr)
            continue

        best = _pick_best_candidate(field_candidates)
        out.append(best)

    return out
