"""Whole-page extraction strategy for non-CE (Template B) performance reports.

Template A (the CE monthly reports) is parsed as two calibrated reading-page
crops with a fixed 14-region layout.  Several other ``Performance Report /
Evaluation`` documents instead carry their Reading table on a single compact
page, or on pages whose section geometry does not match the calibrated CE crop
boxes.  For those documents the crop-based strategy yields almost no fields.

This module implements a genuinely generalized, whole-page strategy:

* the reading page(s) are rendered and OCR'd as a full page;
* the shared row/header machinery is reused (``cluster_rows``,
  ``detect_headers``, the semantic column schemas);
* each registered field's label is matched to a physical row;
* values are collected and mapped to semantic columns by physical
  ``x``-coordinate against the detected header anchors -- never by assuming a
  fixed CE geometry, and never by collapsing ``REF/CALC/ALL/TCn/CYLn`` into a
  single generic ``VALUE``;
* date/time fields are handled as structured datetimes;
* text fields are preserved as text;
* legitimate zeros / negative values are preserved (including OCR tokens whose
  confidence is zero but which nevertheless parse to a clean number inside the
  field's value band).

The output is written with the same ``report.json`` schema that the CE crop
pipeline produces, so the web application, audit engine and exporters consume
both templates identically.  The detected source fields (and populated fields)
are additionally reported so the UI can distinguish "registered" (125) from
"present in this document" and "populated".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pymupdf

from .export import export
from .extract import (
    _compute_value_bounds,
    field_key,
    get_explicit_columns,
    normalize_text,
)
from .headers import detect_headers, nearest_header
from .layout import cluster_rows
from .models import Cell, FieldRecord, OCRRow

# Column names that are not per-cylinder / per-TC / per-bearing data slots and
# therefore never participate in ordinal "column slot" alignment.
META_COLS = {
    "REF", "CALC", "ALL", "AVG", "MEAS", "MOP", "EXPECT", "ENGINE", "OBS",
    "VALUE", "DATETIME", "XPERT", "TORSIOM.", "MCR", "MCR_%", "PUMP", "BY",
    "UNIT",
}

# Known paired scalar fields (SOG/STW, FWD/AFT, MID/TRIM) hold exactly two
# named values; they are not single-value scalars, so they must use their
# explicit pair names rather than generic VALUE/VALn.
_PAIRED_COLUMNS = {
    "ship sog stw": ["SOG", "STW"],
    "draft fwd aft": ["FWD", "AFT"],
    "draft mid trim": ["MID", "TRIM"],
}


def _enrich_schema_anchors(schema, chosen_header, all_headers):
    """Return {schema_column: x} for every schema column.

    ``detect_headers`` occasionally fragments a table's header (e.g. the TC
    row OCR is split so that ``TC1``/``TC3`` are missing), which would make a
    later value snap to the wrong semantic column.  The TC / CYL / BRG value
    columns physically reuse the same x-slots across the whole reading table.
    This helper therefore fills any missing schema column from a sibling header
    of the same anchor family by matching the ordinal rank among its value
    columns (TC1<->CYL1<->BRG1 ..., TC2<->CYL2 ... at the same x).  Values are
    therefore placed in their true physical slot; an absent value leaves a
    blank slot instead of shifting every later value left.
    """
    anchor_x = {c["column"]: c["x"] for c in chosen_header["columns"]}

    # Missing columns that are genuinely blank in the source stay unknown.
    val_cols = [c for c in schema if c not in META_COLS]
    known_names = {c["column"] for c in chosen_header["columns"]}

    # Sibling headers sharing the leading meta anchors (same x family).
    family = []
    for h in all_headers:
        if h is chosen_header:
            continue
        common = [
            c for c in h["columns"]
            if c["column"] in META_COLS and c["column"] in known_names
        ]
        if common and all(
            abs(c["x"] - anchor_x[c["column"]]) < 12 for c in common
        ):
            family.append(h)

    for i, col in enumerate(val_cols):
        if col in anchor_x:
            continue
        for h in family:
            hv = [c for c in h["columns"] if c["column"] not in META_COLS]
            if i < len(hv):
                anchor_x[col] = hv[i]["x"]
                break

    return anchor_x


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_pages(pdf_path: str | Path, reading_pages, dpi: int = 250):
    """Render each 1-based reading page to a PNG file; return (paths, size)."""
    doc = pymupdf.open(str(pdf_path))
    paths = []
    width = height = None
    try:
        for page_no in reading_pages:
            page = doc[page_no - 1]
            scale = dpi / 72.0
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            width, height = pix.width, pix.height
            paths.append((page_no, pix))
    finally:
        doc.close()

    out = []
    for page_no, pix in paths:
        out.append((page_no, pix, width, height))
    return out


def ocr_pages(pixs) -> List[OCRRow]:
    """OCR each rendered page and cluster into rows.

    A single clean PSM pass is used: merging multiple PSM modes produced
    spurious duplicate tokens that corrupted numeric values and column
    assignment.  ``--psm 6`` is the appropriate mode for whole pages."""
    all_rows: List[OCRRow] = []
    for page_no, pix, _w, _h in pixs:
        from io import BytesIO

        from PIL import Image
        pil = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        # psm 11 is the mode the verified crop pipeline uses and captures the
        # tabular value columns best for this document family.
        words = pytesseract_image_to_words(pil, 11, page_no)
        all_rows.extend(cluster_rows(words))
    return all_rows


def pytesseract_image_to_words(pil, psm: int, page_no):
    import pytesseract
    from pytesseract import Output
    data = pytesseract.image_to_data(
        pil, lang="eng", config=f"--oem 1 --psm {psm}",
        output_type=Output.DICT, timeout=60,
    )
    from .models import OCRWord
    words = []
    for i, t in enumerate(data["text"]):
        text = (t or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if not text or conf < 0:
            continue
        words.append(OCRWord(
            page=page_no, region="reading", text=text, conf=conf,
            left=int(data["left"][i]), top=int(data["top"][i]),
            width=int(data["width"][i]), height=int(data["height"][i]),
        ))
    return words


# ---------------------------------------------------------------------------
# Label matching with numeric qualifiers
# ---------------------------------------------------------------------------

DISTINGUISHING = {"aft", "fp", "fwd", "exh"}


def _numeric_qualifiers(name: str) -> List[str]:
    """Standalone integer qualifiers that tail a label (e.g. '298' in
    'TC speed 298').  These disambiguate numbered labels from their unnumbered
    siblings ('TC speed')."""
    return re.findall(r"(?:^|\s)(\d{1,4})(?:\s|$)", str(name or ""))


def _row_numbers(text: str) -> set:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", str(text or "")))


def find_tb_row(rows, field):
    """Match a field label to a physical row, honouring numeric qualifiers
    so that e.g. 'TC speed 298' never collides with 'TC speed'."""
    qualf = _numeric_qualifiers(field.get("name", ""))
    target = normalize_text(field["name"]).split()
    if not target:
        return None, 0.0
    target_special = set(target) & DISTINGUISHING

    def search():
        best = None
        for row in rows:
            cand = normalize_text(row.text).split()
            if not cand:
                continue
            # Required numeric qualifier must be present as a distinct number.
            if qualf:
                row_nums = _row_numbers(row.text)
                if not set(qualf).issubset(row_nums):
                    continue
            cand_special = set(cand) & DISTINGUISHING
            pos = 0
            ok = True
            span = 0
            last = 0
            first = 0
            for token in target:
                try:
                    idx = cand.index(token, pos)
                except ValueError:
                    ok = False
                    break
                if span == 0:
                    first = idx
                last = idx
                span = last - first + 1
                pos = idx + 1
            if not ok:
                continue
            if target_special:
                if not target_special.issubset(cand_special):
                    continue
            else:
                if cand_special:
                    for didx, ct in enumerate(cand):
                        if ct in cand_special and abs(didx - last) <= 1:
                            ok = False
                            break
                    if not ok:
                        continue
            # A plain label must not absorb a numbered sibling row (e.g.
            # 'tc speed' vs row 'tc speed 298 ...').
            if not qualf:
                after = cand[last + 1:] if last + 1 < len(cand) else []
                if after and re.fullmatch(r"\d{1,4}", after[0]):
                    continue
            score = 1.0
            score -= min(0.20, (span - len(target)) * 0.03)
            extra = max(0, len(cand) - len(target))
            score -= min(0.15, extra * 0.02)
            if span == len(target):
                score = 1.0
            if best is None or score > best[0]:
                best = (score, row, span, extra)
        return best

    res = search()
    if res is None:
        return None, 0.0
    score, row, _span, _extra = res
    if score < 0.60:
        return None, score
    return row, score


# ---------------------------------------------------------------------------
# Value collection (zero / conf-0 / negative aware, x-geometry columns)
# ---------------------------------------------------------------------------

_DATE_MD = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})")
# A recording time is only unambiguous when colon-separated (HH:MM or HH:MM:SS).
# A dotted number such as "19.00" is frequently a ship-speed / flow reading (e.g.
# "STW 19.00") on the same OCR line and must NOT be consumed as the recording
# time.  Requiring a colon prevents that cross-field fabrication.
_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def _scan_datetime(words):
    """Scan a list of OCR words for a well-formed date (and optional time).

    The time is accepted either as a single colon-joined token (e.g. ``10:34``)
    or as a colon-lone token flanked by separate hour/minute words (e.g.
    ``10 : 34``), which arise when the OCR engine splits the clock time across
    multiple word boxes.  A bare numeric token with no colon (ship speeds,
    temperatures, draft values) is never treated as a time of day."""
    day = month = year = None
    hh = mm = None
    for w in words:
        t = str(w.text)
        m = _DATE_MD.search(t)
        if m:
            month, day_key, year = m.groups()
            day = day_key
        mt = _TIME.search(t)
        if mt and mt.group(1) and mt.group(2):
            hh, mm = mt.groups()
    if hh is None:
        seq = [str(w.text) for w in words]
        for i, tok in enumerate(seq):
            if tok == ":" and i > 0 and i + 1 < len(seq):
                lft, rgt = seq[i - 1], seq[i + 1]
                if re.fullmatch(r"\d{1,2}", lft) and re.fullmatch(r"\d{2}", rgt):
                    hh, mm = int(lft), int(rgt)
                    break
    parts = []
    if day and month and year:
        y = int(year)
        if y < 100:
            y += 2000
        parts.append(f"{int(year):04d}" if int(year) > 2000 else f"{y:04d}")
        parts.append(f"{int(month):02d}")
        parts.append(f"{int(day):02d}")
    if parts and hh:
        return f"{'-'.join(parts)} {int(hh):02d}:{int(mm):02d}"
    if parts:
        return "-".join(parts)
    return None


def _collect_datetime(row, field, label_end, next_field_left, page_rows=None,
                     psm6_words=None):
    """Collect a structured datetime for the 'Date and time of recording' row.

    A datetime is only emitted from tokens that are structurally associated
    with the field's own logical row.  The date must appear on the matched row;
    the time is only accepted when colon-separated (an unambiguous clock time).
    Arbitrary page tokens (footer dates, STW speeds, title lines) are NEVER
    scanned, so a datetime can not be fabricated from unrelated page content.
    If the matched row carries no well-formed date/time, no datetime is
    returned (OCR_MISSING)."""
    dt = _scan_datetime(row.words)
    if dt is not None:
        return dt
    if psm6_words is None:
        return dt
    near_row = [w for w in psm6_words
                if abs(w.cy - row.y) < 60 and w.page == row.page]
    if near_row:
        dt = _scan_datetime(near_row)
    return dt


def collect_values_tb(row, field, fields, headers, page, page_rows=None,
                      psm6_words=None):
    """Generalized value collection for a matched row.

    Returns a list of Cell objects with semantic columns assigned via physical
    x-coordinates against the nearest header anchor, falling back to the field
    schema / VARCHAR scalar behavior."""
    _numeric_qualifiers(field.get("name", ""))
    schema = get_explicit_columns({"id": field.get("id"), "name": field.get("name")}) or []
    unit = field.get("unit", "")

    name_key = field_key(field.get("name", ""))
    is_datetime = name_key.startswith("date and time")
    allow_text = bool(field.get("allow_text_values"))

    if is_datetime:
        dt = _collect_datetime(row, field, None, None, page_rows=page_rows,
                              psm6_words=psm6_words)
        if dt:
            return [Cell(
                field_id=field.get("id"), field_name=field.get("name"),
                page=page, region=field.get("region", "reading"),
                column="DATETIME", raw_text=dt, value=dt, unit="",
                confidence=1.0, bbox=[0, 0, 0, 0],
                source_row=row.text,
                flags=["datetime"],
            )]
        return []

    try:
        (label_start, label_end, next_field_left) = _compute_value_bounds(row, field, fields)
        if label_end is None:
            label_end = 0
    except Exception:
        label_end = 0
        next_field_left = None

    words = sorted(
        (w for w in row.words if w.left > label_end),
        key=lambda w: w.left,
    )
    # A field with an explicit schema lives on its own dedicated table row
    # (REF/CALC/AVG/CYLn/TCn...).  The row belongs entirely to that field, so
    # the next-field boundary must NOT truncate it (a stray OCR artefact such
    # as a "@" token is often wrongly matched as a neighboring label).  Only
    # non-schema single-line fields (e.g. ENGINE running hours) rely on the
    # next-field boundary to stop before an unrelated packed neighbour.
    if not schema:
        if next_field_left is not None:
            words = [w for w in words if w.left < next_field_left]

    # ---- text fields -------------------------------------------------
    if allow_text:
        # Text fields (Running mode / Governor mode / ENGINE state) hold a
        # single textual value (e.g. "Economy", "RPM", "stable") that may
        # share the physical line with unrelated numeric fields.  Collect only
        # the non-numeric text tokens to the right of the label and take the
        # first one; never let adjacent numeric neighbours leak in.
        STOP = {
            "corrected", "measured", "iso", "ref", "calc", "all", "avg",
        }
        text_words = [
            w for w in words
            if not _very_numeric(w.text) and _clean_text(w.text).lower() not in STOP
        ]
        if text_words:
            w = min(text_words, key=lambda w: w.left)
            raw = str(w.text).strip()
            return [Cell(
                field_id=field.get("id"), field_name=field.get("name"),
                page=page, region=field.get("region", "reading"),
                column="VALUE", raw_text=raw, value=_clean_text(raw), unit=unit,
                confidence=round(w.conf / 100.0, 4), bbox=[w.left, w.top, w.width, w.height],
                source_row=row.text, flags=["text_value"],
            )]
        # No text token present (e.g. "stable" was not captured by OCR).
        # Do not fabricate a value and do not leak numeric neighbours.
        if not any(_very_numeric(x.text) for x in words):
            return []


    # ---- numeric / schematic ------------------------------------------
    candidates = []
    for w in words:
        try:
            val = _parse(w.text)
        except Exception:
            val = None
        if val is None:
            continue
        # skip unit/header tokens
        if str(w.text).lower() in {
            "ref", "calc", "all", "avg", "meas", "mop", "expect", "engine",
            "obs", "bar", "kw", "rpm", "kwh", "mw",
        }:
            continue
        candidates.append((w, val))

    if not candidates:
        return []

    header = None
    # Pick the header that is directly above the row and whose declared
    # columns best match the field's semantic schema.  This stops a value from
    # being anchored to an unrelated section's header (e.g. PUMP2) purely
    # because that header happened to be the nearest.
    above = [h for h in headers if h["y"] < row.y]
    if above:
        if schema:
            above.sort(key=lambda h: (
                -len([c for c in h["columns"] if c["column"] in schema]),
                -h["y"],
            ))
            header = above[0]
            if not any(c["column"] in schema for c in header["columns"]):
                header = above[0]
        else:
            header = min(above, key=lambda h: abs(h["y"] - row.y))

    anchors = header["columns"] if header else None
    anchor_names = [c["column"] for c in anchors] if anchors else []

    # Enrich the anchor set with every schema column so a fragmented header
    # (e.g. a missing TC1/TC3) cannot slide a value into the wrong slot.
    enriched = None
    if schema and anchors:
        ax = _enrich_schema_anchors(schema, header, headers)
        enriched = [{"column": c, "x": x} for c, x in ax.items()]
        anchors = enriched

    # Determine whether to use header geometry or positional schema assignment.
    # If the header provides anchors for fewer schema columns than the schema
    # declares, the header is incomplete for this field -- fall back to
    # positional assignment which respects the declared schema order.
    use_positional = False
    if schema and anchors:
        anchored_schema_cols = set(anchor_names) & set(schema)
        if len(anchored_schema_cols) < len(schema):
            use_positional = True

    # Check for paired scalar fields (no explicit schema but known pairs)
    name_key = field_key(field.get("name", ""))
    paired_cols = _PAIRED_COLUMNS.get(name_key)

    cells = []
    used = set()

    # Geometry-mismatch guard: on an explicit-schema row whose values are
    # placed by header x-geometry, two *distinct* value positions must never
    # land on the same semantic column.  A collision is the signature of a
    # header whose column spacing does not match the value row (e.g. a wider
    # REF/AVG/CYL/MOP header selected above a narrower CYL-only value table),
    # which would duplicate some columns and silently empty others.  When the
    # header geometry is inconsistent in this way, the whole row is re-assigned
    # in schema order instead.  A row with fewer values than the schema (a
    # legitimate blank middle slot) maps each value to a distinct column and is
    # preserved as-is.
    geom_cols = [None] * len(candidates)
    geometry_collision = False
    if schema and anchors and not use_positional:
        geom_used = set()
        for i, (w, _val) in enumerate(candidates):
            col = nearest_header(w.cx, anchors)
            geom_cols[i] = col
            if col is not None and col in schema and col in geom_used:
                geometry_collision = True
            geom_used.add(col)

    for i, (w, val) in enumerate(candidates):
        col = None
        conf = round(w.conf / 100.0, 4)
        flags = []
        if conf <= 0.0:
            flags.append("low_ocr_confidence")
        # x-geometry to header anchors (only when header is complete for schema
        # and the field has an explicit column schema).  A geometry collision
        # falls back to schema-order positional assignment for the whole row.
        if anchors and not use_positional and schema and not geometry_collision:
            col = geom_cols[i]
        if col is None or use_positional or geometry_collision:
            if schema:
                if i < len(schema):
                    col = schema[i]
                else:
                    col = f"VAL{i + 1}"
                    flags.append("column_unresolved")
            elif paired_cols:
                if i < len(paired_cols):
                    col = paired_cols[i]
                else:
                    col = f"VAL{i + 1}"
                    flags.append("column_unresolved")
            elif len(candidates) == 1:
                col = "VALUE"
            else:
                col = f"VAL{i + 1}"
                flags.append("column_unresolved")
        if col in used and schema and col in schema:
            # allow repeat only if duplicates genuinely share the column
            pass
        used.add(col)
        cells.append(Cell(
            field_id=field.get("id"), field_name=field.get("name"),
            page=page, region=field.get("region", "reading"),
            column=col, raw_text=str(w.text), value=val, unit=unit,
            confidence=conf, bbox=[w.left, w.top, w.width, w.height],
            source_row=row.text, flags=flags,
        ))

    # ---- scalar protection -------------------------------------------
    # A non-structured scalar field (no schema, no dual "|" slot) holds a
    # single value.  When OCR packs a corrupted unit token onto the same
    # physical row (e.g. "30.0 2€" where "2€" is the misread "°C" unit), the
    # malformed suffix must never become a second scalar value.  Collapse only
    # when there is exactly one well-formed numeric candidate (the malformed
    # unit suffix is dropped); when several well-formed candidates coexist the
    # row is genuinely multi-column / structured, so every value is preserved
    # and no blank slot is compressed.  This distinguishes Ambient air temp.
    # ("30.0" + junk "2€") from ENGINE power effective ("51070 51.7 % 26410").
    is_scalar_field = not schema and "|" not in name_key and allow_text is False
    if is_scalar_field:
        well_formed = [c for c in cells if _very_numeric(c.raw_text)]
        if len(cells) > 1 and len(well_formed) == 1:
            best = well_formed[0]
            best.flags.append("scalar_ambiguous")
            cells = [best]
        # After scalar protection, a scalar field with exactly one cell
        # should use the VALUE column, not VAL1, and clear column_unresolved.
        if len(cells) == 1:
            cells[0].column = "VALUE"
            if "column_unresolved" in cells[0].flags:
                cells[0].flags.remove("column_unresolved")

    # Cap to schema length when a schema is declared.
    if schema and len(cells) > len(schema):
        cells = cells[:len(schema)]
    return cells


UNITISH = {
    "ref", "calc", "all", "avg", "meas", "mop", "expect", "engine", "obs",
    "bar", "kw", "kwh", "rpm", "mw", "kpa", "hz", "c", "h", "s", "m", "kg",
    "%", "mmwg", "cst", "c°",
}


def _very_numeric(text):
    t = str(text or "").strip()
    return bool(re.match(r"^-?\d+(?:[.,]\d+)?$", t))
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse(text):
    if text is None:
        return None
    m = _NUM.match(str(text).strip())
    if not m:
        return None
    return float(m.group(0))


def _clean_text(raw):
    return re.sub(r"[^A-Za-z0-9/._\-% ]+", " ", raw).strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def process_tb(pdf, fields, output_dir, reading_pages, dpi=250, psm=11):
    """Run Template-B whole-page extraction and export a standard report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(parents=True, exist_ok=True)

    from io import BytesIO

    from PIL import Image

    rendered = render_pages(pdf, reading_pages, dpi)
    pil_by_page = {}
    for i, (page_no, pix, w, h) in enumerate(rendered):
        pix.save(str(out / "pages" / f"reading_{i + 1}_page_{page_no}.png"))
        pil = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        pil_by_page[page_no] = pil

    all_rows = ocr_pages(rendered)
    headers = detect_headers(all_rows)
    rows_by_page = {}
    for r in all_rows:
        rows_by_page.setdefault(r.page, []).append(r)

    psm6_by_page = {}
    for page_no, pil in pil_by_page.items():
        psm6_by_page[page_no] = pytesseract_image_to_words(pil, 6, page_no)

    records = []
    detected_source = []
    populated = []
    for field in fields:
        row, score = find_tb_row(all_rows, field)
        page = row.page if row else field.get("page")
        cells = []
        found = row is not None
        flags = []
        if row is not None:
            cells = collect_values_tb(row, field, fields, headers, page,
                                     rows_by_page.get(page, []),
                                     psm6_words=psm6_by_page.get(page))
            if not cells and field.get("allow_text_values"):
                flags.append("no_text_value_captured")
        else:
            flags.append("field_row_not_found")
        rec = FieldRecord(
            field_id=field.get("id"),
            field_name=field.get("name"),
            page=page,
            region=field.get("region", "reading"),
            unit=field.get("unit", ""),
            found=found,
            row_text=row.text if row else "",
            row_confidence=round(score, 4),
            cells=cells,
            flags=flags,
        )
        records.append(rec)
        if rec.found:
            detected_source.append(rec.field_id)
        if rec.cells:
            populated.append(rec.field_id)

    issues = validate_tb(records, detected_source)
    meta = {
        "source": "whole-page Reading extraction (Template B)",
        "template": "PERFORMANCE_EVALUATION",
        "reading_pages": list(reading_pages),
        "crop_count": 0,
        "detected_source_fields": len(detected_source),
        "populated_fields": len(populated),
        "registered_fields": len(records),
        "parser_version": "3.4.0-template-b",
    }
    summary = export(records, issues, out, meta)
    summary["detected_source_fields"] = len(detected_source)
    summary["populated_fields"] = len(populated)
    summary["registered_fields"] = len(records)
    # Also persist detected source fields for the audit / web layer.
    import json as _json
    (out / "template.json").write_text(_json.dumps({
        "template": "PERFORMANCE_EVALUATION",
        "reading_pages": list(reading_pages),
        "detected_source_fields": detected_source,
        "populated_fields": populated,
        "registered_fields": len(records),
    }, indent=2), encoding="utf-8")
    return summary


def validate_tb(records, detected_source):
    issues = []
    return issues
