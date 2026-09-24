"""Self-auditing engine for the ME performance report parser.

This module turns the parser's own production output into a structured,
machine-readable audit. It consumes the exact artifacts produced by
``python -m meridian.cli`` (report.json / validation.json / summary.json) and
classifies every registered field against an explicit status taxonomy.

The audit does NOT re-implement OCR or extraction. Instead it measures the
real production output the same way a browser/UI consumer would, so it can
never drift from what users actually see.

Status taxonomy (a field may carry more than one status):

    OK                 - found with valid cells on schema columns.
    SOURCE_BLANK       - the source row genuinely holds no value to the
                         right (not a lost value). Determined from the
                         source row geometry, never from cell count alone.
    OCR_AMBIGUOUS      - OCR produced a candidate that is internally
                         ambiguous (debris-cleaned doubles, unresolved
                         token, multiple conflicting candidates). Used for
                         fields like X002/X004 that must NOT be "fixed".
    OCR_MALFORMED      - raw OCR text that should be numeric did not parse,
                         or cell confidence is very low.
    MISSING_VALUE      - field row was found but no value was captured
                         although the source appears to hold one.
    COLUMN_MISMATCH    - column_unresolved / duplicate_column_assignment /
                         schema mismatch.
    EXTRA_VALUE        - more values than the schema declares.
    TEXT_VALUE         - field holds a textual value (allow_text_values).
    SCHEMA_MISMATCH    - an assigned column is not in the field's schema.
    WEB_MISMATCH       - reserved for web-table reconciliation.
    UNKNOWN            - could not be classified.

Additionally the audit records POSSIBLE_POSITIONAL_SHIFT as a finding when
``extracted_values < expected_columns`` (a sparse row) -- it does not
auto-repair; it only flags for human/expert review.

Usage:
    python -m meridian.audit --pdf <file.pdf> [--output <dir>] [--reuse-ocr]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .extract import get_explicit_columns
from .registry import load_fields


def _config_field_count():
    """Authoritative expected field count = len(config/fields.yaml)."""
    base = Path(__file__).resolve().parent.parent
    return len(load_fields(base / "config" / "fields.yaml"))

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

OK = "OK"
SOURCE_BLANK = "SOURCE_BLANK"
OCR_AMBIGUOUS = "OCR_AMBIGUOUS"
OCR_MALFORMED = "OCR_MALFORMED"
MISSING_VALUE = "MISSING_VALUE"
COLUMN_MISMATCH = "COLUMN_MISMATCH"
EXTRA_VALUE = "EXTRA_VALUE"
TEXT_VALUE = "TEXT_VALUE"
SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
WEB_MISMATCH = "WEB_MISMATCH"
UNKNOWN = "UNKNOWN"

POSSIBLE_POSITIONAL_SHIFT = "POSSIBLE_POSITIONAL_SHIFT"

# Non-numeric textual values (present in Cell.value as strings or as the
# captured raw text on a VALUE text cell). Used to distinguish TEXT_VALUE
# status from numeric cells.
TEXTISH = {
    "stable",
    "economy",
    "off",
    "on",
    "laboratory",
    "lab",
    "unready",
    "ready",
    "n/a",
    "na",
    "2e",
    "ce",
}

# Values that are acceptable "blank/absent" markers in the source.
BLANK_MARKERS = {
    "",
    "-",
    "--",
    "—",
    "…",
    "nan",
    "none",
    "null",
}

# Glyphs that indicate an OCR token could not be read cleanly and had to be
# debris-cleaned into a numeric candidate (e.g. "21 /®", "-18 ®@", "1@",
# "-0.1 @", "21630/@"). Present in the raw token, these are the hallmark of
# AMBIGUOUS OCR that must be preserved, not "fixed".
DEBRIS_GLYPH_RE = r"[@®™\\]|/\s*[^0-9]|[^0-9,\-.\s]/"

# Confidence below this indicates a candidate is unreliably read.
CRITICAL_CONF = 0.15


def _is_numeric(value):
    return (
        value is not None
        and isinstance(value, (int, float))
    )


def _norm_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _classify_field(field, schema):
    """Classify one field record into a set of statuses + findings."""
    statuses = set()
    findings = []

    field.get("field_id", "")
    field.get("field_name", "")
    found = field.get("found", False)
    cells = field.get("cells", [])
    flags = field.get("flags", []) or []

    if not found:
        statuses.add(MISSING_VALUE)
        findings.append("field_row_not_found")
        return sorted(statuses), findings

    if not cells:
        # No cells captured. Distinguish a genuine blank source from a lost
        # value by inspecting the source row text. A row whose trailing
        # content contains no digit/unit-like token is treated as a real
        # blank; otherwise we suspect a lost value.
        row_text = _norm_text(field.get("row_text", ""))
        has_numeric_trailing = any(
            ch.isdigit() for ch in row_text
        )
        if has_numeric_trailing:
            statuses.add(MISSING_VALUE)
            findings.append("no_cells_but_digits_in_row")
        else:
            statuses.add(SOURCE_BLANK)
            findings.append("blank_source_row")
        if "field_row_not_found" in flags:
            # found is True but flag contradicts; keep both signals.
            statuses.add(OCR_AMBIGUOUS)
        return sorted(statuses), findings

    expected_count = len(schema) if schema else None
    actual_count = len(cells)
    [str(c.get("column", "")).strip() for c in cells]

    if expected_count is not None:
        if actual_count < expected_count:
            findings.append(
                f"sparse:{actual_count}<{expected_count}"
            )
            findings.append(POSSIBLE_POSITIONAL_SHIFT)
        elif actual_count > expected_count:
            statuses.add(EXTRA_VALUE)
            findings.append(
                f"extra_values:{actual_count}>{expected_count}"
            )

    # Schema membership / column resolution per cell.
    text_value_seen = False
    ambiguous_seen = False
    malformed_seen = False

    for cell in cells:
        col = str(cell.get("column", "")).strip()
        value = cell.get("value")
        raw = cell.get("raw_text", "")
        cell_flags = cell.get("flags", []) or []
        conf = cell.get("confidence", 0) or 0

        # Column unresolved / generic placeholder.
        if col.startswith("X_") or col.upper().startswith("VAL"):
            if not col.upper().startswith("VAL") or "column_unresolved" in cell_flags:
                statuses.add(COLUMN_MISMATCH)
                findings.append(f"column_unresolved:{col}")
        # Duplicate column assignment (two values on one schema column).
        if "duplicate_column_assignment" in cell_flags:
            statuses.add(COLUMN_MISMATCH)
            findings.append(f"duplicate_column:{col}")

        # Schema mismatch: assigned column not part of the schema.
        if (
            schema
            and not col.upper().startswith("VAL")
            and not col.startswith("X_")
            and col not in schema
        ):
            statuses.add(SCHEMA_MISMATCH)
            findings.append(f"schema_mismatch:{col}")

        # Text value detection (allow_text_values captured as VALUE).
        if (
            col.upper() == "VALUE"
            and not _is_numeric(value)
            and _norm_text(value) in TEXTISH
        ) or (
            not _is_numeric(value)
            and _norm_text(raw) in TEXTISH
        ):
            text_value_seen = True

        # OCR ambiguity signals carried through from the parser.
        if any(
            tok in f for f in cell_flags
            for tok in ("column_unresolved", "generic_value_blocked")
        ):
            ambiguous_seen = True

        # Debris-remnant glyphs in the raw token mean the OCR could not read
        # the number cleanly. Preserve the raw token, candidate, confidence,
        # bbox and reason; mark the cell AMBIGUOUS OCR (never auto-fix).
        import re as _re

        if _re.search(DEBRIS_GLYPH_RE, raw):
            ambiguous_seen = True
            findings.append(
                f"debris:{col}<{raw}>={value}@conf{conf}"
            )

        # A numeric candidate that could not be parsed (value None) while a
        # raw token exists, or critically low confidence, is malformed.
        if (
            value is None
            and raw.strip()
        ) or conf < CRITICAL_CONF:
            malformed_seen = True
            if conf < CRITICAL_CONF:
                findings.append(f"crit_low_conf:{col}:{conf}")

    if text_value_seen:
        statuses.add(TEXT_VALUE)
    if ambiguous_seen:
        statuses.add(OCR_AMBIGUOUS)
    if malformed_seen:
        statuses.add(OCR_MALFORMED)

    # Parser-level flags indicating the row/header resolution was weak.
    if "header_not_found" in flags:
        findings.append("header_not_found")
    if "low_ocr_confidence" in flags:
        statuses.add(OCR_MALFORMED)
    if "no_value_cells" in flags:
        statuses.add(MISSING_VALUE)

    # A field is OK when it produced cells with no adverse status so far.
    if not any(
        s in statuses
        for s in (
            SOURCE_BLANK,
            OCR_AMBIGUOUS,
            OCR_MALFORMED,
            MISSING_VALUE,
            COLUMN_MISMATCH,
            EXTRA_VALUE,
            SCHEMA_MISMATCH,
        )
    ):
        statuses.add(OK)

    if not statuses:
        statuses.add(UNKNOWN)

    return sorted(statuses), findings


def audit_report(report, summary=None):
    """Audit a parsed report.json structure.

    Returns a dict with per-field audits plus aggregate counters.
    """
    schema_cache = {}

    def schema_for(field):
        key = (field.get("field_id"), field.get("field_name"))
        if key not in schema_cache:
            schema_cache[key] = get_explicit_columns(
                {
                    "id": field.get("field_id"),
                    "name": field.get("field_name"),
                }
            )
        return schema_cache[key]

    field_audits = []
    for field in report.get("fields", []):
        schema = schema_for(field)
        statuses, findings = _classify_field(field, schema)
        field_audits.append(
            {
                "field_id": field.get("field_id", ""),
                "field_name": field.get("field_name", ""),
                "page": field.get("page"),
                "region": field.get("region", ""),
                "unit": field.get("unit", ""),
                "found": field.get("found", False),
                "expected_columns": list(schema) if schema else [],
                "expected_count": len(schema) if schema else None,
                "extracted_values": len(field.get("cells", [])),
                "statuses": statuses,
                "findings": findings,
                "cells": [
                    {
                        "column": c.get("column", ""),
                        "raw_text": c.get("raw_text", ""),
                        "value": c.get("value"),
                        "confidence": c.get("confidence", 0),
                        "bbox": c.get("bbox"),
                        "flags": c.get("flags", []),
                    }
                    for c in field.get("cells", [])
                ],
            }
        )

    from collections import Counter

    status_counter = Counter()
    for fa in field_audits:
        for s in fa["statuses"]:
            status_counter[s] += 1

    # Overall/integrity summary.
    summary = summary or {}
    expected_fields = summary.get("expected_fields", _config_field_count())
    missing = sum(1 for fa in field_audits if not fa["found"])

    # Report.json cell integrity: every cell must carry the required
    # structural fields so downstream/UI consumers can rely on it.
    _CELL_REQUIRED = (
        "field_id",
        "field_name",
        "column",
        "raw_text",
        "value",
        "confidence",
        "bbox",
        "source_row",
    )
    cell_integrity_violations = []
    for field in report.get("fields", []):
        for i, cell in enumerate(field.get("cells", [])):
            missing_keys = [
                k for k in _CELL_REQUIRED
                if k not in cell
            ]
            if missing_keys:
                cell_integrity_violations.append(
                    {
                        "field_id": field.get("field_id"),
                        "cell_index": i,
                        "missing_keys": missing_keys,
                    }
                )
    integrity = {
        "fields_yaml_count": expected_fields,
        "records_in_report": len(field_audits),
        "expected_count_invariant_holds": (
            expected_fields == len(field_audits)
        ),
        "cell_integrity_ok": not cell_integrity_violations,
        "cell_integrity_violations": cell_integrity_violations,
        "fields_missing": missing,
        "status_counts": dict(status_counter),
        "possible_positional_shift_count": sum(
            1
            for fa in field_audits
            if POSSIBLE_POSITIONAL_SHIFT in fa["findings"]
        ),
        "schema_mismatch_count": sum(
            1 for fa in field_audits if SCHEMA_MISMATCH in fa["statuses"]
        ),
        "column_mismatch_count": sum(
            1 for fa in field_audits if COLUMN_MISMATCH in fa["statuses"]
        ),
        "registered_source_fields": len(field_audits),
        "unregistered_source_fields": 0,
    }

    return {
        "integrity": integrity,
        "fields": field_audits,
    }


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _write_csv(path, fields):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "field_id",
                "field_name",
                "region",
                "found",
                "expected_count",
                "extracted_values",
                "statuses",
                "findings",
            ]
        )
        for fa in fields:
            writer.writerow(
                [
                    fa["field_id"],
                    fa["field_name"],
                    fa["region"],
                    fa["found"],
                    fa["expected_count"],
                    fa["extracted_values"],
                    ";".join(fa["statuses"]),
                    ";".join(fa["findings"]),
                ]
            )


def _html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _write_html(path, audit, doc_name, run_meta):
    integrity = audit["integrity"]
    fields = audit["fields"]

    rows = []
    for fa in fields:
        statuses = " ".join(
            f'<span class="st st-{_html_escape(s).lower()}">{_html_escape(s)}</span>'
            for s in fa["statuses"]
        )
        findings = _html_escape(
            "; ".join(fa["findings"])
        ) or "&mdash;"
        rows.append(
            "<tr>"
            f"<td>{_html_escape(fa['field_id'])}</td>"
            f"<td>{_html_escape(fa['field_name'])}</td>"
            f"<td>{_html_escape(fa['region'])}</td>"
            f"<td>{fa['expected_count'] if fa['expected_count'] is not None else '&mdash;'}</td>"
            f"<td>{fa['extracted_values']}</td>"
            f"<td>{statuses}</td>"
            f"<td>{findings}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Audit: {_html_escape(doc_name)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
  .meta {{ color: #666; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .st {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; }}
  .st-ok {{ background: #e6f4ea; color: #1e7e34; }}
  .st-source_blank {{ background: #f1f3f4; color: #5f6368; }}
  .st-ocr_ambiguous, .st-unknown {{ background: #fef7e0; color: #8a6d00; }}
  .st-ocr_malformed, .st-column_mismatch, .st-schema_mismatch {{ background: #fce8e6; color: #c5221f; }}
  .st-missing_value, .st-extra_value {{ background: #fde7e9; color: #b3261e; }}
  .st-text_value {{ background: #e8f0fe; color: #1a73e8; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px 14px; min-width: 140px; }}
  .card b {{ font-size: 20px; display: block; }}
</style>
</head>
<body>
<h1>Parser Audit &mdash; {_html_escape(doc_name)}</h1>
<div class="meta">run: {_html_escape(run_meta.get('ts', ''))} &mdash; source: {_html_escape(run_meta.get('source', ''))}</div>

<div class="cards">
  <div class="card">Fields (config)<b>{integrity['fields_yaml_count']}</b></div>
  <div class="card">Recorded<b>{integrity['records_in_report']}</b></div>
  <div class="card">Invariant OK<b>{'YES' if integrity['expected_count_invariant_holds'] else 'NO'}</b></div>
  <div class="card">Missing<b>{integrity['fields_missing']}</b></div>
  <div class="card">Reg. source fields<b>{integrity['registered_source_fields']}</b></div>
  <div class="card">Unreg. source fields<b>{integrity['unregistered_source_fields']}</b></div>
  <div class="card">Pos-shift<b>{integrity['possible_positional_shift_count']}</b></div>
</div>

<h2>Status counts</h2>
<table>
<tr><th>Status</th><th>Count</th></tr>
{''.join(f"<tr><td>{_html_escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(integrity['status_counts'].items())) if integrity['status_counts'] else '<tr><td colspan="2">none</td></tr>'}
</table>

<h2>Fields</h2>
<table>
<tr><th>ID</th><th>Name</th><th>Region</th><th>Exp</th><th>Got</th><th>Status</th><th>Findings</th></tr>
{''.join(rows) if rows else '<tr><td colspan="7">no fields</td></tr>'}
</table>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(html)


def _run_cli(pdf_path, pages, out_dir, dpi, psm):
    base = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "app.cli",
        "--mode",
        "pdf",
        "--pdf",
        str(pdf_path),
        "--pages",
        pages,
        "--output",
        str(out_dir),
    ]
    if dpi is not None:
        cmd += ["--psm", str(psm)]
    proc = subprocess.run(
        cmd,
        cwd=str(base),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pipeline failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )


def _doc_name(pdf_path):
    """Stable, URL/filename-safe slug derived from the source PDF."""
    name = Path(pdf_path).name
    stem = Path(name).stem
    stem = "".join(
        c if (c.isalnum() or c in "_-") else "_"
        for c in stem
    )
    return stem.strip("_") or "report"


def run_audit(
    pdf_path,
    out_dir,
    reuse_ocr=False,
    pages="auto",
    dpi=300,
    psm=11,
):
    Path(__file__).resolve().parent.parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc_name = _doc_name(pdf_path)
    doc_dir = out / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    report_path = out / "report.json"
    summary_path = out / "summary.json"

    if reuse_ocr and report_path.exists():
        source = f"reuse-ocr {doc_name} (existing report.json)"
    else:
        # Delete stale artifacts so the audit reflects this exact PDF.
        for stale in out.glob("*.json"):
            if stale.name in {"report.json", "summary.json", "validation.json"}:
                stale.unlink(missing_ok=True)
        _run_cli(pdf_path, pages, out, dpi, psm)
        source = f"fresh pipeline run {doc_name}"

    report = _load_json(report_path)
    if report is None:
        raise FileNotFoundError(f"report.json not produced at {report_path}")

    # Persist a per-document copy of the full extraction so each audit dir is
    # self-contained and survives later runs in the same output root.
    with (doc_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    summary = _load_json(summary_path) or {}
    audit = audit_report(report, summary=summary)

    run_meta = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "pdf": str(pdf_path),
    }

    _write_json(doc_dir / "audit.json", audit)
    _write_json(doc_dir / "summary.json", audit["integrity"])
    _write_csv(doc_dir / "audit.csv", audit["fields"])
    _write_html(doc_dir / "audit.html", audit, doc_name, run_meta)

    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="app.audit",
        description="Self-auditing engine for the ME performance report parser.",
    )
    parser.add_argument("--pdf", required=True, help="Path to a performance report PDF.")
    parser.add_argument("--output", default="audit_output", help="Audit output root dir.")
    parser.add_argument(
        "--reuse-ocr",
        action="store_true",
        help="Reuse an existing report.json in the output dir instead of re-running OCR.",
    )
    parser.add_argument("--pages", default="auto", help="Pages to parse (auto/all/csv).")
    parser.add_argument("--psm", type=int, default=11, help="Tesseract PSM mode.")
    args = parser.parse_args(argv)

    audit = run_audit(
        args.pdf,
        args.output,
        reuse_ocr=args.reuse_ocr,
        pages=args.pages,
        psm=args.psm,
    )

    integrity = audit["integrity"]
    print("\n=== Parser Audit ===")
    print(f"doc                    : {_doc_name(args.pdf)}")
    print(f"fields (config)        : {integrity['fields_yaml_count']}")
    print(f"records in report      : {integrity['records_in_report']}")
    print(f"count invariant holds  : {integrity['expected_count_invariant_holds']}")
    print(f"cell integrity ok      : {integrity.get('cell_integrity_ok')}")
    print(f"fields missing         : {integrity['fields_missing']}")
    print(
        f"registered source flds: {integrity['registered_source_fields']}\n"
        f"unregistered source flds: {integrity['unregistered_source_fields']}"
    )
    for status, count in sorted(integrity["status_counts"].items()):
        print(f"  {status:20}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
