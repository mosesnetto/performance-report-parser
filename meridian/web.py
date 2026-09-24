import json
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from meridian.audit import audit_report
from meridian.extract import get_explicit_columns
from meridian.voice import briefing_text, synthesize, voice_available

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "web_output"
REPORT_INDEX = OUTPUT_DIR / "reports.json"
JOB_INDEX = OUTPUT_DIR / "jobs.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOB_LOCK = threading.Lock()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_report_index():
    if not REPORT_INDEX.exists():
        return []
    try:
        data = load_json(REPORT_INDEX)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_report_index(items):
    save_json(REPORT_INDEX, items)


def load_job_index():
    if not JOB_INDEX.exists():
        return {}
    try:
        data = load_json(JOB_INDEX)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_job_index(items):
    save_json(JOB_INDEX, items)


def update_job(job_id, **changes):
    with JOB_LOCK:
        jobs = load_job_index()
        item = jobs.get(job_id, {})
        item.update(changes)
        jobs[job_id] = item
        save_job_index(jobs)
        return item


def get_job(job_id):
    return load_job_index().get(job_id)


def sort_report_items(items):
    return sorted(items, key=lambda x: x.get("uploaded_at", ""), reverse=True)


def upsert_report(item):
    items = [x for x in load_report_index() if x.get("job_id") != item.get("job_id")]
    items.append(item)
    save_report_index(sort_report_items(items))


def ensure_index_from_existing_jobs():
    items = load_report_index()
    known = {x.get("job_id") for x in items}

    for output_dir in OUTPUT_DIR.iterdir():
        if not output_dir.is_dir() or not (output_dir / "report.json").exists():
            continue
        job_id = output_dir.name
        if job_id in known:
            continue

        filename = None
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.exists() and upload_dir.is_dir():
            pdfs = list(upload_dir.glob("*.pdf"))
            if pdfs:
                filename = pdfs[0].name
        if not filename:
            filename = f"Report {job_id}"

        summary_path = output_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        items.append({
            "job_id": job_id,
            "filename": filename,
            "uploaded_at": datetime.fromtimestamp(output_dir.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "summary": summary,
        })

    save_report_index(sort_report_items(items))
    return sort_report_items(items)


def normalize_web_data(report):
    rows = []
    columns = set()
    regions = set()
    pages = set()

    for field in report.get("fields", []):
        row = {
            "field_id": field.get("field_id", ""),
            "field_name": field.get("field_name", ""),
            "page": field.get("page"),
            "region": field.get("region", ""),
            "unit": field.get("unit", ""),
            "found": field.get("found", False),
            "row_confidence": field.get("row_confidence", 0),
            "row_text": field.get("row_text", ""),
            "flags": field.get("flags", []),
            "cells": {},
            "columns": [],
            "schema": [],
        }

        regions.add(row["region"])
        if row["page"] is not None:
            pages.add(row["page"])

        schema = get_explicit_columns({
            "id": field.get("field_id", ""),
            "name": field.get("field_name", ""),
        })
        if schema:
            row["schema"] = [str(c) for c in schema]
            row["columns"] = [str(c) for c in schema]
            for schema_column in row["columns"]:
                columns.add(schema_column)

        for cell in field.get("cells", []):
            col = str(cell.get("column", "VALUE"))
            if col.startswith("X_") or col.startswith("VALUE_"):
                col = "VALUE"

            # Skip separator cells (visual delimiters like "|" between sub-measurements)
            # These have EXTRACTION_FAILED + status_indicator flags and value=null
            flags = cell.get("flags", [])
            if cell.get("value") is None and "EXTRACTION_FAILED" in flags and "status_indicator" in flags:
                continue

            columns.add(col)
            row["cells"].setdefault(col, []).append({
                "value": cell.get("value"),
                "raw_text": cell.get("raw_text", ""),
                "unit": cell.get("unit", row["unit"]),
                "confidence": cell.get("confidence", 0),
                "flags": cell.get("flags", []),
                "source_row": cell.get("source_row", row["row_text"]),
                "bbox": cell.get("bbox"),
            })

        rows.append(row)

    def col_key(c):
        c = str(c).upper()
        if c == "VALUE": return (0, 0, c)
        if c == "REF": return (1, 0, c)
        if c == "CALC": return (2, 0, c)
        if c == "AVG": return (3, 0, c)
        if c == "MEAS": return (4, 0, c)
        if c == "SUM": return (5, 0, c)
        if c.startswith("CYL") and c[3:].isdigit(): return (10, int(c[3:]), c)
        if c.startswith("TC") and c[2:].isdigit(): return (20, int(c[2:]), c)
        if c.startswith("BRG") and c[3:].isdigit(): return (30, int(c[3:]), c)
        return (90, 0, c)

    return {
        "metadata": report.get("metadata", {}),
        "rows": rows,
        "columns": sorted(columns, key=col_key),
        "regions": sorted(regions),
        "pages": sorted(pages),
    }


def get_report_data(job_id):
    output_dir = OUTPUT_DIR / job_id
    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.json"
    if not report_path.exists():
        return None
    report = load_json(report_path)
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {"summary": summary, "data": normalize_web_data(report)}


def _normalize_cells(rows):
    """Build the per-field cell map used by the audit/metric helpers.

    Returns {field_id: {column: [{"value","raw_text","confidence",...}]}}.
    """
    out = {}
    for row in rows:
        cells = {}
        for cell in row.get("cells", []):
            col = str(cell.get("column", "VALUE"))
            if col.startswith("X_") or col.startswith("VALUE_"):
                col = "VALUE"
            cells.setdefault(col, []).append(cell)
        out[row.get("field_id", "")] = cells
    return out


def report_date(job_id):
    """Derive the report recording date from the filename or G001 date field."""
    filename = None
    item = next((x for x in ensure_index_from_existing_jobs() if x.get("job_id") == job_id), None)
    if item:
        filename = item.get("filename")
    date_value = None
    report = get_report_data(job_id)
    if report:
        for row in report["data"]["rows"]:
            if str(row.get("field_name", "")).strip().lower() == "date and time of recording":
                for col in ("DATETIME", "VAL1", "VALUE"):
                    for cell in row["cells"].get(col, []):
                        if cell.get("value") is not None:
                            date_value = cell.get("value")
                            break
                    if date_value is not None:
                        break
                break
    return {"recording": date_value, "filename_hint": filename}


def get_report_audit(job_id):
    """Expose the audit engine results for a report (informational only).

    Never modifies extracted data.
    """
    output_dir = OUTPUT_DIR / job_id
    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.json"
    audit_path = output_dir / "audit.json"
    if not report_path.exists():
        return None
    report = load_json(report_path)
    summary = load_json(summary_path) if summary_path.exists() else {}

    audit = None
    if audit_path.exists():
        audit = load_json(audit_path)
    else:
        audit = audit_report(report, summary=summary)

    integrity = audit["integrity"]
    status_counts = {
        "registered_source_fields": integrity.get("registered_source_fields"),
        "unregistered_source_fields": integrity.get("unregistered_source_fields"),
        "fields_missing": integrity.get("fields_missing"),
        "field_count_invariant": integrity.get("expected_count_invariant_holds"),
    }

    ocr_ambiguous = 0
    column_mismatch = 0
    text_values = 0
    errors = 0
    field_status = {}
    for fa in audit["fields"]:
        statuses = fa.get("statuses", [])
        field_status[fa.get("field_id", "")] = {
            "statuses": statuses,
            "findings": fa.get("findings", []),
            "ambiguous": any(s == "OCR_AMBIGUOUS" for s in statuses),
            "mismatch": any(s == "COLUMN_MISMATCH" for s in statuses),
        }
        if any(s == "OCR_AMBIGUOUS" for s in statuses):
            ocr_ambiguous += 1
        if any(s == "COLUMN_MISMATCH" for s in statuses):
            column_mismatch += 1
        if any(s == "TEXT_VALUE" for s in statuses):
            text_values += 1
        if any(s in ("OCR_MALFORMED", "MISSING_VALUE") for s in statuses):
            errors += 1

    return {
        "integrity": integrity,
        "counts": {
            **status_counts,
            "ocr_ambiguous": ocr_ambiguous,
            "column_mismatch": column_mismatch,
            "text_values": text_values,
            "errors": errors,
            "status_counts": integrity.get("status_counts", {}),
        },
        "fields": field_status,
    }


def generate_report_pdf(job_id):
    """Generate a PDF of the extracted measurements table for the given job_id.
    
    Returns a BytesIO object containing the PDF, or None if report not found.
    
    Handles wide tables by splitting measurement columns across multiple horizontal
    page groups, repeating fixed columns on each page group.
    """
    from fpdf import FPDF
    
    report_data = get_report_data(job_id)
    if report_data is None:
        return None
    
    data = report_data["data"]
    rows = data["rows"]
    filename = report_data.get("summary", {}).get("filename", f"report_{job_id}")
    
    # Sanitize text for Latin-1 encoding (built-in fonts)
    def sanitize(text):
        if not isinstance(text, str):
            text = str(text)
        return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2026", "...").replace("\u00a0", " ")
    
    # Create PDF with landscape orientation for wide tables
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Collect all unique measurement column labels across all rows
    all_measurement_labels = []
    for row in rows:
        row_cols = _get_row_columns_for_pdf(row)
        for col in row_cols:
            if col["label"] not in all_measurement_labels:
                all_measurement_labels.append(col["label"])
    
    fixed_headers = ["Field", "Unit", "Region", "Page"]
    fixed_col_widths = [60, 20, 30, 15]  # Field, Unit, Region, Page
    min_meas_width = 12  # minimum width for measurement columns
    page_width = pdf.w - pdf.l_margin - pdf.r_margin  # ~277mm for A4 landscape
    fixed_total = sum(fixed_col_widths)
    available_for_meas = page_width - fixed_total
    
    # Calculate how many measurement columns fit per horizontal page group
    max_meas_per_group = max(1, int(available_for_meas / min_meas_width))
    meas_col_width = available_for_meas / max_meas_per_group if max_meas_per_group > 0 else min_meas_width
    
    # Split measurement labels into groups
    meas_groups = []
    for i in range(0, len(all_measurement_labels), max_meas_per_group):
        meas_groups.append(all_measurement_labels[i:i + max_meas_per_group])
    
    # For each horizontal page group, render all rows
    for group_idx, meas_labels in enumerate(meas_groups):
        # Add first page for this group
        if group_idx == 0:
            pdf.add_page()
            # Title only on first page group
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, sanitize(f"Extracted Measurements - {filename}"), new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(2)
            # Metadata only on first page group
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 5, sanitize(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}"), new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, sanitize(f"Report ID: {job_id}"), new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, sanitize(f"Total fields: {len(rows)}"), new_x="LMARGIN", new_y="NEXT")
            if len(meas_groups) > 1:
                pdf.cell(0, 5, sanitize(f"Horizontal page group: {group_idx + 1} of {len(meas_groups)}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        else:
            pdf.add_page()
            # Group header for subsequent horizontal pages
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, sanitize(f"Extracted Measurements - {filename} (Horizontal Group {group_idx + 1} of {len(meas_groups)})"), new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(3)
        
        # Build headers for this group
        group_headers = [sanitize(h) for h in fixed_headers] + [sanitize(h) for h in meas_labels]
        group_col_widths = fixed_col_widths + [meas_col_width] * len(meas_labels)
        
        def draw_group_header():
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_fill_color(240, 240, 240)
            for i, header in enumerate(group_headers):
                pdf.cell(group_col_widths[i], 6, header, border=1, fill=True, align="C")
            pdf.ln()
        
        def draw_group_row(row, row_idx):
            pdf.set_font("Helvetica", "", 6.5)
            
            # Alternate row colors
            if row_idx % 2 == 0:
                pdf.set_fill_color(255, 255, 255)
            else:
                pdf.set_fill_color(248, 250, 252)
            
            # Fixed columns
            field_name = sanitize(row.get("field_name", "")[:40])
            unit = sanitize(row.get("unit", "") or "-")
            region = sanitize(row.get("region", "") or "-")
            page = sanitize(str(row.get("page", "")) or "-")
            
            values = [field_name, unit, region, page]
            
            # Measurement columns for this group
            row_cols = _get_row_columns_for_pdf(row)
            row_col_map = {col["label"]: col for col in row_cols}
            
            for label in meas_labels:
                if label in row_col_map:
                    col_info = row_col_map[label]
                    cells = _get_cells_for_pdf(row, col_info["column"])
                    if col_info["index"] < len(cells):
                        cell = cells[col_info["index"]]
                        val = sanitize(_display_value_for_pdf(cell))
                    else:
                        val = "-"
                else:
                    val = ""
                values.append(val)
            
            # Calculate row height needed
            max_lines = 1
            for i, val in enumerate(values):
                char_width = pdf.get_string_width("x")
                chars_per_line = max(1, group_col_widths[i] / char_width)
                lines = max(1, len(str(val)) / chars_per_line + 0.999)
                max_lines = max(max_lines, int(lines))
            
            row_height = max(6, max_lines * 3.5)
            
            # Check if we need a new page (vertical pagination)
            if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
                pdf.add_page()
                draw_group_header()
            
            y_before = pdf.get_y()
            x_start = pdf.get_x()
            
            for i, val in enumerate(values):
                x = x_start + sum(group_col_widths[:i])
                pdf.set_xy(x, y_before)
                # Draw cell border and background
                pdf.rect(x, y_before, group_col_widths[i], row_height, style="DF")
                pdf.set_xy(x + 1, y_before + 0.5)
                pdf.multi_cell(group_col_widths[i] - 2, 3.5, str(val), border=0, align="R" if i >= 4 else "L")
            
            pdf.set_y(y_before + row_height)
        
        draw_group_header()
        
        for idx, row in enumerate(rows):
            draw_group_row(row, idx)
    
    # Output to BytesIO
    pdf_bytes = pdf.output()
    return BytesIO(pdf_bytes)


def _get_row_columns_for_pdf(row):
    """Get measurement columns for a row, matching the JS rowColumnsForRow logic."""
    if not row or not row.get("cells"):
        return []
    
    keys = row.get("schema", [])
    if not keys:
        keys = list(row["cells"].keys())
    
    # Filter out VALUE
    keys = [k for k in keys if str(k).strip().upper() != "VALUE"]
    
    if not keys:
        return []
    
    result = []
    for column in keys:
        cells = _get_cells_for_pdf(row, column)
        max_cells = max(len(cells), 1)
        
        if max_cells == 1:
            result.append({"key": f"{column}::0", "column": column, "index": 0, "label": column})
        else:
            for i in range(max_cells):
                result.append({"key": f"{column}::{i}", "column": column, "index": i, "label": f"{column} {i+1}"})
    
    return result


def _get_cells_for_pdf(row, column):
    """Get cells for a specific column from a row."""
    cells = row.get("cells", {}).get(column, [])
    if not isinstance(cells, list):
        cells = [cells]
    return cells


def _display_value_for_pdf(cell):
    """Get display value for a cell, matching the JS displayValue logic."""
    if not cell:
        return "—"
    
    if cell.get("value") is not None:
        return str(cell["value"])
    
    if cell.get("raw_text"):
        return str(cell["raw_text"])
    
    return "—"


def parser_worker(job_id, pdf_path, output_dir, filename):
    update_job(job_id, status="processing", stage="running_parser", progress=15, message="Running OCR and extracting fields...")
    command = [
        sys.executable, "-m", "app.cli",
        "--mode", "pdf",
        "--pdf", str(pdf_path),
        "--pages", "auto",
        "--engine", "v4",
        "--output", str(output_dir),
    ]

    try:
        result = subprocess.run(command, cwd=str(BASE_DIR), capture_output=True, text=True)
        report_path = output_dir / "report.json"
        summary_path = output_dir / "summary.json"

        if result.returncode != 0 or not report_path.exists():
            error_text = (result.stdout + "\n" + result.stderr).strip()
            update_job(job_id, status="error", stage="failed", progress=100,
                       message="Parser failed.", error=error_text or "Parser failed.",
                       stdout=result.stdout, stderr=result.stderr)
            return

        summary = load_json(summary_path) if summary_path.exists() else {}
        item = {
            "job_id": job_id,
            "filename": filename,
            "uploaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": summary,
        }
        upsert_report(item)

        update_job(job_id, status="processing", stage="validating", progress=85,
                   message="Validating extraction against source PDF...")

        try:
            from meridian.validate_fidelity import validate_single_pdf
            validation_result = validate_single_pdf(str(pdf_path), str(output_dir))
            if validation_result is not None:
                validation_data = {
                    "pdf_name": validation_result.pdf_name,
                    "template": validation_result.template,
                    "reading_pages": validation_result.reading_pages,
                    "source_physical_cells": validation_result.source_physical_cells,
                    "html_cells_matched": validation_result.html_cells_matched,
                    "discrepancy_count": len(validation_result.discrepancies),
                    "discrepancy_counts": {},
                    "discrepancies": [],
                }
                for d in validation_result.discrepancies:
                    cat = d.category
                    validation_data["discrepancy_counts"][cat] = (
                        validation_data["discrepancy_counts"].get(cat, 0) + 1
                    )
                    validation_data["discrepancies"].append({
                        "category": d.category,
                        "region": d.region,
                        "field_id": d.field_id,
                        "field_name": d.field_name,
                        "source_value": d.source_value,
                        "extraction_value": d.extraction_value,
                        "source_column": d.source_column,
                        "extraction_column": d.extraction_column,
                        "details": d.details,
                    })
                validation_path = output_dir / "validation.json"
                save_json(validation_path, validation_data)
                update_job(job_id, validation=validation_data)
        except Exception as val_exc:
            update_job(job_id, validation_error=str(val_exc))

        update_job(job_id, status="complete", stage="complete", progress=100,
                   message="Complete", summary=summary, filename=filename,
                   stdout=result.stdout, stderr=result.stderr)
        # Phase 7 integration: update registry and evaluate triggers
        try:
            from meridian.analytics import load_union_catalog
            from meridian.registry_manager import discover_from_validator
            from meridian.triggers import evaluate_triggers
            catalog = load_union_catalog()
            if catalog:
                discover_from_validator(catalog)
            evaluate_triggers()
        except Exception:
            pass  # Integration failures should not break parser
    except Exception as exc:
        update_job(job_id, status="error", stage="failed", progress=100,
                   message="Unexpected parser error.", error=str(exc))


@app.get("/")
def index():
    return render_template("index.html", reports=ensure_index_from_existing_jobs())


@app.get("/graphs")
def graphs():
    return render_template("graphs.html", reports=ensure_index_from_existing_jobs())


@app.get("/api/reports")
def reports_api():
    items = []
    for item in ensure_index_from_existing_jobs():
        job_id = item.get("job_id")
        counts = {}
        audit = get_report_audit(job_id) if job_id else None
        if audit:
            counts = audit["counts"]
        items.append({
            **item,
            "audit_counts": counts,
            "report_date": report_date(job_id) if job_id else {"recording": None, "filename_hint": item.get("filename")},
        })
    return jsonify(ok=True, reports=items)


@app.get("/api/report/<job_id>")
def report_api(job_id):
    result = get_report_data(job_id)
    if result is None:
        return jsonify(ok=False, error="Report not found."), 404
    item = next((x for x in ensure_index_from_existing_jobs() if x.get("job_id") == job_id), None)
    audit = get_report_audit(job_id)
    return jsonify(ok=True, job_id=job_id,
                   filename=(item or {}).get("filename", job_id),
                   uploaded_at=(item or {}).get("uploaded_at", ""),
                   summary=result["summary"], data=result["data"],
                   report_date=report_date(job_id),
                   audit_counts=(audit or {}).get("counts", {}))


@app.get("/api/report/<job_id>/audit")
def report_audit_api(job_id):
    audit = get_report_audit(job_id)
    if audit is None:
        return jsonify(ok=False, error="Report not found."), 404
    item = next((x for x in ensure_index_from_existing_jobs() if x.get("job_id") == job_id), None)
    return jsonify(ok=True, job_id=job_id,
                   filename=(item or {}).get("filename", job_id),
                   integrity=audit["integrity"],
                   counts=audit["counts"],
                   field_status=audit["fields"],


)


@app.get("/api/report/<job_id>/validation")
def report_validation_api(job_id):
    """Return fidelity validation results for a report."""
    output_dir = OUTPUT_DIR / job_id
    validation_path = output_dir / "validation.json"
    if validation_path.exists():
        validation_data = load_json(validation_path)
        return jsonify(ok=True, job_id=job_id, validation=validation_data)

    pdf_path = None
    upload_dir = UPLOAD_DIR / job_id
    if upload_dir.exists():
        pdfs = list(upload_dir.glob("*.pdf"))
        if pdfs:
            pdf_path = str(pdfs[0])

    if pdf_path is None:
        return jsonify(ok=False, error="No PDF found for this job."), 404

    try:
        from meridian.validate_fidelity import validate_single_pdf
        result = validate_single_pdf(pdf_path, str(output_dir))
        if result is None:
            return jsonify(ok=False, error="Validation could not be performed."), 404

        validation_data = {
            "pdf_name": result.pdf_name,
            "template": result.template,
            "reading_pages": result.reading_pages,
            "source_physical_cells": result.source_physical_cells,
            "html_cells_matched": result.html_cells_matched,
            "discrepancy_count": len(result.discrepancies),
            "discrepancy_counts": {},
            "discrepancies": [],
        }
        for d in result.discrepancies:
            cat = d.category
            validation_data["discrepancy_counts"][cat] = (
                validation_data["discrepancy_counts"].get(cat, 0) + 1
            )
            validation_data["discrepancies"].append({
                "category": d.category,
                "region": d.region,
                "field_id": d.field_id,
                "field_name": d.field_name,
                "source_value": d.source_value,
                "extraction_value": d.extraction_value,
                "source_column": d.source_column,
                "extraction_column": d.extraction_column,
                "details": d.details,
            })

        save_json(validation_path, validation_data)
        return jsonify(ok=True, job_id=job_id, validation=validation_data)
    except Exception as exc:
        return jsonify(ok=False, error=f"Validation error: {str(exc)}"), 500


@app.get("/api/report/<job_id>/pdf")
def report_pdf_api(job_id):
    """Download extracted measurements as PDF."""
    pdf_buffer = generate_report_pdf(job_id)
    if pdf_buffer is None:
        return jsonify(ok=False, error="Report not found."), 404
    
    item = next((x for x in ensure_index_from_existing_jobs() if x.get("job_id") == job_id), None)
    filename = (item or {}).get("filename", f"report_{job_id}")
    # Sanitize filename for download
    safe_filename = Path(filename).stem + "_extracted_measurements.pdf"
    
    pdf_buffer.seek(0)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=safe_filename
    )


@app.get("/api/job/<job_id>")
def job_api(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify(ok=False, error="Job not found."), 404
    return jsonify(ok=True, job=job)


@app.post("/api/parse")
def parse():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(ok=False, error="No PDF file received."), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify(ok=False, error="Only PDF files are supported."), 400

    safe_name = Path(uploaded.filename).name
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    output_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / safe_name
    uploaded.save(str(pdf_path))

    update_job(job_id, status="queued", stage="queued", progress=5,
               message="PDF received. Waiting for parser...", filename=safe_name,
               created_at=datetime.now().astimezone().isoformat(timespec="seconds"))

    EXECUTOR.submit(parser_worker, job_id, pdf_path, output_dir, safe_name)

    return jsonify(ok=True, job_id=job_id, filename=safe_name, status="queued"), 202


@app.get("/validator")
def validator():
    from meridian.analytics import load_union_catalog
    from meridian.registry_manager import discover_from_validator, get_registry
    registry = get_registry()
    # Auto-discover from existing validator output (Phase 2 discovery trigger)
    catalog = load_union_catalog()
    if catalog:
        discover_from_validator(catalog)
        registry = get_registry()
    fields_list = [{"key": k, **v} for k, v in registry.get("fields", {}).items()]
    tags_list = [{"key": k, **v} for k, v in registry.get("tags", {}).items()]
    return render_template("validator.html",
                           fields=fields_list,
                           tags=tags_list,
                           registry=registry)


@app.post("/validator/field/add")
def validator_add_field():

    from meridian.registry_manager import add_field
    data = request.get_json(silent=True) or {}
    entry = add_field(
        canonical_name=data.get("canonical_name", ""),
        aliases=data.get("aliases", []),
        unit=data.get("unit", ""),
        region=data.get("region", ""),
        description=data.get("description", ""),
        tag=data.get("tag", ""),
        field_id=data.get("field_id", ""),
        status=data.get("status", "MANUAL"),
    )
    return jsonify(ok=True, entry=entry)


@app.post("/validator/field/remove")
def validator_remove_field():

    from meridian.registry_manager import remove_field
    data = request.get_json(silent=True) or {}
    ok = remove_field(data.get("canonical_name", ""))
    return jsonify(ok=ok)


@app.post("/validator/tag/add")
def validator_add_tag():

    from meridian.registry_manager import add_tag
    data = request.get_json(silent=True) or {}
    entry = add_tag(
        canonical_tag=data.get("canonical_tag", ""),
        aliases=data.get("aliases", []),
        description=data.get("description", ""),
        status=data.get("status", "MANUAL"),
    )
    return jsonify(ok=True, entry=entry)


@app.post("/validator/tag/remove")
def validator_remove_tag():

    from meridian.registry_manager import remove_tag
    data = request.get_json(silent=True) or {}
    ok = remove_tag(data.get("canonical_tag", ""))
    return jsonify(ok=ok)


@app.get("/analytics")
def analytics():
    return render_template("analytics.html")


@app.post("/analytics/query")
def analytics_query():

    from meridian.llm_integration import query_with_llm
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not question:
        return jsonify(ok=False, error="No question provided"), 400
    result = query_with_llm(question)
    return jsonify(ok=True, result=result)


@app.get("/analytics/api/fields")
def analytics_fields_api():
    from meridian.query_tools import search_fields
    results = search_fields("")
    return jsonify(ok=True, fields=[{"canonical_name": f.get("canonical_field"), "aliases": f.get("aliases", []), "status": f.get("status", "")} for f in results[:20]])


@app.get("/analytics/api/reports")
def analytics_reports_api():
    from meridian.query_tools import list_reports
    return jsonify(ok=True, reports=list_reports())


@app.get("/triggers")
def triggers():
    from meridian.triggers import _load_triggers, evaluate_triggers
    data = _load_triggers()
    results = evaluate_triggers()
    return render_template("triggers.html", triggers=data.get("triggers", []), results=results)


@app.post("/triggers/add")
def triggers_add():

    from meridian.triggers import add_trigger
    data = request.get_json(silent=True) or {}
    entry = add_trigger(
        name=data.get("name", ""),
        description=data.get("description", ""),
        field=data.get("field", ""),
        tag=data.get("tag", ""),
        operator=data.get("operator", ">"),
        threshold=float(data.get("threshold", 0)),
        grouping=data.get("grouping", ""),
        scope=data.get("scope", "new_reports"),
        active=data.get("active", True),
    )
    return jsonify(ok=True, entry=entry)


@app.post("/triggers/remove")
def triggers_remove():

    from meridian.triggers import remove_trigger
    data = request.get_json(silent=True) or {}
    ok = remove_trigger(data.get("trigger_id", ""))
    return jsonify(ok=ok)


@app.get("/triggers/evaluate")
def triggers_evaluate():
    from meridian.triggers import evaluate_triggers
    results = evaluate_triggers()
    return jsonify(ok=True, results=results)


# --- MERIDIAN Voice (offline Piper briefings) -------------------------

@app.get("/api/report/<job_id>/briefing")
def report_briefing_api(job_id):
    """Natural-language briefing text for a report (no audio)."""
    result = get_report_data(job_id)
    if result is None:
        return jsonify(ok=False, error="Report not found."), 404
    text = briefing_text(result)
    return jsonify(ok=True, job_id=job_id,
                   briefing=text,
                   voice_available=voice_available())


@app.get("/api/report/<job_id>/briefing/audio")
def report_briefing_audio(job_id):
    """Synthesize the report briefing to WAV via offline Piper."""
    if not voice_available():
        return jsonify(ok=False, error="Voice unavailable: install piper and "
                                       "set MERIDIAN_PIPER_MODEL."), 501
    result = get_report_data(job_id)
    if result is None:
        return jsonify(ok=False, error="Report not found."), 404
    wav_path = OUTPUT_DIR / job_id / "briefing.wav"
    if not wav_path.exists():
        ok = synthesize(briefing_text(result), wav_path)
        if not ok:
            return jsonify(ok=False, error="Voice synthesis failed."), 500
    return send_file(str(wav_path), mimetype="audio/wav",
                     as_attachment=False,
                     download_name=f"{job_id}_briefing.wav",
                     conditional=True)


if __name__ == "__main__":
    ensure_index_from_existing_jobs()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
