import shutil
from pathlib import Path

import pymupdf

from .crop_pipeline import process as process_crops
from .reading_layout import boxes_for_page
from .reading_pages import detect_reading_pages
from .registry import load_fields


def _render_page(doc, page_no, dpi):
    page = doc[page_no - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pix


def _dispatch(pdf, pages, fields_yaml, output_dir, dpi, psm, engine="v3"):
    """Route the document to the template it structurally matches.

    Template A (the CE monthly reports) uses the calibrated two-page crop
    pipeline with its backward-compatible reading-page pair.  Template B uses
    the whole-page strategy over the document's true Reading page(s).

    When engine=v4 the V4 multi-zone reconstruction is used; when engine=v3
    (or omitted) the legacy ``process_tb`` path runs.  Only one pipeline runs;
    the two are never merged.
    """
    from .template_routing import (
        TEMPLATE_B,
        detect_template,
        reading_pages_for,
    )

    out = Path(output_dir)
    crops_dir = out / "generated_crops"
    pages_dir = out / "pages"

    if out.exists():
        # Keep the root directory but remove stale generated data.
        shutil.rmtree(crops_dir, ignore_errors=True)
        shutil.rmtree(pages_dir, ignore_errors=True)

    crops_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    template = detect_template(pdf)

    if template == TEMPLATE_B:
        import json as _json

        reading_pages = reading_pages_for(pdf, TEMPLATE_B)
        fields = load_fields(fields_yaml)

        if engine == "v4":
            summary = _process_tb_v4(
                pdf, fields, output_dir, reading_pages, dpi=dpi, psm=psm,
            )
            strategy = "template-b-whole-page-v4"
        else:
            from .template_b import process_tb
            summary = process_tb(
                pdf, fields, output_dir, reading_pages, dpi=dpi, psm=psm,
            )
            strategy = "template-b-whole-page"

        # Persist the routing decision so callers/audit can see which template
        # and Reading pages a job actually used.  Informational only.
        template_path = out / "template_routing.json"
        template_path.write_text(_json.dumps({
            "template": template,
            "reading_pages": list(reading_pages),
            "strategy": strategy,
            "engine": engine,
        }, indent=2), encoding="utf-8")
        return summary

    # ---- Template A: existing calibrated two-page crop pipeline ----
    import json as _json

    reading_pages = detect_reading_pages(pdf)

    doc = pymupdf.open(str(pdf))
    try:
        for slot, page_no in enumerate(reading_pages):
            pix = _render_page(doc, page_no, dpi)
            page_path = pages_dir / f"reading_{slot + 1}_page_{page_no}.png"
            pix.save(str(page_path))

            # Convert the rendered page into PIL for calibrated cropping.
            from io import BytesIO

            from PIL import Image
            pil = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

            for region, x0, y0, x1, y1 in boxes_for_page(slot, pil.width, pil.height):
                crop = pil.crop((x0, y0, x1, y1))
                crop.save(str(crops_dir / f"{region}.png"))
    finally:
        doc.close()

    # Persist the routing decision (informational only).
    template_path = out / "template_routing.json"
    template_path.write_text(_json.dumps({
        "template": "A",
        "reading_pages": list(reading_pages),
        "strategy": "template-a-calibrated-crops",
    }, indent=2), encoding="utf-8")

    return process_crops(
        crops_dir,
        fields_yaml,
        out,
        psm=psm,
        reading_pages=tuple(reading_pages),
        engine=engine,
    )


def process(pdf, pages, fields_yaml, output_dir, dpi=300, psm=11, engine="v3"):
    """Automatic PDF pipeline: detect template, route, OCR and export."""
    return _dispatch(pdf, pages, fields_yaml, output_dir, dpi, psm, engine)


def _process_tb_v4(pdf, fields, output_dir, reading_pages, dpi=300, psm=11):
    """V4 whole-page multi-zone extraction for Template B.

    Renders each reading page as a full-page image, OCRs it, runs the V4
    multi-zone reconstruction, and emits standard export artefacts.
    """
    import json as _json
    from io import BytesIO
    from pathlib import Path

    import pymupdf
    import pytesseract
    from PIL import Image
    from pytesseract import Output as TessOutput

    from .export import export
    from .table_reconstruction import reconstruct_zones
    from .v4_adapter import (
        SOURCE_ABSENT,
        SOURCE_BLANK,
        aggregate_field_records_across_pages,
        build_field_records_canonical,
    )
    from .validate import validate

    out = Path(output_dir)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf))
    try:
        all_records: list = []
        all_page_records: list = []  # per-page FieldRecord lists
        for page_no in reading_pages:
            page = doc[page_no - 1]
            scale = dpi / 72.0
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), alpha=False,
            )
            pil = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

            # Persist the rendered page for audit.
            page_path = pages_dir / f"reading_page_{page_no}.png"
            pil.save(str(page_path))

            # Whole-page OCR at the requested PSM.
            data = pytesseract.image_to_data(
                pil,
                lang="eng",
                config=f"--oem 1 --psm {psm}",
                output_type=TessOutput.DICT,
                timeout=60,
            )

            tokens: list[dict] = []
            for i, t in enumerate(data["text"]):
                text = (t or "").strip()
                conf = float(data["conf"][i])
                if text and conf >= 0:
                    tokens.append({
                        "text": text,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                        "conf": conf,
                        "page": page_no,
                        "region": "reading",
                    })

            # V4 multi-zone reconstruction.
            result, zone_results = reconstruct_zones(tokens)

            # Zone-aware canonical aggregation: process each zone
            # independently with its own column model.
            page_records = build_field_records_canonical(
                zone_results, fields, page=page_no,
            )
            all_page_records.append(page_records)
    finally:
        doc.close()

    # Cross-page canonical aggregation: merge per-page FieldRecords into
    # exactly one record per schema field across ALL reading pages.
    all_records = aggregate_field_records_across_pages(all_page_records, fields)

    # Validate and export (same pipeline as V3/Template-A).
    issues = validate(all_records)
    detected_source = [
        r.field_id for r in all_records
        if r.found
        and SOURCE_ABSENT not in r.flags
        and SOURCE_BLANK not in r.flags
    ]
    populated = [
        r.field_id for r in all_records
        if any(
            c.value is not None
            and SOURCE_ABSENT not in c.flags
            and SOURCE_BLANK not in c.flags
            for c in r.cells
        )
    ]

    meta = {
        "source": "whole-page Reading extraction (Template B V4)",
        "template": "PERFORMANCE_EVALUATION",
        "reading_pages": list(reading_pages),
        "crop_count": 0,
        "detected_source_fields": len(detected_source),
        "populated_fields": len(populated),
        "registered_fields": len(all_records),
        "parser_version": "4.0.0-template-b",
        "engine": "v4",
    }
    summary = export(all_records, issues, out, meta)
    summary["detected_source_fields"] = len(detected_source)
    summary["populated_fields"] = len(populated)
    summary["registered_fields"] = len(all_records)

    # Persist template.json for audit / web layer parity with V3.
    _json_path = out / "template.json"
    _json_path.write_text(_json.dumps({
        "template": "PERFORMANCE_EVALUATION",
        "reading_pages": list(reading_pages),
        "detected_source_fields": detected_source,
        "populated_fields": populated,
        "registered_fields": len(all_records),
    }, indent=2), encoding="utf-8")

    return summary
