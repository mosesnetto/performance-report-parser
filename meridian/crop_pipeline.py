from pathlib import Path

from .export import export
from .extract import extract_field
from .headers import detect_headers
from .layout import cluster_rows
from .ocr import run_ocr
from .registry import build_alias_index, load_fields
from .validate import validate

CROP_ORDER = [
    (0, "general", "general.png", "Screenshot 2026-08-25 102213.png"),
    (0, "power_speed", "power_speed.png", "Screenshot 2026-08-25 102224.png"),
    (0, "electronic_control", "electronic_control.png", "Screenshot 2026-08-25 102236.png"),
    (0, "cylinder_pressure", "cylinder_pressure.png", "Screenshot 2026-08-25 102248.png"),
    (0, "turbocharger", "turbocharger.png", "Screenshot 2026-08-25 102259.png"),
    (0, "scavenge_air", "scavenge_air.png", "Screenshot 2026-08-25 102308.png"),
    (0, "exhaust_gas", "exhaust_gas.png", "Screenshot 2026-08-25 102320.png"),
    (1, "fuel_oil", "fuel_oil.png", "Screenshot 2026-08-25 102338.png"),
    (1, "cylinder_lubrication", "cylinder_lubrication.png", "Screenshot 2026-08-25 102349.png"),
    (1, "cylinder_condition", "cylinder_condition.png", "Screenshot 2026-08-25 102400.png"),
    (1, "crankcase", "crankcase.png", "Screenshot 2026-08-25 102410.png"),
    (1, "liner_wall", "liner_wall.png", "Screenshot 2026-08-25 102423.png"),
    (1, "tc_sac_media", "tc_sac_media.png", "Screenshot 2026-08-25 102436.png"),
    (1, "engine_media_plant", "engine_media_plant.png", "Screenshot 2026-08-25 102447.png"),
]


def _postprocess_record(record):
    """Return extractor evidence unchanged.

    Field extraction owns row boundaries, column identity, and OCR ambiguity.
    A post-extraction stage must never infer decimals/signs, choose an expected
    value, or discard a token based on a field id or numeric literal.
    """
    return record


def _record_quality(record):
    values = [
        c.value for c in record.cells
        if c.value is not None and isinstance(c.value, (int, float))
    ]
    score = 0.0
    if record.found:
        score += 10.0
    score += min(len(record.cells), 12) * 2.0
    if values:
        score += sum(
            1.0 for c in record.cells
            if c.confidence >= 0.60
        )
    if "field_row_not_found" in record.flags:
        score -= 10.0
    if not record.cells:
        score -= 4.0

    # Exact expected width is a strong signal for columnar tables.
    from .extract import COLUMN_SCHEMAS
    expected = len(COLUMN_SCHEMAS.get(record.field_id, []))
    if expected and len(record.cells) == expected:
        score += 12.0
    elif expected and len(record.cells) > expected:
        score -= min(8.0, (len(record.cells) - expected) * 2.0)

    return score

def process(crops_dir, fields_yaml, output_dir, psm=11, reading_pages=(13, 14), engine="v3"):
    if engine == "v4":
        from .v4_pipeline import process as process_v4
        return process_v4(crops_dir, fields_yaml, output_dir, psm=psm, reading_pages=reading_pages)
    if len(reading_pages) == 1:
        page_no_for_slot = lambda slot: reading_pages[0]
    else:
        page_no_for_slot = lambda slot: reading_pages[slot]
    crops = Path(crops_dir)
    out = Path(output_dir)
    ocr_dir = out / "ocr"
    fields = load_fields(fields_yaml)
    field_index = build_alias_index(fields)
    records = []

    # Fast first pass. Only difficult/empty fields trigger additional OCR modes.
    fallback_modes_by_region = {
        # Grayscale/contrast OCR is especially effective on the yellow
        # input cells used by this report. It runs only for weak fields.
        "general": [(11, "gray")],
        "power_speed": [(11, "gray")],
        "fuel_oil": [(11, "gray")],
        "cylinder_lubrication": [(11, "gray")],
        "engine_media_plant": [(11, "gray")],
        "tc_sac_media": [(11, "gray")],
    }

    for slot, region, generated_name, manual_name in CROP_ORDER:
        path = crops / generated_name
        if not path.exists():
            path = crops / manual_name
        if not path.exists():
            raise FileNotFoundError(f"Missing crop: {path}")

        page_no = page_no_for_slot(slot)
        region_fields = [f for f in fields if f["region"] == region]

        def run_mode(mode, preprocess="none"):
            suffix = f"_psm{mode}" if preprocess == "none" else f"_psm{mode}_{preprocess}"
            words = run_ocr(
                path,
                page_no,
                region,
                out_json=ocr_dir / f"{region}{suffix}.json",
                out_tsv=ocr_dir / f"{region}{suffix}.tsv",
                psm=mode,
                preprocess=preprocess,
            )
            rows = cluster_rows(words)
            headers = detect_headers(rows)
            return rows, headers

        rows, headers = run_mode(psm)
        primary = {}
        for field in region_fields:
            rec = extract_field(
                field, rows, region_fields, field_index, headers, page_no, region
            )
            rec = _postprocess_record(rec)
            primary[field["id"]] = rec

        # Only rerun alternate OCR modes for fields that are actually weak.
        needs_fallback = {
            fid for fid, rec in primary.items()
            if (not rec.cells)
            or ("field_row_not_found" in rec.flags)
            or ("no_value_cells" in rec.flags)
            or ("header_not_found" in rec.flags and len(rec.cells) > 0)
            or any("column_unresolved" in c.flags for c in rec.cells)
        }

        best = dict(primary)

        for mode_spec in fallback_modes_by_region.get(region, []):
            if not needs_fallback:
                break
            if isinstance(mode_spec, tuple):
                mode, preprocess = mode_spec
            else:
                mode, preprocess = mode_spec, "none"
            alt_rows, alt_headers = run_mode(mode, preprocess)
            next_needs = set()
            for field in region_fields:
                fid = field["id"]
                if fid not in needs_fallback:
                    continue
                alt = _postprocess_record(
                    extract_field(
                        field, alt_rows, region_fields, field_index, alt_headers, page_no, region
                    )
                )
                current_score = _record_quality(best[fid])
                alt_score = _record_quality(alt)
                if alt_score > current_score:
                    best[fid] = alt
                rec = best[fid]
                if (not rec.cells) or ("field_row_not_found" in rec.flags) or ("no_value_cells" in rec.flags):
                    next_needs.add(fid)
            needs_fallback = next_needs

        for field in region_fields:
            records.append(best[field["id"]])

    issues = validate(records)
    return export(
        records,
        issues,
        out,
        {
            "source": "calibrated Reading-region crops",
            "reading_pages": list(reading_pages),
            "crop_count": len(CROP_ORDER),
            "parser_version": "3.3.0",
        },
    )
