from pathlib import Path

from .export import export
from .ocr import run_ocr
from .registry import load_fields
from .table_reconstruction import reconstruct
from .v4_adapter import build_field_records
from .validate import validate

CROP_ORDER = [
    (0, "general", "general.png"),
    (0, "power_speed", "power_speed.png"),
    (0, "electronic_control", "electronic_control.png"),
    (0, "cylinder_pressure", "cylinder_pressure.png"),
    (0, "turbocharger", "turbocharger.png"),
    (0, "scavenge_air", "scavenge_air.png"),
    (0, "exhaust_gas", "exhaust_gas.png"),
    (1, "fuel_oil", "fuel_oil.png"),
    (1, "cylinder_lubrication", "cylinder_lubrication.png"),
    (1, "cylinder_condition", "cylinder_condition.png"),
    (1, "crankcase", "crankcase.png"),
    (1, "liner_wall", "liner_wall.png"),
    (1, "tc_sac_media", "tc_sac_media.png"),
    (1, "engine_media_plant", "engine_media_plant.png"),
]


def process(crops_dir, fields_yaml, output_dir, psm=11, reading_pages=(13, 14)):
    if len(reading_pages) == 1:
        page_no_for_slot = lambda slot: reading_pages[0]
    else:
        page_no_for_slot = lambda slot: reading_pages[slot]
    crops = Path(crops_dir)
    out = Path(output_dir)
    ocr_dir = out / "ocr"
    fields = load_fields(fields_yaml)
    records = []

    for slot, region, generated_name in CROP_ORDER:
        path = crops / generated_name
        if not path.exists():
            raise FileNotFoundError(f"Missing crop: {path}")

        page_no = page_no_for_slot(slot)
        region_fields = [
            f for f in fields
            if (f.get("region") or "") == region
        ]
        if not region_fields:
            continue

        # Single OCR pass; V4 decides geometry and value provenance directly
        # from token positions. No field-specific reruns or expected-value repair.
        suffix = f"_psm{psm}"
        words = run_ocr(
            path,
            page_no,
            region,
            out_json=ocr_dir / f"{region}{suffix}.json",
            out_tsv=ocr_dir / f"{region}{suffix}.tsv",
            psm=psm,
            preprocess="none",
        )

        token_dicts = [w.to_dict() for w in words]
        result = reconstruct(token_dicts)
        records.extend(
            build_field_records(result, region_fields, page_no, region)
        )

    issues = validate(records)
    return export(
        records,
        issues,
        out,
        {
            "source": "calibrated Reading-region crops (V4 geometry)",
            "reading_pages": list(reading_pages),
            "crop_count": len(CROP_ORDER),
            "parser_version": "4.0.0",
            "engine": "v4",
        },
    )
