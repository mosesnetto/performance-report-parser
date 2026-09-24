import json
from pathlib import Path

import pandas as pd


def export(records, issues, out_dir, meta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    inv = []
    cell_rows = []
    field_rows = []

    for r in records:
        inv.append({
            "id": r.field_id,
            "field": r.field_name,
            "page": r.page,
            "region": r.region,
            "unit": r.unit,
            "found": r.found,
            "cell_count": len(r.cells),
            "row_confidence": r.row_confidence,
            "flags": ";".join(r.flags),
        })

        for c in r.cells:
            cell_rows.append(c.to_dict())

        field_rows.append(r.to_dict())

    (out / "field_inventory.json").write_text(
        json.dumps(inv, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "measurements.json").write_text(
        json.dumps(cell_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "report.json").write_text(
        json.dumps({
            "metadata": meta,
            "fields": field_rows,
            "validation": issues,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "validation.json").write_text(
        json.dumps(issues, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(inv).to_csv(
        out / "field_inventory.csv",
        index=False,
    )

    pd.DataFrame(cell_rows).to_csv(
        out / "measurements.csv",
        index=False,
    )

    with pd.ExcelWriter(
        out / "report.xlsx",
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(inv).to_excel(
            writer,
            sheet_name="FieldInventory",
            index=False,
        )

        pd.DataFrame(cell_rows).to_excel(
            writer,
            sheet_name="Measurements",
            index=False,
        )

        pd.DataFrame(issues).to_excel(
            writer,
            sheet_name="Validation",
            index=False,
        )

        if cell_rows:
            df = pd.DataFrame(cell_rows)
            wide = df.pivot_table(
                index=[
                    "page",
                    "region",
                    "field_id",
                    "field_name",
                    "unit",
                ],
                columns="column",
                values="value",
                aggfunc="first",
            ).reset_index()
        else:
            wide = pd.DataFrame()

        wide.to_excel(
            writer,
            sheet_name="Wide",
            index=False,
        )

    severities = [
        x.get("severity", "warning")
        for x in issues
    ]

    summary = {
        "expected_fields": len(records),
        "fields_found": sum(
            1 for r in records if r.found
        ),
        "fields_missing": sum(
            1 for r in records if not r.found
        ),
        "measurement_cells": sum(
            len(r.cells) for r in records
        ),
        "validation_issues": len(issues),
        "errors": severities.count("error"),
        "warnings": severities.count("warning"),
        "info": severities.count("info"),
        "empty_source_cells": sum(
            1 for i in issues
            if i.get("type") in {
                "empty_source_cell",
                "indicator_only_or_blank",
            }
        ),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
