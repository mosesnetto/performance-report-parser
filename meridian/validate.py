from __future__ import annotations

TEXT_ONLY_FIELDS = {
    "Remarks",
    "Source of analysis",
    "Running mode",
    "Governor mode",
    "ENGINE state",
    "Voy. / title",
    "Record by",
    "Ackn. by",
    "AUX. BLOWER operation",
}


def validate(records):
    """Classify extraction conditions instead of counting every warning as an error."""
    issues = []

    for r in records:
        if not r.found:
            issues.append({
                "severity": "error",
                "type": "missing_field_row",
                "page": r.page,
                "region": r.region,
                "field": r.field_name,
            })
            continue

        if not r.cells:
            if r.field_name in TEXT_ONLY_FIELDS:
                continue
            kind = "indicator_only_or_blank" if "indicator_only_or_blank" in r.flags else "empty_source_cell"
            issues.append({
                "severity": "info",
                "type": kind,
                "page": r.page,
                "region": r.region,
                "field": r.field_name,
            })

        for c in r.cells:
            if "column_unresolved" in c.flags:
                issues.append({
                    "severity": "info",
                    "type": "column_unresolved",
                    "page": c.page,
                    "region": c.region,
                    "field": c.field_name,
                    "column": c.column,
                    "value": c.value,
                })

            if "duplicate_column_assignment" in c.flags:
                issues.append({
                    "severity": "error",
                    "type": "duplicate_column_assignment",
                    "page": c.page,
                    "region": c.region,
                    "field": c.field_name,
                    "column": c.column,
                    "value": c.value,
                })

            if c.confidence < 0.60:
                issues.append({
                    "severity": "warning",
                    "type": "low_ocr_confidence",
                    "page": c.page,
                    "region": c.region,
                    "field": c.field_name,
                    "column": c.column,
                    "confidence": c.confidence,
                })

    return issues
