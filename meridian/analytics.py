"""
Phase 1 — Normalized Analytics Storage (read-only query foundation).

Reuses the existing validator structures (validate_fidelity.py):
  UnionCatalog, UnionField, SourceCell, SourceHeader, PDFValidationResult,
  ValidationSummary.

Does NOT duplicate storage; provides a thin query layer over the
existing validation_output/ artifacts (validation_summary.json,
validation_summary.md, validation_union.json, per-PDF JSON inventories).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse existing validator structures directly

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation_output"


def load_union_catalog() -> Optional[Dict[str, Any]]:
    """Load the validator union catalog if it exists."""
    path = VALIDATION_DIR / "validation_union.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_validation_summary() -> Optional[Dict[str, Any]]:
    """Load the validator summary."""
    path = VALIDATION_DIR / "validation_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def normalized_observations_from_catalog(union_catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert validator union catalog fields into normalized analytics observations.

    Note: The union catalog carries aggregated metadata; actual source observations
    live in validation_summary.json per_pdf_results[].source_inventory[].
    Phase 1 uses both sources.
    """
    observations = []
    for field in union_catalog.get("fields", []):
        observations.append({
            "canonical_field": field.get("field_name", ""),
            "field_id": field.get("field_id", ""),
            "aliases": field.get("aliases", []),
            "unit": field.get("units", [""])[0] if isinstance(field.get("units"), list) and field.get("units") else "",
            "regions": field.get("regions", []),
            "header_structures": field.get("header_structures", []),
            "table_identities": field.get("table_identities", []),
            "source_reports": [],
            "status": "PRESENT" if field.get("field_ids") else "SOURCE_ABSENT",
        })
    # Enrich with actual observations from validation_summary per-pdf inventories
    summary = load_validation_summary()
    if summary and observations:
        for pdf_result in summary.get("per_pdf_results", []):
            pdf_name = pdf_result.get("pdf_name", "")
            for cell in pdf_result.get("source_inventory", []):
                canonical = cell.get("field_name", "").split(" | ")[0].strip()
                # Find matching observation
                for obs in observations:
                    if obs["canonical_field"] == canonical or canonical in obs.get("aliases", []):
                        obs.setdefault("source_reports", []).append({
                            "pdf_name": pdf_name,
                            "region": cell.get("region", ""),
                            "page": cell.get("page", 0),
                            "value": cell.get("value"),
                            "raw_text": cell.get("raw_text", ""),
                            "source_status": cell.get("source_status", "SOURCE_PRESENT"),
                            "unit": cell.get("unit", ""),
                            "physical_row": cell.get("physical_row", 0),
                            "physical_col_index": cell.get("physical_col_index", 0),
                        })
                        # Deduplicate reports per observation
                        {r.get("pdf_name") for r in obs.get("source_reports", []) if r.get("pdf_name")}
                        # Keep unique by (pdf, region)
                        unique_reports = []
                        seen_set = set()
                        for r in obs.get("source_reports", []):
                            key = (r.get("pdf_name"), r.get("region"), r.get("physical_row"))
                            if key not in seen_set:
                                seen_set.add(key)
                                unique_reports.append(r)
                        obs["source_reports"] = unique_reports
            # If a field in union has no observations in any PDF inventory, mark absent
        present_canonicals = {
            cell.get("field_name", "").split(" | ")[0].strip()
            for pdf_result in summary.get("per_pdf_results", [])
            for cell in pdf_result.get("source_inventory", [])
        }
        for obs in observations:
            if obs["canonical_field"] not in present_canonicals and obs["canonical_field"]:
                # Only override if currently PRESENT but no observations found
                if obs["status"] == "PRESENT" and not obs.get("source_reports"):
                    obs["status"] = "SOURCE_ABSENT"
    return observations


def get_analytics_registry() -> Dict[str, Any]:
    """Build the normalized analytics registry (fields + observations)."""
    catalog = load_union_catalog()
    load_validation_summary()
    if not catalog:
        return {"fields": [], "tags": [], "reports": [], "observations": []}
    observations = normalized_observations_from_catalog(catalog)
    # Minimal tag derivation from aliases/header_structures (Phase 1 only)
    tags: Dict[str, List[str]] = {}
    for obs in observations:
        canonical = obs["canonical_field"]
        for alias in obs.get("aliases", []):
            tags.setdefault(alias, []).append(canonical)
    # Report presence derived from observations
    reports_set = set()
    for obs in observations:
        for rpt in obs.get("source_reports", []):
            if rpt.get("pdf_name"):
                reports_set.add(rpt["pdf_name"])
    return {
        "fields": observations,
        "tags": {"derived_tags": tags},
        "reports_present": sorted(reports_set),
        "union_catalog_loaded": True,
        "source_status_map": {
            "PRESENT": "SOURCE_PRESENT",
            "SOURCE_BLANK": "SOURCE_BLANK",
            "SOURCE_ABSENT": "SOURCE_ABSENT",
            "SOURCE_UNREADABLE": "SOURCE_UNREADABLE",
            "SOURCE_AMBIGUOUS": "SOURCE_AMBIGUOUS",
        },
    }


def find_field_in_registry(field_name: str) -> Optional[Dict[str, Any]]:
    registry = get_analytics_registry()
    for f in registry.get("fields", []):
        if f["canonical_field"] == field_name:
            return f
        for alias in f.get("aliases", []):
            if alias == field_name:
                return f
    return None


def find_reports_for_field(field_name: str) -> List[str]:
    result = find_field_in_registry(field_name)
    if not result:
        return []
    return sorted({
        rpt.get("pdf_name", "")
        for rpt in result.get("source_reports", [])
        if rpt.get("pdf_name")
    })


def find_absent_reports_for_field(field_name: str) -> List[str]:
    # Minimal Phase 1: compare against all PDFs in pdfs_test
    all_pdfs = [p.name for p in (BASE_DIR / "pdfs_test").glob("*.pdf")]
    present = set(find_reports_for_field(field_name))
    return sorted([p for p in all_pdfs if p not in present])
