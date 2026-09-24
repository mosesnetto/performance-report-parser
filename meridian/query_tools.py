"""
Phase 3 — Deterministic Query Tools.

Read-only. Uses existing analytics registry (app/analytics) and validator
output (validation_output/). No LLM integration yet (Phase 4).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation_output"


def _load_summary() -> Optional[Dict[str, Any]]:
    path = VALIDATION_DIR / "validation_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def list_reports() -> List[str]:
    """List available source PDF filenames."""
    import glob
    return sorted([Path(p).name for p in glob.glob(str(BASE_DIR / "pdfs_test" / "*.pdf"))])


def search_fields(query: str = "") -> List[Dict[str, Any]]:
    """Search registry fields by canonical name or alias."""
    from meridian.analytics import get_analytics_registry
    registry = get_analytics_registry()
    results = []
    q = (query or "").lower()
    for f in registry.get("fields", []):
        canonical = (f.get("canonical_field") or "").lower()
        aliases = [a.lower() for a in f.get("aliases", [])]
        if not q or q in canonical or any(q in a for a in aliases):
            results.append(f)
    return results


def search_tags(query: str = "") -> List[Dict[str, Any]]:
    """Search tags."""
    from meridian.registry_manager import get_registry
    registry = get_registry()
    results = []
    q = (query or "").lower()
    for key, tag in registry.get("tags", {}).items():
        if not q or q in (tag.get("canonical_tag") or "").lower() or any(q in a.lower() for a in tag.get("aliases", [])):
            results.append({"key": key, **tag})
    return results


def find_present_reports(field_name: str) -> List[str]:
    """Reports where a field has observations."""
    from meridian.analytics import find_reports_for_field
    return find_reports_for_field(field_name)


def find_absent_reports(field_name: str) -> List[str]:
    """Reports where a field is not observed."""
    from meridian.analytics import find_absent_reports_for_field
    return find_absent_reports_for_field(field_name)


def get_field_observations(field_name: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Get observations for a field from validator summary."""
    filters = filters or {}
    summary = _load_summary()
    observations = []
    if not summary:
        return observations
    # Use registry for proper canonical matching instead of splitting incorrectly
    from meridian.analytics import find_field_in_registry
    registry_entry = find_field_in_registry(field_name)
    canonical = field_name.lower().strip()
    for pdf_result in summary.get("per_pdf_results", []):
        pdf_name = pdf_result.get("pdf_name", "")
        for cell in pdf_result.get("source_inventory", []):
            cell_full = cell.get("field_name", "")
            cell_raw = cell.get("raw_text", "")
            # Robust matching: check canonical name appears in full description or raw text
            # and if registry entry exists, also match by aliases/region
            match = False
            if registry_entry:
                reg_name = registry_entry.get("canonical_field", "").lower()
                reg_aliases = [a.lower() for a in registry_entry.get("aliases", [])]
                match = reg_name in cell_full.lower() or reg_name in cell_raw.lower() or any(a in cell_full.lower() or a in cell_raw.lower() for a in reg_aliases)
            else:
                # Fallback: direct substring match
                match = canonical in cell_full.lower() or canonical in cell_raw.lower()
            if filters.get("partial"):
                match = match or (canonical in cell_full.lower() or canonical in cell_raw.lower())
            if match:
                observations.append({
                    "pdf_name": pdf_name,
                    "region": cell.get("region", ""),
                    "page": cell.get("page", 0),
                    "value": cell.get("value"),
                    "raw_text": cell.get("raw_text", ""),
                    "unit": cell.get("unit", ""),
                    "source_status": cell.get("source_status", "SOURCE_PRESENT"),
                    "physical_row": cell.get("physical_row", 0),
                    "physical_col_index": cell.get("physical_col_index", 0),
                })
    return observations


def _numeric_values(observations: List[Dict[str, Any]]) -> List[float]:
    values = []
    for obs in observations:
        val = obs.get("value")
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                pass
    return values


def find_max(field_name: str, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    observations = get_field_observations(field_name, filters or {})
    values = _numeric_values(observations)
    if not values:
        return None
    max_val = max(values)
    # Find the observation(s) with max value
    max_obs = [obs for obs in observations if obs.get("value") is not None and float(obs.get("value")) == max_val]
    return {"field": field_name, "max": max_val, "sources": max_obs}


def find_min(field_name: str, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    observations = get_field_observations(field_name, filters or {})
    values = _numeric_values(observations)
    if not values:
        return None
    min_val = min(values)
    min_obs = [obs for obs in observations if obs.get("value") is not None and float(obs.get("value")) == min_val]
    return {"field": field_name, "min": min_val, "sources": min_obs}


def find_average(field_name: str, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    observations = get_field_observations(field_name, filters or {})
    values = _numeric_values(observations)
    if not values:
        return None
    avg = mean(values)
    return {"field": field_name, "average": avg, "count": len(values), "sources": observations[:5]}


def compare_reports(report_a: str, report_b: str, field_name: str) -> Dict[str, Any]:
    get_field_observations(field_name, {"pdf_filter": [report_a]})
    # Simplified comparison using all observations per report
    values_a = [o.get("value") for o in get_field_observations(field_name) if o.get("pdf_name") == report_a and o.get("value") is not None]
    values_b = [o.get("value") for o in get_field_observations(field_name) if o.get("pdf_name") == report_b and o.get("value") is not None]
    return {
        "report_a": report_a,
        "report_b": report_b,
        "field": field_name,
        "values_a": values_a,
        "values_b": values_b,
        "comparison": "present in both" if values_a and values_b else ("absent in a" if not values_a else "absent in b"),
    }


def aggregate_field(field_name: str, operation: str = "mean", filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    observations = get_field_observations(field_name, filters or {})
    values = _numeric_values(observations)
    if not values:
        return None
    result = {"field": field_name, "operation": operation, "count": len(values)}
    if operation == "mean" or operation == "average":
        result["value"] = mean(values)
    elif operation == "max":
        result["value"] = max(values)
    elif operation == "min":
        result["value"] = min(values)
    elif operation == "sum":
        result["value"] = sum(values)
    else:
        result["value"] = mean(values)
    return result


def get_cell_provenance(observation_id: str) -> Optional[Dict[str, Any]]:
    # Minimal provenance lookup by matching pdf_name + field_name
    observations = get_field_observations(observation_id)
    if observations:
        return observations[0]
    return None
