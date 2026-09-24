"""
Phase 2 — Registry Manager (manual field/tag registry + automatic discovery tracking).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_FILE = BASE_DIR / "registry.json"

DEFAULT_REGISTRY = {
    "fields": {},
    "tags": {},
    "meta": {"created_at": "", "updated_at": ""}
}


def _load_raw() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            pass
    return DEFAULT_REGISTRY.copy()


def _save_raw(data: Dict[str, Any]) -> None:
    data.setdefault("meta", DEFAULT_REGISTRY["meta"])
    data["meta"]["updated_at"] = datetime.now().isoformat()
    REGISTRY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def get_registry() -> Dict[str, Any]:
    return _load_raw()


def save_registry(data: Dict[str, Any]) -> None:
    _save_raw(data)


def add_field(
    canonical_name: str,
    aliases: List[str] = None,
    unit: str = "",
    region: str = "",
    description: str = "",
    tag: str = "",
    field_id: str = "",
    status: str = "MANUAL",
) -> Dict[str, Any]:
    data = get_registry()
    key = canonical_name.lower().strip()
    entry = data["fields"].get(key, {
        "canonical_name": canonical_name,
        "aliases": list(set(aliases or [])),
        "unit": unit,
        "region": region,
        "description": description,
        "tags": [tag] if tag else [],
        "field_id": field_id,
        "status": status,
        "discovered_in": [],
        "manual": True,
        "created_at": datetime.now().isoformat(),
    })
    if entry.get("manual") is None:
        entry["manual"] = True
    entry["canonical_name"] = canonical_name
    entry["unit"] = unit
    entry["region"] = region
    entry["description"] = description
    entry["tags"] = list(set(entry.get("tags", []) + ([tag] if tag else [])))
    entry["field_id"] = field_id or entry.get("field_id", "")
    entry["status"] = status
    entry["aliases"] = list(set(entry.get("aliases", []) + (aliases or [])))
    data["fields"][key] = entry
    save_registry(data)
    return entry


def remove_field(canonical_name: str) -> bool:
    data = get_registry()
    key = canonical_name.lower().strip()
    if key in data["fields"]:
        del data["fields"][key]
        save_registry(data)
        return True
    return False


def add_tag(
    canonical_tag: str,
    aliases: List[str] = None,
    description: str = "",
    status: str = "MANUAL",
) -> Dict[str, Any]:
    data = get_registry()
    key = canonical_tag.lower().strip()
    entry = data["tags"].get(key, {
        "canonical_tag": canonical_tag,
        "aliases": list(set(aliases or [])),
        "description": description,
        "status": status,
        "manual": True,
        "created_at": datetime.now().isoformat(),
    })
    entry.update({
        "canonical_tag": canonical_tag,
        "aliases": list(set(entry.get("aliases", []) + (aliases or []))),
        "description": description,
        "status": status,
        "manual": True,
        "updated_at": datetime.now().isoformat(),
    })
    data["tags"][key] = entry
    save_registry(data)
    return entry


def remove_tag(canonical_tag: str) -> bool:
    data = get_registry()
    key = canonical_tag.lower().strip()
    if key in data["tags"]:
        del data["tags"][key]
        save_registry(data)
        return True
    return False


def discover_from_validator(union_catalog: Dict[str, Any]) -> None:
    """Auto-register discovered fields from validator union catalog."""
    data = get_registry()
    if "fields" not in data:
        data["fields"] = {}
    for field in union_catalog.get("fields", []):
        name = field.get("field_name", "").split(" | ")[0].strip()
        if not name:
            continue
        key = name.lower().strip()
        if key not in data["fields"]:
            data["fields"][key] = {
                "canonical_name": name,
                "aliases": field.get("aliases", []),
                "unit": (field.get("units", [""])[0] if isinstance(field.get("units"), list) and len(field.get("units", [])) > 0 else field.get("unit", "")),                "region": ", ".join(field.get("regions", [])),
                "tags": [],
                "field_id": field.get("field_ids", [""])[0] if isinstance(field.get("field_ids"), list) else "",
                "status": "DISCOVERED",
                "manual": False,
                "discovered_in": [f.get("pdf_name") for f in field.get("source_observations", []) if f.get("pdf_name")],
                "created_at": datetime.now().isoformat(),
            }
        else:
            existing = data["fields"][key]
            if existing.get("status") == "MANUAL" and field.get("field_name"):
                pass  # Keep manual entry
            else:
                existing["aliases"] = list(set(existing.get("aliases", []) + field.get("aliases", [])))
                existing["discovered_in"] = list(set(
                    existing.get("discovered_in", []) +
                    [f.get("pdf_name") for f in field.get("source_observations", []) if f.get("pdf_name")]
                ))
                if existing.get("status") == "MANUAL":
                    existing["status"] = "MANUAL"
                else:
                    existing["status"] = "DISCOVERED"
                existing["updated_at"] = datetime.now().isoformat()
    save_registry(data)
