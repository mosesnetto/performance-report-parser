"""
Phase 6 — Trigger Engine (data-driven triggers over analytics data).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
TRIGGERS_FILE = BASE_DIR / "triggers.json"

DEFAULT_TRIGGERS = {
    "triggers": [],
    "meta": {"updated_at": datetime.now().isoformat()}
}


def _load_triggers() -> Dict[str, Any]:
    if TRIGGERS_FILE.exists():
        try:
            data = json.loads(TRIGGERS_FILE.read_text())
            if "triggers" in data:
                return data
        except Exception:
            pass
    return DEFAULT_TRIGGERS.copy()


def _save_triggers(data: Dict[str, Any]) -> None:
    data.setdefault("meta", DEFAULT_TRIGGERS["meta"])
    data["meta"]["updated_at"] = datetime.now().isoformat()
    TRIGGERS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def add_trigger(
    name: str,
    description: str = "",
    field: str = "",
    tag: str = "",
    operator: str = ">",
    threshold: float = 0.0,
    grouping: str = "",
    scope: str = "new_reports",
    active: bool = True,
) -> Dict[str, Any]:
    data = _load_triggers()
    trigger = {
        "trigger_id": f"TR{len(data['triggers'])+1:04d}",
        "name": name,
        "description": description,
        "field": field,
        "tag": tag,
        "operator": operator,
        "threshold": threshold,
        "grouping": grouping,
        "scope": scope,
        "active": active,
        "created_at": datetime.now().isoformat(),
        "last_evaluated": None,
        "last_result": None,
        "last_fired": None,
    }
    data["triggers"].append(trigger)
    _save_triggers(data)
    return trigger


def remove_trigger(trigger_id: str) -> bool:
    data = _load_triggers()
    original_len = len(data["triggers"])
    data["triggers"] = [t for t in data["triggers"] if t.get("trigger_id") != trigger_id]
    if len(data["triggers"]) < original_len:
        _save_triggers(data)
        return True
    return False


def evaluate_triggers() -> List[Dict[str, Any]]:
    from meridian.query_tools import get_field_observations
    data = _load_triggers()
    results: List[Dict[str, Any]] = []
    for trigger in data.get("triggers", []):
        if not trigger.get("active"):
            continue
        field_name = trigger.get("field", "")
        observations = get_field_observations(field_name) if field_name else []
        values = [o.get("value") for o in observations if o.get("value") is not None]
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass
        fired = False
        result_detail = {"trigger_id": trigger.get("trigger_id"), "field": field_name}
        if trigger.get("operator") == ">":
            fired = any(v > trigger.get("threshold", 0) for v in numeric_values)
        elif trigger.get("operator") == "<":
            fired = any(v < trigger.get("threshold", 0) for v in numeric_values)
        elif trigger.get("operator") == ">=":
            fired = any(v >= trigger.get("threshold", 0) for v in numeric_values)
        elif trigger.get("operator") == "<=":
            fired = any(v <= trigger.get("threshold", 0) for v in numeric_values)
        elif trigger.get("operator") == "==":
            fired = any(v == trigger.get("threshold", 0) for v in numeric_values)
        else:
            # Default: check if value exceeds threshold
            fired = any(v > trigger.get("threshold", 0) for v in numeric_values)
        trigger["last_evaluated"] = datetime.now().isoformat()
        trigger["last_result"] = {"fired": fired, "max": max(numeric_values) if numeric_values else None, "count": len(numeric_values)}
        trigger["last_fired"] = datetime.now().isoformat() if fired else trigger.get("last_fired")
        result_detail["fired"] = fired
        result_detail["threshold"] = trigger.get("threshold")
        result_detail["value_summary"] = trigger.get("last_result")
        results.append(result_detail)
    _save_triggers(data)
    return results
